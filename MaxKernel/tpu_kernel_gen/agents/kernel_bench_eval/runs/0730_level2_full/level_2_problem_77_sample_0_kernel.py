# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl
from jax.experimental.pallas import atomic_add

# Initialization
batch_size = 16
in_channels = 64
out_channels = 32
depth, height, width = 16, 32, 32
kernel_size = 3
scale_factor = 2.0
eps = 1e-5
momentum = 0.1
key = random.PRNGKey(0)
key_x, key_conv, key_bn = random.split(key, 3)
x = random.normal(key_x, (batch_size, depth, height, width, in_channels))
conv_transpose = nn.ConvTranspose(features=out_channels, kernel_size=(kernel_size, kernel_size, kernel_size))
conv_params = conv_transpose.init(key_conv, x)["params"]
batch_norm = nn.BatchNorm(use_running_average=False, epsilon=eps, momentum=momentum)
dummy_conv_out = conv_transpose.apply({"params": conv_params}, x)
bn_variables = batch_norm.init(key_bn, dummy_conv_out)


# Computation
def kernel(
  x_ref,
  kernel_ref,
  conv_bias_ref,
  bn_scale_ref,
  bn_bias_ref,
  mean_ref,
  var_ref,
  out_ref,
  updated_mean_ref,
  updated_var_ref,
):
  """
  Pallas kernel for a sequence of ConvTranspose, scaling, BatchNorm, and pooling.

  NOTE: This implementation adapts the BatchNorm operation to fit the Pallas
  model. True BatchNorm requires statistics across the entire batch, which would
  need communication across the grid. Instead, this kernel:
  1. Normalizes each data instance using its own statistics (like InstanceNorm).
  2. Atomically accumulates these per-instance statistics to update the
     global running mean and variance.
  """
  # Constants derived from the source code
  scale_factor = 2.0
  eps = 1e-5
  momentum = 0.1

  # 1. 3D Transposed Convolution
  # The dimension numbers are set for a 5D tensor in (N, D, H, W, C) format.
  # We assume standard strides and padding for a faithful translation.
  dn = ("NDHWC", "DHWIO", "NDHWC")
  conv_out = jax.lax.conv_transpose(
    x_ref[...],
    kernel_ref[...],
    strides=(1, 1, 1),
    padding="SAME",
    dimension_numbers=dn,
  )
  # Add the convolution bias, reshaping for correct broadcasting.
  conv_out += conv_bias_ref[...].reshape(1, 1, 1, 1, -1)

  # 2. Element-wise Scaling
  scaled_out = conv_out * scale_factor

  # 3. Per-Instance Statistics Calculation
  # We calculate the mean and variance over the spatial dimensions (D, H, W)
  # for the single instance handled by this program.
  instance_mean = jnp.mean(scaled_out, axis=(1, 2, 3), keepdims=True)
  instance_var = jnp.var(scaled_out, axis=(1, 2, 3), keepdims=True)

  # 4. Per-Instance Normalization
  normalized_out = (scaled_out - instance_mean) / jnp.sqrt(instance_var + eps)
  bn_out = normalized_out * bn_scale_ref[...].reshape(1, 1, 1, 1, -1) + bn_bias_ref[...].reshape(1, 1, 1, 1, -1)

  # 5. Update Running Statistics
  # This section uses atomic operations to safely update the shared running
  # statistics from multiple programs.

  # Step 5a: Initialize accumulators for stats (program 0 only).
  # The `out_specs` for updated_mean/var map to the same memory for all programs.
  def _init_body():
    updated_mean_ref[...] = jnp.zeros_like(mean_ref[...])
    updated_var_ref[...] = jnp.zeros_like(var_ref[...])

  pl.when(pl.program_id(axis=0) == 0)(_init_body)
  pl.barrier()  # Ensure initialization is visible to all programs.

  # Step 5b: Atomically accumulate the per-instance stats.
  atomic_add(updated_mean_ref, (), instance_mean.squeeze(axis=(0, 1, 2, 3)))
  atomic_add(updated_var_ref, (), instance_var.squeeze(axis=(0, 1, 2, 3)))
  pl.barrier()  # Ensure all programs have completed their additions.

  # Step 5c: Finalize the running stats update (program 0 only).
  def _finalize_body():
    # Average the accumulated stats across the entire batch.
    avg_instance_mean = updated_mean_ref[...] / batch_size
    avg_instance_var = updated_var_ref[...] / batch_size

    # Apply the momentum update rule.
    final_mean = (1.0 - momentum) * mean_ref[...] + momentum * avg_instance_mean
    final_var = (1.0 - momentum) * var_ref[...] + momentum * avg_instance_var

    # Write the final results to the output buffers.
    updated_mean_ref[...] = final_mean
    updated_var_ref[...] = final_var

  pl.when(pl.program_id(axis=0) == 0)(_finalize_body)

  # 6. Global Average Pooling
  # The mean is taken over the spatial dimensions of the normalized output.
  pooled_out = jnp.mean(bn_out, axis=(1, 2, 3), keepdims=True)
  out_ref[...] = pooled_out


x, updated_mean, updated_var = pl.pallas_call(
  kernel,
  out_shape=(
    jax.ShapeDtypeStruct((batch_size, 1, 1, 1, out_channels), x.dtype),
    jax.ShapeDtypeStruct(bn_variables["batch_stats"]["mean"].shape, bn_variables["batch_stats"]["mean"].dtype),
    jax.ShapeDtypeStruct(bn_variables["batch_stats"]["var"].shape, bn_variables["batch_stats"]["var"].dtype),
  ),
  grid=(batch_size,),
  in_specs=[
    pl.BlockSpec(block_shape=(1, depth, height, width, in_channels), index_map=lambda i: (i, 0, 0, 0, 0)),
    pl.BlockSpec(block_shape=conv_params["kernel"].shape, index_map=lambda i: (0,) * conv_params["kernel"].ndim),
    pl.BlockSpec(block_shape=conv_params["bias"].shape, index_map=lambda i: (0,) * conv_params["bias"].ndim),
    pl.BlockSpec(
      block_shape=bn_variables["params"]["scale"].shape,
      index_map=lambda i: (0,) * bn_variables["params"]["scale"].ndim,
    ),
    pl.BlockSpec(
      block_shape=bn_variables["params"]["bias"].shape,
      index_map=lambda i: (0,) * bn_variables["params"]["bias"].ndim,
    ),
    pl.BlockSpec(
      block_shape=bn_variables["batch_stats"]["mean"].shape,
      index_map=lambda i: (0,) * bn_variables["batch_stats"]["mean"].ndim,
    ),
    pl.BlockSpec(
      block_shape=bn_variables["batch_stats"]["var"].shape,
      index_map=lambda i: (0,) * bn_variables["batch_stats"]["var"].ndim,
    ),
  ],
  out_specs=[
    pl.BlockSpec(block_shape=(1, 1, 1, 1, out_channels), index_map=lambda i: (i, 0, 0, 0, 0)),
    pl.BlockSpec(
      block_shape=bn_variables["batch_stats"]["mean"].shape,
      index_map=lambda i: (0,) * bn_variables["batch_stats"]["mean"].ndim,
    ),
    pl.BlockSpec(
      block_shape=bn_variables["batch_stats"]["var"].shape,
      index_map=lambda i: (0,) * bn_variables["batch_stats"]["var"].ndim,
    ),
  ],
)(
  x,
  conv_params["kernel"],
  conv_params["bias"],
  bn_variables["params"]["scale"],
  bn_variables["params"]["bias"],
  bn_variables["batch_stats"]["mean"],
  bn_variables["batch_stats"]["var"],
).block_until_ready()

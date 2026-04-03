# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Initialization
batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
eps = 1e-5
momentum = 0.1

key = random.PRNGKey(0)
key_x, key_conv, key_bn = random.split(key, 3)

# Use channels-first data format (N, C, H, W) for TPU compatibility
x = random.normal(key_x, (batch_size, in_channels, height, width))

conv = nn.Conv(features=out_channels, kernel_size=(kernel_size, kernel_size), padding="SAME")
# Flax Conv expects NHWC, so we create a dummy input with that layout for init
dummy_x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
conv_params = conv.init(key_conv, dummy_x_nhwc)["params"]
# Kernel has HWIO layout: (3, 3, 3, 16). Transpose to OIHW for NCHW conv.
conv_kernel_transposed = jnp.transpose(conv_params["kernel"], (3, 2, 0, 1))

# BatchNorm input shape must match the output shape of the Conv layer
conv_output_shape = (batch_size, out_channels, height, width)
# For NCHW, batch norm acts on axis 1 (the channel axis)
bn = nn.BatchNorm(use_running_average=False, momentum=momentum, epsilon=eps, axis=1)
bn_vars = bn.init(key_bn, jnp.ones(conv_output_shape))


# Computation
def kernel(
  x_ref,
  conv_kernel_ref,
  conv_bias_ref,
  bn_scale_ref,
  bn_bias_ref,
  bn_mean_ref,
  bn_var_ref,
  x_out_ref,
  new_mean_ref,
  new_var_ref,
):
  """
  Pallas kernel for a fused Conv-Mish-BatchNorm operation.

  This kernel processes a single output channel of the computation. It is
  parallelized over the output channels.

  Args:
    (Refs are for NCHW data layout)
  """
  k = pl.program_id(0)
  # Constants from the original JAX code
  eps = 1e-5
  momentum = 0.1

  # 1. Convolution
  # Use the Pallas TPU-specific convolution primitive.
  conv_out = pltpu.conv(
    x_ref[...],
    conv_kernel_ref[...],
    window_strides=(1, 1),
    padding="SAME",
    dimension_numbers=("NCHW", "OIHW", "NCHW"),
  )
  # Add the per-channel bias. Bias is a full vector, index with k.
  conv_out += conv_bias_ref[k].astype(conv_out.dtype).reshape(1, 1, 1, 1)

  # 2. Mish Activation
  activated_out = jax.nn.mish(conv_out)

  # 3. Batch Normalization
  # Since use_running_average=False, we compute statistics over the current batch.
  # The reduction is over the batch, height, and width axes (0, 2, 3) for NCHW.
  batch_mean = jnp.mean(activated_out, axis=(0, 2, 3))
  batch_var = jnp.var(activated_out, axis=(0, 2, 3))

  # Normalize the output using the batch statistics.
  normalized_out = (activated_out - batch_mean.reshape(1, 1, 1, 1)) / jnp.sqrt(batch_var.reshape(1, 1, 1, 1) + eps)

  # Apply the learned scale and bias parameters.
  bn_out = normalized_out * bn_scale_ref[k].astype(conv_out.dtype).reshape(1, 1, 1, 1) + bn_bias_ref[k].astype(
    conv_out.dtype
  ).reshape(1, 1, 1, 1)

  # Update the running statistics using the momentum.
  # bn_mean_ref and bn_var_ref hold the *old* running averages. We index with k.
  # The result of jnp.mean/var is a (1,)-shaped array. We need a scalar for the update.
  updated_mean = (1 - momentum) * bn_mean_ref[k] + momentum * batch_mean[0]
  updated_var = (1 - momentum) * bn_var_ref[k] + momentum * batch_var[0]

  # Write the final results to the output references.
  x_out_ref[...] = bn_out
  new_mean_ref[k] = updated_mean
  new_var_ref[k] = updated_var


# The original computation involves multiple inputs (x, conv_params, bn_vars)
# and produces multiple outputs (the final transformed x and the updated bn_vars).
# We need to flatten these structures into a list of arrays for the pallas_call.

# Extract individual arrays from the nested parameter structures
# Use the transposed kernel
conv_kernel = conv_kernel_transposed
conv_bias = conv_params["bias"]
bn_scale = bn_vars["params"]["scale"]
bn_bias = bn_vars["params"]["bias"]
bn_mean = bn_vars["batch_stats"]["mean"]
bn_var = bn_vars["batch_stats"]["var"]

# The pallas_call returns a tuple of output arrays.
# The first output is the transformed 'x'.
# The next two are the updated batch normalization statistics.
x_out, new_mean, new_var = pl.pallas_call(
  kernel,
  # Define the shapes and dtypes of the outputs
  out_shape=[
    jax.ShapeDtypeStruct(conv_output_shape, x.dtype),
    jax.ShapeDtypeStruct(bn_mean.shape, bn_mean.dtype),
    jax.ShapeDtypeStruct(bn_var.shape, bn_var.dtype),
  ],
  # Grid: Parallelize over the output channels
  grid=(out_channels,),
  # Input specs: Define how each input array is sliced for the kernel
  in_specs=[
    # x: Pass the full input tensor to each kernel instance
    pl.BlockSpec(x.shape, lambda k: (0, 0, 0, 0)),
    # conv_kernel: OIHW layout. Each instance gets one output channel filter (1, I, H, W)
    pl.BlockSpec(block_shape=(1, in_channels, kernel_size, kernel_size), index_map=lambda k: (k, 0, 0, 0)),
    # 1D vectors: Pass the full vector to each kernel. The kernel will index it.
    pl.BlockSpec(conv_bias.shape, lambda k: (0,)),
    pl.BlockSpec(bn_scale.shape, lambda k: (0,)),
    pl.BlockSpec(bn_bias.shape, lambda k: (0,)),
    pl.BlockSpec(bn_mean.shape, lambda k: (0,)),
    pl.BlockSpec(bn_var.shape, lambda k: (0,)),
  ],
  # Output specs: Define where each kernel instance writes its results
  out_specs=[
    # x_out: NCHW layout. Each instance writes to its output channel slice (N, 1, H, W)
    pl.BlockSpec(block_shape=(batch_size, 1, height, width), index_map=lambda k: (0, k, 0, 0)),
    # 1D vectors: The full vector is an output. Each kernel writes to its element.
    pl.BlockSpec(bn_mean.shape, lambda k: (0,)),
    pl.BlockSpec(bn_var.shape, lambda k: (0,)),
  ],
)(x, conv_kernel, conv_bias, bn_scale, bn_bias, bn_mean, bn_var)

# After the call, we would typically reconstruct the updated_bn_vars dictionary
# from the output arrays new_mean and new_var.
x_out.block_until_ready()
new_mean.block_until_ready()
new_var.block_until_ready()

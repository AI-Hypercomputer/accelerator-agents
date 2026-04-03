# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_features = 1024
out_features = 512
scale_shape = (out_features,)
eps = 1e-5
momentum = 0.1

key = random.PRNGKey(0)
key, x_key, gemm_key, scale_key, bn_key = random.split(key, 5)

x = random.normal(x_key, (batch_size, in_features))

gemm = nn.Dense(features=out_features)
gemm_params = gemm.init(gemm_key, x)["params"]

scale = random.normal(scale_key, scale_shape)

# BatchNorm needs an input of the correct shape to initialize
bn_init_x = jnp.ones((batch_size, out_features))
bn = nn.BatchNorm(use_running_average=False, momentum=momentum, epsilon=eps)
bn_variables = bn.init(bn_key, bn_init_x)

block_n = 128


# Computation
def kernel(
  x_ref,
  gemm_kernel_ref,
  gemm_bias_ref,
  scale_ref,
  bn_scale_ref,
  bn_bias_ref,
  running_mean_ref,
  running_var_ref,
  # Outputs
  out_ref,
  updated_mean_ref,
  updated_var_ref,
):
  """Pallas kernel for a fused GEMM, element-wise scale, and BatchNorm.

  Args:
    x_ref: Input data.
    gemm_kernel_ref: Weights for the dense layer.
    gemm_bias_ref: Bias for the dense layer.
    scale_ref: Element-wise scaling factor.
    bn_scale_ref: Scale parameter (gamma) for BatchNorm.
    bn_bias_ref: Bias parameter (beta) for BatchNorm.
    running_mean_ref: Input running mean for BatchNorm.
    running_var_ref: Input running variance for BatchNorm.
    out_ref: Output buffer for the final transformed data.
    updated_mean_ref: Output buffer for the updated running mean.
    updated_var_ref: Output buffer for the updated running variance.
  """
  # Constants from the source computation
  eps = 1e-5
  momentum = 0.1

  # Get the program ID to manually slice the 1D arrays
  j = pl.program_id(axis=0)
  n_start = j * block_n

  # Manually slice the 1D input arrays using lax.dynamic_slice
  gemm_bias_slice = jax.lax.dynamic_slice(gemm_bias_ref[...], (n_start,), (block_n,))
  scale_slice = jax.lax.dynamic_slice(scale_ref[...], (n_start,), (block_n,))
  bn_scale_slice = jax.lax.dynamic_slice(bn_scale_ref[...], (n_start,), (block_n,))
  bn_bias_slice = jax.lax.dynamic_slice(bn_bias_ref[...], (n_start,), (block_n,))
  running_mean_slice = jax.lax.dynamic_slice(running_mean_ref[...], (n_start,), (block_n,))
  running_var_slice = jax.lax.dynamic_slice(running_var_ref[...], (n_start,), (block_n,))

  # 1. Apply GEMM (Dense layer)
  y = x_ref[...] @ gemm_kernel_ref[...] + gemm_bias_slice

  # 2. Apply element-wise scaling
  y = y * scale_slice

  # 3. Apply BatchNorm
  batch_mean = jnp.mean(y, axis=0)
  batch_var = jnp.var(y, axis=0)

  # Update the running mean and variance
  new_running_mean = running_mean_slice * momentum + batch_mean * (1.0 - momentum)
  new_running_var = running_var_slice * momentum + batch_var * (1.0 - momentum)

  # Normalize the batch
  inv_stddev = jax.lax.rsqrt(batch_var + eps)
  normalized_y = (y - batch_mean) * inv_stddev

  # Apply scale and shift (gamma and beta)
  scaled_y = normalized_y * bn_scale_slice + bn_bias_slice

  # 4. Write results to output buffers
  out_ref[...] = scaled_y
  updated_mean_ref[...] = new_running_mean
  updated_var_ref[...] = new_running_var


# The pallas_call will produce three outputs: the final transformed x,
# the updated running mean, and the updated running variance.
out_shapes = (
  jax.ShapeDtypeStruct((batch_size, out_features), x.dtype),  # Final x
  jax.ShapeDtypeStruct((out_features,), x.dtype),  # Updated mean
  jax.ShapeDtypeStruct((out_features,), x.dtype),  # Updated var
)

x, updated_mean, updated_var = pl.pallas_call(
  kernel,
  out_shape=out_shapes,
  grid=(out_features // block_n,),
  in_specs=[
    # Input x: (batch_size, in_features). Read by all kernels.
    pl.BlockSpec(block_shape=(batch_size, in_features), index_map=lambda j: (0, 0)),
    # GEMM weights: (in_features, out_features). Sliced along out_features.
    pl.BlockSpec(block_shape=(in_features, block_n), index_map=lambda j: (0, j * block_n)),
    # 1D vectors are passed in full to each kernel.
    pl.BlockSpec(block_shape=(out_features,), index_map=lambda j: (0,)),
    pl.BlockSpec(block_shape=(out_features,), index_map=lambda j: (0,)),
    pl.BlockSpec(block_shape=(out_features,), index_map=lambda j: (0,)),
    pl.BlockSpec(block_shape=(out_features,), index_map=lambda j: (0,)),
    pl.BlockSpec(block_shape=(out_features,), index_map=lambda j: (0,)),
    pl.BlockSpec(block_shape=(out_features,), index_map=lambda j: (0,)),
  ],
  out_specs=(
    # Final output x: (batch_size, out_features). Each kernel writes a slice.
    pl.BlockSpec(block_shape=(batch_size, block_n), index_map=lambda j: (0, j * block_n)),
    # Updated 1D vectors are written to slice by slice.
    pl.BlockSpec(block_shape=(block_n,), index_map=lambda j: (j * block_n,)),
    pl.BlockSpec(block_shape=(block_n,), index_map=lambda j: (j * block_n,)),
  ),
)(
  x,
  gemm_params["kernel"],
  gemm_params["bias"],
  scale,
  bn_variables["params"]["scale"],
  bn_variables["params"]["bias"],
  bn_variables["batch_stats"]["mean"],
  bn_variables["batch_stats"]["var"],
).block_until_ready()

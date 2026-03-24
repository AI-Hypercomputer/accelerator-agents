# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_features = 512
out_features = 256
num_groups = 8
bias_shape = (1, out_features, 1, 1)
b_batch = 8

key = random.PRNGKey(0)
key, x_key, bias_key, gemm_key, gn_key = random.split(key, 5)

x = random.normal(x_key, (batch_size, in_features))
gemm = nn.Dense(features=out_features)
gemm_params = gemm.init(gemm_key, x)["params"]

group_norm = nn.GroupNorm(num_groups=num_groups)
# Flax GroupNorm expects channels-last format (NHWC)
dummy_gn_input = jnp.ones((batch_size, 1, 1, out_features))
gn_params = group_norm.init(gn_key, dummy_gn_input)["params"]

# Squeeze the 4D GroupNorm params to 1D. The compiler expects 1D arrays
# for these parameters, as indicated by previous errors.
gn_scale_1d = jnp.squeeze(gn_params["scale"])
gn_bias_1d = jnp.squeeze(gn_params["bias"])

# Create a reduction matrix to perform group-wise summations via matmul.
# This avoids unsupported reshape operations inside the kernel.
group_size = out_features // num_groups
reduction_matrix = jnp.zeros((out_features, num_groups), dtype=x.dtype)
for i in range(num_groups):
  reduction_matrix = reduction_matrix.at[i * group_size : (i + 1) * group_size, i].set(1.0)


bias = random.normal(bias_key, bias_shape)


# Computation
def kernel(
  x_ref,
  gemm_kernel_ref,
  gemm_bias_ref,
  gn_scale_ref,
  gn_bias_ref,
  reduction_matrix_ref,
  bias_ref,
  out_ref,
):
  # Constants for GroupNorm
  num_groups = 8
  epsilon = 1e-5
  out_features = gemm_bias_ref.shape[0]
  group_size = out_features // num_groups

  # 1. Dense layer (GEMM + bias)
  y = x_ref[...] @ gemm_kernel_ref[...]
  y = y + gemm_bias_ref[...]  # y shape: (b_batch, out_features)

  # 2. Group Normalization using matrix multiplication
  # This avoids unsupported reshape operations inside the kernel.
  # sum_y shape: (b_batch, num_groups)
  sum_y = y @ reduction_matrix_ref[...]
  # sum_y_sq shape: (b_batch, num_groups)
  sum_y_sq = (y * y) @ reduction_matrix_ref[...]

  mean = sum_y / group_size
  var = (sum_y_sq / group_size) - (mean * mean)

  # Broadcast mean and variance back to original shape using the transpose
  # of the reduction matrix.
  # mean_bcast shape: (b_batch, out_features)
  mean_bcast = mean @ reduction_matrix_ref[...].T
  var_bcast = var @ reduction_matrix_ref[...].T

  y_normalized = (y - mean_bcast) / jnp.sqrt(var_bcast + epsilon)

  # Apply learned scale and bias for GroupNorm
  # gn_scale_ref and gn_bias_ref are 1D with shape (out_features,)
  z = y_normalized * gn_scale_ref[...] + gn_bias_ref[...]

  # 3. Min reduction over the channel axis
  z_min = jnp.min(z, axis=1, keepdims=True)  # shape: (b_batch, 1)

  # 4. Reshape for final bias addition
  # This reshape adds singleton dimensions, which is generally supported.
  z_min_reshaped = z_min.reshape(z_min.shape[0], 1, 1, 1)

  # 5. Final bias addition
  # bias_ref has shape (1, out_features, 1, 1)
  # Broadcasting results in a shape of (b_batch, out_features, 1, 1)
  final_result = z_min_reshaped + bias_ref[...]

  # 6. Write the final result to the output reference
  out_ref[...] = final_result


x = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, out_features, 1, 1), x.dtype),
  grid=(batch_size // b_batch,),
  in_specs=[
    pl.BlockSpec(block_shape=(b_batch, in_features), index_map=lambda i: (i * b_batch, 0)),
    pl.BlockSpec(block_shape=(in_features, out_features), index_map=lambda i: (0, 0)),
    pl.BlockSpec(block_shape=(out_features,), index_map=lambda i: (0,)),
    pl.BlockSpec(block_shape=(out_features,), index_map=lambda i: (0,)),
    pl.BlockSpec(block_shape=(out_features,), index_map=lambda i: (0,)),
    pl.BlockSpec(block_shape=(out_features, num_groups), index_map=lambda i: (0, 0)),
    pl.BlockSpec(block_shape=(1, out_features, 1, 1), index_map=lambda i: (0, 0, 0, 0)),
  ],
  out_specs=pl.BlockSpec(
    block_shape=(b_batch, out_features, 1, 1),
    index_map=lambda i: (i * b_batch, 0, 0, 0),
  ),
)(
  x,
  gemm_params["kernel"],
  gemm_params["bias"],
  gn_scale_1d,
  gn_bias_1d,
  reduction_matrix,
  bias,
).block_until_ready()

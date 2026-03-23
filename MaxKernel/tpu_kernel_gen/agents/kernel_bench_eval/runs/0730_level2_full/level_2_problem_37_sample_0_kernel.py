# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax import nn
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_features = 512
out_features = 1024
num_groups = 32
bias_shape = (out_features,)
b_size = 8

key = random.PRNGKey(0)
keys = random.split(key, 6)

x = random.normal(keys[0], (batch_size, in_features))
matmul_weight = random.normal(keys[1], (out_features, in_features))
matmul_bias = random.normal(keys[2], (out_features,))
bias = random.normal(keys[3], bias_shape)
group_norm_weight = random.normal(keys[4], (out_features,))
group_norm_bias = random.normal(keys[5], (out_features,))

features_per_group = out_features // num_groups
# Create a matrix to perform grouped sums via matrix multiplication.
# This avoids unsupported reshape or reduce_window operations inside the kernel.
summing_matrix = (jnp.arange(out_features)[:, None] // features_per_group) == jnp.arange(num_groups)[None, :]
summing_matrix = summing_matrix.astype(x.dtype)


# Computation
def kernel(
  x_ref,
  matmul_weight_ref,
  matmul_bias_ref,
  bias_ref,
  group_norm_weight_ref,
  group_norm_bias_ref,
  summing_matrix_ref,
  out_ref,
):
  # Define constants for the GroupNorm operation.
  eps = 1e-5
  features_per_group = out_features // num_groups
  num_groups = 32

  # Perform the linear transformation (matrix multiplication + bias).
  # x_ref is a block of shape (b_size, in_features).
  # matmul_weight_ref is the full weight matrix of shape (out_features, in_features).
  # The result `x` will have shape (b_size, out_features).
  x = jnp.dot(x_ref[...], matmul_weight_ref[...].T) + matmul_bias_ref[...]

  # Apply the SiLU activation function element-wise.
  x = nn.silu(x)

  # Add the residual bias.
  x = x + bias_ref[...]

  # Apply Group Normalization using the summing matrix.
  # Calculate sum and sum of squares per group via dot product.
  group_sums = jnp.dot(x, summing_matrix_ref[...])
  group_sums_sq = jnp.dot(x * x, summing_matrix_ref[...])

  # Calculate mean and variance.
  group_mean = group_sums / features_per_group
  group_var = group_sums_sq / features_per_group - group_mean**2

  # Expand mean and variance to match the original tensor shape.
  mean_expanded = jnp.dot(group_mean, summing_matrix_ref[...].T)
  var_expanded = jnp.dot(group_var, summing_matrix_ref[...].T)

  # Normalize the data.
  x_normalized = (x - mean_expanded) / jnp.sqrt(var_expanded + eps)

  # Apply the learnable scale and shift parameters for GroupNorm.
  output = x_normalized * group_norm_weight_ref[...] + group_norm_bias_ref[...]

  # Write the final result to the output reference.
  out_ref[...] = output


output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, out_features), x.dtype),
  grid=(batch_size // b_size,),
  in_specs=[
    pl.BlockSpec(block_shape=(b_size, in_features), index_map=lambda i: (i * b_size, 0)),
    pl.BlockSpec(block_shape=(out_features, in_features), index_map=lambda i: (0, 0)),
    pl.BlockSpec(block_shape=(out_features,), index_map=lambda i: (0,)),
    pl.BlockSpec(block_shape=(out_features,), index_map=lambda i: (0,)),
    pl.BlockSpec(block_shape=(out_features,), index_map=lambda i: (0,)),
    pl.BlockSpec(block_shape=(out_features,), index_map=lambda i: (0,)),
    pl.BlockSpec(block_shape=(out_features, num_groups), index_map=lambda i: (0, 0)),
  ],
  out_specs=pl.BlockSpec(block_shape=(b_size, out_features), index_map=lambda i: (i * b_size, 0)),
)(x, matmul_weight, matmul_bias, bias, group_norm_weight, group_norm_bias, summing_matrix).block_until_ready()

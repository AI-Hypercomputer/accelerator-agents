# Imports
import jax
import jax.nn as nn
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_features = 1024
out_features = 512
add_value_shape = (out_features,)

key = random.PRNGKey(0)
key, key_x, key_weight, key_bias, key_add = random.split(key, 5)

x = random.normal(key_x, (batch_size, in_features))
matmul_weight = random.normal(key_weight, (out_features, in_features))
matmul_bias = random.normal(key_bias, (out_features,))
add_value = random.normal(key_add, add_value_shape)


# Computation
def kernel(x_ref, matmul_weight_ref, matmul_bias_ref, add_value_ref, out_ref):
  # Get the block index for the output feature dimension.
  j = pl.program_id(axis=1)
  # The size of the block along the output feature dimension.
  out_block_size = out_ref.shape[1]

  # Calculate the slice for the bias and add_value vectors.
  bias_offset = j * out_block_size
  bias_block = jax.lax.dynamic_slice(matmul_bias_ref[...], (bias_offset,), (out_block_size,))

  add_value_offset = j * out_block_size
  add_value_block = jax.lax.dynamic_slice(add_value_ref[...], (add_value_offset,), (out_block_size,))

  # Perform the matrix multiplication and add the bias.
  out = jnp.dot(x_ref[...], matmul_weight_ref[...].T) + bias_block

  # Add the second value, broadcasted across the tile.
  out = out + add_value_block

  # Apply the sequence of activation functions and clipping.
  out = nn.sigmoid(out) * out
  out = nn.tanh(out)
  out = nn.gelu(out)
  out = jnp.clip(out, a_min=-1, a_max=1)

  # Write the final computed tile to the output buffer.
  out_ref[...] = out


x = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, out_features), jnp.float32),
  grid=(batch_size // 128, out_features // 128),
  in_specs=[
    pl.BlockSpec(block_shape=(128, in_features), index_map=lambda i, j: (i, 0)),
    pl.BlockSpec(block_shape=(128, in_features), index_map=lambda i, j: (j, 0)),
    pl.BlockSpec(block_shape=(out_features,), index_map=lambda i, j: (0,)),
    pl.BlockSpec(block_shape=(out_features,), index_map=lambda i, j: (0,)),
  ],
  out_specs=pl.BlockSpec(block_shape=(128, 128), index_map=lambda i, j: (i, j)),
)(x, matmul_weight, matmul_bias, add_value).block_until_ready()

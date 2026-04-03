# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
dim = 16384
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim))


def kernel(x_ref, out_ref):
  """
  Pallas kernel for element-wise tanh.
  """
  # Load the input block from SRAM into a register.
  x = x_ref[...]
  # Apply the tanh function element-wise.
  result = jnp.tanh(x)
  # Write the result block back to SRAM.
  out_ref[...] = result


# For an element-wise operation, we can parallelize across the data.
# We choose a block size for the second dimension that is compatible with TPU constraints.
# 128 is a common and valid choice.
b_dim = 128
# The block size for the first dimension must be divisible by 8.
b_batch = 8

# Computation
# The grid will be (batch_size // b_batch, dim // b_dim), so each kernel instance
# handles a (b_batch, b_dim) chunk of the data.
result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(batch_size // b_batch, dim // b_dim),
  # For element-wise ops, the input and output blocks map directly
  # from the grid indices.
  in_specs=[pl.BlockSpec(block_shape=(b_batch, b_dim), index_map=lambda i, j: (i, j))],
  out_specs=pl.BlockSpec(block_shape=(b_batch, b_dim), index_map=lambda i, j: (i, j)),
)(x).block_until_ready()

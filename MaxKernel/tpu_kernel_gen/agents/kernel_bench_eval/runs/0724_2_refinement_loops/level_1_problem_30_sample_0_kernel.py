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
# The block size for the second dimension must be divisible by 128 for TPU.
block_dim = 128


# Computation
def kernel(x_ref, y_ref):
  """
  Pallas kernel for the element-wise operation y = x / (1 + |x|).

  Args:
    x_ref: A reference to a block of the input array 'x'.
    y_ref: A reference to a block of the output array 'y' where the
           result will be stored.
  """
  # Load the input block from SRAM into registers.
  x = x_ref[...]
  # Perform the element-wise computation on the block.
  y = x / (1.0 + jnp.abs(x))
  # Write the resulting block to the output reference in SRAM.
  y_ref[...] = y


# For an element-wise operation, we can parallelize by chunking the input array.
# We will tile the array along its second dimension.
y = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  # The grid is 1D along the second dimension of the array, as each
  # block will cover the full first dimension.
  grid=(1, x.shape[1] // block_dim),
  # Each kernel instance receives a vertical slice of the input array `x`.
  in_specs=[pl.BlockSpec(block_shape=(x.shape[0], block_dim), index_map=lambda i, j: (i, j))],
  # The output `y` is chunked identically to the input.
  out_specs=pl.BlockSpec(block_shape=(x.shape[0], block_dim), index_map=lambda i, j: (i, j)),
)(x).block_until_ready()

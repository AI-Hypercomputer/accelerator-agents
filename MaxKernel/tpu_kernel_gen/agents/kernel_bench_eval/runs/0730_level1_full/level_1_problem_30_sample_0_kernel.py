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
block_dim = 128


# Computation
def kernel(x_ref, out_ref):
  """
  Pallas kernel to compute x / (1 + abs(x)) element-wise.
  """
  # Perform the computation on the block.
  # x_ref[...] loads the input data for the current block from SRAM into registers.
  # The element-wise computation is performed.
  # The result is written to the output reference (out_ref), which corresponds
  # to a block in the output tensor in SRAM.
  out_ref[...] = x_ref[...] / (1 + jnp.abs(x_ref[...]))


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(1, dim // block_dim),
  in_specs=[pl.BlockSpec(block_shape=(batch_size, block_dim), index_map=lambda i, j: (i, j))],
  out_specs=pl.BlockSpec(block_shape=(batch_size, block_dim), index_map=lambda i, j: (i, j)),
)(x).block_until_ready()

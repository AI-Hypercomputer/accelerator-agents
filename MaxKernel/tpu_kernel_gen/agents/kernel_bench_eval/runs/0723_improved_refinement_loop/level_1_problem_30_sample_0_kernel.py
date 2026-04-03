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
block_size = 1024


def kernel(x_ref, out_ref):
  """Pallas kernel to compute x / (1 + abs(x))."""
  # Perform the element-wise computation on the input block.
  # x_ref[...] loads the data from SRAM into registers.
  # The result is then written to the output buffer out_ref.
  out_ref[...] = x_ref[...] / (1 + jnp.abs(x_ref[...]))


# Computation
result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(1, x.shape[1] // block_size),
  in_specs=[pl.BlockSpec(block_shape=(x.shape[0], block_size), index_map=lambda i, j: (i, j))],
  out_specs=pl.BlockSpec(block_shape=(x.shape[0], block_size), index_map=lambda i, j: (i, j)),
)(x).block_until_ready()

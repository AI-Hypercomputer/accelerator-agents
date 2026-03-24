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
block_batch = 8
block_dim = 1024


# Computation
def kernel(x_ref, y_ref):
  """Pallas kernel for the element-wise operation y = x / (1 + abs(x))."""
  # Load the block of x from HBM into SRAM.
  x = x_ref[...]
  # Perform the computation.
  # Note the use of jnp.abs for the absolute value.
  y = x / (1.0 + jnp.abs(x))
  # Write the result back to the output buffer.
  y_ref[...] = y


y = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(batch_size // block_batch, dim // block_dim),
  in_specs=[pl.BlockSpec(block_shape=(block_batch, block_dim), index_map=lambda i, j: (i, j))],
  out_specs=pl.BlockSpec(block_shape=(block_batch, block_dim), index_map=lambda i, j: (i, j)),
)(x).block_until_ready()

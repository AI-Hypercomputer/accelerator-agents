# Imports
import jax
import jax.nn as nn
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
dim = 16384
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim))
block_dim = 2048
block_rows = 8


# Computation
def kernel(x_ref, y_ref):
  """Pallas kernel for element-wise hard_tanh activation."""
  # Load the input block from SRAM into registers.
  x = x_ref[...]
  # Apply the hard_tanh function, which is equivalent to jnp.clip(x, -1.0, 1.0).
  result = nn.hard_tanh(x)
  # Write the computed block to the output reference in SRAM.
  y_ref[...] = result


y = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(batch_size // block_rows, dim // block_dim),
  in_specs=[pl.BlockSpec(block_shape=(block_rows, block_dim), index_map=lambda i, j: (i * block_rows, j * block_dim))],
  out_specs=pl.BlockSpec(block_shape=(block_rows, block_dim), index_map=lambda i, j: (i * block_rows, j * block_dim)),
)(x).block_until_ready()

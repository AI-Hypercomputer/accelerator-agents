# Imports
import jax
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
dim = 16384
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim))
block_d = 128


# Computation
def kernel(x_ref, out_ref):
  """Pallas kernel for element-wise Swish (SiLU) activation."""
  # Load the input tile from SRAM into registers.
  x = x_ref[...]
  # Perform the element-wise computation: x * sigmoid(x).
  # The result is implicitly stored in registers.
  result = x * jax.nn.sigmoid(x)
  # Write the computed tile from registers back to the output buffer in SRAM.
  out_ref[...] = result


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(1, dim // block_d),
  in_specs=[pl.BlockSpec(block_shape=(batch_size, block_d), index_map=lambda i, j: (i, j))],
  out_specs=pl.BlockSpec(block_shape=(batch_size, block_d), index_map=lambda i, j: (i, j)),
)(x).block_until_ready()

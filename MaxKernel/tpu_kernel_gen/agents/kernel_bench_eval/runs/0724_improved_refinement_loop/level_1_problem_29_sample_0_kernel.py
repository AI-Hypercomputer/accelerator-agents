# Imports
import jax
import jax.nn
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
dim = 16384
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim))

BLOCK_B = 8
BLOCK_D = 128


def kernel(x_ref, out_ref):
  """Pallas kernel for element-wise softplus."""
  # Apply the softplus function to the input block and write to the output block.
  out_ref[...] = jax.nn.softplus(x_ref[...])


# Computation
result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(x.shape[0] // BLOCK_B, x.shape[1] // BLOCK_D),
  in_specs=[pl.BlockSpec(block_shape=(BLOCK_B, BLOCK_D), index_map=lambda i, j: (i, j))],
  out_specs=pl.BlockSpec(block_shape=(BLOCK_B, BLOCK_D), index_map=lambda i, j: (i, j)),
)(x).block_until_ready()

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
batch_block, dim_block = 8, 128


# Computation
def kernel(x_ref, out_ref):
  """Pallas kernel for element-wise ReLU."""
  # Apply the ReLU function to the input block and write to the output block.
  out_ref[...] = jax.nn.relu(x_ref[...])


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(x.shape[0] // batch_block, x.shape[1] // dim_block),
  in_specs=[pl.BlockSpec(block_shape=(batch_block, dim_block), index_map=lambda i, j: (i, j))],
  out_specs=pl.BlockSpec(block_shape=(batch_block, dim_block), index_map=lambda i, j: (i, j)),
)(x).block_until_ready()

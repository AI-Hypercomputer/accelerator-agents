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
block_dim = 128


# Computation
def kernel(x_ref, y_ref):
  """Pallas kernel for element-wise softplus."""
  # Load the input block into registers.
  x = x_ref[...]
  # Compute the softplus function.
  result = jax.nn.softplus(x)
  # Write the result to the output block.
  y_ref[...] = result


y = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(x.shape[1] // block_dim,),
  in_specs=[pl.BlockSpec(block_shape=(x.shape[0], block_dim), index_map=lambda i: (0, i))],
  out_specs=pl.BlockSpec(block_shape=(x.shape[0], block_dim), index_map=lambda i: (0, i)),
)(x).block_until_ready()

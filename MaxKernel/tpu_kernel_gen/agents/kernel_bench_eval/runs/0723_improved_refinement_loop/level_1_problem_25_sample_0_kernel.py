# Imports
import jax
import jax.random as random
from jax.experimental import pallas as pl
from jax.nn import sigmoid

# Initialization
batch_size = 16
dim = 16384
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim))
block_m = 8
block_n = 1024


# Computation
def kernel(x_ref, out_ref):
  x = x_ref[...]
  out_ref[...] = x * sigmoid(x)


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(x.shape[0] // block_m, x.shape[1] // block_n),
  in_specs=[pl.BlockSpec(block_shape=(block_m, block_n), index_map=lambda i, j: (i, j))],
  out_specs=pl.BlockSpec(block_shape=(block_m, block_n), index_map=lambda i, j: (i, j)),
)(x).block_until_ready()

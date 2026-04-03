# Imports
import jax
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
dim = 16384
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim))
BLOCK_DIM = 128


# Computation
def kernel(x_ref, out_ref):
  # The softplus function is log(exp(x) + 1).
  # We apply it element-wise to the input block.
  # Using jax.nn.softplus is preferred for numerical stability.
  out_ref[...] = jax.nn.softplus(x_ref[...])


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(x.shape[1] // BLOCK_DIM,),
  in_specs=[pl.BlockSpec(block_shape=(x.shape[0], BLOCK_DIM), index_map=lambda i: (0, i))],
  out_specs=pl.BlockSpec(block_shape=(x.shape[0], BLOCK_DIM), index_map=lambda i: (0, i)),
)(x).block_until_ready()

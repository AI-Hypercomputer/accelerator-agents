# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
dim = 16384
block_b = 8
block_d = 128
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim))


# Computation
def kernel(x_ref, out_ref):
  out_ref[...] = jnp.maximum(x_ref[...], 0)


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(batch_size // block_b, dim // block_d),
  in_specs=[pl.BlockSpec(block_shape=(block_b, block_d), index_map=lambda i, j: (i, j))],
  out_specs=pl.BlockSpec(block_shape=(block_b, block_d), index_map=lambda i, j: (i, j)),
)(x).block_until_ready()

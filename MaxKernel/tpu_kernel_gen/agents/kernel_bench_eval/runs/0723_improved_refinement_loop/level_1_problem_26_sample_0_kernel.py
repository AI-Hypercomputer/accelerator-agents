# Imports
import jax
import jax.nn
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
dim = 16384
d_block = 128
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim))


def kernel(x_ref, out_ref):
  out_ref[...] = jax.nn.gelu(x_ref[...])


# Computation
result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(dim // d_block,),
  in_specs=[pl.BlockSpec(block_shape=(batch_size, d_block), index_map=lambda i: (0, i))],
  out_specs=pl.BlockSpec(block_shape=(batch_size, d_block), index_map=lambda i: (0, i)),
)(x).block_until_ready()

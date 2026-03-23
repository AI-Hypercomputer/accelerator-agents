# Imports
import jax
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
dim = 16384
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim))

block_batch = 8
block_dim = 128


def kernel(x_ref, out_ref):
  out_ref[...] = jax.nn.gelu(x_ref[...])


# Computation
result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(batch_size // block_batch, dim // block_dim),
  in_specs=[pl.BlockSpec(block_shape=(block_batch, block_dim), index_map=lambda i, j: (i, j))],
  out_specs=pl.BlockSpec(block_shape=(block_batch, block_dim), index_map=lambda i, j: (i, j)),
)(x).block_until_ready()

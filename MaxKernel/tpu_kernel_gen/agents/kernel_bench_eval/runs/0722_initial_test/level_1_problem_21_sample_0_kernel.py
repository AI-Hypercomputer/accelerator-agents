# Imports
import jax
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
dim = 16384
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim))

b_batch = 8
b_dim = 128


def kernel(x_ref, out_ref):
  out_ref[...] = jax.nn.sigmoid(x_ref[...])


# Computation
result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(x.shape[0] // b_batch, x.shape[1] // b_dim),
  in_specs=[pl.BlockSpec(block_shape=(b_batch, b_dim), index_map=lambda i, j: (i, j))],
  out_specs=pl.BlockSpec(block_shape=(b_batch, b_dim), index_map=lambda i, j: (i, j)),
)(x).block_until_ready()

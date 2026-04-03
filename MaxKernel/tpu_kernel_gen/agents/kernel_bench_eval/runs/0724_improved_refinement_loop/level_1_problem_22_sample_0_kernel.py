# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
dim = 16384
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim))


def kernel(x_ref, out_ref):
  out_ref[...] = jnp.tanh(x_ref[...])


# Computation
result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(x.shape[0] // 8, x.shape[1] // 128),
  in_specs=[pl.BlockSpec(block_shape=(8, 128), index_map=lambda i, j: (i, j))],
  out_specs=pl.BlockSpec(block_shape=(8, 128), index_map=lambda i, j: (i, j)),
)(x).block_until_ready()

# Imports
import functools

import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
alpha = 1.0
batch_size = 16
dim = 16384
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim))
block_d = 128
block_b = 8


# Computation
def kernel(x_ref, out_ref, alpha):
  x = x_ref[...]
  out_ref[...] = jnp.where(x > 0, x, alpha * (jnp.exp(x) - 1))


result = pl.pallas_call(
  functools.partial(kernel, alpha=alpha),
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(batch_size // block_b, dim // block_d),
  in_specs=[pl.BlockSpec(block_shape=(block_b, block_d), index_map=lambda i, j: (i, j))],
  out_specs=pl.BlockSpec(block_shape=(block_b, block_d), index_map=lambda i, j: (i, j)),
)(x).block_until_ready()

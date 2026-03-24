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
block_d = 128


def kernel(x_ref, out_ref):
  out_ref[...] = jnp.tanh(x_ref[...])


# Computation
result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(x.shape[1] // block_d,),
  in_specs=[pl.BlockSpec(block_shape=(x.shape[0], block_d), index_map=lambda i: (0, i))],
  out_specs=pl.BlockSpec(block_shape=(x.shape[0], block_d), index_map=lambda i: (0, i)),
)(x).block_until_ready()

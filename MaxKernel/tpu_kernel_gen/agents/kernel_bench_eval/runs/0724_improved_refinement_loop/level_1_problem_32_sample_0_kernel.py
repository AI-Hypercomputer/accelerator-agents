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
block_dim = 2048


# Computation
def kernel(x_ref, out_ref):
  """Pallas kernel for hard_tanh."""
  x = x_ref[...]
  out_ref[...] = jnp.maximum(-1.0, jnp.minimum(1.0, x))


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(1, x.shape[1] // block_dim),
  in_specs=[pl.BlockSpec(block_shape=(x.shape[0], block_dim), index_map=lambda i, j: (i, j))],
  out_specs=pl.BlockSpec(block_shape=(x.shape[0], block_dim), index_map=lambda i, j: (i, j)),
)(x).block_until_ready()

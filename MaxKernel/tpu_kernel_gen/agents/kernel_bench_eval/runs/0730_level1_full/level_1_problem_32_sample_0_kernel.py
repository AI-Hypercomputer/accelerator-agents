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
b_dim = 128


# Computation
def kernel(x_ref, out_ref):
  """Pallas kernel for element-wise hard_tanh."""
  # The hard_tanh function is equivalent to clipping the input between -1 and 1.
  out_ref[...] = jnp.clip(x_ref[...], -1.0, 1.0)


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(1, x.shape[1] // b_dim),
  in_specs=[pl.BlockSpec(block_shape=(x.shape[0], b_dim), index_map=lambda i, j: (i, j))],
  out_specs=pl.BlockSpec(block_shape=(x.shape[0], b_dim), index_map=lambda i, j: (i, j)),
)(x).block_until_ready()

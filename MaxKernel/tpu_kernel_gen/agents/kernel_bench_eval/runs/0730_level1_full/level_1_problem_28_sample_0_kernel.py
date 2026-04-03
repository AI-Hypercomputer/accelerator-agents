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
b_bs = 8
b_dim = 2048


# Computation
def kernel(x_ref, out_ref):
  """Pallas kernel for hard_sigmoid."""
  x = x_ref[...]
  # The hard_sigmoid function is defined as: clip(x + 3, 0, 6) / 6
  result = jnp.clip(x + 3.0, 0.0, 6.0) / 6.0
  out_ref[...] = result


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(batch_size // b_bs, dim // b_dim),
  in_specs=[pl.BlockSpec(block_shape=(b_bs, b_dim), index_map=lambda i, j: (i, j))],
  out_specs=pl.BlockSpec(block_shape=(b_bs, b_dim), index_map=lambda i, j: (i, j)),
)(x).block_until_ready()

# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
dim1 = 256
dim2 = 256
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim1, dim2))
b_dim2 = 128


def kernel(x_ref, out_ref):
  """Pallas kernel to compute sum reduction along axis 1."""
  out_ref[...] = jnp.sum(x_ref[...], axis=1, keepdims=True)


# Computation
result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, 1, dim2), x.dtype),
  grid=(batch_size, dim2 // b_dim2),
  in_specs=[pl.BlockSpec(block_shape=(1, dim1, b_dim2), index_map=lambda i, j: (i, 0, j))],
  out_specs=pl.BlockSpec(block_shape=(1, 1, b_dim2), index_map=lambda i, j: (i, 0, j)),
)(x).block_until_ready()

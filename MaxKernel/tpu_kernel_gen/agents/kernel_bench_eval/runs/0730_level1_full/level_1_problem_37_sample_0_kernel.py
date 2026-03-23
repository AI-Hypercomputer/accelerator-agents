# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
features = 64
dim1 = 256
dim2 = 256
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, features, dim1, dim2))


def kernel(x_ref, norm_ref, y_ref):
  y_ref[...] = x_ref[...] / norm_ref[...]


# Computation
norm = jnp.linalg.norm(x.reshape(x.shape[0], -1), axis=1).reshape(x.shape[0], 1, 1, 1)
y = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(x.shape[0], x.shape[1]),
  in_specs=[
    pl.BlockSpec(block_shape=(1, 1, dim1, dim2), index_map=lambda i, j: (i, j, 0, 0)),
    pl.BlockSpec(block_shape=(1, 1, 1, 1), index_map=lambda i, j: (i, 0, 0, 0)),
  ],
  out_specs=pl.BlockSpec(block_shape=(1, 1, dim1, dim2), index_map=lambda i, j: (i, j, 0, 0)),
)(x, norm).block_until_ready()

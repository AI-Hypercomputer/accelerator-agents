# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
eps = 1e-5
batch_size = 16
features = 64
dim1 = 256
dim2 = 256
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, features, dim1, dim2))


# Computation
def kernel(x_ref, out_ref):
  var = jnp.mean(x_ref[...] ** 2, axis=1, keepdims=True)
  rsqrt = jax.lax.rsqrt(var + eps)
  out_ref[...] = x_ref[...] * rsqrt


output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(batch_size, dim1),
  in_specs=[pl.BlockSpec(block_shape=(1, features, 1, dim2), index_map=lambda i, j: (i, 0, j, 0))],
  out_specs=pl.BlockSpec(block_shape=(1, features, 1, dim2), index_map=lambda i, j: (i, 0, j, 0)),
)(x).block_until_ready()

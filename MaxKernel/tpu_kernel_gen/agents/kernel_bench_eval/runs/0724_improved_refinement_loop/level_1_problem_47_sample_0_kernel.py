# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
dim1 = 256
dim2 = 256
reduce_dim = 1
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim1, dim2))


def kernel(x_ref, out_ref):
  # By adding x = x_ref[...], we explicitly load the data from SRAM into a
  # register as a jax.Array. High-level jnp functions like jnp.sum are
  # designed to operate on jax.Arrays, not on the pallas.Ref objects directly.
  # This explicit load resolves the type mismatch that caused the original error.
  x = x_ref[...]
  out_ref[...] = jnp.sum(x, axis=1, keepdims=True)


# Computation
result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, 1, dim2), x.dtype),
  grid=(batch_size,),
  in_specs=[pl.BlockSpec(block_shape=(1, dim1, dim2), index_map=lambda i: (i, 0, 0))],
  out_specs=pl.BlockSpec(block_shape=(1, 1, dim2), index_map=lambda i: (i, 0, 0)),
)(x).block_until_ready()

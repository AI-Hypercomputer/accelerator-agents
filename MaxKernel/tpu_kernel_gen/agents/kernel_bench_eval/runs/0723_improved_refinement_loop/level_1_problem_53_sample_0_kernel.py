# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
dim = 1
batch_size = 16
dim1 = 256
dim2 = 256
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim1, dim2))


def kernel(x_ref, out_ref):
  """Computes the minimum of each column in a matrix and writes it to the output."""
  out_ref[...] = jnp.min(x_ref, axis=0)


# Computation
result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, dim2), x.dtype),
  grid=(batch_size,),
  in_specs=[pl.BlockSpec(block_shape=(dim1, dim2), index_map=lambda i: (i, 0, 0))],
  out_specs=pl.BlockSpec(block_shape=(dim2,), index_map=lambda i: (i,)),
)(x).block_until_ready()

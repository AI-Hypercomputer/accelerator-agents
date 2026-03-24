# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
input_shape = (4000,)
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, *input_shape))
x_shape = x.shape
x_dtype = x.dtype


def kernel(x_ref, out_ref):
  """Pallas kernel for cumulative sum over axis 1."""
  # Each program instance operates on a block of rows of the input matrix.
  # x_ref is a block of shape (8, 4000).
  # We can directly apply jnp.cumsum to this block along axis 1.
  out_ref[...] = jnp.cumsum(x_ref[...], axis=1)


# Computation
result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x_shape, x_dtype),
  grid=(batch_size // 8,),
  in_specs=[pl.BlockSpec(block_shape=(8, x_shape[1]), index_map=lambda i: (i * 8, 0))],
  out_specs=pl.BlockSpec(block_shape=(8, x_shape[1]), index_map=lambda i: (i * 8, 0)),
)(x).block_until_ready()

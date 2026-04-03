# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
negative_slope = 0.01
batch_size = 16
dim = 16384
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim))
block_dim = 128


# Computation
def kernel(x_ref, out_ref):
  # The 'negative_slope' value is captured from the scope where this kernel is defined.
  # Based on the source code, its value is 0.01.
  negative_slope = 0.01
  x = x_ref[...]
  # The leaky_relu implementation is equivalent to:
  # jnp.where(x > 0, x, negative_slope * x)
  result = jnp.maximum(0, x) + negative_slope * jnp.minimum(0, x)
  out_ref[...] = result


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(dim // block_dim,),
  in_specs=[pl.BlockSpec(block_shape=(batch_size, block_dim), index_map=lambda i: (0, i))],
  out_specs=pl.BlockSpec(block_shape=(batch_size, block_dim), index_map=lambda i: (0, i)),
)(x).block_until_ready()

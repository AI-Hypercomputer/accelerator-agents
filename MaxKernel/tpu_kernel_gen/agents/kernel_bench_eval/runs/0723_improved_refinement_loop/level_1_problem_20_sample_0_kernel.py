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
b_dim = 128


# Computation
def kernel(x_ref, out_ref):
  """Pallas kernel for Leaky ReLU."""
  # Apply the leaky relu operation element-wise on the block.
  # The expression `x_ref[...]` loads the data from SRAM into registers.
  x = x_ref[...]
  # The result is written to the output reference.
  out_ref[...] = jnp.where(x > 0, x, x * negative_slope)


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(batch_size, dim // b_dim),
  in_specs=[pl.BlockSpec(block_shape=(1, b_dim), index_map=lambda i, j: (i, j * b_dim))],
  out_specs=pl.BlockSpec(block_shape=(1, b_dim), index_map=lambda i, j: (i, j * b_dim)),
)(x).block_until_ready()

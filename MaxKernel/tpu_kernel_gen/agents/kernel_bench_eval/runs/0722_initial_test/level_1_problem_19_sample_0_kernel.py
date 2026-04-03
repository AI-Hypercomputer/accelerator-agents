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


def kernel(x_ref, out_ref):
  """Pallas kernel for element-wise ReLU."""
  # Load the input block into registers.
  x = x_ref[...]
  # Compute ReLU and write to the output block.
  out_ref[...] = jnp.maximum(x, 0.0)


# Computation
result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(1, dim // 128),
  in_specs=[pl.BlockSpec(block_shape=(batch_size, 128), index_map=lambda i, j: (i, j))],
  out_specs=pl.BlockSpec(block_shape=(batch_size, 128), index_map=lambda i, j: (i, j)),
)(x).block_until_ready()

# Imports
import jax
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
dim = 16384
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim))


def kernel(x_ref, y_ref):
  """Pallas kernel for element-wise softplus."""
  # Load the input block into registers.
  x = x_ref[...]
  # Apply the softplus function.
  result = jax.nn.softplus(x)
  # Write the result to the output buffer.
  y_ref[...] = result


# Computation
y = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(x.shape[0] // 8, x.shape[1] // 2048),
  in_specs=[pl.BlockSpec(block_shape=(8, 2048), index_map=lambda i, j: (i, j))],
  out_specs=pl.BlockSpec(block_shape=(8, 2048), index_map=lambda i, j: (i, j)),
)(x).block_until_ready()

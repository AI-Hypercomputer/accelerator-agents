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

# Define a block shape that is compatible with TPU constraints
# The second to last dim of the block shape must be divisible by 8
# and the last dim must be divisible by 128.
block_b, block_d = 8, 1024


def hardsigmoid(x):
  """Computes element-wise hard sigmoid."""
  return jnp.clip((x + 3) / 6, 0, 1)


def kernel(x_ref, out_ref):
  """Pallas kernel for element-wise hardsigmoid."""
  # The hardsigmoid function is composed of primitives that Pallas can handle.
  # We apply it element-wise to the input block (x_ref) and store
  # the result in the corresponding output block (out_ref).
  out_ref[...] = hardsigmoid(x_ref[...])


# Computation
# The pallas_call replaces the original computation
result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, dim), jnp.float32),
  grid=(batch_size // block_b, dim // block_d),
  in_specs=[pl.BlockSpec((block_b, block_d), lambda i, j: (i * block_b, j * block_d))],
  out_specs=pl.BlockSpec((block_b, block_d), lambda i, j: (i * block_b, j * block_d)),
)(x).block_until_ready()

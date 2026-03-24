# Imports
import jax
import jax.nn as nn
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
dim = 16384
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim))
BLOCK_SIZE = 128


def kernel(x_ref, out_ref):
  """Pallas kernel for element-wise hard_tanh.

  This kernel computes jax.nn.hard_tanh(x), which is equivalent to
  jnp.clip(x, -1, 1).

  Args:
    x_ref: A reference to the input block.
    out_ref: A reference to the output block to store the result.
  """
  # Load the input data from SRAM into a register.
  x = x_ref[...]
  # Compute the hard_tanh operation.
  result = nn.hard_tanh(x)
  # Write the result back to the output buffer in SRAM.
  out_ref[...] = result


# Computation
result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(1, x.shape[1] // BLOCK_SIZE),
  in_specs=[pl.BlockSpec(block_shape=(x.shape[0], BLOCK_SIZE), index_map=lambda i, j: (i, j))],
  out_specs=pl.BlockSpec(block_shape=(x.shape[0], BLOCK_SIZE), index_map=lambda i, j: (i, j)),
)(x).block_until_ready()

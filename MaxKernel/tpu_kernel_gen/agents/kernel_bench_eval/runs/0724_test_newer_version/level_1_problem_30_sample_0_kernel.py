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

# Define block sizes that adhere to TPU constraints
# For a 2D block, the dimensions must be divisible by 8 and 128 respectively.
# The batch dimension (16) is divisible by 8.
# The feature dimension (16384) is divisible by 128.
block_batch = 8
block_dim = 128


# Computation
def kernel(x_ref, out_ref):
  """
  Pallas kernel to compute x / (1 + abs(x)) element-wise.
  """
  # Load the input block from HBM into SRAM.
  x = x_ref[...]
  # Perform the element-wise computation.
  # The result is stored in registers.
  result = x / (1 + jnp.abs(x))
  # Write the result from registers back to the output block in SRAM.
  out_ref[...] = result


# The computation is an element-wise operation, so we can tile the input and
# output arrays identically. The grid iterates over these tiles.
result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(x.shape[0] // block_batch, x.shape[1] // block_dim),
  in_specs=[pl.BlockSpec(block_shape=(block_batch, block_dim), index_map=lambda i, j: (i, j))],
  out_specs=pl.BlockSpec(block_shape=(block_batch, block_dim), index_map=lambda i, j: (i, j)),
)(x).block_until_ready()

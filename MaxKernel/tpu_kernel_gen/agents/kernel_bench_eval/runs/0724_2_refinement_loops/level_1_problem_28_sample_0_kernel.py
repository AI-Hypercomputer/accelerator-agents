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

# Define block sizes for clarity and adherence to TPU constraints
# The second-to-last dimension of the block must be divisible by 8.
# The last dimension of the block must be divisible by 128.
block_batch = 8
block_dim = 128


# Computation
def kernel(x_ref, y_ref):
  """Pallas kernel for element-wise hard_sigmoid.

  This kernel computes y = max(0, min(1, (x + 3) / 6)) for each element
  in the input block.
  """
  # Load the input block from SRAM into registers.
  x = x_ref[...]
  # Apply the hard_sigmoid function element-wise.
  # Note: Using floating point literals (e.g., 3.0, 6.0) is good practice
  # to ensure floating point division.
  y = jnp.maximum(0.0, jnp.minimum(1.0, (x + 3.0) / 6.0))
  # Write the result to the output block in SRAM.
  y_ref[...] = y


y = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(x.shape[0] // block_batch, x.shape[1] // block_dim),
  in_specs=[pl.BlockSpec(block_shape=(block_batch, block_dim), index_map=lambda i, j: (i, j))],
  out_specs=pl.BlockSpec(block_shape=(block_batch, block_dim), index_map=lambda i, j: (i, j)),
)(x).block_until_ready()

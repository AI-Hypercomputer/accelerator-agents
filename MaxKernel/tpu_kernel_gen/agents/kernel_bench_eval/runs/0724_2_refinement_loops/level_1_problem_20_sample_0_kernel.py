# Imports
import jax
import jax.random as random
from jax.experimental import pallas as pl
from jax.nn import leaky_relu

# Initialization
negative_slope = 0.01
batch_size = 16
dim = 16384
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim))

# Define block sizes that adhere to TPU constraints
# For a 2D block, the second-to-last dim must be divisible by 8,
# and the last dim must be divisible by 128.
BLOCK_BATCH = 8
BLOCK_DIM = 128


# Computation
def kernel(x_ref, out_ref):
  """Pallas kernel for element-wise leaky_relu."""
  # The negative_slope is a parameter from the original computation.
  # Since it's not passed as a dynamic argument to pallas_call,
  # we define it as a constant within the kernel.
  negative_slope = 0.01

  # Load the input block from SRAM into registers.
  x = x_ref[...]

  # Apply the leaky_relu function element-wise on the block.
  result_block = leaky_relu(x, negative_slope=negative_slope)

  # Write the resulting block to the output buffer in SRAM.
  out_ref[...] = result_block


# The leaky_relu operation is element-wise, so we can tile the computation
# across the input array. We create a 2D grid of work items.
# Each work item, identified by (i, j), processes a unique (BLOCK_BATCH, BLOCK_DIM)
# block of the input/output arrays.
result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(x.shape[0] // BLOCK_BATCH, x.shape[1] // BLOCK_DIM),
  in_specs=[pl.BlockSpec(block_shape=(BLOCK_BATCH, BLOCK_DIM), index_map=lambda i, j: (i, j))],
  out_specs=pl.BlockSpec(block_shape=(BLOCK_BATCH, BLOCK_DIM), index_map=lambda i, j: (i, j)),
)(x).block_until_ready()

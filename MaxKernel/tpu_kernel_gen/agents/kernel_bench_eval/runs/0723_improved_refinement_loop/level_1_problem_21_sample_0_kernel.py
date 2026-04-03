# Imports
import jax
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
dim = 16384
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim))


def kernel(x_ref, out_ref):
  """Pallas kernel for element-wise sigmoid.

  Args:
    x_ref: Input block reference.
    out_ref: Output block reference.
  """
  # Load the input block from SRAM into registers, apply the sigmoid function
  # element-wise, and write the result back to the output block in SRAM.
  out_ref[...] = jax.nn.sigmoid(x_ref[...])


# The block size for the second dimension.
# 16384 is divisible by 128, so we can choose a block size of 128
# which satisfies the TPU block size constraint (must be divisible by 128).
block_dim = 128
# The block size for the first dimension. This must be divisible by 8.
block_batch = 8


# The grid is the number of kernel instances that we want to launch.
# We can think of the grid as being flattened into 1D, where each
# kernel instance has a unique index.
grid_x = batch_size // block_batch
grid_y = dim // block_dim
grid = (grid_x, grid_y)


def index_map(i, j):
  # We can then map this 1D index back to a 2D index.
  return (i * block_batch, j * block_dim)


# Computation
result = pl.pallas_call(
  kernel,
  # The output shape is the same as the input shape.
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  # We create a 2D grid.
  grid=grid,
  # For an element-wise operation, the input and output specifications are identical.
  # Each kernel instance processes a unique block of the input tensor.
  in_specs=[
    pl.BlockSpec(
      block_shape=(block_batch, block_dim),
      index_map=index_map,
    )
  ],
  out_specs=pl.BlockSpec(
    block_shape=(block_batch, block_dim),
    index_map=index_map,
  ),
)(x).block_until_ready()

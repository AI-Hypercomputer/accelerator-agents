# Imports
import jax
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
input_shape = (4000,)
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, *input_shape))
# For TPU compatibility, if a block is 2D, its second-to-last dimension
# must be divisible by 8. We choose a block size of 8 for the batch dimension.
b_block_size = 8


# Computation
def kernel(x_ref, out_ref):
  """Pallas kernel for cumulative product along axis=1.

  This kernel processes a block of rows. For each row in the block, it
  computes the cumulative product along the columns.

  Args:
    x_ref: A reference to the input block.
    out_ref: A reference to the output block to be written to.
  """
  # The cumulative product operation has a sequential dependency along the columns.
  # We must iterate to compute the result correctly.

  # Initialize the first column of the output. The cumulative product
  # at the first element is just the element itself.
  out_ref[:, 0] = x_ref[:, 0]

  # Iterate over the remaining columns of the block.
  for j in range(1, x_ref.shape[1]):
    # The cumulative product at column `j` is the product of the
    # cumulative product at the previous column `j-1` and the input
    # value at the current column `j`.
    # This involves reading the value written to `out_ref` in the previous
    # iteration.
    out_ref[:, j] = out_ref[:, j - 1] * x_ref[:, j]


# The computation is a cumulative product along axis=1. This operation is
# independent for each row (i.e., along axis=0). We can parallelize by
# creating one kernel instance per chunk of rows.
result = pl.pallas_call(
  kernel,
  # The output has the same shape and dtype as the input.
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  # The grid is 1D, parallelizing over the batch dimension.
  # Each kernel instance will process a block of `b_block_size` rows.
  grid=(x.shape[0] // b_block_size,),
  # Each kernel instance receives a block of (b_block_size, 4000) from the input.
  # The index_map `lambda i: (i, 0)` maps the grid index `i` to the i-th
  # block of rows in the input array.
  in_specs=[pl.BlockSpec(block_shape=(b_block_size, x.shape[1]), index_map=lambda i: (i, 0))],
  # Each kernel instance writes to a corresponding block in the output.
  out_specs=pl.BlockSpec(block_shape=(b_block_size, x.shape[1]), index_map=lambda i: (i, 0)),
)(x).block_until_ready()

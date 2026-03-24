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


# Computation
def kernel(x_ref, y_ref, tmp_ref):
  """Pallas kernel for row-wise L2 normalization.

  This kernel normalizes each row of the input matrix `x` and writes the
  result to `y`. It handles the reduction (norm calculation) across multiple
  blocks of the same row in a distributed fashion.

  Args:
    x_ref: A reference to the input block.
    y_ref: A reference to the output block.
    tmp_ref: A reference to a temporary scratch buffer used for summing
      the squares of each row across blocks. It is expected to be of shape
      (batch_size,) and initialized to zeros.
  """
  # Get the row index (i) and the block index within the row (j).
  i = pl.program_id(0)
  j = pl.program_id(1)

  # Initialize the scratch space for the current row to zero.
  # This is done only by the first block (j=0) of each row to avoid race conditions.
  if j == 0:
    tmp_ref[i] = 0.0

  # Each kernel instance computes the sum of squares for its local data block.
  # The result is a scalar value.
  x = x_ref[...]
  sum_sq = jnp.sum(x * x)

  # Atomically add the local sum of squares to the shared scratch space for the current row.
  # This ensures that contributions from all blocks of a row are safely aggregated.
  pl.atomic_add(tmp_ref, i, sum_sq)

  # A barrier is implicitly needed here to ensure all atomic adds for a row are
  # complete before proceeding. Pallas on TPU handles this synchronization
  # between the atomic write and the subsequent read from the same memory.
  pl.barrier()

  # Load the total sum of squares for the entire row.
  # This value is the same for all blocks belonging to the same row.
  total_sum_sq = tmp_ref[i]

  # Compute the L2 norm for the row. Add a small epsilon for numerical stability.
  norm = jnp.sqrt(total_sum_sq + 1e-12)

  # Normalize the local block by the row's norm and write to the output.
  y_ref[...] = x / norm


# The block size for the inner dimension.
# This value must be a multiple of 128 for TPU compatibility.
block_size = 128
batch_size, dim = x.shape

# The grid is 2D. The first dimension corresponds to the batch, and the
# second dimension iterates through blocks of the inner dimension.
grid = (batch_size, dim // block_size)

# Each kernel instance processes a (1, block_size) chunk of a row.
# The index_map=(i, j) maps the grid indices directly to the block indices.
# This means kernel (i, j) gets the j-th block of the i-th row.
# The kernel will need to perform a reduction over the `j` dimension
# to compute the norm for the entire row before normalizing its local block.
y = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=grid,
  in_specs=[pl.BlockSpec(block_shape=(1, block_size), index_map=lambda i, j: (i, j))],
  out_specs=pl.BlockSpec(block_shape=(1, block_size), index_map=lambda i, j: (i, j)),
  scratch_shapes=[pl.ScratchPad(shape=(batch_size,), dtype=x.dtype)],
)(x).block_until_ready()

# Imports
import jax
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
N = 4096
key = random.PRNGKey(0)
key_A, key_B = random.split(key)

A = random.normal(key_A, (N, N))
A = (A + A.T) / 2
B = random.normal(key_B, (N, N))
B = (B + B.T) / 2

bN = 128


# Computation
def kernel(a_ref, b_ref, c_ref):
  """
  Computes a block of the matrix multiplication C = A @ B.

  This kernel is designed for a pallas_call with a 2D grid. For each
  program (i, j) in the grid, it receives a full block-row of A and a
  full block-column of B and computes the corresponding (i, j) block of
  the output matrix C.

  Args:
    a_ref: A reference to a block of matrix A, specifically a block-row
      of shape (bN, N).
    b_ref: A reference to a block of matrix B, specifically a block-column
      of shape (N, bN).
    c_ref: A reference to the output block of matrix C, of shape (bN, bN),
      which will be written to in-place.
  """
  # For each output tile, we are given the corresponding row-band of A
  # and column-band of B. The result for the output tile is simply
  # the matrix product of these two input blocks.
  c_ref[...] = a_ref[...] @ b_ref[...]


C = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(A.shape, A.dtype),
  grid=(A.shape[0] // bN, A.shape[1] // bN),
  in_specs=[
    pl.BlockSpec(block_shape=(bN, A.shape[1]), index_map=lambda i, j: (i, 0)),
    pl.BlockSpec(block_shape=(A.shape[0], bN), index_map=lambda i, j: (0, j)),
  ],
  out_specs=pl.BlockSpec(block_shape=(bN, bN), index_map=lambda i, j: (i, j)),
)(A, B).block_until_ready()

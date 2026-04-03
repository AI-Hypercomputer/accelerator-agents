# Imports
import math

import jax
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
M = 8205
K = 2949
N = 5921
key = random.PRNGKey(0)
key_A, key_B = random.split(key)

A = random.normal(key_A, (M, K))
B = random.normal(key_B, (K, N))

# Define block sizes for tiling the computation.
# These values are chosen to be compatible with TPU hardware constraints.
BLOCK_M = 128
BLOCK_N = 128


# Computation
def kernel(x_ref, y_ref, out_ref):
  """Pallas kernel for matrix multiplication.

  This kernel computes a single block of the output matrix C. The pallas_call
  is configured to provide the necessary slices of the input matrices A and B
  for this block computation.

  Args:
    x_ref: A reference to a block of the first input matrix A.
           The shape is (BLOCK_M, K).
    y_ref: A reference to a block of the second input matrix B.
           The shape is (K, BLOCK_N).
    out_ref: A reference to a block of the output matrix C, where the result
             is stored. The shape is (BLOCK_M, BLOCK_N).
  """
  # Perform the matrix multiplication on the input blocks.
  # x_ref[...] loads the (BLOCK_M, K) slice of A from SRAM.
  # y_ref[...] loads the (K, BLOCK_N) slice of B from SRAM.
  # The result of the @ operation is a (BLOCK_M, BLOCK_N) block.
  # This result is then written to the output buffer out_ref.
  out_ref[...] = x_ref[...] @ y_ref[...]


# The pallas_call replaces the jnp.matmul operation.
# It defines how the computation is parallelized across a grid.
C = pl.pallas_call(
  kernel,
  # The output is a matrix of shape (M, N) with the same dtype as input A.
  out_shape=jax.ShapeDtypeStruct((A.shape[0], B.shape[1]), A.dtype),
  # The grid is 2D, tiling the M and N dimensions of the output matrix.
  # Each kernel instance will compute one (BLOCK_M, BLOCK_N) block of the output.
  grid=(math.ceil(A.shape[0] / BLOCK_M), math.ceil(B.shape[1] / BLOCK_N)),
  # in_specs defines how to slice the input matrices for each kernel instance.
  in_specs=[
    # For matrix A, each kernel instance (i, j) gets a horizontal slice.
    # The slice is determined by the grid's row index 'i'.
    # The full K dimension is required for the dot product.
    pl.BlockSpec(block_shape=(BLOCK_M, A.shape[1]), index_map=lambda i, j: (i, 0)),
    # For matrix B, each kernel instance (i, j) gets a vertical slice.
    # The slice is determined by the grid's column index 'j'.
    pl.BlockSpec(block_shape=(B.shape[0], BLOCK_N), index_map=lambda i, j: (0, j)),
  ],
  # out_specs defines where each kernel instance writes its result.
  # The kernel instance (i, j) writes to the (i, j)-th block of the output matrix C.
  out_specs=pl.BlockSpec(block_shape=(BLOCK_M, BLOCK_N), index_map=lambda i, j: (i, j)),
)(A, B).block_until_ready()

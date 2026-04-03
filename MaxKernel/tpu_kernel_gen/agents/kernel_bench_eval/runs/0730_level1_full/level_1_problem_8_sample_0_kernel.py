# Imports
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


def kernel(x_ref, y_ref, out_ref):
  """
  Computes a block of the matrix multiplication C = A @ B.

  Args:
    x_ref: A reference to a (bM, K) block of matrix A.
    y_ref: A reference to a (K, bN) block of matrix B.
    out_ref: A reference to a (bM, bN) block of the output matrix C, to be written to.
  """
  # Perform the matrix multiplication on the input blocks and write the result to the output block.
  # x_ref[...] has shape (bM, K)
  # y_ref[...] has shape (K, bN)
  # The result of the matmul has shape (bM, bN), which matches the shape of out_ref.
  out_ref[...] = x_ref[...] @ y_ref[...]


# Computation
# Define block sizes that satisfy TPU constraints
bM = 128
bN = 128

# Calculate grid dimensions to cover the entire output matrix
grid_m = (M + bM - 1) // bM
grid_n = (N + bN - 1) // bN
grid = (grid_m, grid_n)

C = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((M, N), A.dtype),
  grid=grid,
  in_specs=[
    pl.BlockSpec(block_shape=(bM, K), index_map=lambda i, j: (i, 0)),
    pl.BlockSpec(block_shape=(K, bN), index_map=lambda i, j: (0, j)),
  ],
  out_specs=pl.BlockSpec(block_shape=(bM, bN), index_map=lambda i, j: (i, j)),
)(A, B).block_until_ready()

# Imports
import jax
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
M = 16384
N = 16
key = random.PRNGKey(0)
key_A, key_B = random.split(key)

A = random.normal(key_A, (M, N))
B = random.normal(key_B, (N, M))

# Define a block size for tiling the computation
bM = 128


# Computation
def kernel(a_ref, b_ref, c_ref):
  """Computes a block of the matrix multiplication C = A @ B.

  Args:
    a_ref: A reference to a (bM, N) block of matrix A.
    b_ref: A reference to a (N, bM) block of matrix B.
    c_ref: A reference to a (bM, bM) block of the output matrix C.
  """
  # Perform the matrix multiplication on the input blocks and store the
  # result in the output block.
  # a_ref[...] has shape (bM, N)
  # b_ref[...] has shape (N, bM)
  # The result has shape (bM, bM), which matches c_ref.
  c_ref[...] = a_ref[...] @ b_ref[...]


C = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((M, M), A.dtype),
  grid=(M // bM, M // bM),
  in_specs=[
    # Each kernel instance gets a (bM, N) row-block of A, indexed by `i`.
    pl.BlockSpec(block_shape=(bM, N), index_map=lambda i, j: (i, 0)),
    # Each kernel instance gets a (N, bM) column-block of B, indexed by `j`.
    pl.BlockSpec(block_shape=(N, bM), index_map=lambda i, j: (0, j)),
  ],
  # Each kernel instance (i, j) computes a unique (bM, bM) block of the output C.
  out_specs=pl.BlockSpec(block_shape=(bM, bM), index_map=lambda i, j: (i, j)),
)(A, B).block_until_ready()

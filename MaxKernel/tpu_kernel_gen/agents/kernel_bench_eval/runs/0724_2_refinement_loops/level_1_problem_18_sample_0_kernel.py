# Imports
import jax
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
M = 1024
K = 4096
N = 2048
key = random.PRNGKey(0)
key_A, key_B = random.split(key)

A = random.normal(key_A, (K, M))
B = random.normal(key_B, (N, K))

bM = 128
bN = 128


# Computation
def kernel(a_ref, b_ref, c_ref):
  """Pallas kernel for C = A.T @ B.T.

  Args:
    a_ref: A block of input matrix A with shape (K, bM).
    b_ref: A block of input matrix B with shape (bN, K).
    c_ref: The output block of matrix C with shape (bM, bN) to be computed.
  """
  # The overall computation is C = A.T @ B.T.
  # Each program in the grid computes one block of the output C.
  # The shape of a_ref[...].T is (bM, K).
  # The shape of b_ref[...].T is (K, bN).
  # The matrix multiplication (a_ref[...].T @ b_ref[...].T) results in a block of shape
  # (bM, bN), which matches the shape of the output block c_ref.
  c_ref[...] = a_ref[...].T @ b_ref[...].T


C = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((M, N), A.dtype),
  grid=(M // bM, N // bN),
  in_specs=[
    pl.BlockSpec(block_shape=(K, bM), index_map=lambda i, j: (0, i)),
    pl.BlockSpec(block_shape=(bN, K), index_map=lambda i, j: (j, 0)),
  ],
  out_specs=pl.BlockSpec(block_shape=(bM, bN), index_map=lambda i, j: (i, j)),
)(A, B).block_until_ready()

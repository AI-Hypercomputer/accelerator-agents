# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
M = 1024
K = 4096
N = 2048
key = random.PRNGKey(0)
key_A, key_B = random.split(key)

A = random.normal(key_A, (M, K))
B = random.normal(key_B, (N, K))

bM = 128
bN = 128


# Computation
def kernel(a_ref, b_ref, c_ref):
  """Pallas kernel for C = A @ B.T.

  Args:
    a_ref: A block of matrix A with shape (bM, K).
    b_ref: A block of matrix B with shape (bN, K).
    c_ref: The output block of matrix C with shape (bM, bN) to be computed.
  """
  # The core computation for a single output block is a matrix multiplication
  # of a block of A with the transpose of a block of B.
  # a_ref[...] has shape (bM, K)
  # b_ref[...] has shape (bN, K), so b_ref[...].T has shape (K, bN)
  # The result of the matmul has shape (bM, bN), which matches c_ref.
  c_ref[...] = jnp.matmul(a_ref[...], b_ref[...].T)


# The computation C = A @ B.T is a matrix multiplication.
# We tile the output matrix C of shape (M, N) into (bM, bN) blocks.
# The grid is (M // bM, N // bN) to cover all output blocks.
#
# For each output block C[i, j], we need:
# 1. The i-th block-row of A, which has shape (bM, K).
#    This is selected by the first grid index `i`.
#    in_spec for A: BlockSpec((bM, K), lambda i, j: (i, 0))
#
# 2. The j-th block-row of B, which has shape (bN, K).
#    The kernel will then transpose this to (K, bN) for multiplication.
#    This is selected by the second grid index `j`.
#    in_spec for B: BlockSpec((bN, K), lambda i, j: (j, 0))
#
# The output spec maps each grid instance (i, j) to the corresponding
# (bM, bN) block in the output matrix C.
# out_spec for C: BlockSpec((bM, bN), lambda i, j: (i, j))
C = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((M, N), A.dtype),
  grid=(M // bM, N // bN),
  in_specs=[
    pl.BlockSpec(block_shape=(bM, K), index_map=lambda i, j: (i, 0)),
    pl.BlockSpec(block_shape=(bN, K), index_map=lambda i, j: (j, 0)),
  ],
  out_specs=pl.BlockSpec(block_shape=(bM, bN), index_map=lambda i, j: (i, j)),
)(A, B).block_until_ready()

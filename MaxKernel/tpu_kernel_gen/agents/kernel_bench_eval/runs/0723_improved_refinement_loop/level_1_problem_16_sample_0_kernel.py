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

A = random.normal(key_A, (K, M))
B = random.normal(key_B, (K, N))

bM = 128
bN = 128


# Computation
def kernel(a_ref, b_ref, c_ref):
  """Pallas kernel for C = A.T @ B.

  Args:
    a_ref: A reference to a block of matrix A with shape (K, bM).
    b_ref: A reference to a block of matrix B with shape (K, bN).
    c_ref: A reference to a block of the output matrix C with shape (bM, bN),
      to be written to.
  """
  # The computation for each block is a matrix multiplication of the transposed
  # block of A and the block of B.
  # a_ref[...] has shape (K, bM), so a_ref[...].T has shape (bM, K).
  # b_ref[...] has shape (K, bN).
  # The result of the matmul is of shape (bM, bN), which matches c_ref.
  c_ref[...] = jnp.matmul(a_ref[...].T, b_ref[...])


C = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((M, N), A.dtype),
  grid=(M // bM, N // bN),
  in_specs=[
    pl.BlockSpec(block_shape=(K, bM), index_map=lambda i, j: (0, i)),
    pl.BlockSpec(block_shape=(K, bN), index_map=lambda i, j: (0, j)),
  ],
  out_specs=pl.BlockSpec(block_shape=(bM, bN), index_map=lambda i, j: (i, j)),
)(A, B).block_until_ready()

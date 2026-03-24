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
  """Pallas kernel for matrix multiplication C = A @ B.T.

  Args:
    a_ref: A block of A with shape (bM, K).
    b_ref: A block of B with shape (bN, K).
    c_ref: A block of C with shape (bM, bN) to be computed.
  """
  # The computation is C = A @ B.T.
  # For the given blocks, this translates to:
  # c_block = a_block @ b_block.T
  # The shapes are: (bM, bN) = (bM, K) @ (K, bN).
  # This is equivalent to jnp.matmul(a_ref[...], b_ref[...].T).
  c_ref[...] = jnp.matmul(a_ref[...], b_ref[...].T)


# The K dimension is not blocked in the inputs, but handled inside the kernel.
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

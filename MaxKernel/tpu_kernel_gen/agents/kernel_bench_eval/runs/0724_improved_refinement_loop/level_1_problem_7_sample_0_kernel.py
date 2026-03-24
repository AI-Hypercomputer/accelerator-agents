# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
M = 16384
N = 16384
K = 32
key = random.PRNGKey(0)
key_A, key_B = random.split(key)
A = random.normal(key_A, (M, K))
B = random.normal(key_B, (K, N))
bM = 128
bN = 128


def kernel(a_ref, b_ref, c_ref):
  """
  Computes a block of the matrix multiplication C = A @ B.

  Args:
    a_ref: A reference to a block of matrix A of shape (bM, K).
    b_ref: A reference to a block of matrix B of shape (K, bN).
    c_ref: A reference to a block of the output matrix C of shape (bM, bN),
      which this kernel will compute and write to.
  """
  # Each program in the grid computes one tile of the output matrix C.
  # The pallas_call is configured to load a (bM, K) block of A and a
  # (K, bN) block of B for each program.
  # The matrix multiplication of these two blocks yields a (bM, bN) block,
  # which is the correct size for the output tile.
  # We write this result directly to the output reference.
  c_ref[...] = jnp.matmul(a_ref[...], b_ref[...])


# Computation
C = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((M, N), A.dtype),
  grid=(M // bM, N // bN),
  in_specs=[
    pl.BlockSpec(block_shape=(bM, K), index_map=lambda i, j: (i, 0)),
    pl.BlockSpec(block_shape=(K, bN), index_map=lambda i, j: (0, j)),
  ],
  out_specs=pl.BlockSpec(block_shape=(bM, bN), index_map=lambda i, j: (i, j)),
)(A, B).block_until_ready()

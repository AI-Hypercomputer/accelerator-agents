# Imports
import math

import jax
import jax.numpy as jnp
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

# Define block sizes for the kernel
bM = 128
bN = 128


def kernel(A_ref, B_ref, C_ref):
  """
  Matrix multiplication kernel.

  Each program in the grid computes one block of the output matrix C.
  The pallas_call invocation is configured to load the necessary tiles from
  the input matrices A and B into the SRAM of the TPU.

  Args:
    A_ref: A reference to a tile of matrix A, with shape (bM, K).
    B_ref: A reference to a tile of matrix B, with shape (K, bN).
    C_ref: A reference to a tile of the output matrix C, with shape (bM, bN),
      which this kernel will compute and write to.
  """
  # Perform the matrix multiplication on the tiles loaded in SRAM.
  # A_ref[...] has shape (bM, K) and B_ref[...] has shape (K, bN).
  # The result of the matmul has shape (bM, bN), which matches C_ref.
  C_ref[...] = jnp.matmul(A_ref[...], B_ref[...])


# Computation
# The pallas_call invocation replaces the jnp.matmul
C = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((M, N), A.dtype),
  grid=(math.ceil(M / bM), math.ceil(N / bN)),
  in_specs=[
    pl.BlockSpec(block_shape=(bM, K), index_map=lambda i, j: (i, 0)),
    pl.BlockSpec(block_shape=(K, bN), index_map=lambda i, j: (0, j)),
  ],
  out_specs=pl.BlockSpec(block_shape=(bM, bN), index_map=lambda i, j: (i, j)),
)(A, B).block_until_ready()

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
B = random.normal(key_B, (K, N))

bM = 128
bN = 128


def kernel(x_ref, y_ref, out_ref):
  """
  Computes a block of the matrix multiplication C = A @ B.

  Args:
    x_ref: A reference to a block of matrix A of shape (bM, K).
    y_ref: A reference to a block of matrix B of shape (K, bN).
    out_ref: A reference to a block of the output matrix C of shape (bM, bN),
      where the result will be stored.
  """
  # Perform the matrix multiplication on the input blocks.
  # The result of this operation will have a shape of (bM, bN).
  # This result is then written to the output block reference.
  out_ref[...] = jnp.matmul(x_ref[...], y_ref[...])


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

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


def kernel(x_ref, y_ref, out_ref):
  """Computes a block of the matrix multiplication C = A @ B.

  Args:
    x_ref: A reference to a block of matrix A of shape (bM, K).
    y_ref: A reference to a block of matrix B of shape (K, bN).
    out_ref: A reference to a block of the output matrix C of shape (bM, bN),
      which is modified in-place.
  """
  # Since the whole K dimension is loaded for each block of A and B,
  # we can perform a standard matrix multiplication for the block.
  # x_ref[...] loads the (bM, K) block of A from SRAM into registers.
  # y_ref[...] loads the (K, bN) block of B from SRAM into registers.
  # The result of the dot product is then written to the output block in SRAM.
  out_ref[...] = jnp.dot(x_ref[...], y_ref[...])


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

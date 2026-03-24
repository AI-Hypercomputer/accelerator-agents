# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
N = 16
M = 1024
K = 2048
L = 768
key = random.PRNGKey(0)
key_A, key_B = random.split(key)

A = random.normal(key_A, (N, M, K))
B = random.normal(key_B, (K, L))

bM = 128
bL = 128


# Computation
def kernel(x_ref, y_ref, out_ref):
  """
  Computes a block of the batched matrix multiplication C = A @ B.

  Args:
    x_ref: A reference to a block of input matrix A of shape (1, bM, K).
    y_ref: A reference to a block of input matrix B of shape (K, bL).
    out_ref: A reference to a block of the output matrix C of shape (1, bM, bL),
      which will be written to.
  """
  # Perform the matrix multiplication on the input blocks.
  # The shapes are: (bM, K) @ (K, bL) -> (bM, bL)
  # This result is then written to the output block.
  out_ref[0, :, :] = jnp.matmul(x_ref[0, :, :], y_ref[...])


C = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((N, M, L), A.dtype),
  grid=(N, M // bM, L // bL),
  in_specs=[
    pl.BlockSpec(block_shape=(1, bM, K), index_map=lambda i, j, k: (i, j, 0)),
    pl.BlockSpec(block_shape=(K, bL), index_map=lambda i, j, k: (0, k)),
  ],
  out_specs=pl.BlockSpec(block_shape=(1, bM, bL), index_map=lambda i, j, k: (i, j, k)),
)(A, B).block_until_ready()

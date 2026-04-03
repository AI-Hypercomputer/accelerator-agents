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
B = random.normal(key_B, (N, K))
bM = 128
bN = 128


# Computation
def kernel(x_ref, y_ref, out_ref):
  """Computes a block of the matrix multiplication C = A.T @ B.T.

  Args:
    x_ref: A reference to a block of input matrix A, with shape (K, bM).
    y_ref: A reference to a block of input matrix B, with shape (bN, K).
    out_ref: A reference to a block of the output matrix C, with shape (bM, bN).
  """
  # Load the data from memory references into JAX arrays.
  x = x_ref[...]
  y = y_ref[...]
  # The original computation is C = A.T @ B.T.
  # x is a slice of A with shape (K, bM). Its transpose, x.T,
  # corresponds to a slice of A.T and has shape (bM, K).
  # y is a slice of B with shape (bN, K). Its transpose, y.T,
  # corresponds to a slice of B.T and has shape (K, bN).
  # The matrix multiplication for the block is (bM, K) @ (K, bN),
  # resulting in a (bM, bN) block, which matches the shape of out_ref.
  out_ref[...] = jnp.matmul(x.T, y.T)


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

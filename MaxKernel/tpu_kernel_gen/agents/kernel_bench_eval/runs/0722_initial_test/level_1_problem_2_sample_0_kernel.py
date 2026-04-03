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

A = random.normal(key_A, (M, K))
B = random.normal(key_B, (K, N))

bM = 128
bN = 128


def kernel(x_ref, y_ref, z_ref):
  """Pallas kernel for matrix multiplication.

  This kernel computes a block of the output matrix C by multiplying a block of
  rows from A (x_ref) with a block of columns from B (y_ref).

  Args:
    x_ref: A reference to a block of matrix A of shape (bM, K).
    y_ref: A reference to a block of matrix B of shape (K, bN).
    z_ref: A reference to a block of the output matrix C of shape (bM, bN),
      which this kernel will overwrite.
  """
  # The `pallas_call` is configured to load a (bM, K) block of A and a
  # (K, bN) block of B for each program. The matrix multiplication of these
  # two blocks directly yields the corresponding (bM, bN) output block.
  # Therefore, a single matmul operation is sufficient.
  z_ref[...] = x_ref[...] @ y_ref[...]


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

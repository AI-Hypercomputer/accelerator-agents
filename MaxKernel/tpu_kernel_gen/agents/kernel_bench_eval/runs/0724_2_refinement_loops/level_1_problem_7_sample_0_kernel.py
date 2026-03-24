# Imports
import jax
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


# Computation
def kernel(a_ref, b_ref, c_ref):
  """Computes a block of the matrix multiplication C = A @ B.

  Args:
    a_ref: A reference to a (bM, K) block of matrix A.
    b_ref: A reference to a (K, bN) block of matrix B.
    c_ref: A reference to a (bM, bN) block of the output matrix C, which will
      be written to.
  """
  # The pallas_call maps each program in the grid to a specific output block.
  # It loads the corresponding row-block from A and column-block from B.
  # The core computation is the matrix multiplication of these blocks.
  # The result is written directly to the output block in C.
  c_ref[...] = a_ref[...] @ b_ref[...]


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

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
A = random.normal(key_A, (K, M))
B = random.normal(key_B, (N, K))
bM = 128
bN = 128


# Computation
def kernel(a_t_ref, b_t_ref, c_ref):
  """Pallas kernel for matrix multiplication C = A.T @ B.T.

  This kernel computes a single (bM, bN) tile of the output matrix C.

  Args:
    a_t_ref: A reference to a (bM, K) block of the transposed matrix A.
    b_t_ref: A reference to a (K, bN) block of the transposed matrix B.
    c_ref: A reference to a (bM, bN) block of the output matrix C, to be
      written to.
  """
  # Load the blocks of A.T and B.T from SRAM into registers and perform
  # the matrix multiplication.
  # The result is a (bM, bN) block which is then written to the output C.
  c_ref[...] = a_t_ref[...] @ b_t_ref[...]


C = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((M, N), A.dtype),
  grid=(M // bM, N // bN),
  in_specs=[
    pl.BlockSpec(block_shape=(bM, K), index_map=lambda i, j: (i, 0)),
    pl.BlockSpec(block_shape=(K, bN), index_map=lambda i, j: (0, j)),
  ],
  out_specs=pl.BlockSpec(block_shape=(bM, bN), index_map=lambda i, j: (i, j)),
)(A.T, B.T).block_until_ready()

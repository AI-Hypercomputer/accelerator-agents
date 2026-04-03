# Imports
import jax
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
M = 16384
N = 16
bM = 128
key = random.PRNGKey(0)
key_A, key_B = random.split(key)

A = random.normal(key_A, (M, N))
B = random.normal(key_B, (N, M))


# Computation
def kernel(a_ref, b_ref, c_ref):
  """Computes a tile of the matrix multiplication C = A @ B.

  Args:
    a_ref: A reference to a block of matrix A of shape (bM, N).
    b_ref: A reference to a block of matrix B of shape (N, bM).
    c_ref: A reference to a block of the output matrix C of shape (bM, bM),
      which is written to in-place.
  """
  # Each work-item computes one tile of the output matrix C.
  # The inputs a_ref and b_ref are blocks of A and B respectively.
  # a_ref has shape (bM, N) and b_ref has shape (N, bM).
  # Their product (a_ref @ b_ref) results in a (bM, bM) block.
  # This block corresponds to a tile of the output matrix C, so we can
  # directly assign the result to c_ref.
  c_ref[...] = a_ref[...] @ b_ref[...]


C = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((M, M), A.dtype),
  grid=(M // bM, M // bM),
  in_specs=[
    pl.BlockSpec(block_shape=(bM, N), index_map=lambda i, j: (i, 0)),
    pl.BlockSpec(block_shape=(N, bM), index_map=lambda i, j: (0, j)),
  ],
  out_specs=pl.BlockSpec(block_shape=(bM, bM), index_map=lambda i, j: (i, j)),
)(A, B).block_until_ready()

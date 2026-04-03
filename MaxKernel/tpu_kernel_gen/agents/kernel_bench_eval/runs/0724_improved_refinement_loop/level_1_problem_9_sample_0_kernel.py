# Imports
import jax
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
M = 16384
N = 16
key = random.PRNGKey(0)
key_A, key_B = random.split(key)

A = random.normal(key_A, (M, N))
B = random.normal(key_B, (N, M))

bM = 128


# Computation
def kernel(x_ref, y_ref, out_ref):
  """Computes a tile of the matrix multiplication C = A @ B.

  Args:
    x_ref: A reference to a (bM, N) block of input matrix A.
    y_ref: A reference to a (N, bM) block of input matrix B.
    out_ref: A reference to a (bM, bM) block of the output matrix C.
  """
  # Each program computes a single (bM, bM) output tile.
  # The matrix multiplication of a (bM, N) block from A and a (N, bM) block
  # from B results in the corresponding (bM, bM) block of the output C.
  # This is a direct computation and store, as the entire reduction dimension N
  # is handled within a single operation.
  out_ref[...] = x_ref[...] @ y_ref[...]


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

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

# Define block sizes for tiling
bM = 128
bN = 128

# Shape and dtype information for the output
C_shape = (M, N)
dtype = jnp.float32


# Computation
def kernel(a_ref, b_ref, c_ref):
  """Pallas kernel for matrix multiplication.

  This kernel computes a tile of the output matrix C. The `pallas_call`
  is configured with a 2D grid, where each program in the grid is responsible
  for one output tile.

  Args:
    a_ref: A reference to a block of matrix A of shape (bM, K).
    b_ref: A reference to a block of matrix B of shape (K, bN).
    c_ref: A reference to an output block of matrix C of shape (bM, bN),
      which this kernel will compute and write to.
  """
  # For each program (i, j) in the grid, pallas loads a complete
  # row-block from A and a complete column-block from B.
  # a_ref corresponds to A[i*bM:(i+1)*bM, :]
  # b_ref corresponds to B[:, j*bN:(j+1)*bN]
  # The matrix multiplication of these two blocks produces the
  # corresponding output block for C.
  # The result has shape (bM, bN), which matches c_ref.
  c_ref[...] = a_ref[...] @ b_ref[...]


C = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(C_shape, dtype),
  grid=(M // bM, N // bN),
  in_specs=[
    pl.BlockSpec(block_shape=(bM, K), index_map=lambda i, j: (i, 0)),
    pl.BlockSpec(block_shape=(K, bN), index_map=lambda i, j: (0, j)),
  ],
  out_specs=pl.BlockSpec(block_shape=(bM, bN), index_map=lambda i, j: (i, j)),
)(A, B).block_until_ready()

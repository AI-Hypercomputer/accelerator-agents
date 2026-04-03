# Imports
import jax
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
N = 4096
key = random.PRNGKey(0)
key_A, key_B = random.split(key)

A = random.normal(key_A, (N, N))
A = (A + A.T) / 2
B = random.normal(key_B, (N, N))
B = (B + B.T) / 2

bN = 128


# Computation
def kernel(x_ref, y_ref, z_ref):
  """Computes a block of the matrix multiplication C = A @ B.

  This kernel is designed for a 2D grid of programs. Each program (i, j)
  computes one output block of C.

  Args:
    x_ref: A reference to a (bN, N) block of input matrix A.
    y_ref: A reference to a (N, bN) block of input matrix B.
    z_ref: A reference to a (bN, bN) block of the output matrix C for in-place
      updates.
  """
  # For each program in the 2D grid, x_ref holds a complete row-block of A
  # and y_ref holds a complete column-block of B. Their product gives the
  # corresponding output block of C directly.
  z_ref[...] = x_ref[...] @ y_ref[...]


C = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(A.shape, A.dtype),
  grid=(N // bN, N // bN),
  in_specs=[
    pl.BlockSpec(block_shape=(bN, N), index_map=lambda i, j: (i, 0)),
    pl.BlockSpec(block_shape=(N, bN), index_map=lambda i, j: (0, j)),
  ],
  out_specs=pl.BlockSpec(block_shape=(bN, bN), index_map=lambda i, j: (i, j)),
)(A, B).block_until_ready()

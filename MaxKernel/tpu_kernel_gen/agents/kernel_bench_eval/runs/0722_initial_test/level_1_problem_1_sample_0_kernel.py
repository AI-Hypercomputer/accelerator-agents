# Imports
import jax
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
N = 2048
key = random.PRNGKey(0)
key_A, key_B = random.split(key)

A = random.normal(key_A, (N, N))
B = random.normal(key_B, (N, N))

bN = 128


# Computation
def kernel(x_ref, y_ref, z_ref):
  """Pallas kernel for matrix multiplication.

  This kernel computes a (bN, bN) block of the output matrix C.
  The pallas_call is configured to load a (bN, N) row-panel from A (`x_ref`)
  and a (N, bN) column-panel from B (`y_ref`) for each program. The kernel
  then computes their matrix product and writes the result to the
  corresponding block in C (`z_ref`).

  Args:
    x_ref: A reference to a block of input matrix A.
    y_ref: A reference to a block of input matrix B.
    z_ref: A reference to a block of the output matrix C.
  """
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

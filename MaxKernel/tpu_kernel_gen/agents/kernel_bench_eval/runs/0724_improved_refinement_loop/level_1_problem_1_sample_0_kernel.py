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
  """Matrix multiplication kernel.

  Args:
    x_ref: A reference to a row-panel of the first input matrix.
    y_ref: A reference to a column-panel of the second input matrix.
    z_ref: A reference to a block of the output matrix.
  """
  # The pallas_call invocation maps a (bN, N) block of A (x_ref) and
  # a (N, bN) block of B (y_ref) to each program.
  # The matrix multiplication of these two blocks produces a (bN, bN)
  # output block, which is the correct size for z_ref.
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

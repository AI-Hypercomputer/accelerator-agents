# Imports
import jax
import jax.numpy as jnp
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
def kernel(x_ref, y_ref, out_ref):
  """Computes a block of the matrix multiplication C = A @ B.

  This kernel is designed to be called by a 2D grid of programs. Each program
  (i, j) computes one (bN, bN) block of the output matrix C.

  Args:
    x_ref: A reference to a (bN, N) block of input matrix A.
    y_ref: A reference to a (N, bN) block of input matrix B.
    out_ref: A reference to a (bN, bN) block of the output matrix C.
  """
  # The pallas_call is configured to load a full row-block from A and a full
  # column-block from B. This means the full inner dimension (N) is available
  # for the multiplication.
  # We can therefore compute the output block with a single matmul operation,
  # without needing an explicit accumulation loop over the inner dimension.
  out_ref[...] = jnp.matmul(x_ref[...], y_ref[...])


C = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(A.shape, A.dtype),
  grid=(A.shape[0] // bN, A.shape[1] // bN),
  in_specs=[
    pl.BlockSpec(block_shape=(bN, A.shape[1]), index_map=lambda i, j: (i, 0)),
    pl.BlockSpec(block_shape=(A.shape[0], bN), index_map=lambda i, j: (0, j)),
  ],
  out_specs=pl.BlockSpec(block_shape=(bN, bN), index_map=lambda i, j: (i, j)),
)(A, B).block_until_ready()

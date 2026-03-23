# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
m = 128
k = 256
n = 512
key = random.PRNGKey(0)
key_A, key_B = random.split(key)

A = random.normal(key_A, (batch_size, m, k))
B = random.normal(key_B, (batch_size, k, n))

# Block sizes for tiling the matrix multiplication
# We tile along the 'm' and 'n' dimensions of the output.
# The 'k' dimension is a contraction dimension, so we take the whole slice.
# The batch dimension is handled by the grid.
bm = 128  # Block size for the 'm' dimension
bn = 128  # Block size for the 'n' dimension
bk = k  # Block size for the 'k' dimension


# Computation
def kernel(x_ref, y_ref, out_ref):
  """Pallas kernel for batched matrix multiplication.

  Args:
    x_ref: A reference to a tile of the first input matrix A.
           Shape: (1, bm, k).
    y_ref: A reference to a tile of the second input matrix B.
           Shape: (1, k, bn).
    out_ref: A reference to a tile of the output matrix C, to be computed.
             Shape: (1, bm, bn).
  """
  # Perform the matrix multiplication on the input tiles.
  # The operation is (bm, k) @ (k, bn) -> (bm, bn).
  # The result is written directly to the output tile.
  # No accumulation is needed because the entire 'k' dimension is processed
  # in a single kernel launch for each output tile.
  out_ref[0] = jnp.matmul(x_ref[0], y_ref[0])


C = pl.pallas_call(
  kernel,
  # The output C has shape (batch_size, m, n)
  out_shape=jax.ShapeDtypeStruct((batch_size, m, n), A.dtype),
  # Grid dimensions correspond to (batch, m_tiles, n_tiles)
  grid=(batch_size, m // bm, n // bn),
  in_specs=[
    # For A, each kernel instance (b, i, j) needs the i-th block of rows
    # from the b-th batch element. The block spans the entire 'k' dimension.
    pl.BlockSpec(block_shape=(1, bm, k), index_map=lambda b, i, j: (b, i * bm, 0)),
    # For B, each kernel instance (b, i, j) needs the j-th block of columns
    # from the b-th batch element. The block spans the entire 'k' dimension.
    pl.BlockSpec(block_shape=(1, k, bn), index_map=lambda b, i, j: (b, 0, j * bn)),
  ],
  # Each kernel instance (b, i, j) computes a (bm, bn) block of the output
  # matrix C for the b-th batch element at tile position (i, j).
  out_specs=pl.BlockSpec(block_shape=(1, bm, bn), index_map=lambda b, i, j: (b, i * bm, j * bn)),
)(A, B).block_until_ready()

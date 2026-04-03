# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
N = 16
M = 1024
K = 2048
L = 768
key = random.PRNGKey(0)
key_A, key_B = random.split(key)

A = random.normal(key_A, (N, M, K))
B = random.normal(key_B, (K, L))

# Block sizes
bM = 128
bL = 128


# Computation
def kernel(a_ref, b_ref, c_ref):
  """Pallas kernel for batched matrix multiplication.

  Args:
    a_ref: A reference to a block of the first input tensor A.
    b_ref: A reference to a block of the second input tensor B.
    c_ref: A reference to a block of the output tensor C, to be updated in-place.
  """
  # Perform the matrix multiplication on the input blocks.
  # a_ref has shape (1, bM, K). We slice it to (bM, K) to use a 2D matmul.
  # b_ref has shape (K, bL).
  # The result of the matmul will have shape (bM, bL).
  # We write this result into the corresponding slice of c_ref.
  c_ref[0, :, :] = jnp.matmul(a_ref[0, :, :], b_ref[...])


# Since the computation is a batched matrix multiplication, we can define a 3D grid.
# The first dimension of the grid corresponds to the batch dimension N.
# The second and third dimensions tile the M and L dimensions of the output matrix.
grid = (N, M // bM, L // bL)

C = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((N, M, L), A.dtype),
  grid=grid,
  in_specs=[
    # For input A, each kernel instance (i, j, k) needs the i-th batch element
    # and the j-th row-block. The entire K dimension is required for the matmul.
    # The index_map (i, j, 0) selects the block based on the batch and row indices.
    pl.BlockSpec(block_shape=(1, bM, K), index_map=lambda i, j, k: (i, j, 0)),
    # For input B, each kernel instance (i, j, k) needs the k-th column-block.
    # B is not batched, so the i and j grid indices are ignored for B.
    # The index_map (0, k) selects the block based on the column index.
    pl.BlockSpec(block_shape=(K, bL), index_map=lambda i, j, k: (0, k)),
  ],
  # For the output C, each kernel instance (i, j, k) is responsible for computing
  # a unique (bM, bL) block at the corresponding (i, j, k) position in the output tensor.
  out_specs=pl.BlockSpec(block_shape=(1, bM, bL), index_map=lambda i, j, k: (i, j, k)),
)(A, B).block_until_ready()

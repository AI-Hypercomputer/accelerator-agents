# Imports
import jax
import jax.numpy as jnp
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

  This kernel computes a (bN, bN) block of the output matrix C. The `pallas_call`
  is configured to load a (bN, N) row-panel from input A (x_ref) and a
  (N, bN) column-panel from input B (y_ref) into SRAM for each program.

  The kernel then computes the matrix product of these two panels. To manage
  memory and computation efficiently within the TPU core, it tiles the inner
  shared dimension (N) using a Python for-loop, which Pallas unrolls.

  Args:
    x_ref: A reference to a block of input matrix A of shape (bN, N).
    y_ref: A reference to a block of input matrix B of shape (N, bN).
    z_ref: A reference to the output block of matrix C of shape (bN, bN).
  """
  # The second dimension of x_ref is the shared inner dimension (N).
  N = x_ref.shape[1]
  # Define the size of the blocks for tiling the inner dimension.
  k_block_size = 128

  # Initialize an accumulator for the output block with zeros.
  # This accumulator will reside in the TPU core's registers.
  acc = jnp.zeros_like(z_ref)

  # Iterate over the inner dimension N in chunks of k_block_size.
  for k in range(N // k_block_size):
    # Define the start and end indices for the current inner block.
    k_start = k * k_block_size
    k_end = k_start + k_block_size

    # Load smaller blocks from the input panels in SRAM into registers.
    a_block = x_ref[:, k_start:k_end]
    b_block = y_ref[k_start:k_end, :]

    # Perform matrix multiplication on the smaller blocks and accumulate.
    acc += a_block @ b_block

  # Write the final accumulated result to the output reference in HBM.
  z_ref[...] = acc


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

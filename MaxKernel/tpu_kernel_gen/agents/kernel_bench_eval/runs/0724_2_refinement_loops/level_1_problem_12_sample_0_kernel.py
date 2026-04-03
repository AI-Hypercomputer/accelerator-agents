# Imports
import jax
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
M = 4096
N = 4096
key = random.PRNGKey(0)
key_A, key_B = random.split(key)
# FIX: Reshape A to (N, 1) to use 2D blocking, which is more robust.
A = random.normal(key_A, (N, 1))
B = random.normal(key_B, (N, M))
block_N = 128
block_M = 128


# Computation
def kernel(a_ref, b_ref, out_ref):
  """Pallas kernel to compute diag(A) @ B.

  This is equivalent to scaling each row of B by the corresponding element of A.
  result[i, j] = A[i] * B[i, j]

  Args:
    a_ref: A block of the input vector A.
    b_ref: A block of the input matrix B.
    out_ref: A block of the output matrix to write results to.
  """
  # Load the input blocks from SRAM into registers.
  # a_ref has shape (block_N, 1)
  # b_ref has shape (block_N, block_M)
  a = a_ref[...]
  b = b_ref[...]

  # To scale each row of `b` by the corresponding element of `a`,
  # we can rely on broadcasting.
  # The operation is (block_N, 1) * (block_N, block_M) -> (block_N, block_M).
  # FIX: `a` is already shaped (block_N, 1), so no need for `a[:, None]`.
  result_block = a * b

  # Write the computed block to the output reference in SRAM.
  out_ref[...] = result_block


# The pallas_call replaces the original computation.
# The grid is determined by how many blocks fit into the full arrays.
# The in_specs and out_specs define how data is chunked for each kernel instance.
# - For each output block (i, j), we need the corresponding block from B.
# - We also need the corresponding slice from A, which only depends on the row index i.
result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(B.shape, B.dtype),
  grid=(N // block_N, M // block_M),
  in_specs=[
    # FIX: Update BlockSpec for the 2D shape of A.
    pl.BlockSpec(block_shape=(block_N, 1), index_map=lambda i, j: (i, 0)),
    pl.BlockSpec(block_shape=(block_N, block_M), index_map=lambda i, j: (i, j)),
  ],
  out_specs=pl.BlockSpec(block_shape=(block_N, block_M), index_map=lambda i, j: (i, j)),
)(A, B).block_until_ready()

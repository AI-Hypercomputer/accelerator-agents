# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
N = 4096
key = random.PRNGKey(0)
key_A, key_B = random.split(key)

A = jnp.triu(random.normal(key_A, (N, N)))
B = jnp.triu(random.normal(key_B, (N, N)))

bN = 128


# Computation
def kernel(x_ref, y_ref, out_ref):
  """
  Computes C = jnp.triu(jnp.matmul(A, B)) in a tiled manner.

  Each program in the grid computes one (bN, bN) block of the output matrix C.
  The program's position in the grid (i, j) corresponds to the block's
  position in the output matrix.

  Args:
    x_ref: A reference to a (bN, N) block of the input matrix A.
    y_ref: A reference to a (N, bN) block of the input matrix B.
    out_ref: A reference to a (bN, bN) block of the output matrix C, which will
      be written to by this kernel.
  """
  i = pl.program_id(0)
  j = pl.program_id(1)

  def lower_triangle_branch():
    # If the block's column index `j` is less than its row index `i`,
    # the entire block is in the lower triangle of the result matrix.
    # Therefore, it should be all zeros.
    out_ref[...] = jnp.zeros_like(out_ref)

  def upper_and_diagonal_branch():
    # For blocks on or above the main diagonal (i <= j), we first compute
    # the matrix multiplication.
    result_block = x_ref[...] @ y_ref[...]

    def diagonal_branch():
      # If the block is on the main diagonal (i == j), we need to take the
      # upper triangular part of the computed block.
      out_ref[...] = jnp.triu(result_block)

    def upper_branch():
      # If the block is strictly above the main diagonal (i < j), the entire
      # block is part of the upper triangle, so we write the full result.
      out_ref[...] = result_block

    jax.lax.cond(i == j, diagonal_branch, upper_branch)

  jax.lax.cond(i > j, lower_triangle_branch, upper_and_diagonal_branch)


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

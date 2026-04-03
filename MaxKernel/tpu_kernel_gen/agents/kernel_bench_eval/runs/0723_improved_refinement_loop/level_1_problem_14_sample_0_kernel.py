# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax import lax
from jax.experimental import pallas as pl

# Initialization
N = 4096
bN = 128
key = random.PRNGKey(0)
key_A, key_B = random.split(key)

A = jnp.triu(random.normal(key_A, (N, N)))
B = jnp.triu(random.normal(key_B, (N, N)))


# Computation
def kernel(x_ref, y_ref, out_ref):
  """Pallas kernel to compute C = jnp.triu(jnp.matmul(A, B)).

  This kernel computes a block of the output matrix C. The logic depends
  on the position of the block relative to the main diagonal of the matrix.

  Args:
    x_ref: A block of the first input matrix A.
    y_ref: A block of the second input matrix B.
    out_ref: The output block of matrix C to be computed in-place.
  """
  # Get the 2D program ID, which corresponds to the block's coordinates.
  i, j = pl.program_id(0), pl.program_id(1)

  # The computation depends on whether the block is in the lower triangle,
  # upper triangle, or on the diagonal of the output matrix.
  # We use lax.cond for conditional execution based on tracer values.
  def lower_triangle_case():
    # If the block's column index `j` is less than its row index `i`,
    # the entire block is in the lower triangle of the result matrix.
    # Therefore, all its elements should be zero.
    out_ref[...] = jnp.zeros_like(out_ref)

  def upper_part_case():
    # If the block is on or above the diagonal (j >= i), we first need
    # to compute the matrix multiplication for this block.
    result = x_ref[...] @ y_ref[...]

    def on_diagonal_case():
      # If the block is on the main diagonal (j == i), we apply the
      # upper-triangular operation `jnp.triu` to the resulting block.
      out_ref[...] = jnp.triu(result)

    def strictly_upper_case():
      # If the block is in the upper triangle (j > i), the result of the
      # matmul is written directly to the output block.
      out_ref[...] = result

    lax.cond(j == i, on_diagonal_case, strictly_upper_case)

  lax.cond(j < i, lower_triangle_case, upper_part_case)


C = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((N, N), A.dtype),
  grid=(N // bN, N // bN),
  in_specs=[
    pl.BlockSpec(block_shape=(bN, N), index_map=lambda i, j: (i, 0)),
    pl.BlockSpec(block_shape=(N, bN), index_map=lambda i, j: (0, j)),
  ],
  out_specs=pl.BlockSpec(block_shape=(bN, bN), index_map=lambda i, j: (i, j)),
)(A, B).block_until_ready()

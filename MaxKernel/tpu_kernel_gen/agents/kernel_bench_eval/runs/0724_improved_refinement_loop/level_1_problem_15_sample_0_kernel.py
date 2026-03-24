# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax import lax
from jax.experimental import pallas as pl

# Initialization
M = 4096
key = random.PRNGKey(0)
key_A, key_B = random.split(key)

A = random.normal(key_A, (M, M))
B = random.normal(key_B, (M, M))
A = jnp.tril(A)
B = jnp.tril(B)

bM = 128


# Computation
def kernel(x_ref, y_ref, z_ref):
  """Computes C = tril(A @ B) in a block-wise manner.

  Args:
    x_ref: A reference to a block of matrix A.
    y_ref: A reference to a block of matrix B.
    z_ref: A reference to a block of the output matrix C, which will be
      modified in-place.
  """
  i = pl.program_id(0)
  j = pl.program_id(1)

  # The computation is C = jnp.tril(jnp.matmul(A, B)).
  # We are computing a single block of C at position (i, j).

  def upper_triangle_fn(_):
    # If the block's row index `i` is less than its column index `j`,
    # the entire block is in the upper triangle of the result matrix C.
    # Therefore, all its elements must be zero.
    z_ref[...] = jnp.zeros_like(z_ref)

  def lower_or_diagonal_fn(_):
    # For blocks on or below the diagonal (i >= j), we first need to compute
    # the matrix multiplication. The in_specs are configured to load the
    # necessary slices of A and B to compute one block of the result.
    result_block = x_ref[...] @ y_ref[...]

    def diagonal_fn(_):
      # If the block is on the main diagonal (i == j), we apply jnp.tril
      # to the locally computed result block.
      z_ref[...] = jnp.tril(result_block)

    def lower_fn(_):
      # If the block is strictly in the lower triangle, the entire
      # computed block is part of the final result.
      z_ref[...] = result_block

    lax.cond(i == j, diagonal_fn, lower_fn, operand=None)

  # Use lax.cond to handle control flow based on traced values i and j.
  lax.cond(i < j, upper_triangle_fn, lower_or_diagonal_fn, operand=None)


C = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(A.shape, A.dtype),
  grid=(M // bM, M // bM),
  in_specs=[
    pl.BlockSpec(block_shape=(bM, M), index_map=lambda i, j: (i, 0)),
    pl.BlockSpec(block_shape=(M, bM), index_map=lambda i, j: (0, j)),
  ],
  out_specs=pl.BlockSpec(block_shape=(bM, bM), index_map=lambda i, j: (i, j)),
)(A, B).block_until_ready()

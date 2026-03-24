# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax import lax
from jax.experimental import pallas as pl

# Initialization
M = 4096
bM = 128
key = random.PRNGKey(0)
key_A, key_B = random.split(key)

A = jnp.tril(random.normal(key_A, (M, M)))
B = jnp.tril(random.normal(key_B, (M, M)))


# Computation
def kernel(a_ref, b_ref, c_ref):
  """Computes C = tril(A @ B) in a block-wise manner."""
  i, j = pl.program_id(0), pl.program_id(1)

  # A block C[i, j] is in the lower triangle of the full matrix C if i >= j.
  # If j > i, the block is in the upper triangle and should be zero.
  def upper_triangle_branch():
    c_ref[...] = jnp.zeros_like(c_ref)

  def lower_triangle_branch():
    # For blocks on or below the diagonal, compute the matrix multiplication.
    # The inputs a_ref and b_ref are slices of the original A and B matrices.
    # a_ref corresponds to a block-row of A, and b_ref to a block-column of B.
    result_block = a_ref[...] @ b_ref[...]

    def on_diagonal_branch():
      # If the block is on the main diagonal (i == j), we apply the lower
      # triangular mask to the computed block.
      c_ref[...] = jnp.tril(result_block)

    def below_diagonal_branch():
      # If the block is below the main diagonal (i > j), the entire block
      # is part of the result.
      c_ref[...] = result_block

    lax.cond(i == j, on_diagonal_branch, below_diagonal_branch)

  lax.cond(j > i, upper_triangle_branch, lower_triangle_branch)


C = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(A.shape, A.dtype),
  grid=(A.shape[0] // bM, A.shape[0] // bM),
  in_specs=[
    pl.BlockSpec(block_shape=(bM, A.shape[1]), index_map=lambda i, j: (i, 0)),
    pl.BlockSpec(block_shape=(B.shape[0], bM), index_map=lambda i, j: (0, j)),
  ],
  out_specs=pl.BlockSpec(block_shape=(bM, bM), index_map=lambda i, j: (i, j)),
)(A, B).block_until_ready()

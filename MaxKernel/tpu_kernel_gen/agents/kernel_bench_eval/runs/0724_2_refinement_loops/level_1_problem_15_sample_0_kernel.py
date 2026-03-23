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
def kernel(x_ref, y_ref, out_ref):
  """Pallas kernel for tril(matmul(A, B)).

  This kernel computes the matrix product of two lower-triangular matrices A and B,
  and ensures the output is also lower-triangular. The computation is parallelized
  over a 2D grid of blocks.

  Args:
    x_ref: A reference to a block of rows of matrix A.
    y_ref: A reference to a block of columns of matrix B.
    out_ref: A reference to a block of the output matrix C, to be written to in-place.
  """
  # Get the 2D program ID for the current block.
  i = pl.program_id(0)
  j = pl.program_id(1)

  # The logic is split into two cases based on the block indices.
  # We use jax.lax.cond to handle the conditional logic in a way that is
  # compatible with JAX's tracing mechanism.

  def compute_block(_):
    """Computes the output block for cases where j <= i."""
    # x_ref[...] @ y_ref[...] computes the matrix product for the current block.
    c_block = x_ref[...] @ y_ref[...]
    # For blocks on the main diagonal (i == j), we must apply jnp.tril.
    # For blocks strictly below the diagonal (j < i), we use the full block.
    # jnp.where selects between these two outcomes based on the condition i == j.
    return jnp.where(i == j, jnp.tril(c_block), c_block)

  def zero_block(_):
    """Returns a block of zeros for cases where j > i."""
    return jnp.zeros_like(out_ref[...])

  # jax.lax.cond executes one of the two functions based on the predicate j > i.
  # If j > i, the block is in the upper triangle and should be zero.
  # Otherwise (j <= i), the block is on or below the diagonal and needs to be computed.
  out_ref[...] = lax.cond(j > i, zero_block, compute_block, None)


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

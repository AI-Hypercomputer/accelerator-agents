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

A = random.normal(key_A, (M, M))
B = random.normal(key_B, (M, M))
A = jnp.tril(A)
B = jnp.tril(B)


# Computation
def kernel(x_ref, y_ref, c_ref):
  """Pallas kernel for triangular matrix multiplication."""
  i, j = pl.program_id(0), pl.program_id(1)

  # This kernel computes `C = tril(A @ B)` in a block-wise fashion.
  # The logic changes depending on whether a block is in the lower triangle,
  # upper triangle, or on the diagonal of the output matrix C.

  def upper_triangle_case(_):
    # Blocks in the upper triangle of C are zero.
    c_ref[...] = jnp.zeros_like(c_ref)

  def lower_and_diagonal_case(_):
    # This branch handles i >= j (lower triangle and diagonal)
    res = x_ref[...] @ y_ref[...]

    def diagonal_case(_):
      # For blocks on the diagonal, take the lower triangular part.
      c_ref[...] = jnp.tril(res)

    def lower_case(_):
      # Blocks in the lower triangle are computed fully.
      c_ref[...] = res

    lax.cond(i == j, diagonal_case, lower_case, None)

  lax.cond(i < j, upper_triangle_case, lower_and_diagonal_case, None)


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

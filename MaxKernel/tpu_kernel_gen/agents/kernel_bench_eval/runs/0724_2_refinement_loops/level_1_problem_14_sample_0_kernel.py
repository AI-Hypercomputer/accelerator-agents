# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax import lax
from jax.experimental import pallas as pl

# Initialization
N = 4096
key = random.PRNGKey(0)
key_A, key_B = random.split(key)

A = jnp.triu(random.normal(key_A, (N, N)))
B = jnp.triu(random.normal(key_B, (N, N)))
bN = 128


# Computation
def kernel(x_ref, y_ref, z_ref):
  """Pallas kernel for multiplying two upper-triangular matrices."""
  # Get the block indices for the output block.
  i, j = pl.program_id(0), pl.program_id(1)

  # The product of two upper-triangular matrices is itself upper-triangular.
  # This means that any output block where the row index `i` is greater
  # than the column index `j` will be entirely zero. We can skip the
  # expensive matrix multiplication for these blocks.
  def true_fun(x_ref, y_ref, z_ref):
    z_ref[...] = jnp.zeros_like(z_ref)

  def false_fun(x_ref, y_ref, z_ref):
    # For blocks on or above the main diagonal (i <= j), we perform
    # the matrix multiplication. The result of the matmul for these blocks
    # will correctly form the upper-triangular part of the final matrix.
    # The original computation jnp.triu(jnp.matmul(A, B)) is equivalent to
    # jnp.matmul(A, B) in this case because the inputs A and B are already
    # upper-triangular, making their product upper-triangular.
    z_ref[...] = x_ref[...] @ y_ref[...]

  lax.cond(i > j, true_fun, false_fun, x_ref, y_ref, z_ref)


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

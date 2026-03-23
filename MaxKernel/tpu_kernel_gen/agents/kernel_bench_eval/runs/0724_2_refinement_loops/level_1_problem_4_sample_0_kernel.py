# Imports
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
M = 256
K = 131072
bM = 128
bK = 1024  # Block size for the K dimension
key = random.PRNGKey(0)
key_A, key_B = random.split(key)

A = random.normal(key_A, (M, K), dtype=jnp.float32)
B = random.normal(key_B, (K, 1), dtype=jnp.float32)

# The output C must be initialized to zeros for the accumulation to work.
C_init = jnp.zeros((M, 1), dtype=A.dtype)


# Computation
def kernel(a_ref, b_ref, c_ref):
  """Pallas kernel for matrix-vector multiplication.

  Args:
    a_ref: A block of the input matrix A of shape (bM, bK).
    b_ref: A block of the input vector B of shape (bK, 1).
    c_ref: The output block of C of shape (bM, 1) to be written to.
  """
  # Each program instance computes a partial dot product and accumulates
  # it into the output slice.
  # The shapes are: (bM, bK) @ (bK, 1) -> (bM, 1)
  # The result is added to the existing values in c_ref.
  c_ref[...] = c_ref[...] + jnp.matmul(a_ref[...], b_ref[...])


C = pl.pallas_call(
  kernel,
  # The output C has shape (M, 1) and is initialized to zero.
  out_shape=C_init,
  # The grid will iterate over blocks of M and K dimensions.
  grid=(M // bM, K // bK),
  in_specs=[
    # For A, each kernel instance `(i, j)` gets a block of shape (bM, bK).
    # The index_map `lambda i, j: (i, j)` selects the (i, j)-th block.
    pl.BlockSpec(block_shape=(bM, bK), index_map=lambda i, j: (i, j)),
    # For B, each kernel instance `(i, j)` gets a vertical slice of shape (bK, 1).
    # The index_map `lambda i, j: (j, 0)` selects the j-th block, broadcast
    # across the i dimension of the grid.
    pl.BlockSpec(block_shape=(bK, 1), index_map=lambda i, j: (j, 0)),
  ],
  # For the output C, instances with the same `i` write to the same
  # slice. The results are accumulated.
  out_specs=pl.BlockSpec(block_shape=(bM, 1), index_map=lambda i, j: (i, 0)),
)(A, B).block_until_ready()

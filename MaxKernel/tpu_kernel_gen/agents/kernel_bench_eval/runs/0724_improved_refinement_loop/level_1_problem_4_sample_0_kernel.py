# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
M = 256
K = 131072
key = random.PRNGKey(0)
key_A, key_B = random.split(key)

A = random.normal(key_A, (M, K), dtype=jnp.float32)
B = random.normal(key_B, (K, 1), dtype=jnp.float32)

bM = 8
bK = 2048


# Computation
def kernel(x_ref, y_ref, out_ref):
  """
  Computes a partial block of the matrix-vector product C = A @ B and
  atomically adds it to the output.

  Args:
    x_ref: A block of matrix A of shape (bM, bK).
    y_ref: A block of matrix B of shape (bK, 1).
    out_ref: A block of the output matrix C of shape (bM, 1) to accumulate into.
  """
  # Perform the matrix multiplication on the blocks.
  partial_out = jnp.matmul(x_ref[...], y_ref[...])
  # Atomically add the partial result to the output block.
  pl.atomic_add(out_ref, (slice(None), slice(None)), partial_out)


# The initial value for the output accumulator must be zero.
C_initial = jnp.zeros((A.shape[0], B.shape[1]), dtype=A.dtype)

C = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((A.shape[0], B.shape[1]), A.dtype),
  # Grid is 2D to tile over both M and K dimensions.
  grid=(A.shape[0] // bM, K // bK),
  in_specs=[
    # A is tiled into (bM, bK) blocks.
    pl.BlockSpec(block_shape=(bM, bK), index_map=lambda i, k: (i, k)),
    # B is tiled into (bK, 1) blocks.
    pl.BlockSpec(block_shape=(bK, 1), index_map=lambda i, k: (k, 0)),
  ],
  # Each row-block of the output is an accumulator.
  out_specs=pl.BlockSpec(block_shape=(bM, 1), index_map=lambda i, k: (i, 0)),
)(A, B, C_initial).block_until_ready()

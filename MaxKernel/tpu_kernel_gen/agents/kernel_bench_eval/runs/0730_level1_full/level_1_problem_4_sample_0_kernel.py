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

A = random.normal(key_A, (M, K))
B = random.normal(key_B, (K, 1))

bM = 128
bK = 1024


# Computation
def kernel(a_ref, b_ref, c_ref):
  """
  Computes a block of the matrix-vector product C = A @ B.

  Args:
    a_ref: A block of the input matrix A of shape (bM, bK).
    b_ref: A block of the input vector B of shape (bK, 1).
    c_ref: The output block for C of shape (bM, 1).
  """
  # Perform the matrix multiplication for the current block.
  # This computes (bM, bK) @ (bK, 1) = (bM, 1), which is a partial
  # sum for the final output block c_ref.
  # We accumulate this partial sum into the output.
  c_ref[...] += jnp.matmul(a_ref[...], b_ref[...])


C = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((M, 1), A.dtype),
  grid=(M // bM, K // bK),
  in_specs=[
    pl.BlockSpec(block_shape=(bM, bK), index_map=lambda i, j: (i, j)),
    pl.BlockSpec(block_shape=(bK, 1), index_map=lambda i, j: (j, 0)),
  ],
  out_specs=pl.BlockSpec(block_shape=(bM, 1), index_map=lambda i, j: (i, 0)),
  compiler_params=dict(tpu=dict(dimension_semantics=("parallel", "reduction"))),
)(A, B).block_until_ready()

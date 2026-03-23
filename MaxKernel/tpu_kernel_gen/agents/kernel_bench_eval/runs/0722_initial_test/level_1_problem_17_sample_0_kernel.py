# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
M = 1024
K = 4096
N = 2048
bM = 128
bN = 128
key = random.PRNGKey(0)
key_A, key_B = random.split(key)

A = random.normal(key_A, (M, K))
B = random.normal(key_B, (N, K))


# Computation
def kernel(a_ref, b_ref, c_ref):
  """Computes a tile of C = A @ B.T."""
  # a_ref is a block of A with shape (bM, K)
  # b_ref is a block of B with shape (bN, K)
  # c_ref is a block of C with shape (bM, bN)
  # The computation C = A @ B.T requires multiplying a (bM, K) matrix
  # by a (K, bN) matrix.
  # We can achieve this by taking the transpose of b_ref.
  c_ref[...] = jnp.matmul(a_ref[...], b_ref[...].T)


C = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
  grid=(M // bM, N // bN),
  in_specs=[
    pl.BlockSpec(block_shape=(bM, K), index_map=lambda i, j: (i, 0)),
    pl.BlockSpec(block_shape=(bN, K), index_map=lambda i, j: (j, 0)),
  ],
  out_specs=pl.BlockSpec(block_shape=(bM, bN), index_map=lambda i, j: (i, j)),
)(A, B).block_until_ready()

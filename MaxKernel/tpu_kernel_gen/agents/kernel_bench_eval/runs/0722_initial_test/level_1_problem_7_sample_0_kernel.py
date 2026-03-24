# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
M = 16384
N = 16384
K = 32
key = random.PRNGKey(0)
key_A, key_B = random.split(key)

A = random.normal(key_A, (M, K))
B = random.normal(key_B, (K, N))

bM = 128
bN = 128


# Computation
def kernel(a_ref, b_ref, c_ref):
  """Computes a block of the matrix multiplication C = A @ B."""
  # The block of A has shape (bM, K)
  # The block of B has shape (K, bN)
  # The block of C has shape (bM, bN)
  # The computation is a standard matrix multiplication on the blocks.
  c_ref[...] = jnp.dot(a_ref[...], b_ref[...])


C = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((M, N), A.dtype),
  grid=(M // bM, N // bN),
  in_specs=[
    pl.BlockSpec(block_shape=(bM, K), index_map=lambda i, j: (i, 0)),
    pl.BlockSpec(block_shape=(K, bN), index_map=lambda i, j: (0, j)),
  ],
  out_specs=pl.BlockSpec(block_shape=(bM, bN), index_map=lambda i, j: (i, j)),
)(A, B).block_until_ready()

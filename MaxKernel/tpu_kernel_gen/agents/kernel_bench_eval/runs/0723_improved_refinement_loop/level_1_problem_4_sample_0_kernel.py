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
def kernel(x_ref, y_ref, z_ref):
  """Computes a partial block of the matrix-vector product."""
  # The accumulation over the K dimension is handled by the grid and a
  # sum reduction outside the pallas_call.
  z_ref[...] = jnp.matmul(x_ref[...], y_ref[...])


# The grid now iterates over the K dimension as well.
# The output of the pallas_call will be a set of partial products.
partial_C = pl.pallas_call(
  kernel,
  # Output shape is (num_m_blocks, num_k_blocks, bM, 1)
  out_shape=jax.ShapeDtypeStruct((M // bM, K // bK, bM, 1), A.dtype),
  # Grid iterates over both M and K dimensions
  grid=(M // bM, K // bK),
  in_specs=[
    # Each kernel gets a (bM, bK) block of A
    pl.BlockSpec((bM, bK), lambda i, k: (i * bM, k * bK)),
    # Each kernel gets a (bK, 1) block of B
    pl.BlockSpec((bK, 1), lambda i, k: (k * bK, 0)),
  ],
  # Each kernel writes its (bM, 1) output to a unique slot in the
  # partials grid.
  out_specs=pl.BlockSpec((bM, 1), lambda i, k: (i, k, 0, 0)),
)(A, B)

# Sum the partial products over the K-block dimension and reshape.
C = partial_C.sum(axis=1).reshape((M, 1)).block_until_ready()

# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
N = 4096
key = random.PRNGKey(0)
key_A, key_B = random.split(key)

A = random.normal(key_A, (N, N))
A = (A + A.T) / 2
B = random.normal(key_B, (N, N))
B = (B + B.T) / 2

bN = 128


# Computation
def kernel(x_ref, y_ref, out_ref):
  """Computes a block of the matrix multiplication C = A @ B."""
  # The in_specs load a (bN, N) block of A and an (N, bN) block of B.
  # The matmul of these two blocks results in a (bN, bN) block,
  # which corresponds to a block of the output matrix C.
  out_ref[...] = jnp.matmul(x_ref[...], y_ref[...])


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

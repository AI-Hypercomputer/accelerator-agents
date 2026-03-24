# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
M = 1024
K = 4096
N = 2048
key = random.PRNGKey(0)
key_A, key_B = random.split(key)

A = random.normal(key_A, (K, M))
B = random.normal(key_B, (K, N))

bM = 128
bN = 128


def kernel(x_ref, y_ref, z_ref):
  """
  Computes z = x.T @ y.
  """
  z_ref[...] = jnp.matmul(x_ref[...].T, y_ref[...])


# Computation
C = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((M, N), A.dtype),
  grid=(M // bM, N // bN),
  in_specs=[
    pl.BlockSpec(block_shape=(K, bM), index_map=lambda i, j: (0, i)),
    pl.BlockSpec(block_shape=(K, bN), index_map=lambda i, j: (0, j)),
  ],
  out_specs=pl.BlockSpec(block_shape=(bM, bN), index_map=lambda i, j: (i, j)),
)(A, B).block_until_ready()

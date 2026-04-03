# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
N = 2048
key = random.PRNGKey(0)
key_A, key_B = random.split(key)

A = random.normal(key_A, (N, N))
B = random.normal(key_B, (N, N))

bN = 128


def kernel(x_ref, y_ref, z_ref):
  """Matrix multiplication kernel."""
  z_ref[...] = jnp.matmul(x_ref[...], y_ref[...])


# Computation
C = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(A.shape, A.dtype),
  grid=(A.shape[0] // bN, B.shape[1] // bN),
  in_specs=[
    pl.BlockSpec(block_shape=(bN, A.shape[1]), index_map=lambda i, j: (i, 0)),
    pl.BlockSpec(block_shape=(B.shape[0], bN), index_map=lambda i, j: (0, j)),
  ],
  out_specs=pl.BlockSpec(block_shape=(bN, bN), index_map=lambda i, j: (i, j)),
)(A, B).block_until_ready()

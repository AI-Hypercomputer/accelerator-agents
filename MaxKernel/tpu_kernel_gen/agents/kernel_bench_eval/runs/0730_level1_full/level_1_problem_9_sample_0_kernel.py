# Imports
import jax
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
M = 16384
N = 16
key = random.PRNGKey(0)
key_A, key_B = random.split(key)

A = random.normal(key_A, (M, N))
B = random.normal(key_B, (N, M))

bM = 128


# Computation
def kernel(x_ref, y_ref, z_ref):
  z_ref[...] = x_ref[...] @ y_ref[...]


C = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((M, M), A.dtype),
  grid=(M // bM, M // bM),
  in_specs=[
    pl.BlockSpec(block_shape=(bM, N), index_map=lambda i, j: (i, 0)),
    pl.BlockSpec(block_shape=(N, bM), index_map=lambda i, j: (0, j)),
  ],
  out_specs=pl.BlockSpec(block_shape=(bM, bM), index_map=lambda i, j: (i, j)),
)(A, B).block_until_ready()

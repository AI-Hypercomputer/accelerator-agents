# Imports
import jax
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


def kernel(x_ref, y_ref, z_ref):
  z_ref[...] = x_ref[...] @ y_ref[...]


# Computation
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

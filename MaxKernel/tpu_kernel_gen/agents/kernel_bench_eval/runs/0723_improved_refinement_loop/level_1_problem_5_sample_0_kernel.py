# Imports
import jax
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
M = 16384
N = 4096
key = random.PRNGKey(0)
A = random.normal(key, (M, N))
s = 3.14
bM = 128
bN = 128


def kernel(a_ref, s_ref, c_ref):
  c_ref[...] = a_ref[...] * s_ref[...]


# Computation
C = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(A.shape, A.dtype),
  grid=(M // bM, N // bN),
  in_specs=[
    pl.BlockSpec(block_shape=(bM, bN), index_map=lambda i, j: (i, j)),
    pl.BlockSpec(block_shape=(), index_map=lambda i, j: ()),
  ],
  out_specs=pl.BlockSpec(block_shape=(bM, bN), index_map=lambda i, j: (i, j)),
)(A, s).block_until_ready()

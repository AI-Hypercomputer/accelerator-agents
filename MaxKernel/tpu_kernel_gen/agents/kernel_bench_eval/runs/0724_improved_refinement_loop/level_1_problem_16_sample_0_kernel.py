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


# Computation
def kernel(a_ref, b_ref, c_ref):
  # The computation is C = A.T @ B.
  # `a_ref` is a block of A with shape (K, bM).
  # `b_ref` is a block of B with shape (K, bN).
  # To compute a block of C, we need to transpose the block of A
  # and then matrix multiply it with the block of B.
  # (bM, K) @ (K, bN) -> (bM, bN)
  # This is equivalent to contracting the first (K) dimension of each block.
  a = a_ref[...]
  b = b_ref[...]
  c_ref[...] = jnp.einsum("ki,kj->ij", a, b)


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

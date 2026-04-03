# Imports
import jax
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
M = 4096
N = 4096
key = random.PRNGKey(0)
key_A, key_B = random.split(key)

A = random.normal(key_A, (N,))
B = random.normal(key_B, (N, M))

bN = 128
bM = 128


# Computation
def kernel(a_ref, b_ref, out_ref):
  """Pallas kernel for A[:, jnp.newaxis] * B."""
  out_ref[:, :] = a_ref[:, None] * b_ref[:, :]


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((N, M), A.dtype),
  grid=(N // bN, M // bM),
  in_specs=[
    pl.BlockSpec(block_shape=(bN,), index_map=lambda i, j: (i * bN,)),
    pl.BlockSpec(block_shape=(bN, bM), index_map=lambda i, j: (i * bN, j * bM)),
  ],
  out_specs=pl.BlockSpec(block_shape=(bN, bM), index_map=lambda i, j: (i * bN, j * bM)),
)(A, B).block_until_ready()

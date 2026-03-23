# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
M = 256
N = 256
K = 131072
bM = 128
bN = 128
bK = 128
key = random.PRNGKey(0)
key_A, key_B = random.split(key)
A = random.normal(key_A, (M, K))
B = random.normal(key_B, (K, N))


# Computation
def kernel(x_ref, y_ref, z_ref):
  """Computes z = x @ y."""
  # No need to calculate program IDs, they are handled by BlockSpec.
  acc = jnp.zeros((bM, bN), dtype=jnp.float32)
  for k in range(K // bK):
    # The blockspecs in pallas_call handle the indexing.
    x_block = pl.load(x_ref, (k,))
    y_block = pl.load(y_ref, (k,))
    acc += pl.dot(x_block, y_block)
  pl.store(z_ref, (0, 0), acc)


C = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
  grid=(M // bM, N // bN),
  in_specs=[
    pl.BlockSpec(lambda i, j: (i, 0), (bM, K)),
    pl.BlockSpec(lambda i, j: (0, j), (K, bN)),
  ],
  out_specs=pl.BlockSpec(lambda i, j: (i, j), (bM, bN)),
  # Add memory space information for TPU
  input_output_aliases={2: 0},
)(A, B).block_until_ready()

# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
M = 8205
K = 2949
N = 5921
key = random.PRNGKey(0)
key_A, key_B = random.split(key)

A = random.normal(key_A, (M, K))
B = random.normal(key_B, (K, N))

bM = 128
bN = 128


# Computation
def kernel(x_ref, y_ref, z_ref):
  """Matrix multiplication kernel.

  Args:
    x_ref: A block of matrix A.
    y_ref: A block of matrix B.
    z_ref: The output block.
  """
  z_ref[...] = jnp.dot(x_ref[...], y_ref[...])


C = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((M, N), A.dtype),
  grid=(pl.cdiv(M, bM), pl.cdiv(N, bN)),
  in_specs=[
    pl.BlockSpec(block_shape=(bM, K), index_map=lambda i, j: (i, 0)),
    pl.BlockSpec(block_shape=(K, bN), index_map=lambda i, j: (0, j)),
  ],
  out_specs=pl.BlockSpec(block_shape=(bM, bN), index_map=lambda i, j: (i, j)),
)(A, B).block_until_ready()

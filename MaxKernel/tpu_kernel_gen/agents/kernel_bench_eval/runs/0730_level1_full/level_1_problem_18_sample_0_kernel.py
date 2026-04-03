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
B = random.normal(key_B, (N, K))

bM = 128
bN = 128


# Computation
def kernel(x_ref, y_ref, z_ref):
  """Pallas kernel for C = A.T @ B.T.

  Args:
    x_ref: A block of A with shape (K, bM).
    y_ref: A block of B with shape (bN, K).
    z_ref: A block of the output C with shape (bM, bN) to be computed.
  """
  # The computation is C = A.T @ B.T.
  # x_ref is a slice of A of shape (K, bM). Its transpose has shape (bM, K).
  # y_ref is a slice of B of shape (bN, K). Its transpose has shape (K, bN).
  # The matmul(x_ref.T, y_ref.T) results in a block of shape (bM, bN),
  # which corresponds to a tile of the output matrix C.
  x = x_ref[...]
  y = y_ref[...]
  z_ref[...] = jnp.matmul(x.T, y.T)


C = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((A.shape[1], B.shape[0]), A.dtype),
  grid=(A.shape[1] // bM, B.shape[0] // bN),
  in_specs=[
    pl.BlockSpec(block_shape=(A.shape[0], bM), index_map=lambda i, j: (0, i)),
    pl.BlockSpec(block_shape=(bN, B.shape[1]), index_map=lambda i, j: (j, 0)),
  ],
  out_specs=pl.BlockSpec(block_shape=(bM, bN), index_map=lambda i, j: (i, j)),
)(A, B).block_until_ready()

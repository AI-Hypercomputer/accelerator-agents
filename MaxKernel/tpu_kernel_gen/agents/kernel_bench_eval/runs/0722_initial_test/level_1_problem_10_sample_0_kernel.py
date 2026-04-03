# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
N = 16
M = 1024
K = 2048
L = 768
key = random.PRNGKey(0)
key_A, key_B = random.split(key)

A = random.normal(key_A, (N, M, K))
B = random.normal(key_B, (K, L))

bM = 128
bL = 128


# Computation
def kernel(x_ref, y_ref, out_ref):
  """
  Pallas kernel for batched matrix multiplication.

  Args:
    x_ref: A reference to a block of the first input tensor 'A'.
           Shape: (1, bM, K)
    y_ref: A reference to a block of the second input tensor 'B'.
           Shape: (K, bL)
    out_ref: A reference to a block of the output tensor 'C' to be updated.
             Shape: (1, bM, bL)
  """
  # Perform the matrix multiplication on the blocks.
  # (bM, K) @ (K, bL) -> (bM, bL)
  # We slice the inputs to be 2D, as required by the TPU matmul hardware.
  out_ref[0] = jnp.matmul(x_ref[0], y_ref[...])


C = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((N, M, L), A.dtype),
  grid=(N, M // bM, L // bL),
  in_specs=[
    pl.BlockSpec(block_shape=(1, bM, K), index_map=lambda i, j, k: (i, j * bM, 0)),
    pl.BlockSpec(block_shape=(K, bL), index_map=lambda i, j, k: (0, k * bL)),
  ],
  out_specs=pl.BlockSpec(block_shape=(1, bM, bL), index_map=lambda i, j, k: (i, j * bM, k * bL)),
)(A, B).block_until_ready()

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

C_shape = (N, M, L)
dtype = jnp.float32

# Block sizes for tiling
bM = 128
bL = 128


def kernel(a_ref, b_ref, c_ref):
  """Pallas kernel for batched matrix multiplication.

  This kernel computes a tile of the batched matrix multiplication C = A @ B.
  Each program in the grid is responsible for computing one output tile.

  Args:
    a_ref: A reference to a slice of the input A.
           Shape: (1, bM, K).
    b_ref: A reference to a slice of the input B.
           Shape: (K, bL).
    c_ref: A reference to a slice of the output C, to be written to.
           Shape: (1, bM, bL).
  """
  # Load the input slice from HBM into a register.
  a = a_ref[...]
  # Squeeze the leading dimension to make it a 2D matrix for the matmul.
  a_squeezed = jnp.squeeze(a, axis=0)

  # The core computation is a standard 2D matrix multiplication.
  c = jnp.dot(a_squeezed, b_ref[...])

  # Add the leading dimension back to the result to match the shape of the
  # output slice c_ref and write it back to HBM.
  c_ref[...] = c[None, ...]


# Computation
C = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(C_shape, dtype),
  grid=(N, pl.cdiv(M, bM), pl.cdiv(L, bL)),
  in_specs=[
    # For A, we parallelize over the batch dim (i) and tile the M dim (j).
    # The K dim is the reduction axis, so we take the whole slice.
    pl.BlockSpec(block_shape=(1, bM, K), index_map=lambda i, j, k: (i, j, 0)),
    # For B, we tile the L dim (k). It's broadcast across the batch and M dims.
    pl.BlockSpec(block_shape=(K, bL), index_map=lambda i, j, k: (0, k)),
  ],
  out_specs=pl.BlockSpec(
    # The output C is tiled across all three grid dimensions.
    block_shape=(1, bM, bL),
    index_map=lambda i, j, k: (i, j, k),
  ),
)(A, B).block_until_ready()

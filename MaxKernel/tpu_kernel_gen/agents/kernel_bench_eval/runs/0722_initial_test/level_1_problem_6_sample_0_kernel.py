# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax import lax
from jax.experimental import pallas as pl

# Initialization
M = 256
N = 256
K = 131072
key = random.PRNGKey(0)
key_A, key_B = random.split(key)

A = random.normal(key_A, (M, K))
B = random.normal(key_B, (K, N))

bM = 128
bN = 128
bK = 128


def kernel(x_ref, y_ref, out_ref):
  """Matrix multiplication kernel.

  This kernel computes a block of the output matrix C by multiplying a block of
  A (x_ref) with a block of B (y_ref).

  Args:
    x_ref: A reference to a block of the input matrix A.
    y_ref: A reference to a block of the input matrix B.
    out_ref: A reference to a block of the output matrix C, which will be
      populated by this kernel.
  """
  # Initialize accumulator with zeros
  acc = jnp.zeros((bM, bN), dtype=jnp.float32)

  # Loop over the K dimension in blocks of bK
  def body(k, acc):
    # Load blocks of A and B
    a_block = pl.load(x_ref, (0, k * bK), block_shape=(bM, bK))
    b_block = pl.load(y_ref, (k * bK, 0), block_shape=(bK, bN))
    # Perform matrix multiplication and accumulate
    return acc + a_block @ b_block

  acc = lax.fori_loop(0, K // bK, body, acc)

  # Store the final result
  out_ref[...] = acc


# Computation
C = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((M, N), A.dtype),
  grid=(M // bM, N // bN),
  in_specs=[
    pl.BlockSpec((bM, K), lambda i, j: (i, 0)),
    pl.BlockSpec((K, bN), lambda i, j: (0, j)),
  ],
  out_specs=pl.BlockSpec((bM, bN), lambda i, j: (i, j)),
)(A, B).block_until_ready()

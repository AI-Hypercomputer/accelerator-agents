# Imports
import jax
import jax.random as random
from jax.experimental import pallas as pl
from jax.lax import dot

# Initialization
batch_size = 128
m = 128
k = 256
n = 512
key = random.PRNGKey(0)
key_A, key_B = random.split(key)
A = random.normal(key_A, (batch_size, m, k))
B = random.normal(key_B, (batch_size, k, n))

bM = 128
bN = 128


def kernel(x_ref, y_ref, out_ref):
  """Computes a block of a batched matrix multiplication."""
  # x_ref: A block of shape (bM, k) from the first input matrix.
  # y_ref: A block of shape (k, bN) from the second input matrix.
  # out_ref: A block of shape (bM, bN) for the output matrix.

  # Load the blocks into SRAM.
  x = x_ref[...]
  y = y_ref[...]

  # Perform the matrix multiplication on the loaded blocks.
  # Use pallas.ops.tpu.dot for TPU-optimized matrix multiplication.
  out_ref[...] = dot(x, y)


# Computation
C = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, m, n), A.dtype),
  grid=(batch_size, m // bM, n // bN),
  in_specs=[
    pl.BlockSpec(block_shape=(bM, k), index_map=lambda i, j, k: (i, j, 0)),
    pl.BlockSpec(block_shape=(k, bN), index_map=lambda i, j, k: (i, 0, k)),
  ],
  out_specs=pl.BlockSpec(block_shape=(bM, bN), index_map=lambda i, j, k: (i, j, k)),
)(A, B).block_until_ready()

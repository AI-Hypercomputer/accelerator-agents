# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
m = 128
k = 256
n = 512
key = random.PRNGKey(0)
key_A, key_B = random.split(key)

A = random.normal(key_A, (batch_size, m, k))
B = random.normal(key_B, (batch_size, k, n))


# Computation
def kernel(x_ref, y_ref, out_ref):
  """
  Pallas kernel for batched matrix multiplication.

  Args:
    x_ref: A reference to a slice of the first input matrix A.
    y_ref: A reference to a slice of the second input matrix B.
    out_ref: A reference to a slice of the output matrix C, to be updated in-place.
  """
  # Each program in the grid computes one matrix multiplication from the batch.
  # x_ref corresponds to A[i, :, :] with shape (1, m, k)
  # y_ref corresponds to B[i, :, :] with shape (1, k, n)
  # The result of the matmul will have shape (1, m, n), which matches out_ref.
  x = x_ref[0, :, :]
  y = y_ref[0, :, :]
  out_ref[0, :, :] = jnp.matmul(x, y)


C = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, m, n), A.dtype),
  grid=(batch_size,),
  in_specs=[
    pl.BlockSpec(block_shape=(1, m, k), index_map=lambda i: (i, 0, 0)),
    pl.BlockSpec(block_shape=(1, k, n), index_map=lambda i: (i, 0, 0)),
  ],
  out_specs=pl.BlockSpec(block_shape=(1, m, n), index_map=lambda i: (i, 0, 0)),
)(A, B).block_until_ready()

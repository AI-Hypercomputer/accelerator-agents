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
def kernel(a_ref, b_ref, c_ref):
  """
  Computes the matrix multiplication for a single batch element.

  Args:
    a_ref: A reference to a (1, m, k) block of the first input tensor.
    b_ref: A reference to a (1, k, n) block of the second input tensor.
    c_ref: A reference to a (1, m, n) block of the output tensor.
  """
  # The grid is defined over the batch dimension. Each program instance
  # receives a full (m, k) and (k, n) matrix and is responsible for
  # computing the corresponding (m, n) output matrix.
  c_ref[0] = jnp.matmul(a_ref[0], b_ref[0])


C = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, m, n), A.dtype),
  grid=(batch_size,),
  in_specs=[
    pl.BlockSpec(block_shape=(1, m, k), index_map=lambda b: (b, 0, 0)),
    pl.BlockSpec(block_shape=(1, k, n), index_map=lambda b: (b, 0, 0)),
  ],
  out_specs=pl.BlockSpec(block_shape=(1, m, n), index_map=lambda b: (b, 0, 0)),
)(A, B).block_until_ready()

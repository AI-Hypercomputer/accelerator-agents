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
block_N = 1024
block_M = 128


# Computation
def kernel(a_ref, b_ref, c_ref):
  """
  Computes diag(A) @ B element-wise.

  This kernel is equivalent to the following JAX code:
    A_broadcast = A[:, None]
    C = A_broadcast * B

  Args:
    a_ref: A block of the 'A' vector.
    b_ref: A block of the 'B' matrix.
    c_ref: The output block, which will be populated with the result.
  """
  # Load the block of A and reshape for broadcasting.
  a_block = a_ref[...]
  a_block_broadcast = a_block[:, None]

  # Load the block of B.
  b_block = b_ref[...]

  # Perform element-wise multiplication and write to the output.
  c_ref[...] = a_block_broadcast * b_block


C = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(B.shape, A.dtype),
  grid=(B.shape[0] // block_N, B.shape[1] // block_M),
  in_specs=[
    pl.BlockSpec(block_shape=(block_N,), index_map=lambda i, j: (i,)),
    pl.BlockSpec(block_shape=(block_N, block_M), index_map=lambda i, j: (i, j)),
  ],
  out_specs=pl.BlockSpec(block_shape=(block_N, block_M), index_map=lambda i, j: (i, j)),
)(A, B).block_until_ready()

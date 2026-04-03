# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
M = 4096
N = 4096
key = random.PRNGKey(0)
key_A, key_B = random.split(key)
A = random.normal(key_A, (N,))
B = random.normal(key_B, (N, M))
bN = 128
bM = 128


# Computation
def kernel(a_ref, b_ref, out_ref):
  """Pallas kernel for broadcasted multiplication.

  Args:
    a_ref: A reference to a slice of array A.
    b_ref: A reference to a slice of array B.
    out_ref: A reference to a slice of the output array.
  """
  # Perform the broadcasted multiplication.
  # a_ref is (bN, 1) and b_ref is (bN, bM), so they broadcast correctly.
  out_ref[...] = a_ref[...] * b_ref[...]


# Reshape A to be explicitly 2D for broadcasting.
A_reshaped = jnp.expand_dims(A, axis=1)

result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(B.shape, A.dtype),
  grid=(N // bN, M // bM),
  in_specs=[
    pl.BlockSpec(block_shape=(bN, 1), index_map=lambda i, j: (i, 0)),
    pl.BlockSpec(block_shape=(bN, bM), index_map=lambda i, j: (i, j)),
  ],
  out_specs=pl.BlockSpec(block_shape=(bN, bM), index_map=lambda i, j: (i, j)),
)(A_reshaped, B).block_until_ready()

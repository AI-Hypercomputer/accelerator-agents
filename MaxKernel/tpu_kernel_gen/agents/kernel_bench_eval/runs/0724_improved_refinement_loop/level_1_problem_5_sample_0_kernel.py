# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
M = 16384
N = 4096
key = random.PRNGKey(0)
A = random.normal(key, (M, N))
# Create a 0-dimensional JAX array for the scalar. This is good practice to
# ensure its dtype is well-defined and matches the array's dtype.
s = jnp.array(3.14, dtype=A.dtype)
BLOCK_M = 128
BLOCK_N = 128


# Computation
def kernel(a_ref, s_ref, c_ref):
  """
  Pallas kernel for element-wise multiplication of a matrix by a scalar.

  Args:
    a_ref: A reference to a block of the input matrix A.
    s_ref: A reference to the scalar value.
    c_ref: A reference to a block of the output matrix C, to be written to.
  """
  # Load the block of A from SRAM into registers, multiply by the scalar s,
  # and write the result to the output block in SRAM.
  c_ref[...] = a_ref[...] * s_ref[()]


C = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(A.shape, A.dtype),
  grid=(M // BLOCK_M, N // BLOCK_N),
  in_specs=[
    pl.BlockSpec(block_shape=(BLOCK_M, BLOCK_N), index_map=lambda i, j: (i, j)),
    # For scalar inputs, we provide a BlockSpec with an empty shape `()`
    # and an index_map that returns an empty tuple.
    pl.BlockSpec(block_shape=(), index_map=lambda i, j: ()),
  ],
  out_specs=pl.BlockSpec(block_shape=(BLOCK_M, BLOCK_N), index_map=lambda i, j: (i, j)),
)(A, s).block_until_ready()

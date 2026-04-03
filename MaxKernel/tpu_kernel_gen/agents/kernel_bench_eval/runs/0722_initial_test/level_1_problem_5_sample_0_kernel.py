# Imports
import jax
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
M = 16384
N = 4096
key = random.PRNGKey(0)
A = random.normal(key, (M, N))
s = 3.14
bM = 128


# Computation
def kernel(a_ref, s, c_ref):
  """Pallas kernel for element-wise multiplication by a scalar.

  Args:
    a_ref: A reference to a block of the input matrix A.
    s: The scalar value to multiply by.
    c_ref: A reference to the corresponding block of the output matrix C.
  """
  # Perform element-wise multiplication of the input block by the scalar
  # and write the result to the output block.
  c_ref[...] = a_ref[...] * s


C = pl.pallas_call(
  lambda a_ref, c_ref: kernel(a_ref, s, c_ref),
  out_shape=jax.ShapeDtypeStruct(A.shape, A.dtype),
  grid=(M // bM, 1),
  in_specs=[
    pl.BlockSpec(block_shape=(bM, N), index_map=lambda i, j: (i, 0)),
  ],
  out_specs=pl.BlockSpec(block_shape=(bM, N), index_map=lambda i, j: (i, 0)),
)(A).block_until_ready()

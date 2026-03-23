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
s = jnp.array([3.14])
bM = 128


# Computation
def kernel(a_ref, s_ref, c_ref):
  """Kernel for element-wise multiplication of a matrix by a scalar.

  Args:
    a_ref: A reference to a block of the input matrix A.
    s_ref: A reference to the scalar s.
    c_ref: A reference to a block of the output matrix C, to be written to.
  """
  # Perform the element-wise multiplication.
  # a_ref[...] loads the input block of A from SRAM into registers.
  # s_ref[...] loads the scalar s.
  # JAX handles the broadcasting of the scalar across the matrix block.
  # The result is then written to the output block c_ref in HBM.
  c_ref[...] = a_ref[...] * s_ref[...]


C = pl.pallas_call(
  kernel,
  # The output C has the same shape and dtype as the input A.
  out_shape=jax.ShapeDtypeStruct(A.shape, A.dtype),
  # The grid is 1D, parallelizing over the M dimension of A.
  grid=(M // bM,),
  # in_specs defines how inputs are chunked.
  in_specs=[
    # For A, we take horizontal slices of shape (bM, N). The grid index `i`
    # maps to the i-th block of rows.
    pl.BlockSpec(block_shape=(bM, N), index_map=lambda i: (i, 0)),
    # The scalar `s` is a 1-dimensional array. It's passed in its entirety
    # to each kernel instance.
    pl.BlockSpec(block_shape=(1,), index_map=lambda i: (0,)),
  ],
  # out_specs defines how the output is chunked. It mirrors the spec for A.
  out_specs=pl.BlockSpec(block_shape=(bM, N), index_map=lambda i: (i, 0)),
)(A, s).block_until_ready()

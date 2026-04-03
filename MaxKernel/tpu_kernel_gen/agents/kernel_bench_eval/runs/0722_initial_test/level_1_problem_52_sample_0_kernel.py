# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
dim1 = 256
dim2 = 256
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim1, dim2))


# Computation
def kernel(x_ref, out_ref):
  """Pallas kernel to compute argmin along axis=0 of a 2D block.

  Args:
    x_ref: A reference to the input block of shape (1, dim1, dim2).
    out_ref: A reference to the output block of shape (1, dim2) to store the indices.
  """
  # The pallas_call is configured to slice the input `x` of shape
  # (batch_size, dim1, dim2) into blocks of shape (1, dim1, dim2).
  # The original operation is jnp.argmin(x, axis=1).
  # For each block, this corresponds to finding the argmin along the first
  # dimension of the squeezed block, which is axis=0 for the 2D slice.
  # The result is an array of indices of shape (dim2,), which is then
  # written to the output.
  out_ref[0, :] = jnp.argmin(x_ref[0, ...], axis=0)


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((x.shape[0], x.shape[2]), jnp.int32),
  grid_spec=pl.GridSpec(
    grid=(x.shape[0],),
    in_specs=[pl.BlockSpec((1, x.shape[1], x.shape[2]), lambda i: (i, 0, 0))],
    out_specs=[pl.BlockSpec((1, x.shape[2]), lambda i: (i, 0))],
  ),
)(x).block_until_ready()

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
  """Pallas kernel to compute the maximum value along axis 1.

  Args:
    x_ref: A reference to a block of the input tensor `x`.
    out_ref: A reference to a block of the output tensor, which will be
      written to in-place.
  """
  # The input block x_ref has shape (8, 256, 128).
  # The original operation is jnp.max(x, axis=1).
  # We apply this operation to the block we have loaded into SRAM.
  # jnp.max(x_ref[...], axis=1) will reduce the second dimension,
  # resulting in an array of shape (8, 128).
  # This result is then written to the output block out_ref, which has a
  # matching shape of (8, 128).
  out_ref[...] = jnp.max(x_ref[...], axis=1)


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, dim2), x.dtype),
  grid=(batch_size // 8, dim2 // 128),
  in_specs=[pl.BlockSpec(block_shape=(8, dim1, 128), index_map=lambda i, j: (i, 0, j))],
  out_specs=pl.BlockSpec(block_shape=(8, 128), index_map=lambda i, j: (i, j)),
)(x).block_until_ready()

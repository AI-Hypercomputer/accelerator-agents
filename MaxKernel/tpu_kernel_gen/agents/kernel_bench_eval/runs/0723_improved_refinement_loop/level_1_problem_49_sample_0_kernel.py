# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
dim = 1
batch_size = 16
dim1 = 256
dim2 = 256
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim1, dim2))

# Define block sizes for tiling
b_b = 8
b_d2 = 128


# Computation
def kernel(x_ref, out_ref):
  """Pallas kernel to compute max along axis=1.

  Args:
    x_ref: A reference to a block of the input tensor. The shape of this block
      is (b_b, dim1, b_d2).
    out_ref: A reference to a block of the output tensor. The shape of this
      block is (b_b, b_d2). This kernel will write the result of the max
      reduction into this block.
  """
  # x_ref[...] loads the input block from SRAM into registers.
  # jnp.max(..., axis=1) computes the maximum along the second axis.
  # The result of this operation has shape (b_b, b_d2), which matches
  # the shape of out_ref.
  # The result is then written to the output block in SRAM.
  out_ref[...] = jnp.max(x_ref[...], axis=1)


# The output shape after the reduction along axis=1
out_shape = jax.ShapeDtypeStruct((batch_size, dim2), x.dtype)

result = pl.pallas_call(
  kernel,
  out_shape=out_shape,
  grid=(batch_size // b_b, dim2 // b_d2),
  in_specs=[pl.BlockSpec(block_shape=(b_b, dim1, b_d2), index_map=lambda i, j: (i, 0, j))],
  out_specs=pl.BlockSpec(block_shape=(b_b, b_d2), index_map=lambda i, j: (i, j)),
)(x).block_until_ready()

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
result_shape = jax.ShapeDtypeStruct((batch_size, dim2), x.dtype)


# Computation
def kernel(x_ref, out_ref):
  """
  Computes the mean of x_ref along the second dimension.

  Args:
    x_ref: A reference to a slice of the input tensor `x`.
      The slice has shape (8, 256, 128) due to the BlockSpec.
    out_ref: A reference to a slice of the output tensor.
      The slice has shape (8, 128) and is where the result is stored.
  """
  # The input block x_ref has shape (8, 256, 128).
  # We compute the mean over axis 1.
  mean_val = jnp.mean(x_ref, axis=1)

  # Store the computed mean block into the output block.
  # out_ref has shape (8, 128), same as mean_val.
  out_ref[...] = mean_val


# The grid is defined over blocks of the output array.
# We use a block size of (8, 128) for the output, which is valid on TPU.
out_block_b = 8
out_block_d2 = 128
grid = (batch_size // out_block_b, dim2 // out_block_d2)

result = pl.pallas_call(
  kernel,
  out_shape=result_shape,
  grid=grid,
  in_specs=[pl.BlockSpec((out_block_b, dim1, out_block_d2), lambda i, j: (i * out_block_b, 0, j * out_block_d2))],
  out_specs=pl.BlockSpec((out_block_b, out_block_d2), lambda i, j: (i * out_block_b, j * out_block_d2)),
)(x).block_until_ready()

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

# Define block sizes for tiling, ensuring they meet TPU constraints
# For a 2D block, the first dimension must be divisible by 8 and the second by 128.
b_batch = 8
b_dim2 = 128


# Computation
def kernel(x_ref, out_ref):
  """
  Pallas kernel to compute the mean of a 3D tensor along the second axis.

  Args:
    x_ref: A reference to a tile of the input tensor 'x'. Based on the
           invocation, this tile has shape (b_batch, dim1, b_dim2).
    out_ref: A reference to a tile of the output tensor. This tile has
             shape (b_batch, b_dim2) and is where the result of the
             computation for this tile is stored.
  """
  # The in_spec loads a block of shape (b_batch, dim1, b_dim2).
  # We need to compute the mean across the `dim1` axis (axis=1).
  # jnp.mean(x_ref[...], axis=1) will produce a result of shape (b_batch, b_dim2),
  # which matches the shape of the output block `out_ref`.
  # We write this result directly into the output buffer.
  out_ref[...] = jnp.mean(x_ref[...], axis=1)


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, dim2), x.dtype),
  grid=(batch_size // b_batch, dim2 // b_dim2),
  in_specs=[pl.BlockSpec(block_shape=(b_batch, dim1, b_dim2), index_map=lambda i, j: (i, 0, j))],
  out_specs=pl.BlockSpec(block_shape=(b_batch, b_dim2), index_map=lambda i, j: (i, j)),
)(x).block_until_ready()

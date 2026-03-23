# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
eps = 1e-5
batch_size = 16
features = 64
dim1 = 256
dim2 = 256
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, features, dim1, dim2))
block_d1 = 8
block_d2 = 128


# Computation
def kernel(x_ref, out_ref):
  """Pallas kernel for RMS normalization.

  This kernel computes RMS normalization over the second axis (features)
  of the input tensor `x`. Each program in the grid handles a spatial
  block of the input, processing all features for that block.

  Args:
    x_ref: A reference to a block of the input tensor.
    out_ref: A reference to a block of the output tensor for in-place update.
  """
  # Epsilon for numerical stability, from the original computation.
  eps = 1e-5

  # Load the input block from SRAM into registers for computation.
  x = x_ref[...]

  # Compute the mean of the squares of the input block over the 'features'
  # dimension (axis=1). The axis is relative to the block's shape.
  # keepdims=True ensures the result can be broadcast for the division.
  mean_of_squares = jnp.mean(x * x, axis=1, keepdims=True)

  # Calculate the root mean square (RMS) value, adding epsilon for stability.
  rms = jnp.sqrt(mean_of_squares + eps)

  # Normalize the input block by dividing by the RMS values.
  # JAX handles broadcasting `rms` to the shape of `x`.
  # The result is written directly to the output block in-place.
  out_ref[...] = x / rms


output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(batch_size, dim1 // block_d1, dim2 // block_d2),
  in_specs=[pl.BlockSpec(block_shape=(1, features, block_d1, block_d2), index_map=lambda b, i, j: (b, 0, i, j))],
  out_specs=pl.BlockSpec(block_shape=(1, features, block_d1, block_d2), index_map=lambda b, i, j: (b, 0, i, j)),
)(x).block_until_ready()

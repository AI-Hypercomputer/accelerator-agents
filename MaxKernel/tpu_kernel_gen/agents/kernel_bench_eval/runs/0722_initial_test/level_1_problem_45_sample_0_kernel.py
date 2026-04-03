# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
channels = 64
height = 256
width = 256
kernel_size = 3
# PyTorch's stride=None defaults to kernel_size, which is 3 here.
stride = 3
padding = 0  # Corresponds to 'VALID' padding
key = random.PRNGKey(0)
# JAX/Flax uses the channel-last convention (N, H, W, C)
x = random.normal(key, (batch_size, height, width, channels))

# Calculate output shape for Pallas call
output_height = (height - kernel_size) // stride + 1
output_width = (width - kernel_size) // stride + 1
output_shape = (batch_size, output_height, output_width, channels)

# Define block size for tiling the width dimension
# This must be a multiple of 8 for TPU compatibility
bW = 16


# Computation
def kernel(x_ref, out_ref):
  """Pallas kernel for 2D average pooling with non-overlapping windows.

  This kernel computes the average value over a square window of the input
  and writes the result to the output. The tiling and data loading are
  configured in the pallas_call invocation.

  Args:
    x_ref: A reference to the input tile. The shape is expected to be
      (1, kernel_size, bW * stride, channels), where bW is the block size
      for the width dimension.
    out_ref: A reference to the output tile, to be written to in--place.
      The shape is expected to be (1, 1, bW, channels).
  """
  # Load the input tile from the reference into a JAX array.
  # Operations like reshape and sum need to be on concrete values, not refs.
  x_tile = x_ref[...]

  # Since stride equals kernel_size, we can reshape the input tile to isolate
  # the pooling windows.
  # The input shape is (1, kernel_size, bW * kernel_size, channels).
  # We reshape it to (1, kernel_size, bW, kernel_size, channels) to separate
  # the width dimension into 'bW' blocks of size 'kernel_size'.
  kernel_size = x_tile.shape[1]
  bW = out_ref.shape[2]
  channels = x_tile.shape[-1]
  x_reshaped = x_tile.reshape(1, kernel_size, bW, kernel_size, channels)

  # Sum over the spatial dimensions of the pooling window (the two dimensions
  # with size 'kernel_size', which are at axes 1 and 3).
  # The result has shape (1, bW, channels).
  summed_pool = jnp.sum(x_reshaped, axis=(1, 3))

  # Compute the average by dividing by the number of elements in the window.
  # We use a float to ensure float division.
  num_elements_in_window = (kernel_size * kernel_size) * 1.0
  avg_pool = summed_pool / num_elements_in_window

  # Reshape the result to match the output tile's shape (1, 1, bW, channels)
  # and write it to the output reference.
  out_ref[...] = avg_pool.reshape(1, 1, bW, channels)


# The Pallas kernel is invoked over a grid that maps to the output tiles.
# Each kernel instance computes one tile of the output.
output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(output_shape, x.dtype),
  grid=(batch_size, output_height, (output_width + bW - 1) // bW),
  in_specs=[
    pl.BlockSpec(lambda b, h, w_idx: (b, h * stride, w_idx * bW * stride, 0), (1, kernel_size, bW * stride, channels))
  ],
  out_specs=pl.BlockSpec(lambda b, h, w_idx: (b, h, w_idx * bW, 0), (1, 1, bW, channels)),
)(x).block_until_ready()

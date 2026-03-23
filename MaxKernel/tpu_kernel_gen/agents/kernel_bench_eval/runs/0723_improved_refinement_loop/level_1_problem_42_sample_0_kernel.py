# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
channels = 32
height = 128
width = 128
kernel_size = 2
stride = 2
padding = 1
dilation = 3

key = random.PRNGKey(0)
# JAX uses channels-last convention (N, H, W, C)
x = random.normal(key, (batch_size, height, width, channels))

# Calculate output shape
output_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
output_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


# Computation
def kernel(x_ref, o_ref):
  """Pallas kernel for 2D max pooling with dilation.

  This kernel computes the maximum value within a dilated window for each channel.
  The pallas_call is responsible for sliding this window across the input tensor.

  Args:
    x_ref: A reference to the input tile from the input tensor. Its shape is
      determined by in_specs and is large enough to contain the dilated window.
    o_ref: A reference to the output tile. Its shape is determined by out_specs
      and represents a view into the output tensor.
  """
  # These parameters are fixed by the problem description.
  kernel_size = 2
  dilation = 3

  # Initialize a variable to hold the maximum value for each channel.
  # We initialize it with a very small number (-inf) to ensure correctness
  # for any float dtype and to correctly handle negative input values and padding.
  max_val = jnp.full_like(o_ref[0, 0, 0, :], -jnp.inf)

  # Iterate over the logical kernel window. For a small, fixed size like 2x2,
  # JAX will unroll these Python loops.
  for i in range(kernel_size):
    for j in range(kernel_size):
      # Access the element at the dilated position within the input tile.
      current_val = x_ref[0, i * dilation, j * dilation, :]
      # Update the running maximum for each channel.
      max_val = jnp.maximum(max_val, current_val)

  # Write the final computed maximum values to the output reference.
  # The o_ref view might be larger than 1x1x1xN due to TPU constraints,
  # but each kernel invocation is responsible for a single output pixel.
  # We write to the start of our assigned tile, which corresponds to the
  # correct output coordinate.
  o_ref[0, 0, 0, :] = max_val


output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, output_height, output_width, channels), x.dtype),
  grid=(batch_size, output_height, output_width),
  in_specs=[
    pl.BlockSpec(
      # The block shape must be large enough to contain the dilated window.
      block_shape=(1, (kernel_size - 1) * dilation + 1, (kernel_size - 1) * dilation + 1, channels),
      index_map=lambda b, h, w: (b, h * stride - padding, w * stride - padding, 0),
    )
  ],
  out_specs=pl.BlockSpec(
    # Each kernel invocation computes a single output pixel (across all channels).
    block_shape=(1, 1, 1, channels),
    index_map=lambda b, h, w: (b, h, w, 0),
  ),
)(x).block_until_ready()

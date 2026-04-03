# Imports
import math

import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
channels = 32
depth = 64
height = 64
width = 64
kernel_size = 3
stride = 2
padding = 1
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, depth, height, width, channels))

# Calculate output dimensions
out_depth = math.floor((depth + 2 * padding - kernel_size) / stride) + 1
out_height = math.floor((height + 2 * padding - kernel_size) / stride) + 1
out_width = math.floor((width + 2 * padding - kernel_size) / stride) + 1


def kernel(x_ref, o_ref):
  """Pallas kernel for 3D average pooling with padding.

  This kernel implements the equivalent of flax.linen.avg_pool with
  `count_include_pad=False`.
  """
  # Get the output indices from the grid program IDs.
  # These correspond to the output depth and height.
  od = pl.program_id(1)
  oh = pl.program_id(2)

  # Hardcoded parameters based on the source computation.
  # In a more general kernel, these would be passed as static arguments.
  depth, height, width = 64, 64, 64
  kernel_size = 3
  stride = 2
  padding = 1

  out_width = o_ref.shape[3]
  channels = o_ref.shape[4]

  # Iterate over the output width dimension for the current output block.
  for ow in range(out_width):
    # Initialize an accumulator for the sum of window elements.
    acc = jnp.zeros(channels, dtype=x_ref.dtype)
    # Initialize a counter for valid (non-padded) elements in the window.
    valid_count = 0

    # Iterate over the 3D kernel window.
    for kd in range(kernel_size):
      for kh in range(kernel_size):
        for kw in range(kernel_size):
          # Calculate the corresponding indices in the original input tensor.
          d_idx = od * stride - padding + kd
          h_idx = oh * stride - padding + kh
          w_idx = ow * stride - padding + kw

          # Check if the indices fall within the original tensor's bounds.
          is_valid = (d_idx >= 0) & (d_idx < depth) & (h_idx >= 0) & (h_idx < height) & (w_idx >= 0) & (w_idx < width)

          # Increment the count for each valid element.
          valid_count += jnp.where(is_valid, 1, 0)

          # Read from the input reference and add to the accumulator.
          # The index_map for x_ref is now adjusted by -padding.
          # Pallas handles the out-of-bounds access by returning 0.
          # We use local indices for the read.
          local_w_idx = ow * stride + kw
          acc += x_ref[0, kd, kh, local_w_idx, :]

    # To avoid division by zero for windows entirely in padding,
    # the divisor must be at least 1.
    divisor = jnp.maximum(valid_count, 1.0)
    # Calculate the average and write to the output reference.
    o_ref[0, 0, 0, ow, :] = acc / divisor


# The width of the input slice needs to be large enough to cover all reads
# for an entire output row, including padding.
w_slice_size = (out_width - 1) * stride + kernel_size
# On TPU, the width dimension of a block needs to be a multiple of 8.
w_slice_size_padded = (w_slice_size + 7) & ~7

# Computation
output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, out_depth, out_height, out_width, channels), x.dtype),
  grid=(batch_size, out_depth, out_height),
  in_specs=[
    pl.BlockSpec(
      block_shape=(1, kernel_size, kernel_size, w_slice_size_padded, channels),
      index_map=lambda b, od, oh: (b, od * stride - padding, oh * stride - padding, -padding, 0),
    )
  ],
  out_specs=pl.BlockSpec(block_shape=(1, 1, 1, out_width, channels), index_map=lambda b, od, oh: (b, od, oh, 0, 0)),
)(x).block_until_ready()

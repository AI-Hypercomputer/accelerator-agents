# Imports
import math

import jax
import jax.numpy as jnp
import jax.random as random
from jax import lax
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
channels = 64
height = 256
width = 256
kernel_size = 3
# In PyTorch's AvgPool2d, a stride of None defaults to kernel_size.
stride = (kernel_size, kernel_size)
key = random.PRNGKey(0)
# JAX/Flax convention is (N, H, W, C)
x = random.normal(key, (batch_size, height, width, channels))

# Calculate output shape based on 'VALID' padding
output_height = (height - kernel_size) // stride[0] + 1
output_width = (width - kernel_size) // stride[1] + 1
output_shape_struct = jax.ShapeDtypeStruct(shape=(batch_size, output_height, output_width, channels), dtype=x.dtype)

# Define tile sizes for the Pallas kernel, ensuring TPU compatibility.
# The input tile width must be a multiple of 8.
# (output_tile_dim - 1) * stride + kernel_size must be a multiple of 8.
# (output_tile_dim - 1) * 3 + 3 must be a multiple of 8.
# Choosing 8 for the output tile dimension makes the input tile dimension 24.
output_tile_h, output_tile_w = 8, 8

# Calculate the required input tile size to produce one output tile
input_tile_h = (output_tile_h - 1) * stride[0] + kernel_size
input_tile_w = (output_tile_w - 1) * stride[1] + kernel_size

# Define the execution grid based on the number of output tiles
grid = (
  batch_size,
  math.ceil(output_height / output_tile_h),
  math.ceil(output_width / output_tile_w),
)


# Computation
def kernel(x_ref, out_ref):
  """
  Pallas kernel for average pooling with 'VALID' padding.

  This kernel computes the average value over a sliding window on the input
  and writes the result to the output. It iterates over each pixel of the
  output tile, calculates the corresponding window in the input tile,
  computes the mean, and stores the result.

  Args:
    x_ref: A reference to the input tile. The shape is expected to be
      (1, input_tile_h, input_tile_w, channels).
    out_ref: A reference to the output tile, which will be modified in-place.
      The shape is expected to be (1, output_tile_h, output_tile_w, channels).
  """
  # These parameters are fixed for this specific pooling operation.
  kernel_size = 3
  stride_h, stride_w = 3, 3

  # The shape of the output tile determines the loop bounds.
  _, output_tile_h, output_tile_w, channels = out_ref.shape

  # Iterate over each spatial location in the output tile.
  def body_h(i, _):
    def body_w(j, __):
      # Determine the top-left corner of the pooling window in the input tile.
      start_h = i * stride_h
      start_w = j * stride_w

      # Initialize an accumulator for the sum.
      sum_val = jnp.zeros(channels, dtype=x_ref.dtype)

      # A single fori_loop over the kernel area is more robust than nested loops.
      def kernel_loop_body(k, current_sum):
        # Unravel the 2D kernel index (ki, kj) from the 1D loop index k.
        ki = k // kernel_size
        kj = k % kernel_size
        # Add the value from the input tile to the accumulator.
        return current_sum + x_ref[0, start_h + ki, start_w + kj, :]

      # Sum over the kernel window.
      sum_val = lax.fori_loop(0, kernel_size * kernel_size, kernel_loop_body, sum_val)

      # Compute the mean and assign it to the output.
      pooled_value = sum_val / (kernel_size * kernel_size)
      out_ref[0, i, j, :] = pooled_value

    lax.fori_loop(0, output_tile_w, body_w, None)

  lax.fori_loop(0, output_tile_h, body_h, None)


output = pl.pallas_call(
  kernel,
  out_shape=output_shape_struct,
  grid=grid,
  in_specs=[
    pl.BlockSpec(
      block_shape=(1, input_tile_h, input_tile_w, channels),
      index_map=lambda b, i, j: (
        b,
        i * output_tile_h * stride[0],
        j * output_tile_w * stride[1],
        0,
      ),
    )
  ],
  out_specs=pl.BlockSpec(
    block_shape=(1, output_tile_h, output_tile_w, channels),
    index_map=lambda b, i, j: (b, i * output_tile_h, j * output_tile_w, 0),
  ),
)(x).block_until_ready()

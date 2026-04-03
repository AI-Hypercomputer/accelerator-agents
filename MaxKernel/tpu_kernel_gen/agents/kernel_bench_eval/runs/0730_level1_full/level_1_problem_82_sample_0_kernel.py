# Imports
import math

import jax
import jax.numpy as jnp
import jax.random as random
from flax.linen import Conv
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 3
kernel_size = 3
width = 256
height = 256
stride = 1
padding = "VALID"
key = random.PRNGKey(0)
key_x, key_params = random.split(key)

# JAX uses channels-last convention by default
x = random.normal(key_x, (batch_size, height, width, in_channels))

conv2d = Conv(
  features=in_channels,
  kernel_size=(kernel_size, kernel_size),
  strides=(stride, stride),
  padding=padding,
  feature_group_count=in_channels,
  use_bias=False,
)
params = conv2d.init(key_params, x)["params"]

# Define block sizes for tiling
# We tile along height AND width to satisfy TPU constraints.
out_block_h = 64
out_block_w = 64
out_h = height - kernel_size + 1
out_w = width - kernel_size + 1

# The input block must be larger to provide the full receptive field.
in_block_h = out_block_h + kernel_size - 1
in_block_w_needed = out_block_w + kernel_size - 1
# Pad the input block width to be a multiple of 8 for TPU constraints.
in_block_w = math.ceil(in_block_w_needed / 8) * 8


# Calculate the grid dimensions.
grid_h = math.ceil(out_h / out_block_h)
grid_w = math.ceil(out_w / out_block_w)


# Computation
def kernel(x_ref, kernel_ref, y_ref):
  """Pallas kernel for depthwise convolution.

  Args:
    x_ref: Input tile.
    kernel_ref: Convolution kernel.
    y_ref: Output tile.
  """
  # Since we are not using a bias, we can initialize the output to zeros.
  y_ref[...] = jnp.zeros_like(y_ref)

  # Get the kernel dimensions from the shape of the kernel reference.
  kernel_h, kernel_w, _, _ = kernel_ref.shape

  # Iterate over the spatial dimensions of the kernel.
  for i in range(kernel_h):
    for j in range(kernel_w):
      # Slice the input tile to get the receptive field for the current
      # kernel element. The slice size is the same as the output tile.
      in_slice = x_ref[0, i : i + out_block_h, j : j + out_block_w, :]

      # Get the kernel value for the current position.
      # The shape is (1, in_channels), which will broadcast correctly.
      k_val = kernel_ref[i, j, :, :]

      # Perform element-wise multiplication and accumulate the result.
      y_ref[0, ...] += in_slice * k_val


# The pallas_call replaces the original convolution.
# The kernel function (not defined here) would contain the logic for
# a single tile of the depthwise convolution.
y = pl.pallas_call(
  kernel,
  # The output of the pallas_call has the same shape and type as the original op.
  out_shape=jax.ShapeDtypeStruct((batch_size, out_h, out_w, in_channels), x.dtype),
  # Grid iterates over batch and tiles of output height and width.
  grid=(batch_size, grid_h, grid_w),
  in_specs=[
    # Input x: A slice of (1, in_block_h, in_block_w, in_channels)
    # is passed to each kernel instance.
    pl.BlockSpec(
      block_shape=(1, in_block_h, in_block_w, in_channels),
      index_map=lambda b, h_i, w_i: (b, h_i * out_block_h, w_i * out_block_w, 0),
    ),
    # Input kernel: The full kernel is passed to each instance.
    pl.BlockSpec(
      block_shape=params["kernel"].shape,
      index_map=lambda b, h_i, w_i: (0, 0, 0, 0),
    ),
  ],
  # Output y: Each kernel instance writes to a tile of the output.
  out_specs=pl.BlockSpec(
    block_shape=(1, out_block_h, out_block_w, in_channels),
    index_map=lambda b, h_i, w_i: (b, h_i * out_block_h, w_i * out_block_w, 0),
  ),
)(x, params["kernel"]).block_until_ready()

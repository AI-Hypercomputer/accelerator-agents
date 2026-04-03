# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 3
out_channels = 3
kernel_size = 3
width_in = 256
height_in = 128
stride = 1
padding = 0
bias = False

# Calculate output dimensions
width_out = width_in - kernel_size + 1
height_out = height_in - kernel_size + 1

# Define block sizes for tiling
block_w = 128
block_h = 1

key = random.PRNGKey(0)
key_x, key_params = random.split(key)

x = random.normal(key_x, (batch_size, height_in, width_in, in_channels))

conv2d = nn.Conv(
  features=out_channels,
  kernel_size=(kernel_size, kernel_size),
  strides=stride,
  padding=padding,
  feature_group_count=in_channels,
  use_bias=bias,
)
variables = conv2d.init(key_params, x)


# Computation
def kernel(x_ref, w_ref, y_ref):
  """
  Pallas kernel for 2D depthwise convolution.

  This kernel computes a tile of the output tensor `y` by sliding a kernel `w`
  over a corresponding input patch `x`. It is designed for depthwise
  convolutions where `in_channels == out_channels` and each input channel is
  convolved with its own dedicated filter.

  Args:
    x_ref: A reference to the input tensor tile. The shape is expected to be
      (1, kernel_h, padded_in_block_w, in_channels), which includes
      the necessary padding to compute a full output tile.
    w_ref: A reference to the entire kernel weights tensor. The shape is
      expected to be (kernel_h, kernel_w, 1, out_channels) for a depthwise
      convolution.
    y_ref: A reference to the output tensor tile that this kernel will compute
      and write to. The shape is (1, 1, block_w, out_channels).
  """
  # Initialize an accumulator with zeros, with the same shape as the output tile.
  acc = jnp.zeros(y_ref.shape, dtype=y_ref.dtype)

  # Extract the spatial dimensions of the kernel from the shape of the weights reference.
  kernel_h, kernel_w, _, _ = w_ref.shape
  _, _, block_w_val, out_channels_val = y_ref.shape

  x_pad = x_ref[...]

  # Iterate over the spatial dimensions of the kernel (e.g., for a 3x3 kernel,
  # kh and kw will range from 0 to 2).
  for kh in range(kernel_h):
    for kw in range(kernel_w):
      # Statically slice the input patch `x_pad`. Since the loop over kh and kw
      # is unrolled by JAX, the indices are static from the compiler's perspective.
      x_slice = jax.lax.slice(
        x_pad, start_indices=(0, kh, kw, 0), limit_indices=(1, kh + 1, kw + block_w_val, out_channels_val)
      )

      # Slice the weights tensor to get the filters for the current (kh, kw) position.
      w_slice = w_ref[kh, kw, :, :][...]

      # Perform the core convolution operation: element-wise multiplication and accumulation.
      # `x_slice` has shape (1, 1, block_w, out_channels).
      # `w_slice` has shape (1, out_channels). JAX's broadcasting rules
      # align the channel dimensions, effectively applying each channel's weight
      # to its corresponding feature map in `x_slice`.
      acc += x_slice * w_slice

  # Write the final computed tile from the accumulator to the output reference.
  y_ref[...] = acc


# The kernel's weights are extracted from the variables dictionary
w = variables["params"]["kernel"]

# Padded width for TPU alignment
padded_in_block_w = ((block_w + kernel_size - 1 + 7) // 8) * 8

# pallas_call replaces the original convolution operation
y = pl.pallas_call(
  kernel,
  # The output shape is determined by the convolution parameters
  out_shape=jax.ShapeDtypeStruct((batch_size, height_out, width_out, out_channels), x.dtype),
  # The grid is defined to parallelize over batch, height, and tiled width
  grid=(batch_size, height_out, (width_out + block_w - 1) // block_w),
  in_specs=[
    # BlockSpec for the input tensor 'x'
    # The input patch must be large enough to compute the output tile
    pl.BlockSpec(
      block_shape=(1, kernel_size, padded_in_block_w, in_channels),
      index_map=lambda b, h, w_idx: (b, h, w_idx * block_w, 0),
    ),
    # BlockSpec for the kernel weights 'w'
    # The entire kernel is loaded for each work-item
    pl.BlockSpec(block_shape=w.shape, index_map=lambda *_: (0, 0, 0, 0)),
  ],
  # BlockSpec for the output tensor 'y'
  # The output is written in tiles of size (1, 1, 128, out_channels)
  out_specs=pl.BlockSpec(
    block_shape=(1, block_h, block_w, out_channels), index_map=lambda b, h, w_idx: (b, h, w_idx * block_w, 0)
  ),
)(x, w).block_until_ready()

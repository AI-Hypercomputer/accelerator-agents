# Imports
import math

import flax.linen as nn
import jax
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
width = 256
height = 256
stride = 1
padding = 0
dilation = 1
groups = 1
bias = False

key = random.PRNGKey(0)
key_x, key_params = random.split(key)

x = random.normal(key_x, (batch_size, height, width, in_channels))
conv2d = nn.Conv(
  features=out_channels,
  kernel_size=(kernel_size, kernel_size),
  strides=(stride, stride),
  padding=padding,
  kernel_dilation=(dilation, dilation),
  feature_group_count=groups,
  use_bias=bias,
)

# Initialize parameters for the convolution layer.
# The pallas_call will use these kernel weights.
params = conv2d.init(key_params, x)["params"]


# Computation
def kernel(x_ref, kernel_ref, out_ref):
  """Pallas kernel for 2D convolution.

  This kernel computes a block of the output of a 2D convolution.
  It uses jax.lax.conv_general_dilated to perform the core computation
  on the input and kernel blocks provided by the pallas_call.

  Args:
    x_ref: A reference to a block of the input image.
    kernel_ref: A reference to the convolution kernel weights.
    out_ref: A reference to a block of the output tensor where the result
      is stored.
  """
  # These parameters are fixed based on the source computation.
  stride = 1
  dilation = 1

  # Perform the 2D convolution on the input block (x_ref) with the kernel
  # weights (kernel_ref).
  # - The `lhs` (left-hand side) is the input feature map block.
  # - The `rhs` (right-hand side) is the kernel.
  # - `padding='VALID'` is used because the pallas_call is configured to
  #   load a sufficiently large slice of the input `x` that already accounts
  #   for the kernel's footprint. No further padding is needed inside the kernel.
  # - `dimension_numbers` specifies the data layout for the tensors:
  #   'NHWC' for input/output: Batch, Height, Width, Channels.
  #   'HWIO' for kernel: Height, Width, Input Channels, Output Channels.
  output_block = jax.lax.conv_general_dilated(
    lhs=x_ref[...],
    rhs=kernel_ref[...],
    window_strides=(stride, stride),
    padding="VALID",
    rhs_dilation=(dilation, dilation),
    dimension_numbers=("NHWC", "HWIO", "NHWC"),
  )

  # Write the computed output block to the output reference. This is an
  # in-place operation from the perspective of the kernel.
  out_ref[...] = output_block


# Calculate output dimensions based on convolution parameters
height_out = math.floor((height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1
width_out = math.floor((width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1

# Define block sizes for tiling the computation.
# bH and bW are the height and width of the output block each kernel will compute.
bH = height_out
bW = width_out

# Calculate the required input block size to produce an output block of size (bH, bW).
in_bH = (bH - 1) * stride + kernel_size
in_bW = (bW - 1) * stride + kernel_size

# The pallas_call replaces the original convolution computation.
output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, height_out, width_out, out_channels), x.dtype),
  grid=(batch_size, math.ceil(height_out / bH), math.ceil(width_out / bW)),
  in_specs=[
    # Specification for the input image 'x'
    pl.BlockSpec(
      block_shape=(1, in_bH, in_bW, in_channels),
      index_map=lambda b, h, w: (b, h * bH * stride, w * bW * stride, 0),
    ),
    # Specification for the convolution kernel weights
    pl.BlockSpec(block_shape=(kernel_size, kernel_size, in_channels, out_channels), index_map=lambda *_: (0, 0, 0, 0)),
  ],
  out_specs=pl.BlockSpec(block_shape=(1, bH, bW, out_channels), index_map=lambda b, h, w: (b, h * bH, w * bW, 0)),
)(x, params["kernel"]).block_until_ready()

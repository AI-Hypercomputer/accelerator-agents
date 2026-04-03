# Imports
import jax
import jax.random as random
from flax import linen as nn
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = (3, 5, 7)
width = 64
height = 64
depth = 64

key = random.PRNGKey(0)
key_x, key_params = random.split(key)

# JAX uses channels-last convention: (N, D, H, W, C)
x = random.normal(key_x, (batch_size, depth, height, width, in_channels))

conv3d = nn.Conv(
  features=out_channels,
  kernel_size=kernel_size,
  strides=1,
  padding="VALID",
  kernel_dilation=1,
  feature_group_count=1,
  use_bias=False,
)
variables = conv3d.init(key_params, x)
kernel_weights = variables["params"]["kernel"]

# Define output shape
out_depth = depth - kernel_size[0] + 1
out_height = height - kernel_size[1] + 1
out_width = width - kernel_size[2] + 1
output_shape = (batch_size, out_depth, out_height, out_width, out_channels)

# Define block sizes for the output dimensions
# We choose block sizes for width and height that are multiples of 8 to satisfy TPU constraints.
b_h = 8
b_w = 8


def kernel(x_ref, kernel_weights_ref, out_ref):
  """Pallas kernel for 3D convolution.

  This kernel computes a single block of the 3D convolution output. The grid
  is designed to iterate over the batch and spatial dimensions of the output
  tensor.

  Args:
    x_ref: A reference to the input block. This block is a patch of the
      original input tensor, sized appropriately to compute the corresponding
      output block.
    kernel_weights_ref: A reference to the entire kernel weights tensor. This
      is loaded into SRAM for each kernel invocation.
    out_ref: A reference to the output block, which this kernel will compute
      and fill.
  """
  # The core of the convolution operation is performed by
  # `jax.lax.conv_general_dilated`. This is the same primitive that high-level
  # libraries like Flax use under the hood.
  # We operate on the blocks of data (`x_ref`, `kernel_weights_ref`) that
  # Pallas has loaded into SRAM.
  output_block = jax.lax.conv_general_dilated(
    lhs=x_ref[...],
    rhs=kernel_weights_ref[...],
    window_strides=(1, 1, 1),
    padding="VALID",
    # The dimension_numbers argument is crucial for telling JAX how to
    # interpret the axes of the input arrays.
    # 'NDHWC': Batch, Depth, Height, Width, Channels for input/output.
    # 'DHWIO': Depth, Height, Width, Input Channels, Output Channels for kernel.
    dimension_numbers=("NDHWC", "DHWIO", "NDHWC"),
    feature_group_count=1,
  )

  # The result of the convolution on the input patch has the exact shape of
  # the output block, even for partial blocks at the boundaries.
  out_ref[...] = output_block[...]


# Computation
# Pallas kernel invocation
output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(output_shape, x.dtype),
  grid=(batch_size, out_depth, (out_height + b_h - 1) // b_h, (out_width + b_w - 1) // b_w),
  in_specs=[
    pl.BlockSpec(
      block_shape=(1, kernel_size[0], b_h + kernel_size[1] - 1, b_w + kernel_size[2] - 1, in_channels),
      index_map=lambda n, d, h, w: (n, d, h * b_h, w * b_w, 0),
    ),
    pl.BlockSpec(block_shape=kernel_weights.shape, index_map=lambda *_: (0, 0, 0, 0, 0)),
  ],
  out_specs=pl.BlockSpec(
    block_shape=(1, 1, b_h, b_w, out_channels), index_map=lambda n, d, h, w: (n, d, h * b_h, w * b_w, 0)
  ),
)(x, kernel_weights).block_until_ready()

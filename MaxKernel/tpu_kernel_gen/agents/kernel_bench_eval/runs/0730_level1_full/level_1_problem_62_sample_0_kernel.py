# Imports
import jax
import jax.random as random
from flax import linen as nn
from jax.experimental import pallas as pl
from jax.experimental.pallas.ops import tpu as tpu_ops

# Initialization
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = (3, 5)
width = 256
height = 256


class ConvModel(nn.Module):
  @nn.compact
  def __call__(self, x):
    return nn.Conv(
      features=out_channels,
      kernel_size=kernel_size,
      strides=1,
      padding="VALID",
      kernel_dilation=1,
      feature_group_count=1,
      use_bias=False,
      name="Conv_0",
    )(x)


key = random.PRNGKey(0)
key, init_key, x_key = random.split(key, 3)

conv2d = ConvModel()
x = random.normal(x_key, (batch_size, height, width, in_channels))
params = conv2d.init(init_key, x)["params"]


def kernel(x_ref, kernel_ref, out_ref):
  """Pallas kernel for 2D convolution.

  Args:
    x_ref: A reference to a block of the input tensor.
    kernel_ref: A reference to the convolution kernel tensor.
    out_ref: A reference to a block of the output tensor to be populated.
  """
  # Perform the convolution on the loaded blocks of input and kernel.
  # The dimension numbers specify the layout of the tensors:
  # 'NHWC' for input/output: Batch, Height, Width, Channels.
  # 'HWIO' for the kernel: Height, Width, Input Channels, Output Channels.
  # The padding is 'VALID' as we have already loaded the necessary patch of
  # the input.
  # Use the Pallas-specific TPU convolution primitive.
  # This primitive operates directly on block references, so we don't
  # use [...] to dereference them.
  conv_result = tpu_ops.conv(
    lhs=x_ref,
    rhs=kernel_ref,
    window_strides=(1, 1),
    padding=((0, 0), (0, 0)),
    dimension_numbers=("NHWC", "HWIO", "NHWC"),
  )

  # The input block width (in_block_w) might be larger than what is strictly
  # necessary to compute the output block (out_block_w), a practice known
  # as overfetching, often done to meet hardware alignment constraints.
  # This results in conv_result being wider than out_ref. We must slice
  # conv_result to match the shape of out_ref before writing the result.
  out_ref[...] = conv_result[:, :, : out_ref.shape[2], :]


# Computation
# Calculate output shape and grid dimensions
output_height = height - kernel_size[0] + 1
output_width = width - kernel_size[1] + 1
output_shape = (batch_size, output_height, output_width, out_channels)

# Define block sizes for tiling that satisfy TPU constraints
# Output width block must be divisible by 8.
out_block_w = 8
# Input width block must be divisible by 8. We overfetch to satisfy this.
# Required input width = out_block_w + kernel_width - 1 = 8 + 5 - 1 = 12
# Smallest multiple of 8 >= 12 is 16.
in_block_w = 16
# Input height block
in_block_h = kernel_size[0]

grid = (batch_size, output_height, (output_width + out_block_w - 1) // out_block_w)

output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(output_shape, x.dtype),
  grid=grid,
  in_specs=[
    pl.BlockSpec(
      block_shape=(1, in_block_h, in_block_w, in_channels),
      index_map=lambda b, h, w: (b, h, w * out_block_w, 0),
    ),
    pl.BlockSpec(
      block_shape=(*kernel_size, in_channels, out_channels),
      index_map=lambda b, h, w: (0, 0, 0, 0),
    ),
  ],
  out_specs=pl.BlockSpec(
    block_shape=(1, 1, out_block_w, out_channels),
    index_map=lambda b, h, w: (b, h, w * out_block_w, 0),
  ),
)(x, params["Conv_0"]["kernel"]).block_until_ready()

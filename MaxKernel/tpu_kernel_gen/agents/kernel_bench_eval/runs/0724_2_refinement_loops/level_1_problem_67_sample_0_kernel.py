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
length = 512
stride = 1
padding = "VALID"  # PyTorch padding=0 is equivalent to 'VALID' in JAX/Flax
dilation = 1
groups = 1
bias = False

key = random.PRNGKey(0)
key_params, key_x = random.split(key)

# Note: JAX convention is (batch, length, channels) vs PyTorch's (batch, channels, length)
x = random.normal(key_x, (batch_size, length, in_channels))

conv1d = nn.Conv(
  features=out_channels,
  kernel_size=(kernel_size,),
  strides=(stride,),
  padding=padding,
  kernel_dilation=(dilation,),
  feature_group_count=groups,
  use_bias=bias,
)
params = conv1d.init(key_params, x)["params"]

# Calculate output shape based on 'VALID' padding
output_length = (length - kernel_size) // stride + 1

# Define block sizes for Pallas kernel
# The output block size along the length dimension must be a multiple of 8
# for the BlockSpec to be valid on TPU.
output_block_size = 8
# The required input length to produce `output_block_size` outputs is
# `output_block_size + kernel_size - 1`.
required_input_len = output_block_size + kernel_size - 1
# The input block size must also be a multiple of 8. We round up the
# required length to the next multiple of 8.
input_len_block_size = math.ceil(required_input_len / 8) * 8


# Computation
def kernel(x_ref, kernel_ref, output_ref):
  """
  Pallas kernel for 1D convolution using a blocked approach.

  This kernel processes a block of `output_block_size` output elements for a
  single batch item. The grid iterates over the batch and the blocks along
  the length dimension.

  Args:
    x_ref: A reference to a block of the input tensor.
      Shape: (1, input_len_block_size, in_channels).
    kernel_ref: A reference to the convolution kernel weights.
      Shape: (kernel_size, in_channels, out_channels).
    output_ref: A reference to the output tensor block which will be written to.
      Shape: (1, output_block_size, out_channels).
  """
  # The input is over-fetched to meet alignment constraints. Slice it to the
  # size required for the convolution.
  x_block = jax.lax.slice_in_dim(x_ref, 0, required_input_len, axis=1)

  # Perform the convolution on the block.
  # lhs shape: (1, 10, 3) -> NWC
  # rhs shape: (3, 3, 64) -> WIO
  # out shape: (1, 8, 64) -> NWC
  output_block = jax.lax.conv_general_dilated(
    lhs=x_block,
    rhs=kernel_ref,
    window_strides=(stride,),
    padding="VALID",
    dimension_numbers=("NWC", "WIO", "NWC"),
    feature_group_count=groups,
  )

  # Write the computed block to the output reference.
  output_ref[...] = output_block


# The grid iterates over the batch dimension and blocks of the output length.
grid_y = math.ceil(output_length / output_block_size)
grid = (batch_size, grid_y)

output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, output_length, out_channels), x.dtype),
  grid=grid,
  in_specs=[
    # For each grid item (b, i), load the corresponding input block.
    # The block is larger than needed (over-fetched) to meet TPU alignment
    # requirements. The starting position corresponds to the output block.
    pl.BlockSpec(
      block_shape=(1, input_len_block_size, in_channels), index_map=lambda b, i: (b, i * output_block_size, 0)
    ),
    # The kernel is the same for all grid items.
    pl.BlockSpec(block_shape=(kernel_size, in_channels, out_channels), index_map=lambda b, i: (0, 0, 0)),
  ],
  # For each grid item (b, i), define the output block to write to.
  out_specs=pl.BlockSpec(
    block_shape=(1, output_block_size, out_channels), index_map=lambda b, i: (b, i * output_block_size, 0)
  ),
)(x, params["kernel"]).block_until_ready()

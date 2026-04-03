# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
width = 256
height = 128

key = random.PRNGKey(0)
key_x, key_params = random.split(key)

# JAX uses channels-last convention (N, H, W, C)
# Using bfloat16 as it's often required for optimized TPU kernels.
x = random.normal(key_x, (batch_size, height, width, in_channels), dtype=jnp.bfloat16)


# In Flax, layers are defined inside a Module
class ConvModel(nn.Module):
  @nn.compact
  def __call__(self, x):
    # PyTorch padding=0 is 'VALID' in JAX/Flax
    return nn.Conv(
      features=out_channels,
      kernel_size=(kernel_size, kernel_size),
      strides=1,
      padding="VALID",
      kernel_dilation=1,
      feature_group_count=1,
      use_bias=False,
      dtype=jnp.bfloat16,  # Ensure conv layer uses bfloat16
      param_dtype=jnp.bfloat16,  # Set parameter dtype to bfloat16
    )(x)


conv2d = ConvModel()
params = conv2d.init(key_params, x)["params"]

# Calculate output dimensions based on 'VALID' padding
out_height = height - kernel_size + 1
out_width = width - kernel_size + 1

# Define the width of the output tile. This must be divisible by 8 for TPU
# compatibility, as it corresponds to the second-to-last dimension of the
# output block. We choose 128.
out_w_block_size = 128

# The input width block must be large enough to cover the receptive field for
# the output block. This is `out_w_block_size + kernel_size - 1`.
# For TPU compatibility, this also needs to be divisible by 8.
# 128 + 3 - 1 = 130. The next multiple of 8 is 136.
in_w_block_size = 136

# The grid iterates over batch, output height, and tiles of output width.
grid = (batch_size, out_height, (out_width + out_w_block_size - 1) // out_w_block_size)

# Extract the convolution kernel weights from the initialized parameters.
# Flax nests parameters in a dictionary, with default layer names like 'Conv_0'.
w_conv = params["Conv_0"]["kernel"]


# Computation
def kernel(x_ref, w_ref, out_ref):
  """Pallas kernel for 2D convolution.

  This kernel computes a 'VALID' convolution on a patch of the input image.
  Each kernel instance, as defined by the grid, is responsible for computing
  a horizontal slice of the output feature map.

  Args:
    x_ref: A reference to a block of the input image. The shape is
      (1, kernel_size, in_w_block_size, in_channels), where in_w_block_size
      is padded for TPU compatibility.
    w_ref: A reference to the entire convolution kernel weights tensor.
      The shape is (kernel_size, kernel_size, in_channels, out_channels).
    out_ref: A reference to the output block to be written to. The shape is
      (1, 1, out_w_block_size, out_channels).
  """
  # Define the dimension numbers for the convolution operation, which specifies
  # the layout of the input, weights, and output tensors.
  # 'NHWC' for input/output (Batch, Height, Width, Channels).
  # 'HWIO' for kernel weights (Height, Width, Input Channels, Output Channels).
  dn = ("NHWC", "HWIO", "NHWC")

  # Perform the 2D convolution using JAX's low-level primitive.
  # - lhs: The input patch from x_ref.
  # - rhs: The kernel weights from w_ref.
  # - window_strides: (1, 1) to match the original nn.Conv.
  # - padding: 'VALID' because the input patch is already sliced to be the
  #   exact receptive field required for the output block.
  conv_out = jax.lax.conv_general_dilated(
    lhs=x_ref[...], rhs=w_ref[...], window_strides=(1, 1), padding="VALID", dimension_numbers=dn
  )

  # The input block width (`in_w_block_size`) was padded for TPU memory
  # alignment. This results in the convolution output being slightly wider
  # than the target output block (`out_w_block_size`). We slice the
  # convolution result to match the shape of `out_ref` before writing.
  # For example, an input width of 136 and kernel of 3 gives an output width
  # of 134, which we slice to the target 128.
  out_ref[...] = conv_out[:, :, : out_ref.shape[2], :]


# The convolution is parallelized by creating a grid over the batch dimension,
# the output height, and tiles of the output width. Each kernel instance is
# responsible for computing a horizontal slice of the output feature map.
result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, out_height, out_width, out_channels), x.dtype),
  grid=grid,
  in_specs=[
    # Spec for the input image `x`. Each kernel instance gets a patch.
    # The block shape is padded to be TPU-compatible.
    pl.BlockSpec(
      block_shape=(1, kernel_size, in_w_block_size, in_channels),
      index_map=lambda b, h, w_tile: (b, h, w_tile * out_w_block_size, 0),
    ),
    # Spec for the convolution weights `w_conv`. The entire kernel is passed
    # to each program instance. The block shape equals the full array shape,
    # which satisfies the TPU constraints.
    pl.BlockSpec(block_shape=w_conv.shape, index_map=lambda *_: (0,) * w_conv.ndim),
  ],
  out_specs=pl.BlockSpec(
    # Spec for the output. Each kernel instance writes to a unique
    # (1, 1, 128, 64) block in the output tensor.
    block_shape=(1, 1, out_w_block_size, out_channels),
    index_map=lambda b, h, w_tile: (b, h, w_tile * out_w_block_size, 0),
  ),
)(x, w_conv).block_until_ready()

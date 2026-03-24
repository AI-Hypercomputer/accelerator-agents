# Imports
import jax
import jax.random as random
from flax import linen as nn
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 3
kernel_size = 3
width = 256
height = 256
stride = 1
padding = 0
bias = False

key = random.PRNGKey(0)
key, x_key, params_key = random.split(key, 3)

x = random.normal(x_key, (batch_size, height, width, in_channels))
conv2d = nn.Conv(
  features=in_channels,
  kernel_size=(kernel_size, kernel_size),
  strides=(stride, stride),
  padding=padding,
  feature_group_count=in_channels,
  use_bias=bias,
)
params = conv2d.init(params_key, x)["params"]


# Computation
def kernel(x_ref, kernel_ref, y_ref):
  """Pallas kernel for block-wise depthwise convolution.

  This kernel computes a single block of the output of a depthwise convolution.
  It assumes the input block `x_ref` has the necessary halo (padding) to compute
  the full output block `y_ref` with 'VALID' padding logic.

  Args:
    x_ref: Input block reference from the input image `x`.
    kernel_ref: Convolution kernel reference from `params['kernel']`.
    y_ref: Output block reference to be written to.
  """
  # Infer the number of channels from the last dimension of the input shape.
  # This makes the kernel more general.
  in_channels = x_ref.shape[-1]

  # Define the dimension numbers for the convolution.
  # The kernel from Flax for depthwise convolution has shape (H, W, 1, C).
  # To ensure the convolution produces C output channels (not C*C), we must
  # interpret the kernel's 'O' (output features per group) dimension as having
  # size 1. By specifying the kernel layout as 'HWOI', we map the dimension
  # of size 1 to 'O' and the dimension of size C to 'I'.
  dn = ("NHWC", "HWOI", "NHWC")

  # Perform the depthwise convolution on the input block (x_ref).
  # - The stride is (1, 1), as specified in the original computation.
  # - Padding is 'VALID' because the `pallas_call` invocation prepares the
  #   input block `x_ref` with the necessary halo.
  # - `feature_group_count=in_channels` is the crucial parameter that
  #   instructs `conv_general_dilated` to perform a depthwise convolution.
  result = jax.lax.conv_general_dilated(
    lhs=x_ref,
    rhs=kernel_ref,
    window_strides=(1, 1),
    padding="VALID",
    feature_group_count=in_channels,
    dimension_numbers=dn,
  )

  # Write the computed block to the output reference. The shape of `result`
  # will match the shape of `y_ref` due to the 'VALID' padding and the
  # specific calculation of the input block size in the host code.
  y_ref[...] = result


# The original computation is a depthwise convolution. We will parallelize this
# by creating a grid over the batch and height dimensions. Each kernel instance
# will process a horizontal slice of one image in the batch for all channels.

# Calculate output spatial dimensions based on convolution parameters
out_height = (height - kernel_size) // stride + 1
out_width = (width - kernel_size) // stride + 1

# Define a block size for tiling along the height dimension.
# 128 is a common tile size for TPUs.
out_block_h = 128

# Calculate the required input block height, which includes a "halo"
# to account for the convolution kernel's size.
in_block_h = (out_block_h - 1) * stride + kernel_size

# The grid will have dimensions (batch_size, num_height_blocks).
grid = (batch_size, (out_height + out_block_h - 1) // out_block_h)

y = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, out_height, out_width, in_channels), x.dtype),
  grid=grid,
  in_specs=[
    # BlockSpec for the input tensor 'x'.
    # Each kernel instance receives a horizontal slice of an image.
    # The block shape covers the full width and all channels.
    # The height is `in_block_h` to include the halo.
    # The index_map maps grid index (b, i) to the b-th image and i-th vertical block.
    pl.BlockSpec(
      block_shape=(1, in_block_h, width, in_channels),
      index_map=lambda b, i: (b, i * out_block_h, 0, 0),
    ),
    # BlockSpec for the convolution kernel weights.
    # The kernel is relatively small and is needed by all grid instances.
    # We map all grid instances to the full kernel array.
    pl.BlockSpec(
      block_shape=params["kernel"].shape,
      index_map=lambda b, i: (0, 0, 0, 0),
    ),
  ],
  out_specs=pl.BlockSpec(
    # BlockSpec for the output tensor 'y'.
    # The block shape corresponds to the horizontal slice computed by a kernel.
    # The index_map maps grid index (b, i) to the corresponding output slice.
    block_shape=(1, out_block_h, out_width, in_channels),
    index_map=lambda b, i: (b, i * out_block_h, 0, 0),
  ),
)(x, params["kernel"]).block_until_ready()

# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = (3, 5)
height = 256
width = 128
stride = (1, 1)
padding = "VALID"
dilation = (1, 1)
groups = 1
bias = False

key = random.PRNGKey(0)
key_x, key_params = random.split(key)

# JAX uses channels-last convention (N, H, W, C)
x = random.normal(key_x, (batch_size, height, width, in_channels))

conv2d = nn.Conv(
  features=out_channels,
  kernel_size=kernel_size,
  strides=stride,
  padding=padding,
  kernel_dilation=dilation,
  feature_group_count=groups,
  use_bias=bias,
)
params = conv2d.init(key_params, x)["params"]

# Calculate output dimensions based on 'VALID' padding and stride of 1
out_height = height - kernel_size[0] + 1
out_width = width - kernel_size[1] + 1
output_shape = (batch_size, out_height, out_width, out_channels)
kernel_shape = (kernel_size[0], kernel_size[1], in_channels, out_channels)


# Computation
def kernel(x_ref, kernel_ref, out_ref):
  """Pallas kernel for 2D convolution with 'VALID' padding and stride 1.

  This kernel computes one full row of the output feature map for a single
  batch item. The parallelization is over the batch and output height dimensions.

  Args:
    x_ref: A reference to a slice of the input tensor. The slice contains
      the receptive field needed to compute one output row. Its shape is
      (1, kernel_height, input_width, in_channels).
    kernel_ref: A reference to the complete convolution kernel tensor.
      Its shape is (kernel_height, kernel_width, in_channels, out_channels).
    out_ref: A reference to the output slice to be written to. Its shape is
      (1, 1, output_width, out_channels).
  """
  # Get kernel and output dimensions from the shapes of the references.
  # These shapes are known at compile time.
  kernel_width = kernel_ref.shape[1]
  out_width = out_ref.shape[2]

  # Iterate over each horizontal position in the output row.
  for w_out in range(out_width):
    # Extract the input patch corresponding to the current output pixel.
    # Since the stride is 1, the horizontal starting position of the patch
    # is the same as the output width index, w_out.
    # The slice has shape (kernel_height, kernel_width, in_channels).
    patch = x_ref[0, :, w_out : w_out + kernel_width, :]

    # Perform the convolution for one output pixel location across all output channels.
    # This is done by broadcasting the patch and multiplying with the kernel,
    # then summing over the kernel dimensions.
    out_pixel = jnp.sum(patch[..., None] * kernel_ref[...], axis=(0, 1, 2))

    # Write the computed vector of output channels to the corresponding
    # location in the output reference.
    out_ref[0, 0, w_out, :] = out_pixel


# The grid is designed to parallelize the computation over the batch and output height dimensions.
# Each kernel instance computes one full row of the output feature map.
output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(output_shape, x.dtype),
  grid=(batch_size, out_height),
  in_specs=[
    # Input 'x': Each kernel instance receives a horizontal slice of the input image
    # large enough to compute a full output row.
    # Shape: (1, kernel_height, image_width, in_channels)
    pl.BlockSpec(
      block_shape=(1, kernel_size[0], width, in_channels),
      index_map=lambda b, h: (b, h, 0, 0),
    ),
    # Input 'kernel': The entire convolution kernel is passed to each instance.
    pl.BlockSpec(
      block_shape=kernel_shape,
      index_map=lambda b, h: (0, 0, 0, 0),
    ),
  ],
  # Output: Each kernel instance writes to a slice corresponding to one output row.
  # Shape: (1, 1, out_width, out_channels)
  out_specs=pl.BlockSpec(
    block_shape=(1, 1, out_width, out_channels),
    index_map=lambda b, h: (b, h, 0, 0),
  ),
)(x, params["kernel"]).block_until_ready()

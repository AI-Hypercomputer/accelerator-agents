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
height = 256
stride = 1
padding = 0
dilation = 1
groups = 1
bias = False

key = random.PRNGKey(0)
key_params, key_x = random.split(key)

# JAX/Flax uses channels-last convention
x = random.normal(key_x, (batch_size, height, width, in_channels))

conv2d = nn.Conv(
  features=out_channels,
  kernel_size=(kernel_size, kernel_size),
  strides=stride,
  padding=padding,
  kernel_dilation=dilation,
  feature_group_count=groups,
  use_bias=bias,
)
params = conv2d.init(key_params, x)["params"]

# Calculate output dimensions based on 'VALID' padding logic, as stride=1, padding=0.
# out_dim = (in_dim - kernel_dim) // stride + 1
out_h = (height - kernel_size) // stride + 1
out_w = (width - kernel_size) // stride + 1


# Computation
def kernel(x_ref, kernel_ref, out_ref):
  """
  Pallas kernel for 2D convolution.

  This kernel computes a single row of the output feature map for a given
  image in the batch. The grid parallelizes this over the batch and output
  height dimensions.

  Args:
    x_ref: A reference to a slice of the input image, corresponding to the
      rows needed to compute one output row.
      Shape: (1, kernel_size, width, in_channels).
    kernel_ref: A reference to the full convolution kernel weights.
      Shape: (kernel_size, kernel_size, in_channels, out_channels).
    out_ref: A reference to the output buffer for one row of the feature map.
      Shape: (1, 1, out_w, out_channels).
  """
  # Loop over each pixel in the output row.
  for ox in range(out_w):
    # Extract the input window (patch) for the current output pixel.
    # The window is of size (kernel_size, kernel_size) and starts at `ox`.
    in_window = jax.lax.dynamic_slice(x_ref, (0, 0, ox, 0), (1, kernel_size, kernel_size, in_channels))

    # Reshape the input window to enable broadcasting with the kernel.
    # Shape changes from (1, 3, 3, 3) to (3, 3, 3, 1).
    in_window_reshaped = in_window[0, :, :, :, None]

    # Perform the convolution for one pixel: element-wise multiplication
    # followed by a sum reduction over the kernel dimensions and input channels.
    # This is equivalent to a dot product.
    # (3, 3, 3, 1) * (3, 3, 3, 64) -> (3, 3, 3, 64)
    acc = jnp.sum(in_window_reshaped * kernel_ref, axis=(0, 1, 2))

    # Write the final computed vector to the output row.
    out_ref[0, 0, ox, :] = acc


# The pallas_call replaces the original conv2d.apply computation.
# The parallelization strategy is to map each output row of each batch item
# to a grid instance.
output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, out_h, out_w, out_channels), x.dtype),
  # Grid is over (batch_size, output_height).
  grid=(batch_size, out_h),
  in_specs=[
    # Input spec: for each output row, we need a horizontal slice of the
    # input image that is `kernel_size` high.
    pl.BlockSpec(block_shape=(1, kernel_size, width, in_channels), index_map=lambda b, oy: (b, oy, 0, 0)),
    # Kernel spec: all grid instances use the same full kernel.
    pl.BlockSpec(
      block_shape=(kernel_size, kernel_size, in_channels, out_channels), index_map=lambda b, oy: (0, 0, 0, 0)
    ),
  ],
  out_specs=pl.BlockSpec(
    # Output spec: each instance writes a full output row. The block_shape
    # must have the same rank as the full output array.
    block_shape=(1, 1, out_w, out_channels),
    index_map=lambda b, oy: (b, oy, 0, 0),
  ),
)(x, params["kernel"]).block_until_ready()

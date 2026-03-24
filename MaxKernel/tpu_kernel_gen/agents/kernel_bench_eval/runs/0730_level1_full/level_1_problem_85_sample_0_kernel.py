# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn
from jax import lax
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 3
out_channels = in_channels
kernel_size_h = 3
kernel_size_w = 5
width = 256
height = 128
stride_h = 1
stride_w = 1
padding_h = 0
padding_w = 0
dilation_h = 1
dilation_w = 1
groups = in_channels
bias = False
key = random.PRNGKey(0)
key_x, key_params = random.split(key)
x = random.normal(key_x, (batch_size, height, width, in_channels))
conv2d = nn.Conv(
  features=out_channels,
  kernel_size=(kernel_size_h, kernel_size_w),
  strides=(stride_h, stride_w),
  padding="VALID",
  kernel_dilation=(dilation_h, dilation_w),
  feature_group_count=groups,
  use_bias=bias,
)
params = conv2d.init(key_params, x)["params"]

# Calculate output shape for Pallas call
out_height = (height - kernel_size_h) // stride_h + 1
out_width = (width - kernel_size_w) // stride_w + 1
out_shape_tuple = (batch_size, out_height, out_width, out_channels)
kernel_shape = (kernel_size_h, kernel_size_w, 1, out_channels)


# Computation
def kernel(x_ref, k_ref, out_ref):
  """
  Pallas kernel for depthwise 2D convolution.

  Args:
    x_ref: Input tensor slice.
    k_ref: Kernel tensor.
    out_ref: Output tensor slice to be written to.
  """
  # These constants are derived from the problem description.
  stride_w = 1
  kernel_size_w = k_ref.shape[1]
  out_width = out_ref.shape[2]

  # Load the relevant slices of the input and kernel.
  # x shape: (kernel_size_h, width, in_channels)
  x = x_ref[0, ...]
  # k shape: (kernel_size_h, kernel_size_w, out_channels)
  k = k_ref[:, :, 0, :]

  # Manually implement the convolution along the width dimension.
  for j in range(out_width):
    # Calculate the starting position of the sliding window.
    w_in = j * stride_w
    # Extract the input patch for the convolution.
    x_patch = lax.dynamic_slice_in_dim(x, w_in, kernel_size_w, axis=1)

    # Perform the element-wise product and sum over the spatial dimensions.
    # This computes the convolution for all channels at a given output position.
    out_pixel = jnp.sum(x_patch * k, axis=(0, 1))

    # Write the computed pixel (a vector of channels) to the output.
    out_ref[0, 0, j, :] = out_pixel


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(out_shape_tuple, x.dtype),
  grid=(batch_size, out_height),
  in_specs=[
    pl.BlockSpec(block_shape=(1, kernel_size_h, width, in_channels), index_map=lambda i, j: (i, j * stride_h, 0, 0)),
    pl.BlockSpec(block_shape=kernel_shape, index_map=lambda i, j: (0, 0, 0, 0)),
  ],
  out_specs=pl.BlockSpec(block_shape=(1, 1, out_width, out_channels), index_map=lambda i, j: (i, j, 0, 0)),
)(x, params["kernel"]).block_until_ready()

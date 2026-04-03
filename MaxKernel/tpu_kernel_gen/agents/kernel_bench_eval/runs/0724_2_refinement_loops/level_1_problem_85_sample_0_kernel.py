# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 3
out_channels = 3
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
groups = 3
bias = False

key = random.PRNGKey(0)
key_x, key_params = random.split(key)

# JAX uses channel-last convention (N, H, W, C)
x = random.normal(key_x, (batch_size, height, width, in_channels))

conv2d = nn.Conv(
  features=in_channels,
  kernel_size=(kernel_size_h, kernel_size_w),
  strides=(stride_h, stride_w),
  padding="VALID",  # padding=(0,0) in torch is 'VALID' in JAX
  kernel_dilation=(dilation_h, dilation_w),
  feature_group_count=in_channels,
  use_bias=bias,
)
variables = conv2d.init(key_params, x)


# Computation
def kernel(x_patch_ref, kernel_ref, out_pixel_ref):
  """Pallas kernel for depthwise convolution.

  This kernel computes a single output pixel. The grid is set up to iterate
  over batch, output height, and output width.

  Args:
    x_patch_ref: A reference to the input patch for one output pixel.
      Shape: (1, kernel_size_h, kernel_size_w, in_channels)
    kernel_ref: A reference to the convolution kernel.
      Shape: (kernel_size_h, kernel_size_w, 1, out_channels)
    out_pixel_ref: A reference to a single output pixel location.
      Shape: (1, 1, 1, out_channels)
  """
  # Load the input patch and kernel into registers.
  # Squeeze singleton dimensions to simplify the computation.
  x_patch = x_patch_ref[0, ...]  # Shape: (KH, KW, C)
  kernel_squeezed = jnp.squeeze(kernel_ref[...], axis=2)  # Shape: (KH, KW, C)

  # Perform the depthwise convolution for one output pixel.
  # This is an element-wise product followed by a sum over spatial dimensions.
  out_pixel = jnp.sum(x_patch * kernel_squeezed, axis=(0, 1))  # Shape: (C,)

  # Write the computed pixel to the output buffer.
  out_pixel_ref[0, 0, 0] = out_pixel


# Calculate output dimensions based on 'VALID' padding and stride=1
out_height = height - kernel_size_h + 1
out_width = width - kernel_size_w + 1

output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(shape=(batch_size, out_height, out_width, out_channels), dtype=x.dtype),
  grid=(batch_size, out_height, out_width),
  in_specs=[
    pl.BlockSpec(
      block_shape=(1, kernel_size_h, kernel_size_w, in_channels),
      index_map=lambda b, h_out, w_out: (b, h_out, w_out, 0),
    ),
    pl.BlockSpec(
      block_shape=(kernel_size_h, kernel_size_w, 1, out_channels),
      index_map=lambda b, h_out, w_out: (0, 0, 0, 0),
    ),
  ],
  out_specs=pl.BlockSpec(
    block_shape=(1, 1, 1, out_channels),
    index_map=lambda b, h_out, w_out: (b, h_out, w_out, 0),
  ),
)(x, variables["params"]["kernel"]).block_until_ready()

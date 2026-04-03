# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = (3, 3)
width = 128
height = 128
stride = (1, 1)
padding = "VALID"
groups = 1
bias = False

key = random.PRNGKey(0)
key_x, key_params = random.split(key)

# JAX uses channels-last convention: (N, H, W, C)
x = random.normal(key_x, (batch_size, height, width, in_channels))

conv_transpose2d = nn.ConvTranspose(
  features=out_channels,
  kernel_size=kernel_size,
  strides=stride,
  padding=padding,
  # feature_group_count is not a valid parameter for nn.ConvTranspose in all versions.
  # Since groups is 1 (the default), this parameter can be safely removed.
  # For grouped convolutions, the parameter is actually inherited from nn.Conv
  # and should be feature_group_count, but older Flax versions might not expose it.
  # The simplest fix is removal as it's redundant with the default.
  use_bias=bias,
)
params = conv_transpose2d.init(key_params, x)
kernel_weights = params["params"]["kernel"]

# Calculate output shape for the pallas_call
# For 'VALID' padding and stride (1,1), output shape is (N, H+kH-1, W+kW-1, C_out)
output_height = height + kernel_size[0] - 1
output_width = width + kernel_size[1] - 1
output_shape = (batch_size, output_height, output_width, out_channels)

# A transposed convolution is equivalent to a standard convolution
# on a padded input with a flipped kernel.

# 1. Flip the kernel weights along spatial dimensions (0 and 1).
kernel_flipped = jnp.flip(kernel_weights, axis=(0, 1))

# 2. Pad the input tensor `x`. The padding size is determined by the
# kernel dimensions to ensure the output of the standard convolution
# matches the desired transposed convolution output size.
pad_h = kernel_size[0] - 1
pad_w = kernel_size[1] - 1
x_padded = jnp.pad(x, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)))


# Computation
def kernel(x_ref, kernel_ref, out_ref):
  """Pallas kernel for a single-pixel convolution calculation.

  This kernel computes the output for a single pixel (across all output channels)
  in a standard convolution operation. It takes a patch of the input and the
  full convolution kernel, performs a dot product, and writes the result to
  the corresponding output pixel.

  Note: This kernel implements a standard convolution. For this to correctly
  compute a transposed convolution, the input `x` must be appropriately padded
  and the kernel weights might need to be flipped before the `pallas_call`.

  Args:
    x_ref: A reference to the input patch.
      Shape: (1, kernel_height, kernel_width, in_channels)
    kernel_ref: A reference to the convolution kernel weights.
      Shape: (kernel_height, kernel_width, in_channels, out_channels)
    out_ref: A reference to the output pixel buffer.
      Shape: (1, 1, 1, out_channels)
  """
  # Reshape inputs for matrix multiplication.
  # x_ref: (1, kH, kW, C_in) -> (1, kH * kW * C_in)
  # kernel_ref: (kH, kW, C_in, C_out) -> (kH * kW * C_in, C_out)
  x_flat = x_ref.reshape((1, -1))
  kernel_flat = kernel_ref.reshape((-1, out_channels))

  # Perform the dot product.
  result = jnp.dot(x_flat, kernel_flat)

  # Reshape the result to match the output shape (1, 1, 1, out_channels)
  # and write it to the output buffer.
  out_ref[...] = result.reshape(out_ref.shape)


# Pallas call to replace the standard Flax computation
output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(output_shape, x.dtype),
  grid=(batch_size, output_height, output_width),
  in_specs=[
    pl.BlockSpec(block_shape=(1, kernel_size[0], kernel_size[1], in_channels), index_map=lambda b, h, w: (b, h, w, 0)),
    pl.BlockSpec(
      block_shape=kernel_flipped.shape,
      index_map=lambda b, h, w: (),  # Kernel is broadcast
    ),
  ],
  out_specs=pl.BlockSpec(block_shape=(1, 1, 1, out_channels), index_map=lambda b, h, w: (b, h, w, 0)),
)(x_padded, kernel_flipped).block_until_ready()

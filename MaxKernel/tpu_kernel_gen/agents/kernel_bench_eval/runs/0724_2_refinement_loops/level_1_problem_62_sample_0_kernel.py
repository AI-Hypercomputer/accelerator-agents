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
kernel_size = (3, 5)
width = 256
height = 256

key = random.PRNGKey(0)
key_x, key_params = random.split(key)

# JAX uses channels-last convention (NHWC)
x = random.normal(key_x, (batch_size, height, width, in_channels))


# In Flax, models are defined as classes and are stateless
class ConvNet(nn.Module):
  @nn.compact
  def __call__(self, x):
    # Flax nn.Conv arguments: features, kernel_size, strides, padding, kernel_dilation, feature_group_count, use_bias
    layer = nn.Conv(
      features=out_channels,
      kernel_size=kernel_size,
      strides=1,
      padding="VALID",
      kernel_dilation=1,
      feature_group_count=1,
      use_bias=False,
    )
    return layer(x)


conv2d = ConvNet()
params = conv2d.init(key_params, x)["params"]

# Block size for the output channels dimension, set to satisfy TPU constraints
bC = 64

# Calculate output spatial dimensions based on 'VALID' padding
output_height = height - kernel_size[0] + 1
output_width = width - kernel_size[1] + 1


# Computation
def kernel(x_ref, kernel_ref, out_ref):
  """Pallas kernel for 2D convolution.

  This kernel computes a block of the convolution result. Each program instance
  handles a single image from the batch and a specific block of output channels.

  Args:
    x_ref: A reference to a slice of the input tensor `x`. The shape is
      (1, height, width, in_channels), representing one full image.
    kernel_ref: A reference to a slice of the convolution kernel weights. The
      shape is (kernel_height, kernel_width, in_channels, block_out_channels),
      representing the filters for a block of output channels.
    out_ref: A reference to the output tensor slice to be written to. The shape
      is (1, out_height, out_width, block_out_channels).
  """
  # Create an accumulator for the output block in SRAM.
  acc = jnp.zeros((1, output_height, output_width, bC), dtype=x_ref.dtype)

  # Loop over the kernel's spatial dimensions.
  for kh in range(kernel_size[0]):
    for kw in range(kernel_size[1]):
      # Slice the input image using standard NumPy-style indexing.
      # This is equivalent to a 'VALID' convolution padding.
      in_slice = x_ref[0, kh : kh + output_height, kw : kw + output_width, :]

      # Get the corresponding slice of kernel weights.
      kernel_slice = kernel_ref[kh, kw, :, :]

      # Reshape input for matrix multiplication.
      in_slice_reshaped = jnp.reshape(in_slice, (-1, in_channels))
      # Perform the dot product between the input slice and kernel slice.
      out_slice = jnp.dot(in_slice_reshaped, kernel_slice)
      # Reshape the result back to its spatial form.
      out_slice_reshaped = jnp.reshape(out_slice, (1, output_height, output_width, bC))

      # Accumulate the partial results.
      acc += out_slice_reshaped

  # Write the final accumulated result to the output buffer.
  out_ref[...] = acc


output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, output_height, output_width, out_channels), x.dtype),
  grid=(batch_size, out_channels // bC),
  in_specs=[
    pl.BlockSpec(
      block_shape=(1, height, width, in_channels),
      index_map=lambda b, c: (b, 0, 0, 0),
    ),
    pl.BlockSpec(
      block_shape=(kernel_size[0], kernel_size[1], in_channels, bC),
      index_map=lambda b, c: (0, 0, 0, c * bC),
    ),
  ],
  out_specs=pl.BlockSpec(
    block_shape=(1, output_height, output_width, bC),
    index_map=lambda b, c: (b, 0, 0, c * bC),
  ),
)(x, params["Conv_0"]["kernel"]).block_until_ready()

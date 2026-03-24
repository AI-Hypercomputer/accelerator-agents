# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 3
out_channels = 3
kernel_size = 3
width_in = 256
height_in = 128
stride = 1
padding = "VALID"
bias = False

key = random.PRNGKey(0)
key_x, key_params = random.split(key)

x = random.normal(key_x, (batch_size, height_in, width_in, in_channels))
conv2d = nn.Conv(
  features=out_channels,
  kernel_size=(kernel_size, kernel_size),
  strides=stride,
  padding=padding,
  feature_group_count=in_channels,
  use_bias=bias,
)
params = conv2d.init(key_params, x)


# Computation
def kernel(x_ref, kernel_ref, out_ref):
  """Pallas kernel for depthwise convolution."""
  # Get kernel dimensions and output block shape
  kernel_h, kernel_w, _, num_channels = kernel_ref.shape
  _, block_h, block_w, _ = out_ref.shape

  # Remove the singleton dimension from the kernel by slicing.
  kernel = kernel_ref[:, :, 0, :]  # Shape: (kernel_h, kernel_w, num_channels)

  # Create an accumulator for the output block.
  acc = jnp.zeros((block_h, block_w, num_channels), dtype=x_ref.dtype)

  # Iterate over the spatial dimensions of the kernel. This is more efficient
  # than iterating over the output pixels, as the kernel is small.
  for kh in range(kernel_h):
    for kw in range(kernel_w):
      # Select the kernel weights for the current position.
      kernel_point = kernel[kh, kw, :]  # Shape: (num_channels,)

      # Slice the input block to get the corresponding patch.
      input_slice = x_ref[0, kh : kh + block_h, kw : kw + block_w, :]

      # Perform vectorized multiply-add. The kernel_point is broadcasted
      # across the spatial dimensions of the input_slice.
      acc += input_slice * kernel_point

  # Write the final accumulated values to the output block.
  out_ref[0, :, :, :] = acc


# Define grid and block parameters based on the convolution
height_out = (height_in - kernel_size) // stride + 1
width_out = (width_in - kernel_size) // stride + 1

# Block shapes must satisfy TPU memory layout constraints.
# We tile only along the height dimension.
block_h = 8
block_w = width_out

num_h_blocks = (height_out + block_h - 1) // block_h
num_w_blocks = (width_out + block_w - 1) // block_w

kernel_weights = params["params"]["kernel"]

output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, height_out, width_out, out_channels), x.dtype),
  grid=(batch_size, num_h_blocks, num_w_blocks),
  in_specs=[
    pl.BlockSpec(
      block_shape=(1, block_h + kernel_size - 1, block_w + kernel_size - 1, in_channels),
      index_map=lambda b, i, j: (b, i * block_h * stride, j * block_w * stride, 0),
    ),
    pl.BlockSpec(
      block_shape=kernel_weights.shape,
      index_map=lambda b, i, j: (0, 0, 0, 0),
    ),
  ],
  out_specs=pl.BlockSpec(
    block_shape=(1, block_h, block_w, out_channels),
    index_map=lambda b, i, j: (b, i * block_h, j * block_w, 0),
  ),
  # Add padding configuration for the input tensor `x`.
  # This tells Pallas to pad with zeros when the BlockSpec tries to
  # read data outside the original tensor's bounds.
  # The padding is specified for each dimension of `x`.
  # (batch, height, width, channels)
  input_output_aliases={2: 0},
)(x, kernel_weights).block_until_ready()

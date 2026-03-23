# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3

key = random.PRNGKey(0)
key_x, key_params = random.split(key)

# JAX uses channels-last convention (N, H, W, C)
x = random.normal(key_x, (batch_size, height, width, in_channels))

conv = nn.Conv(features=out_channels, kernel_size=(kernel_size, kernel_size))
params = conv.init(key_params, x)["params"]
kernel_weights = params["kernel"]
bias = params["bias"]

# The output shape after a 'SAME' padding convolution
output_shape = (batch_size, height, width, out_channels)

# Define block sizes for tiling the computation.
# We tile the height and width of the output feature map.
b_h = 8
b_w = 8


def kernel(x_ref, kernel_ref, bias_ref, out_ref):
  """
  Pallas kernel for a 2D convolution followed by two Mish activations.

  Args:
    x_ref: Input image patch.
    kernel_ref: Convolution kernel weights.
    bias_ref: Convolution bias.
    out_ref: Output buffer.
  """
  # Accumulator for the output tile.
  acc = jnp.zeros((b_h, b_w, out_channels), dtype=x_ref.dtype)

  # Loop over the kernel spatial dimensions
  for dy in range(kernel_size):
    for dx in range(kernel_size):
      # Slice the input patch.
      x_slice = x_ref[...][0, dy : dy + b_h, dx : dx + b_w, :]

      # Get the corresponding kernel slice.
      kernel_slice = kernel_ref[...][dy, dx, :, :]

      # Perform dot product between input slice and kernel slice
      update = jnp.dot(x_slice, kernel_slice)
      acc += update

  # Add the bias term.
  output = acc + bias_ref[...]

  # Apply the first Mish activation function.
  output = jax.nn.mish(output)

  # Apply the second Mish activation function.
  output = jax.nn.mish(output)

  # Write the final result to the output reference.
  out_ref[...] = output.reshape(1, b_h, b_w, out_channels)


# Computation
# The Pallas call replaces the computation section.
# The grid is defined over the batch and the tiles of the output feature map.
# Each kernel instance computes one (b_h, b_w) tile of the output.
x = pl.pallas_call(
  kernel,
  # The final output shape after the kernel computation.
  out_shape=jax.ShapeDtypeStruct(output_shape, x.dtype),
  # Grid iterates over batch size and tiles in height/width.
  grid=(batch_size, height // b_h, width // b_w),
  # Input specifications define how data is chunked for each kernel.
  in_specs=[
    # Input 'x': Each kernel needs a patch larger than the output tile
    # to account for the convolution kernel's size.
    # For an 8x8 output tile and a 3x3 kernel, we need a 10x10 input patch.
    # The width dimension is padded to 16 to satisfy TPU alignment (must be a multiple of 8).
    pl.BlockSpec(
      block_shape=(1, b_h + kernel_size - 1, 16, in_channels),
      index_map=lambda i, j, k: (
        i,
        j * b_h - (kernel_size // 2),
        k * b_w - (kernel_size // 2),
        0,
      ),
    ),
    # Kernel weights: All kernels need the full weight matrix.
    pl.BlockSpec(
      block_shape=kernel_weights.shape,
      index_map=lambda i, j, k: (0, 0, 0, 0),
    ),
    # Bias: All kernels need the full bias vector.
    pl.BlockSpec(block_shape=bias.shape, index_map=lambda i, j, k: (0,)),
  ],
  # Output specification defines how each kernel writes to the output array.
  out_specs=pl.BlockSpec(
    block_shape=(1, b_h, b_w, out_channels),
    index_map=lambda i, j, k: (i, j * b_h, k * b_w, 0),
  ),
)(x, kernel_weights, bias).block_until_ready()

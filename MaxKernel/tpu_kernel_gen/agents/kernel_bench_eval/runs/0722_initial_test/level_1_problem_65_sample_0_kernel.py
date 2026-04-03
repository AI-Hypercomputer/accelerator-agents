# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = (3, 5)
width = 128
height = 128
stride = 1
padding = 0
output_padding = 0
groups = 1
bias = False

key = random.PRNGKey(0)
key, x_key, params_key = random.split(key, 3)

# JAX uses channels-last convention (N, H, W, C)
x = random.normal(x_key, (batch_size, height, width, in_channels))

# Get kernel weights from a standard Flax implementation for demonstration
conv_transpose2d = nn.ConvTranspose(
  features=out_channels, kernel_size=kernel_size, strides=stride, padding=padding, use_bias=bias
)
variables = conv_transpose2d.init(params_key, x)
kernel_weights = variables["params"]["kernel"]

# Calculate output shape for the Pallas call
output_height = (height - 1) * stride + kernel_size[0] - 2 * padding + output_padding
output_width = (width - 1) * stride + kernel_size[1] - 2 * padding + output_padding
output_shape = (batch_size, output_height, output_width, out_channels)

# Define block sizes for tiling
# We tile the input's spatial dimensions
bH = 8
bW = 8

# The output block size depends on the input block size and kernel size
# output_block_size = input_block_size + kernel_size - 1
out_bH = bH + kernel_size[0] - 1
out_bW = bW + kernel_size[1] - 1


def kernel(x_ref, w_ref, out_ref):
  """Pallas kernel for 2D transposed convolution.

  This kernel computes a 2D transposed convolution by iterating through each
  pixel of the input tile. For each input pixel, it calculates its contribution
  to the output tile by performing an outer product with the kernel weights.
  These contributions are summed into a local accumulator tile. After all input
  pixels have been processed, the accumulated values are written to the output.

  Args:
    x_ref: Input tile.
    w_ref: Kernel weights.
    out_ref: Output tile.
  """
  # Get kernel dimensions from the scope
  kH, kW = kernel_size
  w = w_ref[...]

  # Initialize a local accumulator with zeros
  acc = jnp.zeros((1, out_bH, out_bW, out_channels), dtype=out_ref.dtype)

  # Iterate over each pixel in the input tile's spatial dimensions
  for h_in in range(bH):
    for w_in in range(bW):
      # Extract the feature vector for the current input pixel
      in_vec = x_ref[0, h_in, w_in, :]  # Shape: (in_channels,)

      # Calculate the contribution of this pixel to the output grid.
      # This is equivalent to an outer product between the input vector and
      # the kernel, summed over the input channels.
      # einsum signature: 'c,kKcC->kKC' where c=in_channels, k,K=kernel_dims, C=out_channels
      contribution = jnp.einsum("c,kKcC->kKC", in_vec, w, preferred_element_type=out_ref.dtype)

      # Add the calculated contribution to the corresponding slice of the accumulator.
      # The .at[...].add(...) pattern performs an out-of-place update.
      acc = acc.at[0, h_in : h_in + kH, w_in : w_in + kW, :].add(contribution)

  # Write the final accumulated result to the output memory block
  out_ref[...] = acc


# Computation
# Pallas kernel invocation
output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(output_shape, x.dtype),
  grid=(batch_size, height // bH, width // bW),
  in_specs=[
    pl.BlockSpec(block_shape=(1, bH, bW, in_channels), index_map=lambda b, i, j: (b, i * bH, j * bW, 0)),
    pl.BlockSpec(block_shape=kernel_weights.shape, index_map=lambda *_: ()),
  ],
  out_specs=pl.AccumulatorSpec(
    index_map=lambda b, i, j: (b, i * bH, j * bW, 0), block_shape=(1, out_bH, out_bW, out_channels)
  ),
)(x, kernel_weights)

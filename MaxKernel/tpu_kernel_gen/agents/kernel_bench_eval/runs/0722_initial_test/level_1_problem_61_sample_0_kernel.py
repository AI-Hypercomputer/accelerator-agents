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
kernel_size = 3
depth = 32
height = 32
width = 32
stride = 1
padding = 0
output_padding = 0
groups = 1
bias = False
bW = 32  # Block size for width dimension

key = random.PRNGKey(0)
key_x, key_init = random.split(key)

# JAX expects channels-last format: (N, D, H, W, C)
x = random.normal(key_x, (batch_size, depth, height, width, in_channels))

# Use Flax to initialize kernel weights
conv_transpose3d = nn.ConvTranspose(
  features=out_channels,
  kernel_size=(kernel_size, kernel_size, kernel_size),
  strides=(stride, stride, stride),
  padding=padding,
  feature_group_count=groups,
  use_bias=bias,
)
params = conv_transpose3d.init(key_init, x)["params"]
w = params["kernel"]

# Calculate output shape for the pallas_call
out_depth = (depth - 1) * stride + kernel_size - 2 * padding + output_padding
out_height = (height - 1) * stride + kernel_size - 2 * padding + output_padding
out_width = (width - 1) * stride + kernel_size - 2 * padding + output_padding
output_shape = (batch_size, out_depth, out_height, out_width, out_channels)


# Computation
def kernel(x_ref, w_ref, out_ref):
  """Pallas kernel for 3D transposed convolution.

  This kernel computes a block of the output tensor at a time. It iterates
  through each point in the convolution kernel, calculates the corresponding
  input coordinates, and accumulates the product of inputs and weights into
  the output block.

  Args:
    x_ref: Reference to the input tensor.
    w_ref: Reference to the kernel weights tensor.
    out_ref: Reference to the output tensor to be written to.
  """
  # Get program IDs which correspond to the output block's index
  d_out = pl.program_id(1)
  h_out = pl.program_id(2)
  w_block = pl.program_id(3)

  # Get dimension sizes from the shapes of the input references
  kernel_depth, kernel_height, kernel_width, _, _ = w_ref.shape
  _, depth, height, width, _ = x_ref.shape
  bW = out_ref.shape[3]  # Block size for the width dimension

  # Initialize an accumulator for the output block with zeros.
  # The accumulator has the same shape as the output block.
  acc = jnp.zeros(out_ref.shape, dtype=out_ref.dtype)

  # Iterate over each spatial position (kd, kh, kw) in the convolution kernel.
  for kd in range(kernel_depth):
    for kh in range(kernel_height):
      for kw in range(kernel_width):
        # Calculate the corresponding input coordinates for the current kernel position.
        # For a transposed convolution, we subtract the kernel index from the output index.
        d_in = d_out - kd
        h_in = h_out - kh

        # Proceed only if the calculated input depth and height are within the valid range.
        if (d_in >= 0) and (d_in < depth) and (h_in >= 0) and (h_in < height):
          # Load the slice of the kernel weights for the current (kd, kh, kw) position.
          w_slice = w_ref[kd, kh, kw, :, :]

          # Iterate over each element along the width of the output block.
          for w_offset in range(bW):
            # Calculate the full output width coordinate.
            w_out = w_block * bW + w_offset
            # Calculate the corresponding input width coordinate.
            w_in = w_out - kw

            # Proceed only if the calculated input width is within the valid range.
            if (w_in >= 0) and (w_in < width):
              # Load the input vector (all input channels for a single spatial position).
              x_val = x_ref[0, d_in, h_in, w_in, :]

              # Perform the vector-matrix product (dot product) between the input
              # vector and the kernel slice.
              update = jnp.dot(x_val, w_slice)

              # Accumulate the result into the corresponding position in the accumulator.
              acc = acc.at[0, 0, 0, w_offset, :].add(update)

  # Write the final accumulated values from the accumulator to the output reference.
  out_ref[...] = acc[...]


# Pallas call to execute the kernel
output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(output_shape, x.dtype),
  grid=(batch_size, out_depth, out_height, (out_width + bW - 1) // bW),
  in_specs=[
    pl.BlockSpec(
      block_shape=(1, depth, height, width, in_channels), index_map=lambda n, d, h, w_block: (n, 0, 0, 0, 0)
    ),
    pl.BlockSpec(block_shape=w.shape, index_map=lambda *_: (0,) * w.ndim),
  ],
  out_specs=pl.BlockSpec(
    block_shape=(1, 1, 1, bW, out_channels), index_map=lambda n, d, h, w_block: (n, d, h, w_block * bW, 0)
  ),
)(x, w).block_until_ready()

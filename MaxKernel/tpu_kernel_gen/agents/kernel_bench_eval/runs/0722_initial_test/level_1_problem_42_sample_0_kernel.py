# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
channels = 32
height = 128
width = 128
kernel_size = 2
stride = 2
padding = 1
dilation = 3
key = random.PRNGKey(0)
# JAX and Pallas expect channels-last format: (N, H, W, C)
x = random.normal(key, (batch_size, height, width, channels))

# Calculate output shape
output_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
output_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
output_shape = (batch_size, output_height, output_width, channels)

# Define the effective window size needed for input reads
window_size = dilation * (kernel_size - 1) + 1

# Define tile size for the kernel, must be a multiple of 8 for TPU
tile_size = 8


# Computation
def kernel(x_ref, o_ref):
  # o_ref corresponds to a tile_size x tile_size tile
  for y_tile in range(tile_size):
    for x_tile in range(tile_size):
      # For each output pixel in the tile, compute the max pool
      max_val = jnp.full((channels,), -jnp.inf, dtype=o_ref.dtype)

      # Top-left corner of the window in the INPUT block (x_ref)
      # for the current output pixel (y_tile, x_tile)
      y_start = y_tile * stride
      x_start = x_tile * stride

      for i in range(kernel_size):
        for j in range(kernel_size):
          # Dilated indices within the window
          row_idx = y_start + i * dilation
          col_idx = x_start + j * dilation

          current_val = x_ref[0, row_idx, col_idx, :]
          max_val = jnp.maximum(max_val, current_val)

      # Write the result for the current pixel to the output tile
      o_ref[0, y_tile, x_tile, :] = max_val


# Calculate the required input block shape for a tile
in_block_height = (tile_size - 1) * stride + window_size
in_block_width = (tile_size - 1) * stride + window_size
# Round up to be divisible by 8 for TPU
in_block_height_padded = (in_block_height + 7) & ~7
in_block_width_padded = (in_block_width + 7) & ~7

output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(output_shape, x.dtype),
  grid=(batch_size, output_height // tile_size, output_width // tile_size),
  in_specs=[
    pl.BlockSpec(
      block_shape=(1, in_block_height_padded, in_block_width_padded, channels),
      index_map=lambda i, j, k: (i, j * tile_size * stride - padding, k * tile_size * stride - padding, 0),
    )
  ],
  out_specs=pl.BlockSpec(
    block_shape=(1, tile_size, tile_size, channels), index_map=lambda i, j, k: (i, j * tile_size, k * tile_size, 0)
  ),
)(x).block_until_ready()

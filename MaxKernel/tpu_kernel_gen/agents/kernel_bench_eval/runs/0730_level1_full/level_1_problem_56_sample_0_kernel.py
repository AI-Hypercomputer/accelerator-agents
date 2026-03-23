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
height = 256
width = 128

key = random.PRNGKey(0)
key_x, key_params = random.split(key)

# JAX/Flax uses channels-last (N, H, W, C) convention
x = random.normal(key_x, (batch_size, height, width, in_channels))

conv2d = nn.Conv(
  features=out_channels,
  kernel_size=kernel_size,
  strides=(1, 1),
  padding="VALID",
  kernel_dilation=(1, 1),
  feature_group_count=1,
  use_bias=False,
)
params = conv2d.init(key_params, x)["params"]

# We need a reference output to get the correct shape and dtype for the pallas_call
output_ref = conv2d.apply({"params": params}, x)
out_height, out_width = output_ref.shape[1:3]


# Computation
def kernel(x_ref, k_ref, o_ref):
  """Pallas kernel for 2D convolution."""
  # Load inputs from SRAM into registers.
  x_tile = x_ref[...]
  k_val = k_ref[...]
  # Reshape kernel once at the beginning.
  k_flat = k_val.reshape(kernel_size[0] * kernel_size[1] * in_channels, out_channels)

  # Get the current tile index from program_id
  h_idx = pl.program_id(axis=1)
  w_idx = pl.program_id(axis=2)

  # Calculate the starting coordinates of the output tile
  h_start = h_idx * o_ref.shape[1]
  w_start = w_idx * o_ref.shape[2]

  # Loop over the output tile
  for i in range(o_ref.shape[1]):
    for j in range(o_ref.shape[2]):
      # Calculate current global output coordinates
      h_curr = h_start + i
      w_curr = w_start + j

      # Define the computation for a single pixel
      def compute_and_write(_):
        # Extract a patch from the input tile.
        patch = jax.lax.dynamic_slice(
          x_tile, start_indices=(0, i, j, 0), slice_sizes=(1, kernel_size[0], kernel_size[1], in_channels)
        )
        # Flatten the patch for matmul.
        patch_flat = patch.reshape(1, kernel_size[0] * kernel_size[1] * in_channels)

        # Perform the convolution for this output pixel via matmul.
        out_pixel = patch_flat @ k_flat

        # Write the result to the output tile.
        o_ref[0, i, j, :] = out_pixel.squeeze(axis=0)

      # Define a no-op for out-of-bounds pixels
      def do_nothing(_):
        pass

      # Conditionally execute the computation only for valid pixels
      # This avoids the TracerBoolConversionError by using JAX's control flow
      jax.lax.cond(
        jnp.logical_and(h_curr < out_height, w_curr < out_width), compute_and_write, do_nothing, operand=None
      )


# Helper for ceiling division
def ceil_div(x, y):
  return (x + y - 1) // y


# Define tile sizes for the convolution operation
# Choose output tile dimensions that are TPU-friendly
tile_H_out = 8
# The width of the output tile must be chosen carefully to ensure that the
# corresponding input tile width satisfies TPU alignment constraints.
# Setting it to the exact output width of the convolution ensures that the
# block shape dimension equals the array dimension, satisfying the constraint.
tile_W_out = out_width

# Corresponding input tile dimensions
# H_in = H_out + kH - 1
# W_in = W_out + kW - 1
tile_H_in = tile_H_out + kernel_size[0] - 1
tile_W_in = tile_W_out + kernel_size[1] - 1

# The grid iterates over the tiles of the output
grid_h = ceil_div(out_height, tile_H_out)
grid_w = ceil_div(out_width, tile_W_out)
grid = (batch_size, grid_h, grid_w)

output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(output_ref.shape, output_ref.dtype),
  grid=grid,
  in_specs=[
    # Input image spec: loads one tile per kernel instance.
    # The input tile is anchored at the same top-left corner as the output tile.
    pl.BlockSpec(
      block_shape=(1, tile_H_in, tile_W_in, in_channels),
      index_map=lambda b, h, w: (b, h * tile_H_out, w * tile_W_out, 0),
    ),
    # Kernel spec: loads the entire kernel for each instance
    pl.BlockSpec(block_shape=params["kernel"].shape, index_map=lambda *_: (0,) * len(params["kernel"].shape)),
  ],
  out_specs=pl.BlockSpec(
    # Output spec: defines the shape of the output tile.
    block_shape=(1, tile_H_out, tile_W_out, out_channels),
    index_map=lambda b, h, w: (b, h * tile_H_out, w * tile_W_out, 0),
  ),
)(x, params["kernel"]).block_until_ready()

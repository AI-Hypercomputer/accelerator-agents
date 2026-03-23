# Imports
import flax.linen as nn
import jax
import jax.lax
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

key = random.PRNGKey(0)
key_x, key_params = random.split(key)

# JAX uses channels-last convention by default (N, H, W, C)
x = random.normal(key_x, (batch_size, height, width, in_channels))


class Model(nn.Module):
  @nn.compact
  def __call__(self, x):
    # Flax's ConvTranspose takes output channels as the `features` argument.
    # PyTorch's padding=0 corresponds to Flax's padding='VALID'.
    # PyTorch's bias=False corresponds to Flax's use_bias=False.
    # Other parameters like stride=1, groups=1 are defaults in Flax.
    conv_transpose2d = nn.ConvTranspose(
      features=out_channels, kernel_size=kernel_size, strides=(1, 1), padding="VALID", use_bias=False
    )
    return conv_transpose2d(x)


model = Model()
params = model.init(key_params, x)["params"]

# The 'VALID' padding in ConvTranspose means the output size is calculated as:
# output_size = (input_size - 1) * stride + kernel_size
output_height = (height - 1) * 1 + kernel_size[0]
output_width = (width - 1) * 1 + kernel_size[1]
output_shape = (batch_size, output_height, output_width, out_channels)

# Pallas kernel implementation
KH, KW = kernel_size
# Define tile sizes for the output block.
tile_h, tile_w = 32, 32


def kernel(x_ref, kernel_ref, out_ref):
  """Pallas kernel for 2D transposed convolution using output tiling."""
  # Get grid indices to identify which output tile this kernel is computing.
  b_idx, oh_tile_idx, ow_tile_idx = pl.program_id(0), pl.program_id(1), pl.program_id(2)

  # Calculate the base coordinates of the output tile.
  oh_base = oh_tile_idx * tile_h
  ow_base = ow_tile_idx * tile_w

  # Accumulator for the output tile, held in fast local memory (VMEM).
  acc = jnp.zeros((tile_h, tile_w, out_channels), dtype=x_ref.dtype)

  # Calculate padding required for the input tile based on kernel size.
  in_pad_h = KH - 1
  in_pad_w = KW - 1

  # Loop over each pixel within the output tile.
  for oh_offset in range(tile_h):
    for ow_offset in range(tile_w):
      # Global output coordinates.
      oh = oh_base + oh_offset
      ow = ow_base + ow_offset

      # Boundary check to avoid writing outside the output feature map.
      if oh >= output_height or ow >= output_width:
        continue

      pixel_acc = jnp.zeros(out_channels, dtype=x_ref.dtype)
      # Perform the convolution for a single output pixel.
      for kh in range(KH):
        for kw in range(KW):
          # Corresponding input coordinates.
          ih = oh - kh
          iw = ow - kw

          # Check if the input coordinates are within the valid input area.
          if 0 <= ih < height and 0 <= iw < width:
            # Calculate where to read from in the loaded input tile (x_ref).
            x_ref_h = ih - (oh_base - in_pad_h)
            x_ref_w = iw - (ow_base - in_pad_w)

            in_vec = x_ref[0, x_ref_h, x_ref_w, :]
            kernel_slice = kernel_ref[kh, kw, :, :]
            pixel_acc += jnp.dot(in_vec, kernel_slice)
      acc = acc.at[oh_offset, ow_offset, :].set(pixel_acc)

  # Write the completed tile from local memory back to HBM.
  out_ref[0, ...] = acc


# The default layer name assigned by Flax is 'ConvTranspose_0'.
kernel_weights = params["ConvTranspose_0"]["kernel"]

# Define the grid over which the kernel will be executed.
grid_h = (output_height + tile_h - 1) // tile_h
grid_w = (output_width + tile_w - 1) // tile_w
grid = (batch_size, grid_h, grid_w)

# Define the shape of the input block, accounting for kernel-induced padding.
in_block_h = tile_h + KH - 1
in_block_w = tile_w + KW - 1
in_block_shape = (1, in_block_h, in_block_w, in_channels)

# Padding values for calculating input tile indices.
in_pad_h = KH - 1
in_pad_w = KW - 1

output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(output_shape, x.dtype),
  grid=grid,
  in_specs=[
    pl.BlockSpec(in_block_shape, lambda b, gh, gw: (b, gh * tile_h - in_pad_h, gw * tile_w - in_pad_w, 0)),
    pl.BlockSpec(kernel_weights.shape, lambda *_: (0,) * kernel_weights.ndim),
  ],
  out_specs=pl.BlockSpec(
    (1, tile_h, tile_w, out_channels),
    lambda b, gh, gw: (b, gh * tile_h, gw * tile_w, 0),
  ),
)(x, kernel_weights).block_until_ready()

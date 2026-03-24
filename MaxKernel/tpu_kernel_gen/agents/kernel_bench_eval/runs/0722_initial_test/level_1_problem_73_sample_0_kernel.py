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
kernel_size = 3
depth = 16
height = 32
width = 32
stride = 2
padding = 3
groups = 4

key = random.PRNGKey(0)
key_x, key_params = random.split(key)

# JAX uses channels-last convention (N, D, H, W, C)
x = random.normal(key_x, (batch_size, depth, height, width, in_channels))

conv_transpose3d = nn.ConvTranspose(
  features=out_channels,
  kernel_size=(kernel_size, kernel_size, kernel_size),
  strides=stride,
  padding=padding,
  feature_group_count=groups,
  use_bias=False,
)
params = conv_transpose3d.init(key_params, x)
kernel_param = params["params"]["kernel"]

# Calculate output shape
output_shape_struct = jax.eval_shape(lambda: conv_transpose3d.apply(params, x))

# Define block sizes for tiling the output
bW = 16
bH = 16
bD = 4
# The number of output channels per group
bC = out_channels // groups

# The grid iterates over tiles of the output tensor
grid = (
  batch_size,
  (output_shape_struct.shape[1] + bD - 1) // bD,
  (output_shape_struct.shape[2] + bH - 1) // bH,
  (output_shape_struct.shape[3] + bW - 1) // bW,
  groups,
)


def kernel(x_ref, kernel_ref, out_ref):
  """
  Pallas kernel for 3D transposed convolution with feature grouping.
  """
  # Hardcoded convolution parameters
  stride = 2
  padding = 3
  groups = 4
  in_channels = 32
  out_channels = 64

  # Get the unique identifiers for this kernel instance from the grid.
  n_idx = pl.program_id(0)
  d_idx = pl.program_id(1)
  h_idx = pl.program_id(2)
  w_idx = pl.program_id(3)
  g_idx = pl.program_id(4)

  # Get tensor dimensions from the input and output references.
  _, in_depth, in_height, in_width, _ = x_ref.shape
  k_depth, k_height, k_width, _, _ = kernel_ref.shape
  # Shape of the output block this kernel instance is responsible for.
  bD, bH, bW, bC = out_ref.shape

  # Determine the input channel range for the current group.
  in_channels_per_group = in_channels // groups
  in_channel_start = g_idx * in_channels_per_group
  in_channel_end = in_channel_start + in_channels_per_group

  # Initialize an accumulator for the output tile with zeros.
  acc = jnp.zeros(out_ref.shape, dtype=out_ref.dtype)

  # Iterate over each position (bd, bh, bw) in the output block.
  for bd, bh, bw in pl.grid(bD, bH, bW):
    # Calculate the global output coordinates.
    od = d_idx * bD + bd
    oh = h_idx * bH + bh
    ow = w_idx * bW + bw

    pixel_acc = jnp.zeros((bC,), dtype=x_ref.dtype)

    # Iterate over the kernel's spatial dimensions.
    for kd, kh, kw in pl.grid(k_depth, k_height, k_width):
      id_num = od + kd - padding
      ih_num = oh + kh - padding
      iw_num = ow + kw - padding

      # Check if the numerator is divisible by stride
      if (id_num % stride == 0) and (ih_num % stride == 0) and (iw_num % stride == 0):
        id = id_num // stride
        ih = ih_num // stride
        iw = iw_num // stride

        # Check if the calculated input coordinates are within valid bounds.
        if (id >= 0) and (id < in_depth) and (ih >= 0) and (ih < in_height) and (iw >= 0) and (iw < in_width):
          # Slice the relevant input channels for the current group.
          x_slice = x_ref[n_idx, id, ih, iw, in_channel_start:in_channel_end]
          # Slice the relevant kernel weights for the current group.
          out_channel_start = g_idx * bC
          out_channel_end = out_channel_start + bC
          kernel_slice = kernel_ref[kd, kh, kw, :, out_channel_start:out_channel_end]
          # Perform the dot product and accumulate.
          pixel_acc += x_slice @ kernel_slice

    # Store the final accumulated value for the pixel in the output accumulator.
    acc = acc.at[bd, bh, bw, :].set(pixel_acc)

  # Write the computed block to the output reference.
  out_ref[...] = acc


# Computation
output = pl.pallas_call(
  kernel,
  out_shape=output_shape_struct,
  grid=grid,
  in_specs=[
    pl.BlockSpec(lambda *_: (0, 0, 0, 0, 0), x.shape),
    pl.BlockSpec(lambda *_: (0, 0, 0, 0, 0), kernel_param.shape),
  ],
  out_specs=pl.BlockSpec(block_shape=(bD, bH, bW, bC), index_map=lambda n, d, h, w, g: (n, d, h, w, g * bC)),
  compiler_params=dict(),
)(x, kernel_param).block_until_ready()

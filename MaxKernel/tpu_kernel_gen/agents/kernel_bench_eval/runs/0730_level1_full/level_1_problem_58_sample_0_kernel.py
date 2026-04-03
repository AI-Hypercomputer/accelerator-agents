# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 32
out_channels = 16
kernel_size = (3, 5, 7)
depth_in = 16
height_in = 32
width_in = 64

key = random.PRNGKey(0)
key_x, key_params = random.split(key)

# JAX uses channels-last (NDHWC) format by default
x = random.normal(key_x, (batch_size, depth_in, height_in, width_in, in_channels))

# In Flax, layers are stateless. We initialize them to get the parameters.
# PyTorch padding=0 is 'VALID' in Flax.
# PyTorch groups=1 is the default behavior in Flax ConvTranspose and doesn't require a specific argument.
conv_transpose3d = nn.ConvTranspose(
  features=out_channels, kernel_size=kernel_size, strides=(1, 1, 1), padding="VALID", use_bias=False
)
params = conv_transpose3d.init(key_params, x)["params"]
kernel_weights = params["kernel"]
kd, kh, kw, _, _ = kernel_weights.shape

# Calculate output dimensions based on 'VALID' padding and a stride of 1.
output_depth = depth_in + kd - 1
output_height = height_in + kh - 1
output_width = width_in + kw - 1
output_shape = (batch_size, output_depth, output_height, output_width, out_channels)

# Define tile sizes for the output block.
# These must be multiples of 8 to satisfy TPU constraints.
BLOCK_D = 8
BLOCK_H = 8
BLOCK_W = 8


def conv_transpose_3d_kernel(x_ref, kernel_ref, out_ref):
  # This kernel computes one tile of the output using a "gather" approach.
  # It implements a direct convolution with a flipped kernel, which is
  # equivalent to a transposed convolution.
  acc = jnp.zeros(out_ref.shape, dtype=x_ref.dtype)

  kd, kh, kw, _, _ = kernel_ref.shape
  _, out_tile_d, out_tile_h, out_tile_w, _ = out_ref.shape

  # Load the entire input patch needed for this output tile.
  x_tile = x_ref[...]

  # Iterate over the kernel dimensions.
  for kdi in range(kd):
    for khi in range(kh):
      for kwi in range(kw):
        # The kernel is flipped for the convolution.
        k_flipped_d = kd - 1 - kdi
        k_flipped_h = kh - 1 - khi
        k_flipped_w = kw - 1 - kwi

        # Extract the slice of the input that will be multiplied
        # by the current kernel element.
        x_slice = jax.lax.dynamic_slice(
          x_tile,
          start_indices=(0, kdi, khi, kwi, 0),
          slice_sizes=(1, out_tile_d, out_tile_h, out_tile_w, in_channels),
        )

        # Get the current kernel element's weights across all channels.
        kernel_matrix = kernel_ref[k_flipped_d, k_flipped_h, k_flipped_w, :, :]

        # Perform matrix multiplication: (..., in_channels) @ (in_channels, out_channels)
        update = jnp.dot(x_slice, kernel_matrix)
        acc += update

  out_ref[...] = acc


# A 'VALID' transposed convolution is equivalent to a 'FULL' direct convolution.
# This requires symmetric padding of size (k-1) on the input tensor.
pad_d, pad_h, pad_w = kd - 1, kh - 1, kw - 1
x_padded = jnp.pad(x, ((0, 0), (pad_d, pad_d), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="constant")

# Define the grid over which the kernel will be executed.
grid_d = (output_depth + BLOCK_D - 1) // BLOCK_D
grid_h = (output_height + BLOCK_H - 1) // BLOCK_H
grid_w = (output_width + BLOCK_W - 1) // BLOCK_W
grid = (batch_size, grid_d, grid_h, grid_w)


# Calculate the required input block shapes, padding them to the next multiple of 8
# to satisfy TPU hardware constraints.
def _to_multiple(val, N):
  return (val + N - 1) // N * N


in_block_d = _to_multiple(BLOCK_D + kd - 1, 8)
in_block_h = _to_multiple(BLOCK_H + kh - 1, 8)
in_block_w = _to_multiple(BLOCK_W + kw - 1, 8)

# Execute the Pallas kernel.
output = pl.pallas_call(
  conv_transpose_3d_kernel,
  out_shape=jax.ShapeDtypeStruct(output_shape, x.dtype),
  grid=grid,
  in_specs=[
    pl.BlockSpec(
      block_shape=(1, in_block_d, in_block_h, in_block_w, in_channels),
      index_map=lambda b, d, h, w: (b, d * BLOCK_D, h * BLOCK_H, w * BLOCK_W, 0),
    ),
    pl.BlockSpec(block_shape=kernel_weights.shape, index_map=lambda *_: tuple([0] * kernel_weights.ndim)),
  ],
  out_specs=pl.BlockSpec(
    block_shape=(1, BLOCK_D, BLOCK_H, BLOCK_W, out_channels),
    index_map=lambda b, d, h, w: (b, d * BLOCK_D, h * BLOCK_H, w * BLOCK_W, 0),
  ),
)(x_padded, kernel_weights).block_until_ready()

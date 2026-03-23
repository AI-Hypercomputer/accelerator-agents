# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 3
kernel_size = 3
width = 256
height = 256
stride = 1
padding = 0
dilation = 1
bias = False

key = random.PRNGKey(0)
key_params, key_x = random.split(key)

# JAX uses channels-last convention (NHWC) by default
x = random.normal(key_x, (batch_size, height, width, in_channels))

# In Flax, nn.Conv with feature_group_count=in_channels is a depthwise convolution
conv2d = nn.Conv(
  features=in_channels,
  kernel_size=(kernel_size, 1),
  strides=stride,
  padding=padding,
  kernel_dilation=dilation,
  feature_group_count=in_channels,
  use_bias=bias,
)
params = conv2d.init(key_params, x)["params"]


# Computation
def kernel(x_ref, w_ref, out_ref):
  """Pallas kernel for vertical depthwise convolution.

  Args:
    x_ref: Input tile.
    w_ref: Kernel weights.
    out_ref: Output tile to be written to.
  """
  # Get kernel height and output height from the shapes of the references.
  kernel_height = w_ref.shape[0]
  # out_ref has shape (1, tile_h, tile_w, in_channels)
  out_height = out_ref.shape[1]

  # Initialize the output tile with zeros.
  out_ref[...] = jnp.zeros_like(out_ref)

  # Iterate over the vertical dimension of the kernel.
  for kh in range(kernel_height):
    # Slice the input tile vertically. x_ref has shape (1, tile_h + kernel_size - 1, tile_w, C)
    # The slice starts at `kh` and has height `out_height` (which is tile_h).
    x_slice = x_ref[0, kh : kh + out_height, :, :]

    # Get the corresponding kernel weights for this vertical position.
    # w_ref has shape (KH, 1, 1, C_in), so we slice it to get (C_in,).
    w_slice = w_ref[kh, 0, 0, :]

    # Perform the depthwise multiplication and accumulate.
    # The w_slice of shape (C_in,) is broadcast across the (H, W) dimensions
    # of x_slice.
    out_ref[0, ...] += x_slice * w_slice


# Define tiling parameters based on the problem
# To avoid OOM, we tile both width and height dimensions.
tile_w = 128
out_height = height - kernel_size + 1
# To avoid an out-of-bounds read, we select a tile_h that divides out_height.
# out_height = 254, so we can use 127.
tile_h = 127

# Define shapes based on the initialization block
out_width = width
output_shape = (batch_size, out_height, out_width, in_channels)

# The kernel weights from Flax have shape (KH, KW, 1, C_in) for depthwise conv
kernel_shape = (kernel_size, 1, 1, in_channels)

# Calculate grid size, using ceiling division to handle all pixels
grid_h = (out_height + tile_h - 1) // tile_h
grid_w = (out_width + tile_w - 1) // tile_w

result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(output_shape, x.dtype),
  # Grid iterates over batch, height tiles, and width tiles
  grid=(batch_size, grid_h, grid_w),
  in_specs=[
    # Input 'x' spec: Each grid element gets a tile of the input with a halo
    # for the convolution.
    pl.BlockSpec(
      block_shape=(1, tile_h + kernel_size - 1, tile_w, in_channels),
      index_map=lambda b, i, j: (b, i * tile_h, j * tile_w, 0),
    ),
    # Kernel spec: The entire kernel is passed to each grid element
    pl.BlockSpec(block_shape=kernel_shape, index_map=lambda b, i, j: (0, 0, 0, 0)),
  ],
  # Output spec: Each grid element computes a corresponding output tile.
  out_specs=pl.BlockSpec(
    block_shape=(1, tile_h, tile_w, in_channels), index_map=lambda b, i, j: (b, i * tile_h, j * tile_w, 0)
  ),
)(x, params["kernel"]).block_until_ready()

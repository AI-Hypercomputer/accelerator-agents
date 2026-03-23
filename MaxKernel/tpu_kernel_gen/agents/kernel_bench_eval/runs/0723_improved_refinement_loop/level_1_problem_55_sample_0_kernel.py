# Imports
import flax.linen as nn
import jax
import jax.random as random
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

# Initialization
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
width = 256
height = 128
stride = 1
padding = "VALID"  # Pallas call will handle tiling for a 'VALID' convolution
dilation = 1
groups = 1
bias = False

key = random.PRNGKey(0)
key_x, key_params = random.split(key)

# Note: JAX uses channels-last (NHWC) convention by default
x = random.normal(key_x, (batch_size, height, width, in_channels))

conv2d = nn.Conv(
  features=out_channels,
  kernel_size=(kernel_size, kernel_size),
  strides=(stride, stride),
  padding=padding,
  kernel_dilation=(dilation, dilation),
  feature_group_count=groups,
  use_bias=bias,
)
variables = conv2d.init(key_params, x)


# Computation
def kernel(x_ref, kernel_ref, out_ref):
  """
  Pallas kernel for 2D convolution.

  This kernel computes a tile of the convolution output. It uses
  `jax.lax.conv_general_dilated` to perform the core computation for each
  tile. The `pallas_call` orchestrates the tiling of the input and output
  tensors from HBM into SRAM, and this kernel operates on those SRAM tiles.

  Args:
    x_ref: A reference to the input tile (e.g., shape [1, 32, 128, 3]).
    kernel_ref: A reference to the complete convolution kernel (e.g., shape [3, 3, 3, 64]).
    out_ref: A reference to the output tile to be written to (e.g., shape [1, 32, 128, 64]).
  """
  # Perform the 2D convolution on the input tile.
  # 'VALID' padding is used here because the input tile `x_ref` is already
  # padded with the halo required for the convolution. Applying a 'VALID'
  # convolution to this padded tile produces an output of the correct size
  # for `out_ref`.
  out_ref[...] = tpu.conv(
    x_ref,
    kernel_ref,
    window_strides=(stride, stride),
    padding="VALID",
    dimension_numbers=("NHWC", "HWIO", "NHWC"),
    feature_group_count=groups,
    rhs_dilation=(dilation, dilation),
  )


# Define tiling and grid parameters
h_block, w_block = 32, 128
out_height = height - kernel_size + 1
out_width = width - kernel_size + 1
output_shape = (batch_size, out_height, out_width, out_channels)

grid_h = (out_height + h_block - 1) // h_block
grid_w = (out_width + w_block - 1) // w_block
grid = (batch_size, grid_h, grid_w)

# This call replaces the original conv2d.apply computation.
output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(output_shape, x.dtype),
  grid=grid,
  in_specs=[
    pl.BlockSpec(
      (1, h_block + kernel_size - 1, w_block + kernel_size - 1, in_channels),
      lambda n, h, w: (n, h * h_block, w * w_block, 0),
    ),
    pl.BlockSpec(
      variables["params"]["kernel"].shape,
      lambda *_: tuple(0 for _ in range(variables["params"]["kernel"].ndim)),
    ),
  ],
  out_specs=pl.BlockSpec((1, h_block, w_block, out_channels), lambda n, h, w: (n, h * h_block, w * w_block, 0)),
  compiler_params=dict(mosaic=dict(dimension_semantics=("parallel", "parallel", "parallel"))),
)(x, variables["params"]["kernel"]).block_until_ready()

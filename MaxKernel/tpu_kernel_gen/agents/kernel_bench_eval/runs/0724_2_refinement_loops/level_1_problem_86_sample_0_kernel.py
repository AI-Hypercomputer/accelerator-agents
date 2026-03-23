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
kernel_size = 3
width = 256
height = 256
stride = 1
padding = 0
dilation = 1
bias = False

key = random.PRNGKey(0)
key_params, key_x = random.split(key)

# JAX uses channels-last convention (N, H, W, C)
x = random.normal(key_x, (batch_size, height, width, in_channels))

# In Flax, a sequence of layers is typically defined using nn.Sequential
# or a custom nn.Module. This is the idiomatic way to represent the
# original code's sequential application of two convolutions.
model = nn.Sequential(
  [
    nn.Conv(
      features=in_channels,
      kernel_size=(kernel_size, kernel_size),
      strides=(stride, stride),
      padding=padding,
      kernel_dilation=(dilation, dilation),
      feature_group_count=in_channels,
      use_bias=bias,
    ),
    nn.Conv(features=out_channels, kernel_size=(1, 1), use_bias=bias),
  ]
)

params = model.init(key_params, x)["params"]


# Computation
def kernel(x_ref, dw_kernel_ref, pw_kernel_ref, out_ref):
  """Pallas kernel for depthwise separable convolution.

  This kernel performs a depthwise convolution followed by a pointwise (1x1)
  convolution on a tile of the input image.

  Args:
    x_ref: A reference to a tile of the input image. It includes a halo
      region necessary for the convolution and is padded for hardware
      alignment.
    dw_kernel_ref: A reference to the depthwise convolution kernel.
    pw_kernel_ref: A reference to the pointwise convolution kernel.
    out_ref: A reference to the output tile where the result is stored.
  """
  # Load the input tile and convolution kernels from SRAM into registers.
  x = x_ref[...]
  dw_kernel = dw_kernel_ref[...]
  pw_kernel = pw_kernel_ref[...]

  # The input tile `x` is padded for hardware memory alignment. We must
  # slice it to the actual dimensions required for the convolution before
  # proceeding. The required width for a 3x3 convolution producing an
  # output of width TILE_W is TILE_W + 2.
  tile_h, tile_w = out_ref.shape[1:3]
  x_unpadded = x[:, : tile_h + 2, : tile_w + 2, :]

  # Allocate an intermediate buffer for the output of the depthwise convolution.
  dw_out = jnp.zeros((1, tile_h, tile_w, x.shape[-1]), dtype=x.dtype)

  # 1. Perform the depthwise convolution manually by iterating through the kernel.
  # This avoids using lax.scatter, which is not supported on TPU.
  squeezed_dw_kernel = jnp.squeeze(dw_kernel, axis=2)
  for kh in range(3):
    for kw in range(3):
      k = squeezed_dw_kernel[kh, kw, :]
      in_slice = x_unpadded[:, kh : kh + tile_h, kw : kw + tile_w, :]
      dw_out += in_slice * k

  # 2. Perform the pointwise (1x1) convolution as a batched matrix multiplication.
  pw_kernel_squeezed = jnp.squeeze(pw_kernel, axis=(0, 1))
  pw_out = dw_out @ pw_kernel_squeezed

  # Write the final result to the output tile.
  out_ref[...] = pw_out


# The original computation performs a depthwise separable convolution, which consists
# of a depthwise convolution followed by a pointwise (1x1) convolution.
# The pallas_call will replace this two-layer operation.

# The kernel takes three inputs: the image `x`, the depthwise kernel, and the pointwise kernel.
dw_kernel = params["layers_0"]["kernel"]
pw_kernel = params["layers_1"]["kernel"]

# Output dimensions after a 3x3 convolution with stride 1 and no padding on a 256x256 image.
# The second 1x1 convolution does not change spatial dimensions.
H_OUT, W_OUT = 254, 254

# We tile the computation over the output's spatial dimensions.
TILE_H, TILE_W = 16, 16

# The input patch width must be padded to be compatible with TPU hardware constraints.
# Required width = TILE_W + kernel_size - 1 = 16 + 3 - 1 = 18.
# Padded width = 24, the next multiple of 8.
INPUT_TILE_W_PADDED = 24

x = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, H_OUT, W_OUT, out_channels), x.dtype),
  # The grid is 3D, parallelizing over the batch and the H/W tiles of the output.
  grid=(batch_size, (H_OUT + TILE_H - 1) // TILE_H, (W_OUT + TILE_W - 1) // TILE_W),
  in_specs=[
    # Input image `x`. Each kernel instance receives a patch. The patch is
    # larger than the output tile to account for the convolution window (3x3).
    # The width is padded to 24 to meet TPU alignment requirements (must be divisible by 8).
    pl.BlockSpec(
      block_shape=(1, TILE_H + 2, INPUT_TILE_W_PADDED, in_channels),
      index_map=lambda n, i, j: (n, i * TILE_H, j * TILE_W, 0),
    ),
    # The depthwise and pointwise kernels are small and read in their entirety
    # by every kernel instance, so they are not tiled.
    pl.BlockSpec(block_shape=dw_kernel.shape, index_map=lambda n, i, j: (0, 0, 0, 0)),
    pl.BlockSpec(block_shape=pw_kernel.shape, index_map=lambda n, i, j: (0, 0, 0, 0)),
  ],
  out_specs=pl.BlockSpec(
    # Each kernel instance computes one tile of the output.
    block_shape=(1, TILE_H, TILE_W, out_channels),
    index_map=lambda n, i, j: (n, i * TILE_H, j * TILE_W, 0),
  ),
)(x, dw_kernel, pw_kernel).block_until_ready()

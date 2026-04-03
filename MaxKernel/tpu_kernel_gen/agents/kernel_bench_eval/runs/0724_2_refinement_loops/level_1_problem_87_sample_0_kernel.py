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
width = 256
height = 256
bias = False
key = random.PRNGKey(0)
key_x, key_params = random.split(key)
# Note: JAX uses channels-last convention by default (N, H, W, C)
x = random.normal(key_x, (batch_size, height, width, in_channels))
conv2d = nn.Conv(features=out_channels, kernel_size=(1, 1), strides=(1, 1), padding="VALID", use_bias=bias)
params = conv2d.init(key_params, x)["params"]

# Define block sizes for spatial dimensions that are TPU-compatible.
# The second-to-last dimension of a block must be divisible by 128.
block_h = 64
block_w = 128


# Computation
def kernel(x_ref, kernel_ref, out_ref):
  """Pallas kernel for a 1x1 convolution.

  This kernel computes a 1x1 convolution, which is mathematically equivalent
  to a matrix multiplication between the input channels and the output channels
  at each spatial location.

  Args:
    x_ref: A reference to a tile of the input tensor. The shape is
      (1, block_h, block_w, in_channels).
    kernel_ref: A reference to the convolution kernel tensor. The shape is
      (1, 1, in_channels, out_channels).
    out_ref: A reference to the output tensor tile, where the result is written
      in-place. The shape is (1, block_h, block_w, out_channels).
  """
  # Load the input tile and the full convolution kernel from SRAM into registers.
  x_tile = x_ref[...]
  conv_kernel = kernel_ref[...]

  # A 1x1 convolution is equivalent to a matrix multiplication across the
  # channel dimension. To perform this with jnp.dot, we reshape the
  # convolution kernel from (1, 1, C_in, C_out) to (C_in, C_out).
  kernel_2d = jnp.squeeze(conv_kernel, axis=(0, 1))

  # The TPU matmul unit expects 2D inputs. We reshape the input tile by
  # flattening the batch and spatial dimensions into a single dimension.
  x_tile_2d = jnp.reshape(x_tile, (-1, in_channels))

  # Perform the matrix multiplication.
  # The result has shape (1 * block_h * block_w, out_channels).
  output_tile_2d = jnp.dot(x_tile_2d, kernel_2d)

  # Reshape the output back to its original 4D tile shape.
  output_tile = jnp.reshape(output_tile_2d, x_ref.shape[:-1] + (out_channels,))

  # Write the computed tile to the output buffer in HBM.
  out_ref[...] = output_tile


# The 1x1 convolution can be parallelized over the batch, height, and width
# dimensions. Each kernel instance will process a tile of the input image.
# The grid is 3D, corresponding to the tiles of the N, H, and W dimensions.
grid = (batch_size, height // block_h, width // block_w)

output = pl.pallas_call(
  kernel,
  # The output has the same spatial dimensions as the input, but different channel count.
  out_shape=jax.ShapeDtypeStruct((batch_size, height, width, out_channels), x.dtype),
  grid=grid,
  in_specs=[
    # Spec for input 'x':
    # We tile along the batch, height, and width dimensions. Each kernel instance gets a
    # (1, block_h, block_w, in_channels) block of the input.
    # The index_map specifies the block indices, and Pallas computes the
    # element offset by multiplying the block indices by the block shape.
    pl.BlockSpec(block_shape=(1, block_h, block_w, in_channels), index_map=lambda b, i, j: (b, i, j, 0)),
    # Spec for convolution kernel 'params['kernel']':
    # The 1x1 convolution kernel is small and used by all grid instances,
    # so we pass the entire kernel to each instance without chunking.
    pl.BlockSpec(block_shape=(1, 1, in_channels, out_channels), index_map=lambda b, i, j: (0, 0, 0, 0)),
  ],
  # Spec for the output:
  # The output is tiled similarly to the input 'x', mapping each grid
  # instance (b, i, j) to a unique block in the output tensor.
  out_specs=pl.BlockSpec(block_shape=(1, block_h, block_w, out_channels), index_map=lambda b, i, j: (b, i, j, 0)),
)(x, params["kernel"]).block_until_ready()

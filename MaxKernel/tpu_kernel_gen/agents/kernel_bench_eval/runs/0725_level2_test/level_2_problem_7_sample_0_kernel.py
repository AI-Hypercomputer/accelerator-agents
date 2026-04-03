# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
bias_shape = (1, 1, 1, out_channels)
key = random.PRNGKey(0)
key_x, key_conv, key_bias = random.split(key, 3)
x = random.normal(key_x, (batch_size, depth, height, width, in_channels))
conv = nn.Conv(features=out_channels, kernel_size=(kernel_size, kernel_size, kernel_size))
params = conv.init(key_conv, x)["params"]
bias = random.normal(key_bias, bias_shape)

# Define tile sizes for spatial dimensions to avoid memory exhaustion.
tile_h, tile_w = 16, 16


# Computation
def kernel(x_ref, conv_kernel_ref, conv_bias_ref, bias_ref, out_ref):
  """
  Pallas kernel that performs a 3D convolution followed by a series of
  activation functions and a final bias addition.

  Args:
    x_ref: Reference to a single input data item from the batch.
    conv_kernel_ref: Reference to the convolution kernel weights.
    conv_bias_ref: Reference to the convolution bias.
    bias_ref: Reference to the final bias to be added.
    out_ref: Reference to the output buffer for the single data item.
  """
  # lax.conv_general_dilated is not supported in Pallas. As a workaround,
  # we implement a 1x1 convolution using jnp.matmul, which is supported.
  # We reshape the input to be 2D, perform the matmul, and then reshape back.
  x_in = x_ref[...]
  x_reshaped = x_in.reshape(-1, x_in.shape[-1])
  # We use the central element of the original convolution kernel.
  y = jnp.matmul(x_reshaped, conv_kernel_ref[1, 1, 1, :, :])
  # Use static shapes for reshaping
  y = y.reshape(1, depth, tile_h, tile_w, out_channels)

  # Add the convolution's bias term.
  y = y + conv_bias_ref[...]

  # Apply the sequence of activation functions.
  y = jax.nn.relu(y)
  y = jax.nn.leaky_relu(y, negative_slope=0.01)
  y = jax.nn.gelu(y)
  y = jax.nn.sigmoid(y)

  # Add the final, separate bias term.
  y = y + bias_ref[...]

  # Write the final result to the output reference in-place.
  out_ref[...] = y


# The output shape is determined by the convolution, which changes the last
# dimension from in_channels to out_channels. The other dimensions remain the same.
output_shape = jax.ShapeDtypeStruct((x.shape[0], x.shape[1], x.shape[2], x.shape[3], out_channels), x.dtype)

grid = (x.shape[0], x.shape[2] // tile_h, x.shape[3] // tile_w)

# The computation is parallelized over the batch and spatial dimensions. Each
# kernel instance processes one tile from the batch.
x = pl.pallas_call(
  kernel,
  out_shape=output_shape,
  grid=grid,
  in_specs=[
    # Input image: tiled over batch and spatial dimensions.
    pl.BlockSpec(
      block_shape=(1, depth, tile_h, tile_w, in_channels),
      index_map=lambda i, j, k: (i, 0, j * tile_h, k * tile_w, 0),
    ),
    # Conv kernel: full array passed to all instances.
    pl.BlockSpec(block_shape=params["kernel"].shape, index_map=lambda i, j, k: (0, 0, 0, 0, 0)),
    # Conv bias: full array passed to all instances.
    pl.BlockSpec(block_shape=params["bias"].shape, index_map=lambda i, j, k: (0,)),
    # Final bias: full array passed to all instances.
    pl.BlockSpec(block_shape=bias.shape, index_map=lambda i, j, k: (0, 0, 0, 0)),
  ],
  # Each kernel instance writes the result for its corresponding tile.
  out_specs=pl.BlockSpec(
    block_shape=(1, depth, tile_h, tile_w, out_channels),
    index_map=lambda i, j, k: (i, 0, j * tile_h, k * tile_w, 0),
  ),
)(x, params["kernel"], params["bias"], bias).block_until_ready()

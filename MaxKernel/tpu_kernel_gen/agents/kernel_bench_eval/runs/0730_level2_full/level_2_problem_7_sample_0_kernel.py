# Imports
import flax.linen as nn
import jax
import jax.random as random
from jax.experimental import pallas as pl
from jax.experimental.pallas.ops.tpu import convolution

# Initialization
batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
bias_shape = (out_channels,)
key = random.PRNGKey(0)
key_conv, key_bias, key_x = random.split(key, 3)

conv = nn.Conv(features=out_channels, kernel_size=(kernel_size, kernel_size, kernel_size))
x = random.normal(key_x, (batch_size, depth, height, width, in_channels))
params = conv.init(key_conv, x)["params"]
bias = random.normal(key_bias, bias_shape)


def kernel(x_ref, kernel_ref, conv_bias_ref, bias_ref, out_ref):
  # Perform 3D convolution
  # The dimension numbers specify the layout of the tensors:
  # 'NDHWC' for input/output: Batch, Depth, Height, Width, Channels
  # 'DHWIO' for kernel: Depth, Height, Width, Input channels, Output channels
  dimension_numbers = ("NDHWC", "DHWIO", "NDHWC")
  y = convolution(
    x_ref[...],
    kernel_ref[...],
    window_strides=(1, 1, 1),
    padding="SAME",
    dimension_numbers=dimension_numbers,
  )

  # The convolution with 'SAME' padding on an input block of depth 3 produces
  # an output block of depth 3. We take the central slice to get the
  # actual output for the current position in the grid.
  y = y[:, 1:2, ...]

  # Add the convolution's bias
  y = y + conv_bias_ref[...]

  # Apply activation functions
  y = nn.relu(y)
  y = nn.leaky_relu(y, negative_slope=0.01)
  y = nn.gelu(y)
  y = nn.sigmoid(y)

  # Add the final bias
  y = y + bias_ref[...]

  # Store the result
  out_ref[...] = y


# Computation
x = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, depth, height, width, out_channels), x.dtype),
  grid=(batch_size, depth),
  in_specs=[
    pl.BlockSpec(
      block_shape=(1, kernel_size, height, width, in_channels),
      index_map=lambda b, d: (b, d - 1, 0, 0, 0),
    ),
    pl.BlockSpec(
      block_shape=params["kernel"].shape,
      index_map=lambda b, d: tuple([0] * params["kernel"].ndim),
    ),
    pl.BlockSpec(
      block_shape=params["bias"].shape,
      index_map=lambda b, d: (0,),
    ),
    pl.BlockSpec(block_shape=bias.shape, index_map=lambda b, d: (0,)),
  ],
  out_specs=pl.BlockSpec(
    block_shape=(1, 1, height, width, out_channels),
    index_map=lambda b, d: (b, d, 0, 0, 0),
  ),
)(x, params["kernel"], params["bias"], bias).block_until_ready()

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
key = random.PRNGKey(0)
key_x, key_conv, key_scale, key_bias = random.split(key, 4)

# Note: JAX uses channel-last convention (N, D, H, W, C)
x = random.normal(key_x, (batch_size, depth, height, width, in_channels))
conv = nn.Conv(features=out_channels, kernel_size=(kernel_size, kernel_size, kernel_size), padding="VALID")
conv_params = conv.init(key_conv, x)["params"]
bias_shape = (1, 1, 1, out_channels)
scaling_factor = random.normal(key_scale, bias_shape)
bias = random.normal(key_bias, bias_shape)

# Calculate output dimensions after 'VALID' convolution
out_depth = depth - kernel_size + 1
out_height = height - kernel_size + 1
out_width = width - kernel_size + 1


# Computation
def kernel(x_ref, kernel_ref, conv_bias_ref, scaling_factor_ref, bias_ref, out_ref):
  """
  Pallas kernel implementing a sequence of 3D convolution and activation functions.
  """
  # Define the dimension numbers for the 3D convolution. This tells JAX how to
  # interpret the dimensions of the input, kernel, and output tensors.
  # 'NDHWC': Batch, Depth, Height, Width, Channels (for input and output)
  # 'DHWIO': Depth, Height, Width, Input Channels, Output Channels (for the kernel)
  dimension_numbers = ("NDHWC", "DHWIO", "NDHWC")

  # 1. Perform the 3D convolution.
  # `lhs` (left-hand side) is the input tensor.
  # `rhs` (right-hand side) is the convolution kernel.
  # `window_strides` defines the step size of the kernel over the input.
  # `padding='VALID'` means no padding is applied, so the output size is reduced.
  y = jax.lax.conv_general_dilated(
    lhs=x_ref[...],
    rhs=kernel_ref[...],
    window_strides=(1, 1, 1),
    padding="VALID",
    dimension_numbers=dimension_numbers,
  )

  # Add the convolution bias. JAX handles the broadcasting of the bias
  # tensor across the spatial and batch dimensions of the output `y`.
  y = y + conv_bias_ref[...]

  # 2. Apply element-wise scaling.
  y = y * scaling_factor_ref[...]

  # 3. Apply the tanh activation function.
  y = jnp.tanh(y)

  # 4. Apply the final element-wise bias.
  y = y * bias_ref[...]

  # 5. Apply the sigmoid activation function.
  y = nn.sigmoid(y)

  # 6. Write the final result to the output buffer in-place.
  out_ref[...] = y


x = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, out_depth, out_height, out_width, out_channels), x.dtype),
  grid=(batch_size,),
  in_specs=[
    # Input image tensor 'x'
    pl.BlockSpec(
      block_shape=(1, depth, height, width, in_channels),
      index_map=lambda i: (i, 0, 0, 0, 0),
    ),
    # Convolution kernel weights
    pl.BlockSpec(
      block_shape=conv_params["kernel"].shape,
      index_map=lambda i: (0, 0, 0, 0, 0),
    ),
    # Convolution bias
    pl.BlockSpec(block_shape=conv_params["bias"].shape, index_map=lambda i: (0,)),
    # Scaling factor
    pl.BlockSpec(block_shape=scaling_factor.shape, index_map=lambda i: (0, 0, 0, 0)),
    # Final bias
    pl.BlockSpec(block_shape=bias.shape, index_map=lambda i: (0, 0, 0, 0)),
  ],
  out_specs=pl.BlockSpec(
    block_shape=(1, out_depth, out_height, out_width, out_channels),
    index_map=lambda i: (i, 0, 0, 0, 0),
  ),
  interpret=True,
)(x, conv_params["kernel"], conv_params["bias"], scaling_factor, bias).block_until_ready()

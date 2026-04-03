# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_channels = 32
out_channels = 64
height, width = 16, 16
kernel_size = 4
stride = 2
padding = 1
output_padding = 1
bias_shape = (1, 1, 1, out_channels)  # JAX uses channel-last (N, H, W, C)
scaling_factor = 2.0

key = random.PRNGKey(0)
key_params, key_bias, key_x = random.split(key, 3)

# For a transposed convolution with stride 2, kernel size 4, to go from 16x16
# to 32x32, the correct explicit padding is ((1, 1), (1, 1)).
# Output size = (Input - 1) * Stride + Kernel - Padding_Start - Padding_End
# 32 = (16 - 1) * 2 + 4 - 1 - 1 = 30 + 4 - 2 = 32
conv_transpose_padding = ((1, 1), (1, 1))

conv_transpose = nn.ConvTranspose(
  features=out_channels,
  kernel_size=(kernel_size, kernel_size),
  strides=(stride, stride),
  padding=conv_transpose_padding,
)
x_shape = (batch_size, height, width, in_channels)  # JAX uses channel-last
params = conv_transpose.init(key_params, jnp.ones(x_shape))["params"]

bias = random.normal(key_bias, bias_shape)
x = random.normal(key_x, x_shape)


# Computation
def kernel(x_ref, kernel_ref, bias_ref, out_ref):
  """
  Pallas kernel for a sequence of neural network operations.

  This kernel performs the following steps:
  1. Transposed convolution on the input `x_ref` with `kernel_ref`.
  2. Softmax activation along the channel dimension.
  3. Addition of a `bias_ref`.
  4. Multiplication by a constant scaling factor (2.0).
  5. Sigmoid activation.

  Args:
    x_ref: A reference to the input tensor with shape (1, H, W, C_in).
    kernel_ref: A reference to the convolution kernel with shape (KH, KW, C_in, C_out).
    bias_ref: A reference to the bias tensor, broadcastable to the output shape.
    out_ref: A reference to the output tensor for storing the result in-place.
  """
  # Define constants based on the source computation.
  strides = (2, 2)
  # This padding configuration ensures the output spatial dimensions are 32x32.
  padding = ((1, 1), (1, 1))
  scaling_factor = 2.0
  dimension_numbers = ("NHWC", "HWIO", "NHWC")

  # Step 1: Perform the transposed convolution.
  # The inputs x_ref and kernel_ref are loaded from SRAM into registers
  # and the computation is performed.
  y = jax.lax.conv_transpose(
    x_ref[...], kernel_ref[...], strides=strides, padding=padding, dimension_numbers=dimension_numbers
  )

  # Step 2: Apply softmax activation along the last (channel) axis.
  y = jax.nn.softmax(y, axis=-1)

  # Step 3: Add the bias term. The bias is broadcast across the spatial dimensions.
  y = y + bias_ref[...]

  # Step 4: Multiply by the scaling factor.
  y = y * scaling_factor

  # Step 5: Apply the sigmoid activation function.
  y = jax.nn.sigmoid(y)

  # Step 6: Write the final result to the output buffer.
  out_ref[...] = y


x = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, 32, 32, out_channels), x.dtype),
  grid=(batch_size,),
  in_specs=[
    # Input image `x` is chunked along the batch dimension.
    pl.BlockSpec(block_shape=(1, height, width, in_channels), index_map=lambda i: (i, 0, 0, 0)),
    # The full kernel is passed to each program instance.
    pl.BlockSpec(block_shape=params["kernel"].shape, index_map=lambda _: (0, 0, 0, 0)),
    # The full bias is passed to each program instance.
    pl.BlockSpec(block_shape=bias.shape, index_map=lambda _: (0, 0, 0, 0)),
  ],
  # The output is chunked along the batch dimension.
  out_specs=pl.BlockSpec(block_shape=(1, 32, 32, out_channels), index_map=lambda i: (i, 0, 0, 0)),
)(x, params["kernel"], bias).block_until_ready()

# Imports
from functools import partial

import flax.linen as nn
import jax
import jax.random as random
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
add_value = 0.5
scale = 2

key = random.PRNGKey(0)
key_x, key_params = random.split(key)

x = random.normal(key_x, (batch_size, height, width, in_channels))

conv_transpose = nn.ConvTranspose(
  features=out_channels, kernel_size=(kernel_size, kernel_size), strides=(stride, stride), padding="SAME"
)
params = conv_transpose.init(key_params, x)["params"]

output_shape = (batch_size, height * stride, width * stride, out_channels)


# Computation
def kernel(add_value, scale, x_ref, kernel_ref, bias_ref, out_ref):
  # Perform the 2D transposed convolution.
  # The 'padding' is set to 'SAME' to match the behavior of the original Flax ConvTranspose layer,
  # which, with the given stride, doubles the input height and width.
  # The dimension numbers specify the layout of the input, kernel, and output tensors.
  # 'NHWC' for input/output and 'HWIO' for the kernel are standard for JAX.
  dn = ("NHWC", "HWIO", "NHWC")
  # The strides for the convolution are (2, 2) for the spatial dimensions.
  strides = (2, 2)

  # The conv_transpose operation is the core of the computation.
  y = jax.lax.conv_transpose(lhs=x_ref[...], rhs=kernel_ref[...], strides=strides, padding="SAME", dimension_numbers=dn)

  # Add the bias term to the result of the convolution.
  y = y + bias_ref[...]

  # Apply the Mish activation function.
  y = jax.nn.mish(y)

  # Add the constant value.
  y = y + add_value

  # Apply the hard_tanh activation function.
  # jax.nn.hard_tanh defaults to minval=-1 and maxval=1, which matches nn.hard_tanh.
  y = jax.nn.hard_tanh(y)

  # Scale the result.
  y = y * scale

  # Write the final result to the output buffer.
  out_ref[...] = y


x = pl.pallas_call(
  partial(kernel, add_value, scale),
  out_shape=jax.ShapeDtypeStruct(output_shape, x.dtype),
  grid=(batch_size,),
  in_specs=[
    pl.BlockSpec(block_shape=(1, height, width, in_channels), index_map=lambda i: (i, 0, 0, 0)),  # x
    pl.BlockSpec(
      block_shape=(kernel_size, kernel_size, in_channels, out_channels), index_map=lambda i: (0, 0, 0, 0)
    ),  # kernel
    pl.BlockSpec(block_shape=(out_channels,), index_map=lambda i: (0,)),  # bias
  ],
  out_specs=pl.BlockSpec(
    block_shape=(1, output_shape[1], output_shape[2], out_channels), index_map=lambda i: (i, 0, 0, 0)
  ),
)(x, params["kernel"], params["bias"]).block_until_ready()

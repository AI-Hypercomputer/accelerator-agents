# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
bias_shape = (1, 1, 1, 1, 1)

key = random.PRNGKey(0)
key_x, key_params, key_bias = random.split(key, 3)

# JAX uses a channels-last convention (NDHWC)
x = random.normal(key_x, (batch_size, depth, height, width, in_channels))
conv_transpose = nn.ConvTranspose(
  features=out_channels,
  kernel_size=(kernel_size, kernel_size, kernel_size),
  strides=(stride, stride, stride),
  padding=padding,
)
params = conv_transpose.init(key_params, x)["params"]
bias = random.normal(key_bias, bias_shape)


# Computation
def kernel(x_ref, out_ref, kernel_param, conv_bias_param, final_bias_param):
  # This kernel processes one element of the batch at a time.
  # The input x_ref corresponds to a single item from the input batch x.
  # kernel_param, conv_bias_param, and final_bias_param are passed as regular
  # JAX arrays, not Pallas Refs.

  # 1. Apply 3D transposed convolution.
  # We use 'SAME' padding which results in an output spatial size of
  # `input_size * stride`, matching the expected output shape.
  # The dimension numbers 'NDHWC' for input/output and 'DHWIO' for the kernel
  # match the standard JAX/Flax channel-last convention.
  strides = (2, 2, 2)
  dn = ("NDHWC", "DHWIO", "NDHWC")
  y = jax.lax.conv_transpose(x_ref[...], kernel_param, strides=strides, padding="SAME", dimension_numbers=dn)

  # Add the convolution bias.
  y = y + conv_bias_param

  # 2. Apply logsumexp along the channel axis.
  # This reduces the channel dimension from `out_channels` to 1.
  y = jax.nn.logsumexp(y, axis=-1, keepdims=True)

  # 3. Apply element-wise activation function.
  y = y * jax.nn.sigmoid(y + 3) / 6

  # 4. Subtract the final bias term.
  y = y - final_bias_param

  # 5. Clip the result to the range [-1, 1].
  y = jnp.clip(y, a_min=-1, a_max=1)

  # 6. Apply max along the channel axis.
  # Note: This is a no-op since the channel dimension is already 1 after
  # the logsumexp operation, but we include it to faithfully replicate
  # the original computation.
  y = jnp.max(y, axis=-1, keepdims=True)

  # 7. Write the final result to the output reference.
  out_ref[...] = y


pl.pallas_call(
  lambda x_ref, out_ref: kernel(x_ref, out_ref, params["kernel"], params["bias"], bias),
  out_shape=jax.ShapeDtypeStruct((batch_size, depth * stride, height * stride, width * stride, 1), x.dtype),
  grid=(batch_size,),
  in_specs=[
    pl.BlockSpec(
      block_shape=(1, depth, height, width, in_channels),
      index_map=lambda i: (i, 0, 0, 0, 0),
    )
  ],
  out_specs=pl.BlockSpec(
    block_shape=(1, depth * stride, height * stride, width * stride, 1),
    index_map=lambda i: (i, 0, 0, 0, 0),
  ),
)(x).block_until_ready()

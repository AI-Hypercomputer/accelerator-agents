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
height, width = 32, 32
kernel_size = 3
b_size = 8

key = random.PRNGKey(0)
key, x_key, params_key = random.split(key, 3)

# Note: JAX uses channels-last convention by default
x = random.normal(x_key, (batch_size, height, width, in_channels))
conv = nn.Conv(features=out_channels, kernel_size=(kernel_size, kernel_size))
params = conv.init(params_key, x)["params"]


# Computation
def kernel(x_ref, kernel_ref, bias_ref, out_ref):
  # Manual implementation of convolution using im2col (conv_general_dilated_patches) and dot.
  # 1. Pad the input for 'SAME' convolution.
  pad_width = kernel_size // 2
  padded_x = jnp.pad(
    x_ref[...],
    (
      (0, 0),
      (pad_width, pad_width),
      (pad_width, pad_width),
      (0, 0),
    ),
  )

  # 2. Extract image patches.
  dn = ("NHWC", "OIHW", "NHWC")
  patches = jax.lax.conv_general_dilated_patches(
    padded_x, filter_shape=(kernel_size, kernel_size), window_strides=(1, 1), padding="VALID", dimension_numbers=dn
  )

  # 3. Reshape kernel for matrix multiplication.
  kernel_reshaped = kernel_ref[...].reshape((-1, out_channels))

  # 4. Perform the convolution as a matrix multiplication.
  conv_out = jnp.dot(patches, kernel_reshaped)

  # Add the bias, which will be broadcasted.
  conv_with_bias = conv_out + bias_ref[...]

  # Apply the GELU activation function.
  activated_out = jax.nn.gelu(conv_with_bias)

  # Perform global average pooling by taking the mean over the spatial axes (1, 2).
  # The result will have the shape (b_size, out_channels), matching out_ref.
  final_out = jnp.mean(activated_out, axis=(1, 2))

  # Write the final result to the output buffer.
  out_ref[...] = final_out


x = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, out_channels), x.dtype),
  grid=(batch_size // b_size,),
  in_specs=[
    pl.BlockSpec(
      block_shape=(b_size, height, width, in_channels),
      index_map=lambda i: (i * b_size, 0, 0, 0),
    ),
    pl.BlockSpec(
      block_shape=(kernel_size, kernel_size, in_channels, out_channels),
      index_map=lambda i: (0, 0, 0, 0),
    ),
    pl.BlockSpec(block_shape=(out_channels,), index_map=lambda i: (0,)),
  ],
  out_specs=pl.BlockSpec(block_shape=(b_size, out_channels), index_map=lambda i: (i * b_size, 0)),
)(x, params["kernel"], params["bias"]).block_until_ready()

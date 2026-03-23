# Imports
import flax.linen as nn
import jax
import jax.lax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl
from jax.experimental.pallas.ops.tpu import conv_general_dilated as conv

# Initialization
batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = (3, 3, 3)
divisor = 2.0
pool_size = (2, 2, 2)
bias_shape = (1, 1, 1, 1, out_channels)
sum_dim = -1

key = random.PRNGKey(0)
key_x, key_conv, key_bias = random.split(key, 3)

x = random.normal(key_x, (batch_size, depth, height, width, in_channels))
conv_layer = nn.Conv(features=out_channels, kernel_size=kernel_size)
conv_params = conv_layer.init(key_conv, x)["params"]

bias = random.normal(key_bias, bias_shape)


# Computation
def kernel(x_ref, kernel_ref, conv_bias_ref, bias_ref, out_ref):
  # Define constants from the source code
  divisor = 2.0
  pool_size = (2, 2, 2)
  sum_dim = -1

  # Perform 3D convolution
  # The dimension numbers correspond to the (N, D, H, W, C) input format
  dimension_numbers = ("NDHWC", "DHWIO", "NDHWC")
  y = conv(
    lhs=x_ref[...],
    rhs=kernel_ref[...],
    window_strides=(1, 1, 1),
    padding="SAME",
    dimension_numbers=dimension_numbers,
  )

  # Add the convolution bias
  y = y + conv_bias_ref[...]

  # Divide by a constant
  y = y / divisor

  # Perform max pooling
  # The window dimensions and strides are set up to pool over the spatial
  # dimensions (D, H, W) only.
  window_dims = (1, *pool_size, 1)
  strides = (1, *pool_size, 1)
  y = jax.lax.reduce_window(y, -jnp.inf, jax.lax.max, window_dims, strides, "VALID")

  # Compute the mean over the spatial dimensions
  y = jnp.mean(y, axis=(1, 2, 3), keepdims=True)

  # Add the final bias
  y = y + bias_ref[...]

  # Sum along the final dimension
  y = jnp.sum(y, axis=sum_dim)

  # Store the final scalar result in the output buffer
  out_ref[...] = y


x = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, 1, 1, 1), x.dtype),
  grid=(batch_size,),
  in_specs=[
    pl.BlockSpec((1, depth, height, width, in_channels), lambda i: (i, 0, 0, 0, 0)),
    pl.BlockSpec(
      conv_params["kernel"].shape,
      lambda i: tuple([0] * conv_params["kernel"].ndim),
    ),
    pl.BlockSpec(conv_params["bias"].shape, lambda i: tuple([0] * conv_params["bias"].ndim)),
    pl.BlockSpec(bias.shape, lambda i: tuple([0] * bias.ndim)),
  ],
  out_specs=pl.BlockSpec((1, 1, 1, 1), lambda i: (i, 0, 0, 0)),
)(x, conv_params["kernel"], conv_params["bias"], bias).block_until_ready()

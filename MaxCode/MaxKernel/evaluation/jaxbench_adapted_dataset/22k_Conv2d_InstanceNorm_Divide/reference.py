# pylint: skip-file
# Imports
import jax
import jax.numpy as jnp


# Initialization
def get_inputs():
  batch_size = 128
  in_channels = 64
  out_channels = 128
  kernel_size = 3
  divide_by_value = 2.0

  dtype = jnp.float32
  key = jax.random.key(0)
  height = width = 128
  x = jax.random.uniform(
      key, (batch_size, in_channels, height, width), dtype=dtype
  )
  weight = jnp.zeros(
      (out_channels, in_channels, kernel_size, kernel_size), dtype=dtype
  )
  conv_bias = jnp.zeros(out_channels, dtype=dtype)
  in_weight = jnp.ones(out_channels, dtype=dtype)
  in_bias = jnp.zeros(out_channels, dtype=dtype)

  dynamic_args = [x, weight, conv_bias, in_weight, in_bias]
  static_args = [divide_by_value]
  return dynamic_args, static_args


# Computation
def computation(x, weight, conv_bias, in_weight, in_bias, divide_by_value):
  x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
  kernel = jnp.transpose(weight, (2, 3, 1, 0))
  x = jax.lax.conv_general_dilated(
      x_nhwc,
      kernel,
      window_strides=(1, 1),
      padding='VALID',
      dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
  )
  x = x + conv_bias.reshape(1, 1, 1, -1)
  x = jnp.transpose(x, (0, 3, 1, 2))

  mean = jnp.mean(x, axis=(2, 3), keepdims=True)
  var = jnp.var(x, axis=(2, 3), keepdims=True)
  x = (x - mean) / jnp.sqrt(var + 1e-5)
  x = x * in_weight.reshape(1, -1, 1, 1) + in_bias.reshape(1, -1, 1, 1)

  x = x / divide_by_value
  return x

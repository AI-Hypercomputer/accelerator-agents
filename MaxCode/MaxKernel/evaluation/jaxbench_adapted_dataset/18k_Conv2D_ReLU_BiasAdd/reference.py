# pylint: skip-file
# Imports
import jax
import jax.numpy as jnp


# Initialization
def get_inputs(dtype=jnp.float32):
  batch_size = 128
  in_channels = 64
  out_channels = 128
  kernel_size = 3
  height = width = 128

  key = jax.random.key(0)
  k1, k2 = jax.random.split(key)
  x = jax.random.uniform(
      k1, (batch_size, in_channels, height, width), dtype=dtype
  )
  weight = jnp.zeros(
      (out_channels, in_channels, kernel_size, kernel_size), dtype=dtype
  )
  conv_bias = jnp.zeros(out_channels, dtype=dtype)
  bias = jnp.zeros((out_channels, 1, 1), dtype=dtype)

  dynamic_args = [x, weight, conv_bias, bias]
  static_args = []
  return dynamic_args, static_args


# Computation
def computation(x, weight, conv_bias, bias):
  x = jnp.transpose(x, (0, 2, 3, 1))
  kernel = jnp.transpose(weight, (2, 3, 1, 0))
  x = jax.lax.conv_general_dilated(
      x,
      kernel,
      window_strides=(1, 1),
      padding='VALID',
      dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
  )
  x = x + conv_bias.reshape(1, 1, 1, -1)
  x = jax.nn.relu(x)
  x = jnp.transpose(x, (0, 3, 1, 2))
  x = x + bias
  return x

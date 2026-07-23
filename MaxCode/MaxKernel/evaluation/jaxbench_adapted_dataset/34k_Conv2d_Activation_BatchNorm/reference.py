# pylint: skip-file
# Imports
import jax
import jax.lax as lax
import jax.numpy as jnp


# Initialization
def get_inputs(dtype=jnp.float32):
  batch_size = 64
  in_channels = 64
  out_channels = 128
  kernel_size = 3
  height = 128
  width = 128

  key = jax.random.key(0)
  k1, k2, k3 = jax.random.split(key, 3)
  x = jax.random.uniform(
      k1, (batch_size, in_channels, height, width), dtype=dtype
  )
  conv_weight = jax.random.normal(
      k2, (out_channels, in_channels, kernel_size, kernel_size), dtype=dtype
  )
  conv_bias = jax.random.normal(k3, (out_channels,), dtype=dtype)
  bn_weight = jnp.ones(out_channels, dtype=dtype)
  bn_bias = jnp.zeros(out_channels, dtype=dtype)

  dynamic_args = [x, conv_weight, conv_bias, bn_weight, bn_bias]
  static_args = []
  return dynamic_args, static_args


# Computation
def computation(x, conv_weight, conv_bias, bn_weight, bn_bias):
  eps = 1e-5
  x = jnp.transpose(x, (0, 2, 3, 1))
  weight = jnp.transpose(conv_weight, (2, 3, 1, 0))
  x = lax.conv_general_dilated(
      x,
      weight,
      window_strides=(1, 1),
      padding='VALID',
      dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
  )
  x = x + conv_bias.reshape(1, 1, 1, -1)
  x = jnp.transpose(x, (0, 3, 1, 2))

  softplus_x = jax.nn.softplus(x)
  x = jnp.multiply(jnp.tanh(softplus_x), x)

  mean = jnp.mean(x, axis=(0, 2, 3), keepdims=True)
  var = jnp.mean((x - mean) ** 2, axis=(0, 2, 3), keepdims=True)
  w = bn_weight.reshape(1, -1, 1, 1)
  b = bn_bias.reshape(1, -1, 1, 1)
  x = (x - mean) / jnp.sqrt(var + eps) * w + b
  return x

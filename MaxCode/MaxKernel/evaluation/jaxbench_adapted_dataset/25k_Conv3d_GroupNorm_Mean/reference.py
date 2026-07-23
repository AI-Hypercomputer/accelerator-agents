# pylint: skip-file
# Imports
import jax
import jax.numpy as jnp


# Initialization
def get_inputs(dtype=jnp.float32):
  key = jax.random.key(0)
  batch_size = 128
  in_channels = 3
  out_channels = 24
  kernel_size = 3
  D, H, W = 24, 32, 32
  x = jax.random.uniform(key, (batch_size, in_channels, D, H, W), dtype=dtype)
  weight = jnp.zeros(
      (out_channels, in_channels, kernel_size, kernel_size, kernel_size),
      dtype=dtype,
  )
  conv_bias = jnp.zeros(out_channels, dtype=dtype)
  gamma = jnp.ones(out_channels, dtype=dtype)
  beta = jnp.zeros(out_channels, dtype=dtype)
  dynamic_args = [x, weight, conv_bias, gamma, beta]
  static_args = []
  return dynamic_args, static_args


# Computation
def computation(x, weight, conv_bias, gamma, beta):
  num_groups = 8
  x = jnp.transpose(x, (0, 2, 3, 4, 1))
  kernel = jnp.transpose(weight, (2, 3, 4, 1, 0))
  x = jax.lax.conv_general_dilated(
      x,
      kernel,
      window_strides=(1, 1, 1),
      padding='VALID',
      dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
  )
  x = x + conv_bias.reshape(1, 1, 1, 1, -1)
  x = jnp.transpose(x, (0, 4, 1, 2, 3))
  N, C, D, H, W = x.shape
  G = num_groups
  x = x.reshape(N, G, C // G, D, H, W)
  mean = jnp.mean(x, axis=(2, 3, 4, 5), keepdims=True)
  var = jnp.var(x, axis=(2, 3, 4, 5), keepdims=True)
  x = (x - mean) / jnp.sqrt(var + 1e-5)
  x = x.reshape(N, C, D, H, W)
  x = x * gamma.reshape(1, -1, 1, 1, 1) + beta.reshape(1, -1, 1, 1, 1)
  x = jnp.mean(x, axis=(1, 2, 3, 4))
  return x

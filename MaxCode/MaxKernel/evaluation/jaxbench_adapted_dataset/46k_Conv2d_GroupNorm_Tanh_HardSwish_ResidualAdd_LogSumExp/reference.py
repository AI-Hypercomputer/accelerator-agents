# pylint: skip-file
# Imports
import jax
import jax.numpy as jnp
import jax.scipy.special


# Initialization
def get_inputs(dtype=jnp.float32):
  key = jax.random.key(0)
  batch_size = 128
  in_channels = 8
  out_channels = 64
  kernel_size = 3
  height = 128
  width = 128
  x = jax.random.uniform(
      key, (batch_size, in_channels, height, width), dtype=dtype
  )
  conv_weight = jnp.zeros(
      (out_channels, in_channels, kernel_size, kernel_size), dtype=dtype
  )
  conv_bias = jnp.zeros(out_channels, dtype=dtype)
  gn_weight = jnp.ones(out_channels, dtype=dtype)
  gn_bias = jnp.zeros(out_channels, dtype=dtype)
  dynamic_args = [x, conv_weight, conv_bias, gn_weight, gn_bias]
  static_args = []
  return dynamic_args, static_args


# Computation
def computation(x, conv_weight, conv_bias, gn_weight, gn_bias):
  groups = 16
  eps = 1e-5
  x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
  kernel = jnp.transpose(conv_weight, (2, 3, 1, 0))
  x_conv = jax.lax.conv_general_dilated(
      x_nhwc,
      kernel,
      window_strides=(1, 1),
      padding='VALID',
      dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
  )
  x_conv = x_conv + conv_bias.reshape(1, 1, 1, -1)
  x_conv = jnp.transpose(x_conv, (0, 3, 1, 2))
  N, C, H, W = x_conv.shape
  x = x_conv.reshape(N, groups, C // groups, H, W)
  mean = jnp.mean(x, axis=(2, 3, 4), keepdims=True)
  var = jnp.var(x, axis=(2, 3, 4), keepdims=True)
  x = (x - mean) / jnp.sqrt(var + eps)
  x = x.reshape(N, C, H, W)
  x_norm = x * gn_weight.reshape(1, -1, 1, 1) + gn_bias.reshape(1, -1, 1, 1)
  x_tanh = jnp.tanh(x_norm)
  x_hard_swish = x_tanh * jnp.minimum(jnp.maximum(x_tanh + 3, 0), 6) / 6
  x_res = x_conv + x_hard_swish
  x_logsumexp = jax.scipy.special.logsumexp(x_res, axis=1, keepdims=True)
  return x_logsumexp

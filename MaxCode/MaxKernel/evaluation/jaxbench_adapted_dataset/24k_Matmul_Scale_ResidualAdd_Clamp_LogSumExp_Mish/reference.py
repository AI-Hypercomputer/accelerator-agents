# pylint: skip-file
# Imports
import jax
import jax.numpy as jnp


# Initialization
def get_inputs():
  batch_size = 4096
  input_size = 8192
  hidden_size = 8192
  scale_factor = 2.0
  clamp_min = -10.0
  clamp_max = 10.0
  dtype = jnp.float32

  key = jax.random.key(0)
  x = jax.random.uniform(key, (batch_size, input_size), dtype=dtype)
  weight = jnp.zeros((input_size, hidden_size), dtype=dtype)
  bias = jnp.zeros(hidden_size, dtype=dtype)

  dynamic_args = [x, weight, bias]
  static_args = [scale_factor, clamp_min, clamp_max]
  return dynamic_args, static_args


# Computation
def computation(x, weight, bias, scale_factor, clamp_min, clamp_max):
  x = jnp.matmul(x, weight.T) + bias
  x = x * scale_factor
  x = x + x
  x = jnp.clip(x, clamp_min, clamp_max)
  x = jax.scipy.special.logsumexp(x, axis=1, keepdims=True)
  softplus_x = jnp.logaddexp(x, 0.0)
  mish_x = x * jnp.tanh(softplus_x)
  x = x * mish_x
  return x

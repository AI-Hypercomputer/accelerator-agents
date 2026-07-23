# pylint: skip-file
# Imports
import jax
import jax.numpy as jnp


# Initialization
def get_inputs(dtype=jnp.float32):
  batch_size = 4096
  in_features = 8192
  out_features = 8192

  key = jax.random.key(0)
  x = jax.random.uniform(key, (batch_size, in_features), dtype=dtype)
  weight = jnp.zeros((in_features, out_features), dtype=dtype)
  bias = jnp.zeros(out_features, dtype=dtype)

  dynamic_args = [x, weight, bias]
  static_args = []
  return dynamic_args, static_args


# Computation
def computation(x, weight, bias):
  multiplier = 2.0
  negative_slope = 0.1
  x = jnp.matmul(x, weight) + bias
  x = x * multiplier
  x = jnp.where(x >= 0, x, x * negative_slope)
  return x

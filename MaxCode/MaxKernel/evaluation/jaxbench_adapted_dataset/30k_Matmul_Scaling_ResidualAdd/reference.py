# pylint: skip-file
# Imports
import jax
import jax.numpy as jnp


# Initialization
def get_inputs(dtype=jnp.float32):
  batch_size = 16384
  in_features = 4096
  out_features = 4096
  key = jax.random.key(0)
  x = jax.random.uniform(key, (batch_size, in_features), dtype=dtype)
  weight = jnp.zeros((in_features, out_features), dtype=dtype)
  bias = jnp.zeros(out_features, dtype=dtype)
  dynamic_args = [x, weight, bias]
  static_args = []
  return dynamic_args, static_args


# Computation
def computation(x, weight, bias):
  scaling_factor = 0.5
  x = jnp.matmul(x, weight) + bias
  original_x = x
  x = x * scaling_factor
  x = x + original_x
  return x

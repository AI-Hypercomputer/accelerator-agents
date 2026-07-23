# pylint: skip-file
# Imports
import jax
import jax.numpy as jnp


# Initialization
def get_inputs():
  config = {
      'name': '80_Gemm_Max_Subtract_GELU',
      'batch_size': 4096,
      'in_features': 8192,
      'out_features': 8192,
      'max_dim': 1,
  }
  dtype = jnp.float32
  key = jax.random.key(0)
  x = jax.random.uniform(
      key, (config['batch_size'], config['in_features']), dtype=dtype
  )
  weight = jnp.zeros(
      (config['in_features'], config['out_features']), dtype=dtype
  )
  bias = jnp.zeros(config['out_features'], dtype=dtype)
  dynamic_args = [x, weight, bias]
  static_args = []
  return dynamic_args, static_args


# Computation
def computation(x, weight, bias):
  x = jnp.matmul(x, weight) + bias
  x = jnp.max(x, axis=1, keepdims=True)
  x = x - jnp.mean(x, axis=1, keepdims=True)
  x = jax.nn.gelu(x)
  return x

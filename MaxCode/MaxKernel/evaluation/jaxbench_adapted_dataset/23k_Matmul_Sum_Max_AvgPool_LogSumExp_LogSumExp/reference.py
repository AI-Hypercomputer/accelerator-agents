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
  x = jnp.matmul(x, weight.T) + bias
  x = jnp.sum(x, axis=1, keepdims=True)
  x = jnp.max(x, axis=1, keepdims=True)
  x = jnp.mean(x, axis=1, keepdims=True)
  x = jax.scipy.special.logsumexp(x, axis=1, keepdims=True)
  x = jax.scipy.special.logsumexp(x, axis=1, keepdims=True)
  return x

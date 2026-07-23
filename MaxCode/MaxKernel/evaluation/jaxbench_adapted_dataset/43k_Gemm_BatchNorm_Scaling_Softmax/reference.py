# pylint: skip-file
# Imports
import jax
import jax.numpy as jnp


# Initialization
def get_inputs():
  batch_size = 4096
  in_features = 8192
  out_features = 8192
  dtype = jnp.float32

  key = jax.random.key(0)
  x = jax.random.uniform(key, (batch_size, in_features), dtype=dtype)
  weight = jnp.zeros((in_features, out_features), dtype=dtype)
  bias = jnp.zeros(out_features, dtype=dtype)
  bn_scale = jnp.ones(out_features, dtype=dtype)
  bn_bias = jnp.zeros(out_features, dtype=dtype)
  bn_mean = jnp.zeros(out_features, dtype=dtype)
  bn_var = jnp.ones(out_features, dtype=dtype)
  scale = jnp.ones((1,), dtype=dtype)

  dynamic_args = [x, weight, bias, bn_scale, bn_bias, bn_mean, bn_var, scale]
  static_args = []
  return dynamic_args, static_args


# Computation
def computation(x, weight, bias, bn_scale, bn_bias, bn_mean, bn_var, scale):
  bn_eps = 1e-5
  x = jnp.matmul(x, weight) + bias
  x_normalized = (x - bn_mean) / jnp.sqrt(bn_var + bn_eps)
  x = bn_scale * x_normalized + bn_bias
  x = scale * x
  x = jax.nn.softmax(x, axis=1)
  return x

# pylint: skip-file
# Imports
import jax
import jax.numpy as jnp


# Initialization
def get_inputs(dtype=jnp.float32):
  key = jax.random.key(0)
  batch_size = 8192
  in_features = 4096
  out_features = 4096
  num_groups = 64
  x = jax.random.uniform(key, (batch_size, in_features), dtype=dtype)
  weight = jnp.zeros((in_features, out_features), dtype=dtype)
  bias = jnp.zeros(out_features, dtype=dtype)
  gn_weight = jnp.ones(out_features, dtype=dtype)
  gn_bias = jnp.zeros(out_features, dtype=dtype)
  dynamic_args = [x, weight, bias, gn_weight, gn_bias]
  static_args = [num_groups, out_features]
  return dynamic_args, static_args


# Computation
def computation(x, weight, bias, gn_weight, gn_bias, num_groups, out_features):
  x = jnp.matmul(x, weight)
  x = jax.nn.sigmoid(x) * x
  x = x + bias
  group_size = out_features // num_groups
  x = x.reshape(-1, num_groups, group_size)
  mean = jnp.mean(x, axis=-1, keepdims=True)
  var = jnp.var(x, axis=-1, keepdims=True)
  x = (x - mean) / jnp.sqrt(var + 1e-5)
  x = x.reshape(-1, out_features)
  x = x * gn_weight + gn_bias
  return x

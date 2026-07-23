# pylint: skip-file
# Imports
import jax
import jax.numpy as jnp


# Initialization
def get_inputs(dtype=jnp.float32):
  batch_size = 16384
  in_features = 8192
  out_features = 8192
  key = jax.random.key(0)
  k1, k2, k3 = jax.random.split(key, 3)
  x = jax.random.uniform(k1, (batch_size, in_features), dtype=dtype)
  gemm_weight = jax.random.normal(k2, (out_features, in_features), dtype=dtype)
  gemm_bias = jax.random.normal(k3, (out_features,), dtype=dtype)
  bn_weight = jnp.ones(out_features, dtype=dtype)
  bn_bias = jnp.zeros(out_features, dtype=dtype)
  dynamic_args = [x, gemm_weight, gemm_bias, bn_weight, bn_bias]
  static_args = []
  return dynamic_args, static_args


# Computation
def computation(x, gemm_weight, gemm_bias, bn_weight, bn_bias):
  eps = 1e-5
  x = jnp.matmul(x, gemm_weight.T) + gemm_bias
  mean = jnp.mean(x, axis=0, keepdims=True)
  var = jnp.mean((x - mean) ** 2, axis=0, keepdims=True)
  x = (x - mean) / jnp.sqrt(var + eps) * bn_weight + bn_bias
  x = jax.nn.gelu(x)
  x = jax.nn.relu(x)
  return x

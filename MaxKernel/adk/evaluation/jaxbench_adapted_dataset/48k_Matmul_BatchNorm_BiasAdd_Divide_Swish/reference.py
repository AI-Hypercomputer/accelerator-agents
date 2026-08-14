# Imports
import jax
import jax.numpy as jnp

# Initialization
def get_inputs(dtype=jnp.float32):
    batch_size = 4096
    in_features = 8192
    out_features = 8192
    key = jax.random.key(0)
    rand_key = jax.random.key(0xBADC0DE)
    ka, kb, kc = jax.random.split(rand_key, 3)
    x = jax.random.uniform(key, (batch_size, in_features), dtype=dtype)
    weight = jax.random.normal(ka, (in_features, out_features), dtype=dtype) * 0.02
    linear_bias = jax.random.normal(kb, out_features, dtype=dtype) * 0.02
    bn_scale = jnp.ones(out_features, dtype=dtype)
    bn_bias = jnp.zeros(out_features, dtype=dtype)
    bn_mean = jnp.zeros(out_features, dtype=dtype)
    bn_var = jnp.ones(out_features, dtype=dtype)
    bias = jax.random.normal(kc, (1,), dtype=dtype) * 0.02
    dynamic_args = [x, weight, linear_bias, bn_scale, bn_bias, bn_mean, bn_var, bias]
    static_args = []
    return dynamic_args, static_args

# Computation
def computation(x, weight, linear_bias, bn_scale, bn_bias, bn_mean, bn_var, bias):
    bn_eps = 1e-5
    divide_value = 1.0
    x = jnp.matmul(x, weight) + linear_bias
    x_normalized = (x - bn_mean) / jnp.sqrt(bn_var + bn_eps)
    x = bn_scale * x_normalized + bn_bias
    x = x + bias
    x = x / divide_value
    x = x * jax.nn.sigmoid(x)
    return x
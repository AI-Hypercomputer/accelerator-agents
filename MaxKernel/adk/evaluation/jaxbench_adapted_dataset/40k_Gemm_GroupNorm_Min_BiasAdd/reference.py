# Imports
import jax
import jax.numpy as jnp

# Initialization
def get_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    rand_key = jax.random.key(0xBADC0DE)
    ka, kb, kc = jax.random.split(rand_key, 3)
    batch_size, in_features, out_features, num_groups = 4096, 8192, 8192, 512
    x = jax.random.uniform(key, (batch_size, in_features), dtype=dtype)
    weight = jax.random.normal(ka, (out_features, in_features), dtype=dtype) * 0.02
    linear_bias = jax.random.normal(kb, out_features, dtype=dtype) * 0.02
    gn_weight = jnp.ones(out_features, dtype=dtype)
    gn_bias = jnp.zeros(out_features, dtype=dtype)
    bias = jax.random.normal(kc, (1, out_features, 1, 1), dtype=dtype) * 0.02
    dynamic_args = [x, weight, linear_bias, gn_weight, gn_bias, bias]
    static_args = []
    return dynamic_args, static_args

# Computation
def computation(x, weight, linear_bias, gn_weight, gn_bias, bias):
    num_groups = 512
    eps = 1e-5
    x = jnp.matmul(x, weight.T) + linear_bias
    N, C = x.shape
    G = num_groups
    x = x.reshape(N, G, C // G)
    mean = jnp.mean(x, axis=2, keepdims=True)
    var = jnp.var(x, axis=2, keepdims=True)
    x = (x - mean) / jnp.sqrt(var + eps)
    x = x.reshape(N, C)
    x = x * gn_weight + gn_bias
    x = jnp.min(x, axis=1, keepdims=True)
    x = x.reshape(1, 1, N, 1)
    x = x + bias
    return x
# Imports
import jax
import jax.numpy as jnp

# Initialization
def get_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k1, k2 = jax.random.split(key)
    batch_size, in_features, out_features = 4096, 8192, 8192
    x = jax.random.uniform(k1, (batch_size, in_features), dtype=dtype)
    y = jax.random.uniform(k2, (batch_size, out_features), dtype=dtype)
    bmm_weight = jnp.zeros((out_features, in_features), dtype=dtype)
    bmm_bias = jnp.zeros(out_features, dtype=dtype)
    in_weight = jnp.ones(out_features, dtype=dtype)
    in_bias = jnp.zeros(out_features, dtype=dtype)

    dynamic_args = [x, y, bmm_weight, bmm_bias, in_weight, in_bias]
    static_args = []
    return dynamic_args, static_args

# Computation
def computation(x, y, bmm_weight, bmm_bias, in_weight, in_bias):
    eps = 1e-5
    x = x @ bmm_weight.T + bmm_bias
    x = jnp.expand_dims(jnp.expand_dims(x, 2), 3)
    mean = jnp.mean(x, axis=(2, 3), keepdims=True)
    var = jnp.var(x, axis=(2, 3), keepdims=True)
    x = (x - mean) / jnp.sqrt(var + eps)
    x = x * jnp.reshape(in_weight, (1, -1, 1, 1)) + jnp.reshape(in_bias, (1, -1, 1, 1))
    x = jnp.squeeze(jnp.squeeze(x, axis=3), axis=2)
    x = x + y
    x = x * y
    return x
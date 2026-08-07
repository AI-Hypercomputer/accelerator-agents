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
    rand_key = jax.random.key(42)
    ka, kb = jax.random.split(rand_key, 2)
    x = jax.random.uniform(key, (batch_size, in_features), dtype=dtype)
    weight = jax.random.normal(ka, (out_features, in_features), dtype=dtype) * 0.02
    bias = jax.random.normal(kb, out_features, dtype=dtype) * 0.02

    dynamic_args = [x, weight, bias]
    static_args = []

    return dynamic_args, static_args

# Computation
def computation(x, weight, bias):
    x = x @ weight.T + bias
    x = jax.nn.softmax(x, axis=1)
    return x
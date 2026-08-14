# Imports
import jax
import jax.numpy as jnp

# Initialization
def get_inputs():
    batch_size = 4096
    input_size = 8192
    hidden_size = 8192
    dtype = jnp.float32

    key = jax.random.key(0)
    rand_key = jax.random.key(42)
    ka, kb = jax.random.split(rand_key, 2)
    x = jax.random.uniform(key, (batch_size, input_size), dtype=dtype)
    weight = jax.random.normal(ka, (input_size, hidden_size), dtype=dtype) * 0.02
    bias = jax.random.normal(kb, hidden_size, dtype=dtype) * 0.02

    dynamic_args = [x, weight, bias]
    static_args = []
    return dynamic_args, static_args

# Computation
def computation(x, weight, bias):
    x = jnp.matmul(x, weight) + bias
    x = jax.nn.sigmoid(x)
    x = jnp.sum(x, axis=1, keepdims=True)
    return x
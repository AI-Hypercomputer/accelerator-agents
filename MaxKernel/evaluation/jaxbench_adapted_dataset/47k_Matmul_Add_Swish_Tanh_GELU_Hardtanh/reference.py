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
    add_value = jnp.zeros(out_features, dtype=dtype)

    dynamic_args = [x, weight, bias, add_value]
    static_args = []
    return dynamic_args, static_args

# Computation
def computation(x, weight, bias, add_value):
    x = x @ weight + bias
    x = x + add_value
    x = jax.nn.swish(x)
    x = jnp.tanh(x)
    x = jax.nn.gelu(x)
    x = jnp.clip(x, -1.0, 1.0)
    return x
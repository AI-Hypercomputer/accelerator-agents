# Imports
import jax
import jax.numpy as jnp

# Initialization
def get_inputs():
    config = {
        'name': '80_Gemm_Max_Subtract_GELU',
        'batch_size': 4096,
        'in_features': 8192,
        'out_features': 8192,
        'max_dim': 1,
    }
    dtype = jnp.float32
    key = jax.random.key(0)
    k_x, k_w, k_b = jax.random.split(key, 3)
    x = jax.random.uniform(k_x, (config['batch_size'], config['in_features']), dtype=dtype)
    weight = jax.random.normal(k_w, (config['in_features'], config['out_features']), dtype=dtype) * 0.02
    bias = jax.random.normal(k_b, config['out_features'], dtype=dtype) * 0.02
    dynamic_args = [x, weight, bias]
    static_args = []
    return dynamic_args, static_args

# Computation
def computation(x, weight, bias):
    x = jnp.matmul(x, weight) + bias
    x_max = jnp.max(x, axis=1, keepdims=True)
    x = x - x_max
    x = jax.nn.gelu(x)
    return x
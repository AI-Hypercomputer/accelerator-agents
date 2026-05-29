# Imports
import jax
import jax.numpy as jnp

# Initialization
def get_inputs(dtype=jnp.float32):
    CONFIG = {
        'name': '53_Gemm_Scaling_Hardtanh_GELU',
        'batch_size': 4096,
        'in_features': 8192,
        'out_features': 8192,
        'scaling_factor': 0.5,
        'hardtanh_min': -2,
        'hardtanh_max': 2,
    }

    batch_size = CONFIG['batch_size']
    in_features = CONFIG['in_features']
    out_features = CONFIG['out_features']
    scaling_factor = CONFIG['scaling_factor']
    hardtanh_min = CONFIG['hardtanh_min']
    hardtanh_max = CONFIG['hardtanh_max']

    key = jax.random.key(0)
    x = jax.random.uniform(key, (batch_size, in_features), dtype=dtype)
    weight = jnp.zeros((in_features, out_features), dtype=dtype)
    bias = jnp.zeros(out_features, dtype=dtype)

    dynamic_args = [x, weight, bias]
    static_args = [scaling_factor, hardtanh_min, hardtanh_max]

    return dynamic_args, static_args

# Computation
def computation(x, weight, bias, scaling_factor, hardtanh_min, hardtanh_max):
    x = jnp.matmul(x, weight) + bias
    x = x * scaling_factor
    x = jnp.clip(x, hardtanh_min, hardtanh_max)
    x = x * 0.5 * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * x**3)))
    return x
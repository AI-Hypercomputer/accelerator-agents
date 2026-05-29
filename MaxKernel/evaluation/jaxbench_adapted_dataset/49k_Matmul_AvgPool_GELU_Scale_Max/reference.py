# Imports
import jax
import jax.numpy as jnp

# Initialization
def get_inputs():
    batch_size = 4096
    in_features = 8192
    out_features = 8192
    pool_kernel_size = 16
    scale_factor = 2.0
    dtype = jnp.float32

    key = jax.random.key(0)
    x = jax.random.uniform(key, (batch_size, in_features), dtype=dtype)
    weight = jnp.zeros((in_features, out_features), dtype=dtype)
    bias = jnp.zeros(out_features, dtype=dtype)

    dynamic_args = [x, weight, bias]
    static_args = [pool_kernel_size, scale_factor]
    return dynamic_args, static_args

# Computation
def computation(x, weight, bias, pool_kernel_size, scale_factor):
    x = jnp.matmul(x, weight) + bias
    x = jnp.expand_dims(x, axis=1)
    x = jax.lax.reduce_window(
        x,
        init_value=0.0,
        computation=jax.lax.add,
        window_dimensions=(1, 1, pool_kernel_size),
        window_strides=(1, 1, pool_kernel_size),
        padding='VALID'
    ) / pool_kernel_size
    x = jnp.squeeze(x, axis=1)
    x = jax.nn.gelu(x)
    x = x * scale_factor
    x = jnp.max(x, axis=1)
    return x
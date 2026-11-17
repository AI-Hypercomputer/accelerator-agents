# Imports
import jax
import jax.numpy as jnp

# Initialization
def get_inputs():
    key = jax.random.key(0)
    rand_key = jax.random.key(0xBADC0DE)
    ka, kb = jax.random.split(rand_key, 2)
    batch_size, in_channels, out_channels, kernel_size = 128, 8, 64, 3
    height, width = 256, 256
    x = jax.random.uniform(key, (batch_size, in_channels, height, width), dtype=jnp.float32)
    weight = jax.random.normal(ka, (out_channels, in_channels, kernel_size, kernel_size), dtype=jnp.float32) * 0.02
    bias = jax.random.normal(kb, out_channels, dtype=jnp.float32) * 0.02
    dynamic_args = [x, weight, bias]
    static_args = []
    return dynamic_args, static_args

# Computation
def computation(x, weight, bias):
    x = jnp.transpose(x, (0, 2, 3, 1))
    kernel = jnp.transpose(weight, (2, 3, 1, 0))
    x = jax.lax.conv_general_dilated(
        x, kernel,
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )
    x = x + bias.reshape(1, 1, 1, -1)
    x = jax.nn.gelu(x)
    x = jnp.mean(x, axis=(1, 2))
    return x
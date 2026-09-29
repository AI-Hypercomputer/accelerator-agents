# Imports
import jax
import jax.numpy as jnp

# Initialization
def get_inputs(dtype=jnp.float32):
    batch_size = 128
    in_channels = 64
    out_channels = 128
    kernel_size = 3
    height = width = 128

    key = jax.random.key(0)
    rand_key = jax.random.key(0xBADC0DE)
    ka, kb, kc = jax.random.split(rand_key, 3)
    k1, k2 = jax.random.split(key)
    x = jax.random.uniform(k1, (batch_size, in_channels, height, width), dtype=dtype)
    weight = jax.random.normal(ka, (out_channels, in_channels, kernel_size, kernel_size), dtype=dtype) * 0.02
    conv_bias = jax.random.normal(kb, out_channels, dtype=dtype) * 0.02
    bias = jax.random.normal(kc, (out_channels, 1, 1), dtype=dtype) * 0.02

    dynamic_args = [x, weight, conv_bias, bias]
    static_args = []
    return dynamic_args, static_args

# Computation
def computation(x, weight, conv_bias, bias):
    x = jnp.transpose(x, (0, 2, 3, 1))
    kernel = jnp.transpose(weight, (2, 3, 1, 0))
    x = jax.lax.conv_general_dilated(
        x, kernel,
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )
    x = x + conv_bias.reshape(1, 1, 1, -1)
    x = jax.nn.relu(x)
    x = jnp.transpose(x, (0, 3, 1, 2))
    x = x + bias
    return x
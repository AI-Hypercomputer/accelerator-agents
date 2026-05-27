# Imports
import jax
import jax.numpy as jnp

# Initialization
def get_inputs(dtype=jnp.float32):
    batch_size = 16
    in_channels = 32
    out_channels = 64
    kernel_size = 3
    D, H, W = 32, 64, 64

    key = jax.random.key(0)
    x = jax.random.uniform(key, (batch_size, in_channels, D, H, W), dtype=dtype)
    weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size, kernel_size), dtype=dtype)
    bias = jnp.zeros(out_channels, dtype=dtype)

    dynamic_args = [x, weight, bias]
    static_args = []
    return dynamic_args, static_args

# Computation
def computation(x, weight, bias):
    x = jnp.transpose(x, (0, 2, 3, 4, 1))
    kernel = jnp.transpose(weight, (2, 3, 4, 1, 0))
    x = jax.lax.conv_general_dilated(
        x, kernel,
        window_strides=(1, 1, 1),
        padding=((0, 0), (0, 0), (0, 0)),
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC')
    )
    x = x + bias.reshape(1, 1, 1, 1, -1)
    x = x * jnp.tanh(jnp.log(1 + jnp.exp(x)))
    x = jnp.tanh(x)
    x = jnp.transpose(x, (0, 4, 1, 2, 3))
    return x
# Imports
import jax
import jax.numpy as jnp
import jax.lax as lax

# Initialization
def get_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k1, k2 = jax.random.split(key)
    batch_size = 32
    in_channels = 32
    out_channels = 64
    kernel_size_val = 4
    D, H, W = 16, 32, 32

    x = jax.random.uniform(k1, (batch_size, in_channels, D, H, W), dtype=dtype)
    conv_weight = jax.random.normal(k2, (in_channels, out_channels, kernel_size_val, kernel_size_val, kernel_size_val), dtype=dtype)
    conv_bias = jnp.zeros(out_channels, dtype=dtype)
    ln_weight = jnp.ones(out_channels, dtype=dtype)
    ln_bias = jnp.zeros(out_channels, dtype=dtype)

    stride_val = 2
    padding_val = 1
    eps_val = 1e-5
    scaling_factor_val = 1.0

    dynamic_args = [x, conv_weight, conv_bias, ln_weight, ln_bias]
    static_args = [stride_val, padding_val, kernel_size_val, eps_val, scaling_factor_val]

    return dynamic_args, static_args

# Computation
def computation(x, conv_weight, conv_bias, ln_weight, ln_bias, stride, padding, kernel_size, eps, scaling_factor):
    x = jnp.transpose(x, (0, 2, 3, 4, 1))
    kernel = jnp.transpose(conv_weight, (2, 3, 4, 1, 0))
    kernel = jnp.flip(kernel, axis=(0, 1, 2))

    batch_size, d_in, h_in, w_in, channels = x.shape
    k = kernel_size

    d_dilated = d_in + (d_in - 1) * (stride - 1)
    h_dilated = h_in + (h_in - 1) * (stride - 1)
    w_dilated = w_in + (w_in - 1) * (stride - 1)
    x_dilated = jnp.zeros((batch_size, d_dilated, h_dilated, w_dilated, channels), dtype=x.dtype)
    x_dilated = x_dilated.at[:, ::stride, ::stride, ::stride, :].set(x)
    x = x_dilated

    pad = k - 1 - padding
    jax_padding = ((pad, pad), (pad, pad), (pad, pad))

    x = lax.conv_general_dilated(
        x, kernel,
        window_strides=(1, 1, 1),
        padding=jax_padding,
        dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC')
    )
    x = x + conv_bias.reshape(1, 1, 1, 1, -1)

    x = jnp.transpose(x, (0, 4, 1, 2, 3))

    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    x = (x - mean) / jnp.sqrt(var + eps)
    x = x * ln_weight + ln_bias

    x = jax.nn.gelu(x)
    x = x * scaling_factor
    return x
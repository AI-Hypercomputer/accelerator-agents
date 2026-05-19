"""34_ConvTranspose3d_LayerNorm_GELU_Scaling — JAXBench fused operator workload."""
import jax
import jax.numpy as jnp
import jax.lax as lax

CONFIG = {
    'name': '34_ConvTranspose3d_LayerNorm_GELU_Scaling',
    'batch_size': 32,
    'in_channels': 32,
    'out_channels': 64,
    'kernel_size': 4,
    'stride': 2,
    'padding': 1,
    'bias': True,
    'eps': 1e-05,
    'scaling_factor': 1.0,
}


def create_inputs(dtype=jnp.float32):
    """Create all inputs including weights."""
    key = jax.random.key(0)
    k1, k2 = jax.random.split(key)
    batch_size, in_channels, out_channels, kernel_size = 32, 32, 64, 4
    D, H, W = 16, 32, 32
    x = jax.random.uniform(k1, (batch_size, in_channels, D, H, W), dtype=dtype)
    conv_weight = jax.random.normal(k2, (in_channels, out_channels, kernel_size, kernel_size, kernel_size), dtype=dtype)
    conv_bias = jnp.zeros(out_channels, dtype=dtype)
    ln_weight = jnp.ones(out_channels, dtype=dtype)
    ln_bias = jnp.zeros(out_channels, dtype=dtype)
    return x, conv_weight, conv_bias, ln_weight, ln_bias


def workload(x, conv_weight, conv_bias, ln_weight, ln_bias):
    """ConvTranspose3d + LayerNorm + GELU + Scaling."""
    stride = 2
    padding = 1
    kernel_size = 4
    eps = 1e-5
    scaling_factor = 1.0

    # NCDHW -> NDHWC
    x = jnp.transpose(x, (0, 2, 3, 4, 1))
    kernel = jnp.transpose(conv_weight, (2, 3, 4, 1, 0))
    kernel = jnp.flip(kernel, axis=(0, 1, 2))

    batch_size, d_in, h_in, w_in, channels = x.shape
    k = kernel_size

    # Dilate input for transposed conv
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

    # NDHWC -> NCDHW
    x = jnp.transpose(x, (0, 4, 1, 2, 3))

    # LayerNorm over last dimension
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    x = (x - mean) / jnp.sqrt(var + eps)
    x = x * ln_weight + ln_bias

    # GELU + Scaling
    x = jax.nn.gelu(x)
    x = x * scaling_factor
    return x

def benchmark(num_warmup=5, num_iters=100):
    """Benchmark and return results dict."""
    import time
    inputs = create_inputs()
    fn = jax.jit(workload)
    for _ in range(num_warmup):
        out = fn(*inputs)
        if hasattr(out, 'block_until_ready'):
            out.block_until_ready()
    times = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        out = fn(*inputs)
        if hasattr(out, 'block_until_ready'):
            out.block_until_ready()
        times.append(time.perf_counter() - t0)
    import numpy as np
    times_ms = np.array(times) * 1000
    avg = float(np.mean(times_ms))
    return {
        'name': CONFIG['name'],
        'config': {k: v for k, v in CONFIG.items() if k != 'name'},
        'time_ms': round(avg, 4),
        'std_ms': round(float(np.std(times_ms)), 4),
        'output_shape': list(out.shape) if hasattr(out, 'shape') else [],
        'status': 'success',
    }


if __name__ == '__main__':
    import json
    print(json.dumps(benchmark()))

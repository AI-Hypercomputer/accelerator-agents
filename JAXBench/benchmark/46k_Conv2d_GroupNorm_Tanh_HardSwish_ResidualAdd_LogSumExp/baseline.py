"""92_Conv2d_GroupNorm_Tanh_HardSwish_ResidualAdd_LogSumExp — JAXBench fused operator workload."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '92_Conv2d_GroupNorm_Tanh_HardSwish_ResidualAdd_LogSumExp',
    'batch_size': 128,
    'in_channels': 8,
    'out_channels': 64,
    'kernel_size': 3,
    'groups': 16,
}


def create_inputs(dtype=jnp.float32):
    """Create all inputs including weights."""
    key = jax.random.key(0)
    batch_size, in_channels, out_channels, kernel_size, groups = 128, 8, 64, 3, 16
    height, width = 128, 128
    x = jax.random.uniform(key, (batch_size, in_channels, height, width), dtype=dtype)
    conv_weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size), dtype=dtype)
    conv_bias = jnp.zeros(out_channels, dtype=dtype)
    gn_weight = jnp.ones(out_channels, dtype=dtype)
    gn_bias = jnp.zeros(out_channels, dtype=dtype)
    return x, conv_weight, conv_bias, gn_weight, gn_bias


def workload(x, conv_weight, conv_bias, gn_weight, gn_bias):
    """Conv2d + GroupNorm + Tanh + HardSwish + ResidualAdd + LogSumExp."""
    groups = 16
    eps = 1e-5
    # Conv2d
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    kernel = jnp.transpose(conv_weight, (2, 3, 1, 0))
    x_conv = jax.lax.conv_general_dilated(
        x_nhwc, kernel,
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    x_conv = x_conv + conv_bias.reshape(1, 1, 1, -1)
    x_conv = jnp.transpose(x_conv, (0, 3, 1, 2))  # NHWC -> NCHW

    # GroupNorm
    N, C, H, W = x_conv.shape
    x = x_conv.reshape(N, groups, C // groups, H, W)
    mean = jnp.mean(x, axis=(2, 3, 4), keepdims=True)
    var = jnp.var(x, axis=(2, 3, 4), keepdims=True)
    x = (x - mean) / jnp.sqrt(var + eps)
    x = x.reshape(N, C, H, W)
    x_norm = x * gn_weight.reshape(1, -1, 1, 1) + gn_bias.reshape(1, -1, 1, 1)

    # Tanh
    x_tanh = jnp.tanh(x_norm)
    # HardSwish
    x_hard_swish = x_tanh * jnp.minimum(jnp.maximum(x_tanh + 3, 0), 6) / 6
    # Residual Add
    x_res = x_conv + x_hard_swish
    # LogSumExp
    x_logsumexp = jax.scipy.special.logsumexp(x_res, axis=1, keepdims=True)
    return x_logsumexp

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

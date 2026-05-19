"""23_Conv3d_GroupNorm_Mean — JAXBench fused operator workload."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '23_Conv3d_GroupNorm_Mean',
    'batch_size': 128,
    'in_channels': 3,
    'out_channels': 24,
    'kernel_size': 3,
    'num_groups': 8,
}


def create_inputs(dtype=jnp.float32):
    """Create all inputs including weights."""
    key = jax.random.key(0)
    batch_size, in_channels, out_channels, kernel_size, num_groups = 128, 3, 24, 3, 8
    D, H, W = 24, 32, 32
    x = jax.random.uniform(key, (batch_size, in_channels, D, H, W), dtype=dtype)
    weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size, kernel_size), dtype=dtype)
    conv_bias = jnp.zeros(out_channels, dtype=dtype)
    gamma = jnp.ones(out_channels, dtype=dtype)
    beta = jnp.zeros(out_channels, dtype=dtype)
    return x, weight, conv_bias, gamma, beta


def workload(x, weight, conv_bias, gamma, beta):
    """Conv3d + GroupNorm + Mean."""
    num_groups = 8
    # Conv3d: NCDHW -> NDHWC
    x = jnp.transpose(x, (0, 2, 3, 4, 1))
    kernel = jnp.transpose(weight, (2, 3, 4, 1, 0))
    x = jax.lax.conv_general_dilated(
        x, kernel,
        window_strides=(1, 1, 1),
        padding='VALID',
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC')
    )
    x = x + conv_bias.reshape(1, 1, 1, 1, -1)
    x = jnp.transpose(x, (0, 4, 1, 2, 3))  # NDHWC -> NCDHW

    # Group Normalization
    N, C, D, H, W = x.shape
    G = num_groups
    x = x.reshape(N, G, C // G, D, H, W)
    mean = jnp.mean(x, axis=(2, 3, 4, 5), keepdims=True)
    var = jnp.var(x, axis=(2, 3, 4, 5), keepdims=True)
    x = (x - mean) / jnp.sqrt(var + 1e-5)
    x = x.reshape(N, C, D, H, W)
    x = x * gamma.reshape(1, -1, 1, 1, 1) + beta.reshape(1, -1, 1, 1, 1)

    # Mean across all dims except batch
    x = jnp.mean(x, axis=(1, 2, 3, 4))
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

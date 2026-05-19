"""47_Conv3d_Mish_Tanh — JAXBench fused operator workload."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '47_Conv3d_Mish_Tanh',
    'batch_size': 16,
    'in_channels': 32,
    'out_channels': 64,
    'kernel_size': 3,
}


def create_inputs(dtype=jnp.float32):
    """Create all inputs including weights."""
    key = jax.random.key(0)
    batch_size, in_channels, out_channels, kernel_size = 16, 32, 64, 3
    D, H, W = 32, 64, 64
    x = jax.random.uniform(key, (batch_size, in_channels, D, H, W), dtype=dtype)
    weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size, kernel_size), dtype=dtype)
    bias = jnp.zeros(out_channels, dtype=dtype)
    return x, weight, bias


def workload(x, weight, bias):
    """Conv3d + Mish + Tanh."""
    # NCDHW -> NDHWC
    x = jnp.transpose(x, (0, 2, 3, 4, 1))
    kernel = jnp.transpose(weight, (2, 3, 4, 1, 0))
    x = jax.lax.conv_general_dilated(
        x, kernel,
        window_strides=(1, 1, 1),
        padding=((0, 0), (0, 0), (0, 0)),
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC')
    )
    x = x + bias.reshape(1, 1, 1, 1, -1)
    # Mish
    x = x * jnp.tanh(jnp.log(1 + jnp.exp(x)))
    # Tanh
    x = jnp.tanh(x)
    # NDHWC -> NCDHW
    x = jnp.transpose(x, (0, 4, 1, 2, 3))
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

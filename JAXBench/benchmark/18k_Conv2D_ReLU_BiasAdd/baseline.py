"""1_Conv2D_ReLU_BiasAdd — JAXBench fused operator workload."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '1_Conv2D_ReLU_BiasAdd',
    'batch_size': 128,
    'in_channels': 64,
    'out_channels': 128,
    'kernel_size': 3,
}


def create_inputs(dtype=jnp.float32):
    """Create all inputs including weights."""
    key = jax.random.key(0)
    k1, k2 = jax.random.split(key)
    batch_size, in_channels, out_channels, kernel_size = 128, 64, 128, 3
    height = width = 128
    x = jax.random.uniform(k1, (batch_size, in_channels, height, width), dtype=dtype)
    weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size), dtype=dtype)
    conv_bias = jnp.zeros(out_channels, dtype=dtype)
    bias = jnp.zeros((out_channels, 1, 1), dtype=dtype)
    return x, weight, conv_bias, bias


def workload(x, weight, conv_bias, bias):
    """Conv2D + ReLU + BiasAdd."""
    # NCHW -> NHWC
    x = jnp.transpose(x, (0, 2, 3, 1))
    kernel = jnp.transpose(weight, (2, 3, 1, 0))  # OIHW -> HWIO
    x = jax.lax.conv_general_dilated(
        x, kernel,
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )
    x = x + conv_bias.reshape(1, 1, 1, -1)
    x = jax.nn.relu(x)
    x = jnp.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
    x = x + bias
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

"""40_Matmul_Scaling_ResidualAdd — JAXBench fused operator workload."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '40_Matmul_Scaling_ResidualAdd',
    'batch_size': 16384,
    'in_features': 4096,
    'out_features': 4096,
    'scaling_factor': 0.5,
}


def create_inputs(dtype=jnp.float32):
    """Create all inputs including weights."""
    key = jax.random.key(0)
    x = jax.random.uniform(key, (16384, 4096), dtype=dtype)
    weight = jnp.zeros((4096, 4096), dtype=dtype)
    bias = jnp.zeros(4096, dtype=dtype)
    return x, weight, bias


def workload(x, weight, bias):
    """Matmul + Scaling + ResidualAdd."""
    x = jnp.matmul(x, weight) + bias
    original_x = x
    x = x * 0.5
    x = x + original_x
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

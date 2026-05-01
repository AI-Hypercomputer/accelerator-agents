"""99_Matmul_GELU_Softmax — JAXBench fused operator workload."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '99_Matmul_GELU_Softmax',
    'batch_size': 4096,
    'in_features': 8192,
    'out_features': 8192,
}


def create_inputs(dtype=jnp.float32):
    """Create all inputs including weights."""
    key = jax.random.key(0)
    x = jax.random.uniform(key, (4096, 8192), dtype=dtype)
    weight = jnp.zeros((8192, 8192), dtype=dtype)
    bias = jnp.zeros(8192, dtype=dtype)
    return x, weight, bias


def workload(x, weight, bias):
    """Matmul + GELU + Softmax."""
    x = jnp.matmul(x, weight) + bias
    x = jax.nn.gelu(x)
    x = jax.nn.softmax(x, axis=1)
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

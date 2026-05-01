"""45_Gemm_Sigmoid_LogSumExp — JAXBench fused operator workload."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '45_Gemm_Sigmoid_LogSumExp',
    'batch_size': 16384,
    'input_size': 2048,
    'hidden_size': 4096,
    'output_size': 1024,
}


def create_inputs(dtype=jnp.float32):
    """Create all inputs including weights."""
    key = jax.random.key(0)
    batch_size, input_size, hidden_size, output_size = 16384, 2048, 4096, 1024
    x = jax.random.uniform(key, (batch_size, input_size), dtype=dtype)
    w1 = jnp.zeros((hidden_size, input_size), dtype=dtype)
    b1 = jnp.zeros(hidden_size, dtype=dtype)
    w2 = jnp.zeros((output_size, hidden_size), dtype=dtype)
    b2 = jnp.zeros(output_size, dtype=dtype)
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    """Gemm + Sigmoid + Gemm + LogSumExp."""
    x = jnp.matmul(x, w1.T) + b1
    x = jax.nn.sigmoid(x)
    x = jnp.matmul(x, w2.T) + b2
    x = jax.scipy.special.logsumexp(x, axis=1)
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

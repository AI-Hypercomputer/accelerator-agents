"""14_Gemm_Divide_Sum_Scaling — JAXBench fused operator workload."""
import jax
import jax.numpy as jnp
import jax.lax as lax

CONFIG = {
    'name': '14_Gemm_Divide_Sum_Scaling',
    'batch_size': 4096,
    'input_size': 8192,
    'hidden_size': 8192,
    'scaling_factor': 1.5,
}


def create_inputs(dtype=jnp.float32):
    """Create all inputs including weights."""
    key = jax.random.key(0)
    k1, k2 = jax.random.split(key)
    x = jax.random.uniform(k1, (4096, 8192), dtype=dtype)
    weight = jax.random.normal(k2, (8192, 8192), dtype=dtype)
    return x, weight


def workload(x, weight):
    """Gemm + Divide + Sum + Scaling."""
    x = lax.dot_general(
        x, weight.T,
        dimension_numbers=(((1,), (0,)), ((), ())),
        precision=lax.Precision.HIGHEST
    )
    x = x / 2.0
    x = jnp.sum(x, axis=1, keepdims=True)
    x = x * 1.5
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

"""22_Matmul_Scale_ResidualAdd_Clamp_LogSumExp_Mish — JAXBench fused operator workload."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '22_Matmul_Scale_ResidualAdd_Clamp_LogSumExp_Mish',
    'batch_size': 4096,
    'input_size': 8192,
    'hidden_size': 8192,
    'scale_factor': 2.0,
    'clamp_min': -10.0,
    'clamp_max': 10.0,
}


def create_inputs(dtype=jnp.float32):
    """Create all inputs including weights."""
    key = jax.random.key(0)
    x = jax.random.uniform(key, (4096, 8192), dtype=dtype)
    weight = jnp.zeros((8192, 8192), dtype=dtype)
    bias = jnp.zeros(8192, dtype=dtype)
    return x, weight, bias


def workload(x, weight, bias):
    """Matmul + Scale + ResidualAdd + Clamp + LogSumExp + Mish."""
    x = jnp.matmul(x, weight.T) + bias
    x = x * 2.0
    x = x + x
    x = jnp.clip(x, -10.0, 10.0)
    x = jax.scipy.special.logsumexp(x, axis=1, keepdims=True)
    softplus_x = jnp.logaddexp(x, 0.0)
    mish_x = x * jnp.tanh(softplus_x)
    x = x * mish_x
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

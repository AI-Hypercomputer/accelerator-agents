"""75_Gemm_GroupNorm_Min_BiasAdd — JAXBench fused operator workload."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '75_Gemm_GroupNorm_Min_BiasAdd',
    'batch_size': 4096,
    'in_features': 8192,
    'out_features': 8192,
    'num_groups': 512,
}


def create_inputs(dtype=jnp.float32):
    """Create all inputs including weights."""
    key = jax.random.key(0)
    batch_size, in_features, out_features, num_groups = 4096, 8192, 8192, 512
    x = jax.random.uniform(key, (batch_size, in_features), dtype=dtype)
    weight = jnp.zeros((out_features, in_features), dtype=dtype)
    linear_bias = jnp.zeros(out_features, dtype=dtype)
    gn_weight = jnp.ones(out_features, dtype=dtype)
    gn_bias = jnp.zeros(out_features, dtype=dtype)
    bias = jnp.zeros((1, out_features, 1, 1), dtype=dtype)
    return x, weight, linear_bias, gn_weight, gn_bias, bias


def workload(x, weight, linear_bias, gn_weight, gn_bias, bias):
    """Gemm + GroupNorm + Min + BiasAdd."""
    num_groups = 512
    eps = 1e-5
    # Linear
    x = jnp.matmul(x, weight.T) + linear_bias
    # GroupNorm
    N, C = x.shape
    G = num_groups
    x = x.reshape(N, G, C // G)
    mean = jnp.mean(x, axis=2, keepdims=True)
    var = jnp.var(x, axis=2, keepdims=True)
    x = (x - mean) / jnp.sqrt(var + eps)
    x = x.reshape(N, C)
    x = x * gn_weight + gn_bias
    # Min along features
    x = jnp.min(x, axis=1, keepdims=True)
    # Reshape and bias add
    x = x.reshape(1, 1, N, 1)
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

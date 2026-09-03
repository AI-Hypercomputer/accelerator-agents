"""41_Gemm_BatchNorm_GELU_ReLU — JAXBench fused operator workload."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '41_Gemm_BatchNorm_GELU_ReLU',
    'batch_size': 16384,
    'in_features': 8192,
    'out_features': 8192,
}


def create_inputs(dtype=jnp.float32):
    """Create all inputs including weights."""
    key = jax.random.key(0)
    k1, k2, k3 = jax.random.split(key, 3)
    batch_size, in_features, out_features = 16384, 8192, 8192
    x = jax.random.uniform(k1, (batch_size, in_features), dtype=dtype)
    gemm_weight = jax.random.normal(k2, (out_features, in_features), dtype=dtype)
    gemm_bias = jax.random.normal(k3, (out_features,), dtype=dtype)
    bn_weight = jnp.ones(out_features, dtype=dtype)
    bn_bias = jnp.zeros(out_features, dtype=dtype)
    return x, gemm_weight, gemm_bias, bn_weight, bn_bias


def workload(x, gemm_weight, gemm_bias, bn_weight, bn_bias):
    """Gemm + BatchNorm + GELU + ReLU."""
    eps = 1e-5
    # Linear
    x = jnp.matmul(x, gemm_weight.T) + gemm_bias
    # BatchNorm1d (training mode)
    mean = jnp.mean(x, axis=0, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=0, keepdims=True)
    x = (x - mean) / jnp.sqrt(var + eps) * bn_weight + bn_bias
    # GELU + ReLU
    x = jax.nn.gelu(x)
    x = jax.nn.relu(x)
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

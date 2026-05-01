"""88_Gemm_GroupNorm_Swish_Multiply_Swish — JAXBench fused operator workload."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '88_Gemm_GroupNorm_Swish_Multiply_Swish',
    'batch_size': 4096,
    'in_features': 8192,
    'out_features': 8192,
    'num_groups': 256,
}


def create_inputs(dtype=jnp.float32):
    """Create all inputs including weights."""
    key = jax.random.key(0)
    batch_size, in_features, out_features, num_groups = 4096, 8192, 8192, 256
    x = jax.random.uniform(key, (batch_size, in_features), dtype=dtype)
    gemm_weight = jnp.zeros((out_features, in_features), dtype=dtype)
    gemm_bias = jnp.zeros(out_features, dtype=dtype)
    gn_weight = jnp.ones(out_features, dtype=dtype)
    gn_bias = jnp.zeros(out_features, dtype=dtype)
    multiply_weight = jnp.zeros(out_features, dtype=dtype)
    return x, gemm_weight, gemm_bias, gn_weight, gn_bias, multiply_weight


def workload(x, gemm_weight, gemm_bias, gn_weight, gn_bias, multiply_weight):
    """Gemm + GroupNorm + Swish + Multiply + Swish."""
    num_groups = 256
    out_features = 8192
    # Linear
    x = jnp.matmul(x, gemm_weight.T) + gemm_bias
    # GroupNorm
    batch_size = x.shape[0]
    group_size = out_features // num_groups
    x_grouped = x.reshape(batch_size, num_groups, group_size)
    mean = jnp.mean(x_grouped, axis=-1, keepdims=True)
    var = jnp.var(x_grouped, axis=-1, keepdims=True)
    x_normalized = (x_grouped - mean) / jnp.sqrt(var + 1e-5)
    x = x_normalized.reshape(batch_size, out_features)
    x = x * gn_weight + gn_bias
    # Swish
    x = x * jax.nn.sigmoid(x)
    # Multiply
    x = x * multiply_weight
    # Swish
    x = x * jax.nn.sigmoid(x)
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

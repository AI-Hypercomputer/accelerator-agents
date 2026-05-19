"""37_Matmul_Swish_Sum_GroupNorm — JAXBench fused operator workload."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '37_Matmul_Swish_Sum_GroupNorm',
    'batch_size': 8192,
    'in_features': 4096,
    'out_features': 4096,
    'num_groups': 64,
}


def create_inputs(dtype=jnp.float32):
    """Create all inputs including weights."""
    key = jax.random.key(0)
    batch_size, in_features, out_features, num_groups = 8192, 4096, 4096, 64
    x = jax.random.uniform(key, (batch_size, in_features), dtype=dtype)
    weight = jnp.zeros((in_features, out_features), dtype=dtype)
    bias = jnp.zeros(out_features, dtype=dtype)
    gn_weight = jnp.ones(out_features, dtype=dtype)
    gn_bias = jnp.zeros(out_features, dtype=dtype)
    return x, weight, bias, gn_weight, gn_bias


def workload(x, weight, bias, gn_weight, gn_bias):
    """Matmul + Swish + Sum(bias) + GroupNorm."""
    num_groups = 64
    out_features = 4096
    # Linear
    x = jnp.matmul(x, weight)
    # Swish
    x = jax.nn.sigmoid(x) * x
    # Add bias
    x = x + bias
    # GroupNorm
    group_size = out_features // num_groups
    x = x.reshape(-1, num_groups, group_size)
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    x = (x - mean) / jnp.sqrt(var + 1e-5)
    x = x.reshape(-1, out_features)
    x = x * gn_weight + gn_bias
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

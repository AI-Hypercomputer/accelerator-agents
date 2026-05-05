"""28_BMM_InstanceNorm_Sum_ResidualAdd_Multiply — JAXBench fused operator workload."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '28_BMM_InstanceNorm_Sum_ResidualAdd_Multiply',
    'batch_size': 4096,
    'in_features': 8192,
    'out_features': 8192,
}


def create_inputs(dtype=jnp.float32):
    """Create all inputs including weights."""
    key = jax.random.key(0)
    k1, k2 = jax.random.split(key)
    batch_size, in_features, out_features = 4096, 8192, 8192
    x = jax.random.uniform(k1, (batch_size, in_features), dtype=dtype)
    y = jax.random.uniform(k2, (batch_size, out_features), dtype=dtype)
    bmm_weight = jnp.zeros((out_features, in_features), dtype=dtype)
    bmm_bias = jnp.zeros(out_features, dtype=dtype)
    in_weight = jnp.ones(out_features, dtype=dtype)
    in_bias = jnp.zeros(out_features, dtype=dtype)
    return x, y, bmm_weight, bmm_bias, in_weight, in_bias


def workload(x, y, bmm_weight, bmm_bias, in_weight, in_bias):
    """BMM + InstanceNorm + Sum + ResidualAdd + Multiply."""
    eps = 1e-5
    # Linear
    x = x @ bmm_weight.T + bmm_bias

    # InstanceNorm2d on (N, C, 1, 1)
    x = jnp.expand_dims(jnp.expand_dims(x, 2), 3)
    mean = jnp.mean(x, axis=(2, 3), keepdims=True)
    var = jnp.var(x, axis=(2, 3), keepdims=True)
    x = (x - mean) / jnp.sqrt(var + eps)
    x = x * jnp.reshape(in_weight, (1, -1, 1, 1)) + jnp.reshape(in_bias, (1, -1, 1, 1))
    x = jnp.squeeze(jnp.squeeze(x, axis=3), axis=2)

    # Residual add and multiply
    x = x + y
    x = x * y
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

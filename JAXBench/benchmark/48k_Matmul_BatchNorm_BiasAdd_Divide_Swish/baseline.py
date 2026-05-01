"""97_Matmul_BatchNorm_BiasAdd_Divide_Swish — JAXBench fused operator workload."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '97_Matmul_BatchNorm_BiasAdd_Divide_Swish',
    'batch_size': 4096,
    'in_features': 8192,
    'out_features': 8192,
    'bn_eps': 1e-05,
    'bn_momentum': 0.1,
    'divide_value': 1.0,
}


def create_inputs(dtype=jnp.float32):
    """Create all inputs including weights."""
    key = jax.random.key(0)
    batch_size, in_features, out_features = 4096, 8192, 8192
    x = jax.random.uniform(key, (batch_size, in_features), dtype=dtype)
    weight = jnp.zeros((in_features, out_features), dtype=dtype)
    linear_bias = jnp.zeros(out_features, dtype=dtype)
    bn_scale = jnp.ones(out_features, dtype=dtype)
    bn_bias = jnp.zeros(out_features, dtype=dtype)
    bn_mean = jnp.zeros(out_features, dtype=dtype)
    bn_var = jnp.ones(out_features, dtype=dtype)
    bias = jnp.zeros((1,), dtype=dtype)
    return x, weight, linear_bias, bn_scale, bn_bias, bn_mean, bn_var, bias


def workload(x, weight, linear_bias, bn_scale, bn_bias, bn_mean, bn_var, bias):
    """Matmul + BatchNorm + BiasAdd + Divide + Swish."""
    bn_eps = 1e-5
    divide_value = 1.0
    # Linear
    x = jnp.matmul(x, weight) + linear_bias
    # BatchNorm (eval mode)
    x_normalized = (x - bn_mean) / jnp.sqrt(bn_var + bn_eps)
    x = bn_scale * x_normalized + bn_bias
    # Bias + divide
    x = x + bias
    x = x / divide_value
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

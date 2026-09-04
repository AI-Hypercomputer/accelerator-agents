"""Grouped Matmul (Ragged Dot) for MoE — Mixtral 8x7B. From openxla/tokamax."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'mixtral_8x7b_ragged_dot',
    'model': 'Mixtral-8x7B',
    'operator': 'ragged_dot',
    'num_groups': 8,
    'M': 8192,
    'K': 4096,
    'N': 14336,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (x, weights, group_sizes) for grouped matmul."""
    key = jax.random.key(42)
    k1, k2 = jax.random.split(key, 2)
    G, M, K, N = CONFIG['num_groups'], CONFIG['M'], CONFIG['K'], CONFIG['N']
    # Each group gets M/G rows
    x = jax.random.normal(k1, (G, M // G, K), dtype=dtype)
    weights = jax.random.normal(k2, (G, K, N), dtype=dtype) * 0.02
    return x, weights


def workload(x, weights):
    """Grouped matmul: each group does independent matmul. Equivalent to ragged dot."""
    # x: (G, M/G, K), weights: (G, K, N) -> out: (G, M/G, N)
    return jnp.einsum('gmk,gkn->gmn', x, weights)


def benchmark(num_warmup=5, num_iters=100):
    """Benchmark and return results dict."""
    import time
    inputs = create_inputs()
    fn = jax.jit(workload)
    for _ in range(num_warmup):
        out = fn(*inputs)
        out.block_until_ready()
    times = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        out = fn(*inputs)
        out.block_until_ready()
        times.append(time.perf_counter() - t0)
    import numpy as np
    times = np.array(times) * 1000
    G, M, K, N = CONFIG['num_groups'], CONFIG['M'], CONFIG['K'], CONFIG['N']
    flops = G * (M // G) * K * N * 2
    avg = float(np.mean(times))
    return {
        'name': CONFIG['name'],
        'model': CONFIG['model'],
        'operator': CONFIG['operator'],
        'config': {k: v for k, v in CONFIG.items() if k not in ('name', 'model', 'operator')},
        'time_ms': round(avg, 4),
        'std_ms': round(float(np.std(times)), 4),
        'tflops': round(flops / (avg / 1000) / 1e12, 2),
        'output_shape': list(out.shape),
        'status': 'success',
    }


if __name__ == '__main__':
    import json
    print(json.dumps(benchmark()))

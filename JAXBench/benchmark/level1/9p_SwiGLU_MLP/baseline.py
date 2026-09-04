"""SwiGLU MLP — Llama 3.1 70B. Extracted from MaxText."""
import jax
import jax.numpy as jnp
from functools import partial

CONFIG = {
    'name': 'llama3_70b_swiglu',
    'model': 'Llama-3.1-70B',
    'operator': 'swiglu_mlp',
    'batch': 2,
    'seq_len': 4096,
    'emb_dim': 8192,
    'mlp_dim': 28672,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (x, gate_kernel, up_kernel, down_kernel)."""
    key = jax.random.key(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    B, S, E, M = CONFIG['batch'], CONFIG['seq_len'], CONFIG['emb_dim'], CONFIG['mlp_dim']
    x = jax.random.normal(k1, (B, S, E), dtype=dtype)
    gate = jax.random.normal(k2, (E, M), dtype=dtype) * 0.02
    up = jax.random.normal(k3, (E, M), dtype=dtype) * 0.02
    down = jax.random.normal(k4, (M, E), dtype=dtype) * 0.02
    return x, gate, up, down


def workload(x, gate_kernel, up_kernel, down_kernel):
    """SwiGLU: output = (SiLU(x @ gate) * (x @ up)) @ down"""
    gate = jax.nn.silu(jnp.dot(x, gate_kernel))
    up = jnp.dot(x, up_kernel)
    return jnp.dot(gate * up, down_kernel)


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
    B, S, E, M = CONFIG['batch'], CONFIG['seq_len'], CONFIG['emb_dim'], CONFIG['mlp_dim']
    flops = B * S * E * M * 2 * 3
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

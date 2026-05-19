"""RMSNorm — Llama-3.1-70B pre-attention normalization.

Root Mean Square Layer Normalization at Llama-3.1-70B scale.
Input shape: (batch=1, seq_len=2048, emb_dim=8192).
From MaxText layers/normalizations.py.
"""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'llama3_70b_rmsnorm',
    'model': 'Llama-3.1-70B',
    'operator': 'rms_norm',
    'batch': 8,
    'seq_len': 4096,
    'emb_dim': 8192,
    'epsilon': 1e-5,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (x, scale) tensors."""
    key = jax.random.key(42)
    k1, k2 = jax.random.split(key, 2)
    B, S, D = CONFIG['batch'], CONFIG['seq_len'], CONFIG['emb_dim']
    x = jax.random.normal(k1, (B, S, D), dtype=dtype)
    scale = jax.random.normal(k2, (D,), dtype=dtype) * 0.1 + 1.0
    return x, scale


def workload(x, scale):
    """RMSNorm: x * rsqrt(mean(x^2) + eps) * scale"""
    x_f32 = jnp.asarray(x, jnp.float32)
    mean2 = jnp.mean(lax.square(x_f32), axis=-1, keepdims=True)
    normed = x_f32 * lax.rsqrt(mean2 + CONFIG['epsilon'])
    normed = jnp.asarray(normed, x.dtype)
    return normed * scale


def get_flops():
    """RMSNorm FLOPs: square + mean + rsqrt + 2x mul over B*S*D."""
    B, S, D = CONFIG['batch'], CONFIG['seq_len'], CONFIG['emb_dim']
    return 5 * B * S * D


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
    B, S, D = CONFIG['batch'], CONFIG['seq_len'], CONFIG['emb_dim']
    # square + mean + rsqrt + mul + mul: ~5 elementwise ops over B*S*D
    flops = 5 * B * S * D
    avg = float(np.mean(times))
    return {
        'name': CONFIG['name'],
        'model': CONFIG['model'],
        'operator': CONFIG['operator'],
        'config': {k: v for k, v in CONFIG.items() if k not in ('name', 'model', 'operator')},
        'time_ms': round(avg, 4),
        'std_ms': round(float(np.std(times)), 4),
        'tflops': round(flops / (avg / 1000) / 1e12, 4),
        'output_shape': list(out.shape),
        'status': 'success',
    }


if __name__ == '__main__':
    import json
    print(json.dumps(benchmark()))

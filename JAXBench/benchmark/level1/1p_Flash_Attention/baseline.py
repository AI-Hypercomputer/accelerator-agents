"""Vanilla Multi-Head Causal Attention — Baseline (64 heads, seq=2048).

Standard scaled dot-product attention with causal mask.
No GQA, no softcap, no sliding window — pure MHA baseline.
Matches Pallas flash_attention kernel config.
"""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'flash_attention_baseline',
    'model': 'Baseline-MHA',
    'operator': 'causal_mha',
    'batch': 4,
    'seq_len': 4096,
    'num_heads': 64,
    'head_dim': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (query, key, value) tensors."""
    key = jax.random.key(42)
    k1, k2, k3 = jax.random.split(key, 3)
    B, S = CONFIG['batch'], CONFIG['seq_len']
    H, D = CONFIG['num_heads'], CONFIG['head_dim']
    query = jax.random.normal(k1, (B, H, S, D), dtype=dtype)
    key_t = jax.random.normal(k2, (B, H, S, D), dtype=dtype)
    value = jax.random.normal(k3, (B, H, S, D), dtype=dtype)
    return query, key_t, value


def workload(query, key, value):
    """Standard causal multi-head attention: QK^T -> mask -> softmax -> AV."""
    B, H, S, D = query.shape
    scale = D ** -0.5
    # QK^T
    attn = jnp.einsum('bhqd,bhkd->bhqk', query, key) * scale
    # Causal mask
    mask = jnp.tril(jnp.ones((S, S)))
    attn = jnp.where(mask, attn, -1e9)
    # Softmax
    attn = jax.nn.softmax(attn, axis=-1)
    # AV
    output = jnp.einsum('bhqk,bhkd->bhqd', attn, value)
    return output


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
    B, S, H, D = CONFIG['batch'], CONFIG['seq_len'], CONFIG['num_heads'], CONFIG['head_dim']
    # QK^T: 2*B*H*S*S*D, AV: 2*B*H*S*S*D
    flops = 4 * B * H * S * S * D
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

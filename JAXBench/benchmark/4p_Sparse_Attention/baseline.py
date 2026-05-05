"""Sparse (Splash) Attention — Llama-3.1-70B GQA with causal mask.

Vanilla JAX baseline for sparse/splash attention: standard dot-product
attention with causal masking and grouped-query attention (GQA).
From MaxText kernels/attention/splash_attention_kernel.py.
"""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'llama3_70b_sparse_attention',
    'model': 'Llama-3.1-70B',
    'operator': 'sparse_attention',
    'batch': 4,
    'seq_len': 4096,
    'num_query_heads': 64,
    'num_kv_heads': 8,
    'head_dim': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (q, k, v) tensors in [num_heads, seq_len, head_dim] layout."""
    key = jax.random.key(42)
    k1, k2, k3 = jax.random.split(key, 3)
    S = CONFIG['seq_len']
    H_q = CONFIG['num_query_heads']
    H_kv = CONFIG['num_kv_heads']
    D = CONFIG['head_dim']
    q = jax.random.normal(k1, (H_q, S, D), dtype=dtype) * (D ** -0.5)
    k = jax.random.normal(k2, (H_kv, S, D), dtype=dtype) * 0.02
    v = jax.random.normal(k3, (H_kv, S, D), dtype=dtype) * 0.02
    return q, k, v


def workload(q, k, v):
    """Causal GQA attention: splash attention baseline."""
    S = CONFIG['seq_len']
    H_q = CONFIG['num_query_heads']
    H_kv = CONFIG['num_kv_heads']
    num_q_per_kv = H_q // H_kv

    # Repeat KV heads for GQA
    k = jnp.repeat(k, num_q_per_kv, axis=0)  # (H_q, S, D)
    v = jnp.repeat(v, num_q_per_kv, axis=0)  # (H_q, S, D)

    # Attention scores: (H_q, S, D) x (H_q, S, D) -> (H_q, S, S)
    attn = jnp.einsum('hqd,hkd->hqk', q, k)

    # Causal mask
    causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
    attn = jnp.where(causal[None, :, :], attn, -1e30)

    attn = jax.nn.softmax(attn, axis=-1)

    # Output: (H_q, S, S) x (H_q, S, D) -> (H_q, S, D)
    out = jnp.einsum('hqk,hkd->hqd', attn, v)
    return out


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
    H_q = CONFIG['num_query_heads']
    S = CONFIG['seq_len']
    D = CONFIG['head_dim']
    # QK dot: 2*H*S*S*D, AV dot: 2*H*S*S*D
    flops = 4 * H_q * S * S * D
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

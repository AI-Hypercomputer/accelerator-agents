"""Ragged Paged Attention — Llama-3.1-70B mixed prefill+decode.

Reference implementation with data-dependent slicing on per-sequence boundaries.
Processes each sequence independently with variable-length queries and paged KV cache.
From JAX experimental pallas ops (ref_ragged_paged_attention).

Not jit-compatible: uses data-dependent slicing.
"""

import math

import jax
import jax.numpy as jnp

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)

CONFIG = {
    'name': 'ragged_paged_attention_llama70b',
    'model': 'Llama-3.1-70B',
    'operator': 'ragged_paged_attention',
    'max_num_batched_tokens': 4096,
    'max_num_seqs': 64,
    'num_q_heads': 64,
    'num_kv_heads': 8,
    'head_dim': 128,
    'page_size': 16,
    'pages_per_seq': 256,
}

_skip_jit = True


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(42)
    k1, k2 = jax.random.split(key, 2)
    max_tokens = CONFIG['max_num_batched_tokens']
    max_seqs = CONFIG['max_num_seqs']
    H_q = CONFIG['num_q_heads']
    H_kv = CONFIG['num_kv_heads']
    D = CONFIG['head_dim']
    page_size = CONFIG['page_size']
    pages_per_seq = CONFIG['pages_per_seq']
    num_combined_kv_heads = 2 * H_kv
    total_num_pages = max_seqs * pages_per_seq
    q = jax.random.normal(k1, (max_tokens, H_q, D), dtype=dtype)
    kv_pages = jax.random.normal(
        k2, (total_num_pages, page_size, num_combined_kv_heads, D), dtype=dtype
    )
    tokens_per_seq = max_tokens // max_seqs
    kv_len_per_seq = pages_per_seq * page_size
    kv_lens = jnp.full((max_seqs,), kv_len_per_seq, dtype=jnp.int32)
    page_indices = jnp.arange(total_num_pages, dtype=jnp.int32).reshape(
        max_seqs, pages_per_seq
    )
    cu_q_lens = jnp.arange(max_seqs + 1, dtype=jnp.int32) * tokens_per_seq
    num_seqs = jnp.array([max_seqs], dtype=jnp.int32)
    return q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs


def workload(queries, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs):
    """Reference ragged paged attention from upstream JAX.

    Processes each sequence independently with data-dependent slicing.
    Must be run eagerly (not under jax.jit).
    """
    sm_scale = 1.0 / math.sqrt(CONFIG['head_dim'])
    mask_value = DEFAULT_MASK_VALUE
    _, _, num_combined_kv_heads, head_dim = kv_pages.shape
    num_kv_heads = num_combined_kv_heads // 2
    num_q_heads = queries.shape[1]
    num_query_per_kv = num_q_heads // num_kv_heads

    outputs = []
    for i in range(num_seqs[0]):
        q_start = cu_q_lens[i]
        q_end = cu_q_lens[i + 1]
        q_len = q_end - q_start
        kv_len = kv_lens[i]
        indices = page_indices[i]

        q = queries[q_start:q_end]
        k = kv_pages[indices, :, 0::2, :].reshape(-1, num_kv_heads, head_dim)[:kv_len]
        v = kv_pages[indices, :, 1::2, :].reshape(-1, num_kv_heads, head_dim)[:kv_len]

        k = jnp.repeat(k, num_query_per_kv, axis=1)
        v = jnp.repeat(v, num_query_per_kv, axis=1)

        attn = jnp.einsum(
            "qhd,khd->hqk", q, k, preferred_element_type=jnp.float32
        )
        attn *= sm_scale

        q_span = (kv_len - q_len) + jax.lax.broadcasted_iota(
            jnp.int32, attn.shape, 1
        )
        kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
        mask = q_span < kv_span
        attn += jnp.where(mask, mask_value, 0.0)

        attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)
        out = jnp.einsum("hqk,khd->qhd", attn, v).astype(queries.dtype)
        outputs.append(out)

    return jnp.concatenate(outputs, axis=0)


def get_flops():
    """Ragged paged attention FLOPs: per seq QK^T + AV matmuls."""
    max_seqs = CONFIG['max_num_seqs']
    H_q = CONFIG['num_q_heads']
    D = CONFIG['head_dim']
    tokens_per_seq = CONFIG['max_num_batched_tokens'] // max_seqs
    kv_len = CONFIG['pages_per_seq'] * CONFIG['page_size']
    return max_seqs * H_q * (4 * tokens_per_seq * kv_len * D)


def benchmark(num_warmup=2, num_iters=10):
    """Benchmark eagerly (no JIT — data-dependent control flow)."""
    import time
    inputs = create_inputs()
    # Warmup
    for _ in range(num_warmup):
        out = workload(*inputs)
        out.block_until_ready()
    times = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        out = workload(*inputs)
        out.block_until_ready()
        times.append(time.perf_counter() - t0)
    import numpy as np
    times = np.array(times) * 1000
    max_seqs = CONFIG['max_num_seqs']
    H_q = CONFIG['num_q_heads']
    D = CONFIG['head_dim']
    tokens_per_seq = CONFIG['max_num_batched_tokens'] // max_seqs
    kv_len = CONFIG['pages_per_seq'] * CONFIG['page_size']
    flops = max_seqs * H_q * (4 * tokens_per_seq * kv_len * D)
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

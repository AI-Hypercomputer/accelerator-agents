"""Ragged Paged Attention — Llama-3.1-70B inference decode.

Paged KV-cache attention with variable-length sequences, as used in
serving frameworks. Supports grouped-query attention (GQA).
From MaxText inference/paged_attention_kernel_v2.py.
"""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'llama3_70b_paged_attention',
    'model': 'Llama-3.1-70B',
    'operator': 'paged_attention',
    'num_seqs': 64,
    'max_seq_len': 4096,
    'num_query_heads': 64,
    'num_kv_heads': 8,
    'head_dim': 128,
    'page_size': 16,
    'pages_per_seq': 256,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (queries, k_pages, v_pages, kv_lens, page_indices, cu_q_lens)."""
    key = jax.random.key(42)
    keys = jax.random.split(key, 5)
    num_seqs = CONFIG['num_seqs']
    max_seq_len = CONFIG['max_seq_len']
    num_q_heads = CONFIG['num_query_heads']
    num_kv_heads = CONFIG['num_kv_heads']
    head_dim = CONFIG['head_dim']
    page_size = CONFIG['page_size']
    pages_per_seq = CONFIG['pages_per_seq']
    total_pages = num_seqs * pages_per_seq

    # Each sequence has 1 query token (decode mode)
    max_num_tokens = num_seqs
    queries = jax.random.normal(keys[0], (max_num_tokens, num_q_heads, head_dim), dtype=dtype)
    k_pages = jax.random.normal(keys[1], (total_pages, page_size, num_kv_heads, head_dim), dtype=dtype) * 0.02
    v_pages = jax.random.normal(keys[2], (total_pages, page_size, num_kv_heads, head_dim), dtype=dtype) * 0.02

    # All sequences have full length
    kv_lens = jnp.full((num_seqs,), max_seq_len, dtype=jnp.int32)
    # Page indices: sequence i owns pages [i*pages_per_seq, (i+1)*pages_per_seq)
    page_indices = jnp.arange(total_pages, dtype=jnp.int32).reshape(num_seqs, pages_per_seq)
    # Cumulative query lengths: each sequence has 1 query token
    cu_q_lens = jnp.arange(num_seqs + 1, dtype=jnp.int32)

    return queries, k_pages, v_pages, kv_lens, page_indices, cu_q_lens


def workload(queries, k_pages, v_pages, kv_lens, page_indices, cu_q_lens):
    """Ragged paged attention: gather pages, compute GQA attention per sequence."""
    num_seqs = CONFIG['num_seqs']
    num_q_heads = CONFIG['num_query_heads']
    num_kv_heads = CONFIG['num_kv_heads']
    head_dim = CONFIG['head_dim']
    page_size = CONFIG['page_size']
    max_seq_len = CONFIG['pages_per_seq'] * page_size
    pages_per_seq = CONFIG['pages_per_seq']
    num_q_per_kv = num_q_heads // num_kv_heads
    sm_scale = head_dim ** -0.5

    def attend_one_seq(seq_idx):
        q_start = cu_q_lens[seq_idx]
        q_end = cu_q_lens[seq_idx + 1]
        q = jax.lax.dynamic_slice(queries, (q_start, 0, 0), (1, num_q_heads, head_dim))  # (1, H_q, D)
        # Gather KV pages
        seq_pages = page_indices[seq_idx]  # (pages_per_seq,)
        k = k_pages[seq_pages].reshape(max_seq_len, num_kv_heads, head_dim)  # (S, H_kv, D)
        v = v_pages[seq_pages].reshape(max_seq_len, num_kv_heads, head_dim)
        # Repeat KV heads for GQA
        k = jnp.repeat(k, num_q_per_kv, axis=1)  # (S, H_q, D)
        v = jnp.repeat(v, num_q_per_kv, axis=1)
        # Attention: (1, H_q, D) x (S, H_q, D) -> (H_q, 1, S)
        attn = jnp.einsum('qhd,khd->hqk', q, k) * sm_scale
        # Causal mask: query at position kv_len-1, keys at 0..kv_len-1
        kv_len = kv_lens[seq_idx]
        mask = jnp.arange(max_seq_len) < kv_len
        attn = jnp.where(mask[None, None, :], attn, -1e30)
        attn = jax.nn.softmax(attn, axis=-1)
        out = jnp.einsum('hqk,khd->qhd', attn, v)  # (1, H_q, D)
        return out.squeeze(0)  # (H_q, D)

    # Process all sequences
    outputs = jax.vmap(attend_one_seq)(jnp.arange(num_seqs))  # (num_seqs, H_q, D)
    return outputs


def get_flops():
    """Decode attention FLOPs: per seq QK^T + AV matmuls."""
    num_seqs = CONFIG['num_seqs']
    num_q_heads = CONFIG['num_query_heads']
    max_seq_len = CONFIG['max_seq_len']
    head_dim = CONFIG['head_dim']
    return num_seqs * num_q_heads * (4 * max_seq_len * head_dim)


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
    num_seqs = CONFIG['num_seqs']
    num_q_heads = CONFIG['num_query_heads']
    max_seq_len = CONFIG['max_seq_len']
    head_dim = CONFIG['head_dim']
    # Per sequence: QK dot (2*1*S*D*H) + softmax + AV dot (2*1*S*D*H)
    flops = num_seqs * num_q_heads * (4 * max_seq_len * head_dim)
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

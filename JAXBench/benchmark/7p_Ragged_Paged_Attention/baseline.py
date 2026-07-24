"""Ragged Paged Attention — Llama-3.1-70B mixed prefill+decode.

The input buffers have static maximum shapes, but contain a deterministic mix of
decode and prefill requests with variable query/KV lengths, inactive sequence
slots, and a shuffled physical page table.
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

# IMPORTANT: This benchmark tests ONE representative serving scenario, not a
# distribution of all possible RPA inputs. It models a saturated continuous-
# batching step for Llama-3.1-70B: decode requests are scheduled first, then
# the remaining 4096-token budget is filled by 512-token chunked prefills.
# Specifically: 48 one-token decodes + seven 512-token prefill chunks + one
# 464-token remainder, leaving eight of the 64 sequence slots inactive.
ACTIVE_Q_LENS = (1,) * 48 + (512,) * 7 + (464,)

# Decode contexts span the available cache range and always end on partial
# pages. Prefill KV lengths represent the context length after each chunk.
ACTIVE_KV_LENS = tuple(
    257 + ((i * 73) % 240) * 16 for i in range(48)
) + (1023, 1535, 2047, 2559, 3071, 3583, 4095, 4095)
NUM_ACTIVE_SEQS = len(ACTIVE_Q_LENS)


def create_inputs(dtype=jnp.bfloat16):
    """Create the single canonical input used for correctness and timing."""
    key = jax.random.key(42)
    k1, k2, k3 = jax.random.split(key, 3)
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
    q_lens = jnp.array(ACTIVE_Q_LENS, dtype=jnp.int32)
    kv_lens = jnp.pad(
        jnp.array(ACTIVE_KV_LENS, dtype=jnp.int32),
        (0, max_seqs - NUM_ACTIVE_SEQS),
    )
    active_cu_q_lens = jnp.concatenate(
        (jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(q_lens))
    )
    cu_q_lens = jnp.pad(
        active_cu_q_lens,
        (0, max_seqs + 1 - active_cu_q_lens.shape[0]),
        constant_values=active_cu_q_lens[-1],
    )

    # A global permutation makes each logical sequence's pages physically
    # noncontiguous while keeping every page-table entry in bounds and unique.
    page_indices = jax.random.permutation(
        k3, total_num_pages, independent=True
    ).astype(jnp.int32).reshape(max_seqs, pages_per_seq)
    num_seqs = jnp.array([NUM_ACTIVE_SEQS], dtype=jnp.int32)
    return q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs


def workload(queries, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs):
    """Ragged paged attention using static shapes and masking for JIT compatibility.

    Processes each sequence independently, avoiding data-dependent slicing.
    """
    sm_scale = 1.0 / math.sqrt(CONFIG['head_dim'])
    mask_value = DEFAULT_MASK_VALUE
    _, _, num_combined_kv_heads, head_dim = kv_pages.shape
    num_kv_heads = num_combined_kv_heads // 2
    num_q_heads = queries.shape[1]
    num_query_per_kv = num_q_heads // num_kv_heads

    max_tokens = CONFIG['max_num_batched_tokens']
    output = jnp.zeros_like(queries)

    # ACTIVE_Q_LENS supplies static per-sequence capacities so XLA can compile
    # the reference without data-dependent shapes.  cu_q_lens remains the
    # source of truth for the effective lengths and packed-buffer positions.
    for i, q_capacity in enumerate(ACTIVE_Q_LENS):
        q_start = cu_q_lens[i]
        q_len = cu_q_lens[i + 1] - q_start
        kv_len = kv_lens[i]
        indices = page_indices[i]

        q_offsets = jnp.arange(q_capacity, dtype=jnp.int32)
        q_positions = jnp.minimum(q_start + q_offsets, max_tokens - 1)
        q = queries[q_positions]

        k = kv_pages[indices, :, 0::2, :].reshape(-1, num_kv_heads, head_dim)
        v = kv_pages[indices, :, 1::2, :].reshape(-1, num_kv_heads, head_dim)

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

        mask = (q_span < kv_span) | (kv_span >= kv_len)
        attn = jnp.where(mask, mask_value, attn)

        attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)
        out = jnp.einsum("hqk,khd->qhd", attn, v).astype(queries.dtype)

        q_valid = q_offsets < q_len
        is_valid = i < num_seqs[0]
        out = jnp.where(q_valid[:, None, None] & is_valid, out, 0.0)
        output = output.at[q_positions].add(out)

    return output


def get_flops():
    """Ragged paged attention FLOPs: per seq QK^T + AV matmuls."""
    H_q = CONFIG['num_q_heads']
    D = CONFIG['head_dim']
    return H_q * 4 * D * sum(
        q_len * kv_len
        for q_len, kv_len in zip(ACTIVE_Q_LENS, ACTIVE_KV_LENS)
    )


def benchmark(num_warmup=2, num_iters=10):
    """Benchmark with JIT."""
    import time
    inputs = create_inputs()
    fn = jax.jit(workload)
    # Warmup
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
    flops = get_flops()
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

import jax
import jax.numpy as jnp
from functools import partial

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)

DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024

def cdiv(a, b):
    assert b != 0
    return (a + b - 1) // b

def align_to(x, a):
    return cdiv(x, a) * a

def get_dtype_packing(dtype):
    bits = jax.dtypes.itemsize_bits(dtype)
    return 32 // bits

CONFIG = {
    'name': 'MLA',
    'batch_size': 128,
    'q_len': 1,
    'kv_len_val': 9216,
    'page_size': 256,
    'symbol': 'd',
}

def create_inputs():
    key = jax.random.PRNGKey(0)

    num_heads = 128
    lkv_dim = 512
    r_dim = 64
    q_dtype = jnp.bfloat16
    kv_dtype = jnp.bfloat16

    padded_r_dim = align_to(r_dim, 128)
    padded_lkv_dim = align_to(lkv_dim, 128)
    padded_kv_dim = padded_lkv_dim + padded_r_dim
    packing = get_dtype_packing(kv_dtype)

    def gen_random(k, shape, dtype):
        return jax.random.uniform(k, shape, dtype=jnp.float32).astype(dtype)

    total_kv_tokens = CONFIG['batch_size'] * CONFIG['kv_len_val']
    num_pages = cdiv(total_kv_tokens, CONFIG['page_size']) + CONFIG['batch_size']

    total_q_len = CONFIG['batch_size'] * CONFIG['q_len']
    cu_q_lens_list = [i * CONFIG['q_len'] for i in range(CONFIG['batch_size'] + 1)]

    pages_per_seq = cdiv(CONFIG['kv_len_val'], CONFIG['page_size'])
    page_indices_list = []
    page_count = 0
    for _ in range(CONFIG['batch_size']):
        num_seq_pages = cdiv(CONFIG['kv_len_val'], CONFIG['page_size'])
        indices = list(range(page_count, page_count + num_seq_pages))
        page_indices_list.extend(indices + [-1] * (pages_per_seq - num_seq_pages))
        page_count += num_seq_pages

    total_num_pages = max(num_pages, page_count)

    key, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
    ql_nope = gen_random(k1, (total_q_len, num_heads, lkv_dim), q_dtype)
    q_pe = gen_random(k2, (total_q_len, num_heads, r_dim), q_dtype)
    new_kv_c = gen_random(k3, (total_q_len, lkv_dim), kv_dtype)
    new_k_pe = gen_random(k4, (total_q_len, r_dim), kv_dtype)

    cache_kv = gen_random(
        k5,
        (total_num_pages, CONFIG['page_size'] // packing, packing, padded_kv_dim),
        kv_dtype,
    )

    kv_lens = jnp.array([CONFIG['kv_len_val']] * CONFIG['batch_size'], dtype=jnp.int32)
    page_indices = jnp.array(page_indices_list, dtype=jnp.int32)
    cu_q_lens = jnp.array(cu_q_lens_list, dtype=jnp.int32)

    num_decode_seqs = CONFIG['batch_size'] if CONFIG['q_len'] == 1 else 0
    distribution = jnp.array([num_decode_seqs, num_decode_seqs, CONFIG['batch_size']], dtype=jnp.int32)

    return (
        ql_nope, q_pe, new_kv_c, new_k_pe, cache_kv, kv_lens,
        page_indices, cu_q_lens, distribution
    )

def update_kv_cache(
        new_kv_c: jax.Array,  # [num_tokens, actual_lkv_dim]
        new_k_pe: jax.Array,  # [num_tokens, actual_r_dim]
        cache_kv: jax.Array,  # [total_num_pages, page_size_per_kv_packing, kv_packing, lkv_dim+r_dim]
        kv_lens: jax.Array,  # i32[max_num_seqs]
        page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
        cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
        distribution: jax.Array,  # i32[3]
) -> jax.Array:
    """Update KV cache with new tokens."""
    actual_r_dim = new_k_pe.shape[-1]
    r_dim = align_to(actual_r_dim, 128)
    if actual_r_dim != r_dim:
        new_k_pe = jnp.pad(new_k_pe, ((0, 0), (0, r_dim - actual_r_dim)),
                           constant_values=0)
    actual_lkv_dim = new_kv_c.shape[-1]
    lkv_dim = align_to(actual_lkv_dim, 128)
    if actual_lkv_dim != lkv_dim:
        new_kv_c = jnp.pad(new_kv_c, ((0, 0), (0, lkv_dim - actual_lkv_dim)),
                           constant_values=0)
    kv_dim = r_dim + lkv_dim
    _, page_size_per_kv_packing, kv_packing, cache_kv_dim = cache_kv.shape
    assert kv_dim == cache_kv_dim
    page_size = page_size_per_kv_packing * kv_packing

    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    pages_per_seq = num_page_indices // max_num_seqs

    def seq_loop_body(i, cache_kv):
        q_start, q_end = cu_q_lens[i], cu_q_lens[i + 1]
        q_len = q_end - q_start
        kv_len = kv_lens[i]

        def token_loop_body(j, cache_kv_):
            token_idx_in_seq = kv_len - q_len + j
            page_num_in_seq = token_idx_in_seq // page_size
            page_indices_start = i * pages_per_seq
            page_idx = page_indices[page_indices_start + page_num_in_seq]
            row = (token_idx_in_seq % page_size) // kv_packing
            col = (token_idx_in_seq % page_size) % kv_packing

            cache_kv_ = cache_kv_.at[page_idx, row, col, :lkv_dim].set(new_kv_c[q_start + j])
            cache_kv_ = cache_kv_.at[page_idx, row, col, lkv_dim:].set(new_k_pe[q_start + j])
            return cache_kv_

        return jax.lax.fori_loop(0, q_len, token_loop_body, cache_kv)

    cache_kv = jax.lax.fori_loop(0, distribution[-1], seq_loop_body, cache_kv)

    return cache_kv

def ref_mla_ragged_paged_attention(
    ql_nope: jax.Array,  # [num_tokens, actual_num_q_heads, actual_lkv_dim]
    q_pe: jax.Array,  # [num_tokens, actual_num_q_heads, actual_r_dim]
    new_kv_c: jax.Array,  # [num_tokens, actual_lkv_dim]
    new_k_pe: jax.Array,  # [num_tokens, actual_r_dim]
    cache_kv: jax.Array,  # [total_num_pages, page_size_per_kv_packing, kv_packing, lkv_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
):

    if mask_value is None:
        mask_value = DEFAULT_MASK_VALUE

    updated_cache_kv = update_kv_cache(
        new_kv_c,
        new_k_pe,
        cache_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
    )

    # Pad ql_nope and q_pe to make the last dimension 128-byte aligned.
    actual_lkv_dim = ql_nope.shape[-1]
    lkv_dim = align_to(actual_lkv_dim, 128)
    if lkv_dim != actual_lkv_dim:
        ql_nope = jnp.pad(
            ql_nope,
            ((0, 0), (0, 0), (0, lkv_dim - actual_lkv_dim)),
            constant_values=0,
        )
    actual_r_dim = q_pe.shape[-1]
    r_dim = align_to(actual_r_dim, 128)
    if actual_r_dim != r_dim:
        q_pe = jnp.pad(q_pe, ((0, 0), (0, 0), (0, r_dim - actual_r_dim)),
                       constant_values=0)

    q = jnp.concatenate([ql_nope, q_pe], axis=-1)
    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    assert num_page_indices % max_num_seqs == 0
    pages_per_seq = num_page_indices // max_num_seqs

    total_num_pages, page_size_per_kv_packing, kv_packing, _ = updated_cache_kv.shape
    page_size = page_size_per_kv_packing * kv_packing
    assert lkv_dim == ql_nope.shape[-1]
    assert r_dim == q_pe.shape[-1]
    assert lkv_dim + r_dim == updated_cache_kv.shape[-1]

    kv_c_cache = updated_cache_kv[..., :lkv_dim].reshape(
        total_num_pages, page_size, lkv_dim)
    k_pe_cache = updated_cache_kv[...,
                                  lkv_dim:].reshape(total_num_pages, page_size,
                                                    r_dim)

    B = max_num_seqs
    Q_max = q.shape[0] // B
    KV_max = pages_per_seq * page_size

    out_dtype = q.dtype

    def process_seq(i):
        q_start = cu_q_lens[i]

        # Dynamic lengths for this specific sequence
        q_len = cu_q_lens[i+1] - cu_q_lens[i]
        kv_len = kv_lens[i]

        # Use static Q_max for dynamic slice to satisfy XLA
        q_i = jax.lax.dynamic_slice(
            q, (q_start, 0, 0), (Q_max, q.shape[1], q.shape[2]))

        indices_start = i * pages_per_seq
        # Use static pages_per_seq to satisfy XLA
        indices = jax.lax.dynamic_slice(
            page_indices, (indices_start,), (pages_per_seq,))

        # Gather paged kv_c and k_pe
        gathered_kv_c = kv_c_cache[indices]  # [pages_per_seq, page_size, lkv_dim]
        gathered_k_pe = k_pe_cache[indices]  # [pages_per_seq, page_size, r_dim]

        # Flatten pages to sequence
        flat_kv_c = gathered_kv_c.reshape(-1, lkv_dim)  # [KV_max, lkv_dim]
        flat_k_pe = gathered_k_pe.reshape(-1, r_dim)  # [KV_max, r_dim]

        # Prepare k and v for attention (using the fully padded arrays)
        k_i = jnp.concatenate([flat_kv_c, flat_k_pe], axis=-1)  # [KV_max, lkv_dim+r_dim]
        v_i = flat_kv_c  # [KV_max, lkv_dim]

        # MQA attention:
        # q:[Q_max, actual_num_q_heads, lkv_dim+r_dim]
        # k:[KV_max, lkv_dim+r_dim]
        # v:[KV_max, lkv_dim]
        # attn: [actual_num_q_heads, Q_max, KV_max]
        attn = jnp.einsum("qnh,kh->nqk",
                          q_i,
                          k_i,
                          preferred_element_type=jnp.float32)
        attn *= sm_scale
        if k_scale is not None:
            attn *= k_scale
        if q_scale is not None:
            attn *= q_scale

        q_iota = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 1)
        kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
        q_span = kv_len - q_len + q_iota

        # Causal mask: a query cannot attend to a future key.
        mask = q_span < kv_span
        # Ragged mask: KV slots beyond kv_len are page padding.
        mask = jnp.logical_or(mask, kv_span >= kv_len)
        # Ragged mask: query rows beyond q_len are batch padding.
        mask = jnp.logical_or(mask, q_iota >= q_len)

        if sliding_window is not None:
            mask = jnp.logical_or(mask, q_span - sliding_window >= kv_span)
        if soft_cap is not None:
            attn = soft_cap * jnp.tanh(attn / soft_cap)

        attn = jnp.where(mask, mask_value, attn)
        # Keep the probabilities in f32. Rounding them to the fp8 kv dtype here
        # flushes everything below 2**-9 to zero, which silently deletes the
        # tail of the distribution without renormalizing.
        attn = jax.nn.softmax(attn, axis=-1)

        # out_i: [Q_max, actual_num_q_heads, lkv_dim]
        out_i = jnp.einsum("nqk,kl->qnl",
                           attn,
                           v_i.astype(jnp.float32),
                           preferred_element_type=jnp.float32)
        if v_scale is not None:
            out_i *= v_scale
        # Cast to the output dtype only once, after all arithmetic is done.
        return out_i.astype(out_dtype)

    # Vmap over the sequence index to perfectly preserve original loop semantics
    out_batched = jax.vmap(process_seq)(jnp.arange(B))

    return out_batched.reshape(B * Q_max, out_batched.shape[2], out_batched.shape[3])



def workload(
    ql_nope: jax.Array,
    q_pe: jax.Array,
    new_kv_c: jax.Array,
    new_k_pe: jax.Array,
    cache_kv: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
):
    return ref_mla_ragged_paged_attention(
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution
    )

def benchmark(num_warmup=5, num_iters=100):
    """Benchmark and return results dict."""
    import time
    import numpy as np
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
    return {
        'times': [round(float(t * 1000), 4) for t in times],
        'time_ms': round(float(np.mean(times) * 1000), 4),
        'std_ms': round(float(np.std(times) * 1000), 4),
        'output_shape': list(out.shape),
        'status': 'success',
    }

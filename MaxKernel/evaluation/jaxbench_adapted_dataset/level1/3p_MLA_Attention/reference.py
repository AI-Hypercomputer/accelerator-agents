# Imports
import jax
import jax.numpy as jnp
from functools import partial
import time
import numpy as np

# Initialization
def get_inputs():
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
    
    DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024

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

    dynamic_args = [
        ql_nope, q_pe, new_kv_c, new_k_pe, cache_kv, kv_lens,
        page_indices, cu_q_lens, distribution
    ]
    static_args = []

    return dynamic_args, static_args

# Computation
def cdiv(a, b):
    assert b != 0
    return (a + b - 1) // b

def align_to(x, a):
    return cdiv(x, a) * a

def update_kv_cache(
    new_kv_c: jax.Array,
    new_k_pe: jax.Array,
    cache_kv: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
) -> jax.Array:
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

            cache_kv_ = cache_kv_.at[page_idx, row, col,
                                     ..., :lkv_dim].set(new_kv_c[q_start + j])
            cache_kv_ = cache_kv_.at[page_idx, row, col, ...,
                                     lkv_dim:].set(new_k_pe[q_start + j])
            return cache_kv_

        return jax.lax.fori_loop(0, q_len, token_loop_body, cache_kv)

    cache_kv = jax.lax.fori_loop(0, distribution[-1], seq_loop_body, cache_kv)

    return cache_kv

def computation(
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
    DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)
    sm_scale = 1.0
    sliding_window = None
    soft_cap = None
    mask_value = DEFAULT_MASK_VALUE
    q_scale = None
    k_scale = None
    v_scale = None

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

        q_len = cu_q_lens[i+1] - cu_q_lens[i]
        kv_len = kv_lens[i]

        q_i = jax.lax.dynamic_slice(
            q, (q_start, 0, 0), (Q_max, q.shape[1], q.shape[2]))

        indices_start = i * pages_per_seq
        indices = jax.lax.dynamic_slice(
            page_indices, (indices_start,), (pages_per_seq,))

        gathered_kv_c = kv_c_cache[indices]
        gathered_k_pe = k_pe_cache[indices]

        flat_kv_c = gathered_kv_c.reshape(-1, lkv_dim)
        flat_k_pe = gathered_k_pe.reshape(-1, r_dim)

        k_i = jnp.concatenate([flat_kv_c, flat_k_pe], axis=-1)
        v_i = flat_kv_c

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

        mask = q_span < kv_span
        mask = jnp.logical_or(mask, kv_span >= kv_len)
        mask = jnp.logical_or(mask, q_iota >= q_len)

        if sliding_window is not None:
            mask = jnp.logical_or(mask, q_span - sliding_window >= kv_span)
        if soft_cap is not None:
            attn = soft_cap * jnp.tanh(attn / soft_cap)

        attn = jnp.where(mask, mask_value, attn)
        attn = jax.nn.softmax(attn, axis=-1)

        out_i = jnp.einsum("nqk,kl->qnl",
                           attn,
                           v_i.astype(jnp.float32),
                           preferred_element_type=jnp.float32)
        if v_scale is not None:
            out_i *= v_scale
        return out_i.astype(out_dtype)

    out_batched = jax.vmap(process_seq)(jnp.arange(B))

    return out_batched.reshape(B * Q_max, out_batched.shape[2], out_batched.shape[3])

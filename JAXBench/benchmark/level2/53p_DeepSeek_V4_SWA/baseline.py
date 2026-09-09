# Imports
import numpy as np
import time
import functools
from enum import Enum
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Initialization
CONFIG = {
    'name': 'deepseek_swa',
    'model': 'deepseek',
    'operator': 'sliding_window_attention',
    'configs': {
        "decode_8k_1k": (256, 1, 9216, 1024, 8, 512, 128),
        "prefill_first_chunk": (1, 1024, 1024, 1024, 8, 512, 128),
        "prefill_last_chunk": (1, 1024, 8192, 1024, 8, 512, 128),
        "medium_batch_decode": (128, 1, 9216, 1024, 8, 512, 128),
        "prefill_256": (1, 256, 1024, 1024, 8, 512, 128),
        "prefill_512": (1, 512, 4096, 1024, 8, 512, 128),
        "alt_page_decode": (128, 1, 9216, 256, 8, 512, 128),
        "alt_page_prefill": (1, 256, 1024, 256, 8, 512, 128),
    },
    'atol': 0.01,
    'rtol': 0.01,
}

def cdiv_val(a, b):
    assert b != 0
    return (a + b - 1) // b


def create_inputs():
    configs = CONFIG['configs']
    if isinstance(configs, dict):
        configs = [(name, *params) for name, params in configs.items()]
    inputs = []
    key = jax.random.PRNGKey(42)

    for (
        name,
        batch_size,
        q_len,
        kv_len_val,
        page_size,
        num_q_heads,
        head_dim,
        sliding_window,
    ) in configs:
        key, k1, k2, k3 = jax.random.split(key, 4)

        num_tokens = batch_size * q_len
        kv_lens = jnp.full((batch_size,), kv_len_val, dtype=jnp.int32)
        cu_q_lens = jnp.arange(0, num_tokens + 1, q_len, dtype=jnp.int32)

        pages_per_seq = cdiv_val(kv_len_val, page_size) + 2
        total_pages = batch_size * pages_per_seq
        page_indices = jnp.arange(total_pages, dtype=jnp.int32)

        q = jax.random.normal(
            k1, (num_tokens, num_q_heads, head_dim), dtype=jnp.bfloat16
        )
        new_kv = jax.random.normal(k2, (num_tokens, head_dim), dtype=jnp.bfloat16)
        attention_sinks = jax.random.normal(k3, (num_q_heads,), dtype=jnp.float32)

        num_decode_seqs = batch_size if q_len == 1 else 0
        distribution = jnp.array(
            [num_decode_seqs, num_decode_seqs, batch_size], dtype=jnp.int32
        )
        kernel_cache = jnp.zeros((total_pages, page_size * 2, 4, 128), dtype=jnp.uint8)

        args = [
            q,
            new_kv,
            kernel_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            attention_sinks,
        ]
        inputs.append(args)

    return inputs

# Computation
def cdiv(a, b):
    assert b != 0
    return (a + b - 1) // b

def align_to(x, a):
    return cdiv(x, a) * a

def get_dtype_bitwidth(dtype):
    return jax.dtypes.itemsize_bits(dtype)

def get_dtype_packing(dtype):
    bits = get_dtype_bitwidth(dtype)
    return 32 // bits

class MlaCase(Enum):
    DECODE = 0
    PREFILL = 1
    MIXED = 2

    @property
    def symbol(self):
        return {
            MlaCase.DECODE: "d",
            MlaCase.PREFILL: "p",
            MlaCase.MIXED: "m",
        }[self]

def _mla_sliding_window_ragged_paged_attention_kernel(
    kv_lens_ref,
    page_indices_ref,
    cu_q_lens_ref,
    start_end_seq_idx_ref,
    sem_ids_ref,
    bo_ids_ref,
    bkv_update_ids_ref,
    attention_sinks_ref,
    q_hbm_ref,
    new_kv_hbm_ref,
    cache_kv_hbm_ref,
    in_output_hbm_ref,
    in_l_hbm_ref,
    in_m_hbm_ref,
    o_hbm_ref,
    updated_cache_kv_hbm_ref,
    l_hbm_ref,
    m_hbm_ref,
    bkv_x2_ref,
    bq_x2_ref,
    bo_x2_ref,
    bl_x2_ref,
    bm_x2_ref,
    sems,
    l_ref,
    m_ref,
    acc_ref,
    *,
    static_q_len: int,
    sm_scale: float,
    sliding_window: int,
    logical_page_size: int,
    unnormalized_output: bool,
    q_compute_block_size: int | None,
    bkv_p,
    bq_sz,
):
    assert q_hbm_ref.shape == o_hbm_ref.shape
    assert sliding_window > 0

    _, num_q_heads, head_dim = q_hbm_ref.shape
    q_packing = get_dtype_packing(q_hbm_ref.dtype)
    assert num_q_heads % q_packing == 0
    num_q_heads_per_q_packing = num_q_heads // q_packing

    total_num_pages, physical_page_size_per_kv_packing, kv_packing, lkv_dim = cache_kv_hbm_ref.shape
    q_dtype = q_hbm_ref.dtype
    assert o_hbm_ref.dtype == q_dtype
    assert head_dim % 128 == 0
    token_bytes = head_dim * get_dtype_bitwidth(q_dtype) // 8
    slot_bytes = kv_packing * lkv_dim
    assert token_bytes % slot_bytes == 0
    slots_per_token = token_bytes // slot_bytes
    phys_tokens_per_page = physical_page_size_per_kv_packing // slots_per_token

    max_num_seqs = kv_lens_ref.shape[0]
    num_page_indices = page_indices_ref.shape[0]
    assert num_page_indices % max_num_seqs == 0
    pages_per_seq = num_page_indices // max_num_seqs

    bkv_sz = bkv_p * logical_page_size
    page_size = logical_page_size

    max_bkv_blocks = cdiv(bq_sz + sliding_window - 1, bkv_sz)
    single_bkv_block = max_bkv_blocks == 1

    start_seq_idx = start_end_seq_idx_ref[0]
    end_seq_idx = start_end_seq_idx_ref[1]
    seq_idx = pl.program_id(0) + start_seq_idx
    q_start = cu_q_lens_ref[seq_idx]
    q_end = cu_q_lens_ref[seq_idx + 1]
    q_len = q_end - q_start
    kv_len = kv_lens_ref[seq_idx]

    def flash_attention(
        q,
        kv,
        *,
        bq_idx,
        bkv_idx,
        start_offset,
    ):
        assert len(q.shape) == 2
        assert len(kv.shape) == 2
        assert q.shape[0] % num_q_heads == 0
        assert q.shape[1] == head_dim
        assert kv.shape == (bkv_sz, head_dim)
        n = q.shape[0] // num_q_heads

        if q_compute_block_size is None:
            chunk_sz = n
        else:
            chunk_sz = q_compute_block_size if n % q_compute_block_size == 0 else n
        num_chunks = n // chunk_sz

        def load_with_init(ref, init_val):
            if single_bkv_block:
                return jnp.full_like(ref, init_val)
            else:
                return jnp.where(bkv_idx == 0, jnp.full_like(ref, init_val), ref[...])

        k_span = start_offset + bkv_idx * bkv_sz + lax.broadcasted_iota(jnp.int32, (1, bkv_sz), 1)
        k_pos = start_offset + bkv_idx * bkv_sz + lax.broadcasted_iota(jnp.int32, (bkv_sz, 1), 0)
        kv = jnp.where(k_pos < kv_len, kv, 0.0)

        chunk_size = chunk_sz * num_q_heads
        for c in range(num_chunks):
            start_row = c * chunk_size
            qc = q[start_row : start_row + chunk_size]
            cl_ref = l_ref.at[start_row : start_row + chunk_size]
            cm_ref = m_ref.at[start_row : start_row + chunk_size]
            cacc_ref = acc_ref.at[start_row : start_row + chunk_size]

            s = jnp.einsum("nd,md->nm", qc, kv, preferred_element_type=jnp.float32)
            s *= sm_scale

            q_span = kv_len - q_len + bq_idx * bq_sz + (start_row + lax.broadcasted_iota(jnp.int32, (chunk_size, 1), 0)) // num_q_heads
            keep = (q_span - k_span).astype(jnp.uint32) < jnp.uint32(sliding_window)

            s = jnp.where(keep, s, jnp.finfo(s.dtype).min)
            s_rowmax = jnp.max(s, axis=1, keepdims=True)
            m_prev = load_with_init(cm_ref, jnp.finfo(jnp.float32).min)
            m_curr = jnp.maximum(m_prev, s_rowmax)
            cm_ref[...] = m_curr
            p = jnp.exp(s - broadcast_minor(m_curr, s.shape))
            p = jnp.where(keep, p, 0.0)

            pv = jnp.einsum("nm,md->nd", p, kv, preferred_element_type=jnp.float32)

            p_rowsum = jnp.sum(p, axis=1, keepdims=True)
            exp_m_diff = jnp.exp(m_prev - m_curr)
            l_prev = load_with_init(cl_ref, 0.0)
            l_curr = exp_m_diff * l_prev + p_rowsum
            cl_ref[...] = l_curr
            o_prev = load_with_init(cacc_ref, 0.0)
            o_curr = broadcast_minor(exp_m_diff, o_prev.shape) * o_prev + pv
            cacc_ref[...] = o_curr

    def _async_copy(src, dst, sem, wait):
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

    def _get_kv_len(seq_idx):
        return jnp.where(seq_idx < end_seq_idx, kv_lens_ref[seq_idx], 0)

    def _get_q_len(seq_idx):
        return jnp.where(seq_idx < end_seq_idx, cu_q_lens_ref[seq_idx + 1] - cu_q_lens_ref[seq_idx], 0)

    def _start_offset(seq_idx, bq_idx):
        return jnp.maximum(_get_kv_len(seq_idx) - _get_q_len(seq_idx) + bq_idx * bq_sz - sliding_window + 1, 0)

    def _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, start_offset, *, wait=False):
        sem = sems.at[0, bkv_sem_idx]
        bkv_vmem_ref = bkv_x2_ref.at[bkv_sem_idx]

        reshaped_cache_hbm_ref = cache_kv_hbm_ref.reshape(total_num_pages * phys_tokens_per_page, slots_per_token * kv_packing, lkv_dim)

        kv_len = kv_lens_ref[seq_idx]
        kv_len_start = start_offset + bkv_idx * bkv_sz
        kv_p_start = kv_len_start // page_size
        page_off = kv_len_start - kv_p_start * page_size

        q_start = cu_q_lens_ref[seq_idx]
        q_end = cu_q_lens_ref[seq_idx + 1]
        q_len = q_end - q_start

        kv_left = jnp.maximum(kv_len - kv_len_start, 0)
        kv_left_frm_cache = jnp.maximum(kv_left - q_len, 0)
        kv_left_frm_new = kv_left - kv_left_frm_cache

        bkv_sz_frm_cache = jnp.minimum(kv_left_frm_cache, bkv_sz)
        bkv_sz_frm_new = jnp.minimum(bkv_sz - bkv_sz_frm_cache, kv_left_frm_new)
        page_indices_offset = seq_idx * pages_per_seq + kv_p_start

        new_kv_len_start = q_end - kv_left_frm_new
        dma_bkv_sz = bkv_sz_frm_cache + bkv_sz_frm_new

        if not wait:
            wait_update_kv_cache(bkv_sem_idx)

            for i in range(bkv_p + 1):
                if i == 0:
                    in_page_off = page_off
                    vmem_off = jnp.int32(0)
                    avail = page_size - page_off
                else:
                    in_page_off = jnp.int32(0)
                    vmem_off = i * page_size - page_off
                    avail = jnp.int32(page_size)
                sz = jnp.clip(bkv_sz_frm_cache - vmem_off, 0, avail)
                page_idx = jnp.minimum(page_indices_offset + i, num_page_indices - 1)
                _async_copy(
                    reshaped_cache_hbm_ref.at[pl.ds(page_indices_ref[page_idx] * phys_tokens_per_page + in_page_off, sz)],
                    bkv_vmem_ref.at[pl.ds(vmem_off, sz)],
                    sem,
                    wait,
                )

            _async_copy(
                new_kv_hbm_ref.at[pl.ds(new_kv_len_start, bkv_sz_frm_new)],
                bkv_vmem_ref.at[pl.ds(bkv_sz_frm_cache, bkv_sz_frm_new)],
                sem,
                wait,
            )

        else:
            dst_kv = bkv_vmem_ref.at[pl.ds(0, dma_bkv_sz)]
            _async_copy(src=dst_kv, dst=dst_kv, sem=sem, wait=True)

        return kv_len_start + bkv_sz_frm_cache, bkv_sz_frm_new, bkv_sz_frm_cache

    def _update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz, *, in_vmem_start=0, wait=False):
        sem = sems.at[3, bkv_sem_idx]
        bkv_vmem_ref = bkv_x2_ref.at[bkv_sem_idx]

        update_kv_packing_iters = update_sz

        reshaped_cache_kv_hbm_ref = updated_cache_kv_hbm_ref.reshape(total_num_pages * phys_tokens_per_page, slots_per_token * kv_packing, lkv_dim)

        if not wait:
            kv_p_start = offset // page_size
            kv_p_end = cdiv(offset + update_sz, page_size)
            start_word_in_page = offset % page_size
            start_word_in_vmem = in_vmem_start
            words_to_transfer = update_kv_packing_iters
            page_indices_offset = seq_idx * pages_per_seq + kv_p_start

            def loop_body(i, states):
                curr_word_in_page, words_to_transfer, curr_word_in_vmem = states
                sz = jnp.minimum(page_size - curr_word_in_page, words_to_transfer)
                page_idx = page_indices_ref[page_indices_offset + i]

                _async_copy(
                    bkv_vmem_ref.at[pl.ds(curr_word_in_vmem, sz)],
                    reshaped_cache_kv_hbm_ref.at[pl.ds(page_idx * phys_tokens_per_page + curr_word_in_page, sz)],
                    sem,
                    wait=False,
                )
                return 0, words_to_transfer - sz, curr_word_in_vmem + sz

            lax.fori_loop(
                0,
                kv_p_end - kv_p_start,
                loop_body,
                (start_word_in_page, words_to_transfer, start_word_in_vmem),
                unroll=False,
            )
        else:
            dma_sz_words = update_kv_packing_iters
            dst_kv = bkv_vmem_ref.at[pl.ds(0, dma_sz_words)]
            _async_copy(src=dst_kv, dst=dst_kv, sem=sem, wait=True)

    def _fetch_bq(seq_idx, bq_idx, bq_sem_idx, *, wait=False):
        sem = sems.at[1, bq_sem_idx]
        bq_vmem_ref = bq_x2_ref.at[bq_sem_idx]

        q_len_start = cu_q_lens_ref[seq_idx] + bq_idx * bq_sz
        q_end = cu_q_lens_ref[seq_idx + 1]
        sz = jnp.minimum(bq_sz, q_end - q_len_start)

        _async_copy(q_hbm_ref.at[pl.ds(q_len_start, sz)], bq_vmem_ref.at[pl.ds(0, sz)], sem, wait)

    def _send_bo(seq_idx, bo_idx, bo_sem_idx, *, wait=False):
        sem = sems.at[2, bo_sem_idx]
        vmem_ref = bo_x2_ref.at[bo_sem_idx]
        q_len_start = cu_q_lens_ref[seq_idx] + bo_idx * bq_sz
        q_end = cu_q_lens_ref[seq_idx + 1]
        sz = jnp.minimum(bq_sz, q_end - q_len_start)

        _async_copy(vmem_ref.at[pl.ds(0, sz)], o_hbm_ref.at[pl.ds(q_len_start, sz)], sem, wait)

    def _send_l(seq_idx, bo_idx, bo_sem_idx, *, wait=False):
        sem = sems.at[4, bo_sem_idx]
        vmem_ref = bl_x2_ref.at[bo_sem_idx]
        q_len_start = cu_q_lens_ref[seq_idx] + bo_idx * bq_sz
        q_end = cu_q_lens_ref[seq_idx + 1]
        sz = jnp.minimum(bq_sz, q_end - q_len_start)

        _async_copy(vmem_ref.at[pl.ds(0, sz)], l_hbm_ref.at[pl.ds(q_len_start, sz)], sem, wait)

    def _send_m(seq_idx, bo_idx, bo_sem_idx, *, wait=False):
        sem = sems.at[5, bo_sem_idx]
        vmem_ref = bm_x2_ref.at[bo_sem_idx]
        q_len_start = cu_q_lens_ref[seq_idx] + bo_idx * bq_sz
        q_end = cu_q_lens_ref[seq_idx + 1]
        sz = jnp.minimum(bq_sz, q_end - q_len_start)

        _async_copy(vmem_ref.at[pl.ds(0, sz)], m_hbm_ref.at[pl.ds(q_len_start, sz)], sem, wait)

    def start_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, start_offset):
        return _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, start_offset)

    def wait_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, start_offset):
        return _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, start_offset, wait=True)

    def start_fetch_bq(seq_idx, bq_idx, bq_sem_idx):
        return _fetch_bq(seq_idx, bq_idx, bq_sem_idx)

    def wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx):
        return _fetch_bq(seq_idx, bq_idx, bq_sem_idx, wait=True)

    def start_send_bo(seq_idx, bo_idx, bo_sem_idx):
        bo_ids_ref[bo_sem_idx] = seq_idx
        bo_ids_ref[bo_sem_idx + 2] = bo_idx
        _send_bo(seq_idx, bo_idx, bo_sem_idx)
        _send_l(seq_idx, bo_idx, bo_sem_idx)
        _send_m(seq_idx, bo_idx, bo_sem_idx)

    def wait_send_bo(bo_sem_idx):
        old_seq_idx = bo_ids_ref[bo_sem_idx]
        old_bo_idx = bo_ids_ref[bo_sem_idx + 2]

        @pl.when(jnp.logical_and(0 <= old_seq_idx, old_seq_idx <= seq_idx))
        def _():
            _send_bo(old_seq_idx, old_bo_idx, bo_sem_idx, wait=True)
            _send_l(old_seq_idx, old_bo_idx, bo_sem_idx, wait=True)
            _send_m(old_seq_idx, old_bo_idx, bo_sem_idx, wait=True)

    def start_update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz, vmem_start):
        bkv_update_ids_ref[bkv_sem_idx] = seq_idx
        bkv_update_ids_ref[bkv_sem_idx + 2] = offset
        bkv_update_ids_ref[bkv_sem_idx + 4] = update_sz
        _update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz, in_vmem_start=vmem_start)

    def wait_update_kv_cache(bkv_sem_idx):
        update_sz = bkv_update_ids_ref[bkv_sem_idx + 4]

        @pl.when(update_sz > 0)
        def _():
            seq_idx = bkv_update_ids_ref[bkv_sem_idx]
            offset = bkv_update_ids_ref[bkv_sem_idx + 2]
            bkv_update_ids_ref[bkv_sem_idx + 4] = 0
            _update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz, wait=True)

    def load_bq(bq_sem_idx):
        q_ref = bq_x2_ref.bitcast(jnp.uint32).at[bq_sem_idx].reshape(bq_sz * num_q_heads_per_q_packing, head_dim)
        q = pltpu.bitcast(q_ref[: bq_sz * num_q_heads_per_q_packing], q_dtype).reshape(bq_sz * num_q_heads, head_dim)
        return q

    def load_bkv(bkv_sem_idx, bkv_idx, start_offset):
        bkv_u8 = bkv_x2_ref.at[bkv_sem_idx][...]
        bkv = pltpu.bitcast(bkv_u8, jnp.bfloat16).reshape(bkv_sz, head_dim)
        return bkv

    def broadcast_minor(src, shape):
        if src.shape == shape:
            return src
        assert src.shape[:-1] == shape[:-1]
        assert src.shape[-1] % 128 == 0
        target_minor = align_to(shape[-1], src.shape[-1])
        return jnp.concatenate([src for _ in range(target_minor // src.shape[-1])], axis=-1)[..., : shape[-1]]

    def process():
        if static_q_len is None:
            num_bq = jnp.maximum(1, cdiv(q_len, bq_sz))
        else:
            num_bq = jnp.maximum(1, cdiv(static_q_len, bq_sz))

        def get_next_bq_ids(seq_idx, bq_idx, bq_sem_idx):
            next_bq_idx = bq_idx + 1
            is_last_bq = next_bq_idx == num_bq
            next_bq_idx = lax.select(is_last_bq, 0, next_bq_idx)
            next_seq_idx = lax.select(is_last_bq, seq_idx + 1, seq_idx)
            next_bq_sem_idx = lax.select(bq_sem_idx == 0, 1, 0)
            return next_seq_idx, next_bq_idx, next_bq_sem_idx

        def compute_with_bq(bq_idx, _):
            cur_start_offset = _start_offset(seq_idx, bq_idx)
            start_bkv_idx = 0
            if single_bkv_block:
                end_bkv_idx = 1
            else:
                end_bkv_idx = jnp.maximum(cdiv(jnp.minimum(kv_len - q_len + (bq_idx + 1) * bq_sz, kv_len) - cur_start_offset, bkv_sz), 1)

            def get_next_bkv_ids(seq_idx, bq_idx, bkv_idx, bkv_sem_idx):
                next_bkv_idx = bkv_idx + 1
                is_last_bkv = next_bkv_idx == end_bkv_idx
                next_bq_idx = lax.select(is_last_bkv, bq_idx + 1, bq_idx)
                is_last_bq = next_bq_idx == num_bq
                next_bq_idx = lax.select(is_last_bq, 0, next_bq_idx)
                next_seq_idx = lax.select(is_last_bq, seq_idx + 1, seq_idx)
                next_bkv_idx = lax.select(is_last_bkv, 0, next_bkv_idx)
                next_bkv_sem_idx = lax.select(bkv_sem_idx == 0, 1, 0)
                return next_seq_idx, next_bq_idx, next_bkv_idx, next_bkv_sem_idx

            bq_sem_idx = sem_ids_ref[0]
            next_seq_idx, next_bq_idx, next_bq_sem_idx = get_next_bq_ids(seq_idx, bq_idx, bq_sem_idx)

            @pl.when(next_seq_idx < end_seq_idx)
            def prefetch_next_bq():
                sem_ids_ref[0] = next_bq_sem_idx
                start_fetch_bq(next_seq_idx, next_bq_idx, next_bq_sem_idx)

            def compute_with_bkv(bkv_idx, _):
                bkv_sem_idx = sem_ids_ref[1]
                next_seq_idx, next_bq_idx, next_bkv_idx, next_bkv_sem_idx = get_next_bkv_ids(seq_idx, bq_idx, bkv_idx, bkv_sem_idx)

                @pl.when(next_seq_idx < end_seq_idx)
                def prefetch_next_bkv():
                    sem_ids_ref[1] = next_bkv_sem_idx
                    next_start_offset = _start_offset(next_seq_idx, next_bq_idx)
                    start_fetch_bkv(next_seq_idx, next_bkv_idx, next_bkv_sem_idx, next_start_offset)

                offset, update_sz, vmem_start = wait_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, cur_start_offset)

                @pl.when(update_sz > 0)
                def update_cur_bkv_to_cache():
                    start_update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz, vmem_start)

                bkv = load_bkv(bkv_sem_idx, bkv_idx, cur_start_offset)
                bq = load_bq(bq_sem_idx)

                flash_attention(bq, bkv, bq_idx=bq_idx, bkv_idx=bkv_idx, start_offset=cur_start_offset)

            wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx)
            if single_bkv_block:
                compute_with_bkv(0, None)
            else:
                lax.fori_loop(start_bkv_idx, end_bkv_idx, compute_with_bkv, None, unroll=False)

            acc = acc_ref[...]

            if unnormalized_output:
                l = broadcast_minor(l_ref[...], acc.shape)
                out = acc.astype(q_dtype)
            else:
                attention_sinks = jnp.concat([attention_sinks_ref[...] for _ in range(bq_sz)])[..., None]
                exp_attention_sinks = jnp.exp(attention_sinks - m_ref[...])
                l = l_ref[...] + exp_attention_sinks
                l = broadcast_minor(l, acc.shape)
                out = lax.div(acc, l) if q_dtype == jnp.float32 else (acc * pl.reciprocal(l, approx=True)).astype(q_dtype)

            bo_sem_idx = sem_ids_ref[2]
            sem_ids_ref[2] = lax.select(bo_sem_idx == 0, 1, 0)
            wait_send_bo(bo_sem_idx)

            bo_x2_ref.at[bo_sem_idx].bitcast(jnp.int32).reshape(bq_sz * num_q_heads_per_q_packing, head_dim)[...] = pltpu.bitcast(out, jnp.int32)
            bl_x2_ref.at[bo_sem_idx][:bq_sz, :num_q_heads] = l_ref[..., 0].reshape(bq_sz, num_q_heads)
            bm_x2_ref.at[bo_sem_idx][:bq_sz, :num_q_heads] = m_ref[..., 0].reshape(bq_sz, num_q_heads)

            start_send_bo(seq_idx, bq_idx, bo_sem_idx)

        lax.fori_loop(0, num_bq, compute_with_bq, None, unroll=False)

    @pl.when(seq_idx == start_seq_idx)
    def prologue():
        start_fetch_bq(start_seq_idx, 0, 0)
        start_fetch_bkv(start_seq_idx, 0, 0, _start_offset(start_seq_idx, 0))

    process()

    @pl.when(seq_idx == end_seq_idx - 1)
    def epilogue():
        for i in range(2):
            wait_send_bo(i)
            wait_update_kv_cache(i)

def prepare_q_inputs(q: jax.Array):
    max_num_tokens, actual_num_q_heads, actual_head_dim = q.shape
    q_packing = get_dtype_packing(q.dtype)
    num_q_heads = align_to(actual_num_q_heads, q_packing)
    head_dim = align_to(actual_head_dim, 128)
    q = jnp.pad(
        q,
        ((0, 0), (0, num_q_heads - actual_num_q_heads), (0, head_dim - actual_head_dim)),
        constant_values=0,
    )
    return q

def prepare_kv_inputs(kv: jax.Array):
    assert kv.dtype == jnp.bfloat16
    tokens, head_dim = kv.shape
    assert head_dim % 128 == 0
    kv_u16 = jax.lax.bitcast_convert_type(kv, jnp.uint16)
    kv_hi = ((kv_u16 >> 8) & 0xFF).astype(jnp.uint8)
    kv_lo = (kv_u16 & 0xFF).astype(jnp.uint8)
    nb = head_dim // 128
    kv_hi = kv_hi.reshape(tokens, nb, 128)
    kv_lo = kv_lo.reshape(tokens, nb, 128)
    interleaved = jnp.stack([kv_lo, kv_hi], axis=2)
    return interleaved.reshape(tokens, head_dim * 2)

def prepare_outputs(out, actual_num_q_heads: int, actual_head_dim: int):
    return out[:, :actual_num_q_heads, :actual_head_dim]

@functools.partial(
    jax.jit,
    static_argnames=(
        "sm_scale",
        "sliding_window",
        "chunk_prefill_size",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "vmem_limit_bytes",
        "logical_page_size",
        "unnormalized_output",
        "q_compute_block_size",
    ),
    donate_argnames=("cache_kv",),
)
def mla_sliding_window_ragged_paged_attention(
    q: jax.Array,
    new_kv: jax.Array,
    cache_kv: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    attention_sinks: jax.Array,
    *,
    sm_scale: float = 1.0,
    sliding_window: int,
    logical_page_size: int,
    chunk_prefill_size: int | None = None,
    num_kv_pages_per_block: tuple[int, int, int] | int | None = None,
    num_queries_per_block: tuple[int, int, int] | int | None = None,
    q_compute_block_size: int | None = None,
    vmem_limit_bytes: int = 100 * 1024 * 1024,
    unnormalized_output: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    if num_kv_pages_per_block is None or num_queries_per_block is None:
        raise ValueError("num_kv_pages_per_block and num_queries_per_block must be specified.")
    
    if isinstance(num_kv_pages_per_block, int):
        num_kv_pages_per_blocks = [num_kv_pages_per_block for _ in range(3)]
    else:
        num_kv_pages_per_blocks = num_kv_pages_per_block

    if isinstance(num_queries_per_block, int):
        num_queries_per_blocks = [num_queries_per_block for _ in range(3)]
    else:
        num_queries_per_blocks = num_queries_per_block

    _, actual_num_q_heads, actual_head_dim = q.shape

    q = prepare_q_inputs(q)
    attention_sinks = jnp.pad(
        attention_sinks,
        (0, q.shape[1] - actual_num_q_heads),
        constant_values=jnp.finfo(attention_sinks.dtype).min,
    )
    assert new_kv.dtype == jnp.bfloat16
    assert cache_kv.dtype == jnp.uint8
    head_dim = q.shape[-1]
    _, physical_page_size_per_kv_packing, kv_packing, lkv_dim = cache_kv.shape

    slot_bytes = kv_packing * lkv_dim
    token_bytes = head_dim * get_dtype_bitwidth(new_kv.dtype) // 8
    assert token_bytes % slot_bytes == 0
    slots_per_token = token_bytes // slot_bytes
    phys_tokens_per_page = physical_page_size_per_kv_packing // slots_per_token

    new_kv = prepare_kv_inputs(new_kv)
    new_kv = new_kv.reshape(new_kv.shape[0], slots_per_token * kv_packing, lkv_dim)
    assert logical_page_size <= phys_tokens_per_page

    _, num_q_heads, _ = q.shape
    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    assert num_page_indices % max_num_seqs == 0

    def run_mla_kernel(
        q: jax.Array,
        new_kv: jax.Array,
        cache_kv: jax.Array,
        kv_lens: jax.Array,
        page_indices: jax.Array,
        cu_q_lens: jax.Array,
        start_seq_idx: jax.Array,
        end_seq_idx: jax.Array,
        in_output: jax.Array,
        in_l: jax.Array,
        in_m: jax.Array,
        attention_sinks: jax.Array,
        static_q_len: int | None,
        unnormalized_output: bool,
        num_kv_pages_per_block: int,
        num_queries_per_block: int,
        case: MlaCase = MlaCase.MIXED,
    ):
        bkv_p = num_kv_pages_per_block
        if static_q_len is not None:
            bq_sz = min(num_queries_per_block, static_q_len)
        else:
            bq_sz = num_queries_per_block
        bkv_sz = bkv_p * logical_page_size
        grid = (end_seq_idx - start_seq_idx,)

        in_specs = [
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ]

        out_specs = [
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ]

        bkv_double_buf = pltpu.VMEM((2, bkv_sz, slots_per_token * kv_packing, lkv_dim), cache_kv.dtype)
        bq_double_bufq = pltpu.VMEM((2, bq_sz, num_q_heads, head_dim), q.dtype)
        bo_double_buf = bq_double_bufq

        num_l_heads = align_to(num_q_heads, 128)
        bl_double_buf = pltpu.VMEM((2, bq_sz, num_l_heads), jnp.float32)
        bm_double_buf = bl_double_buf

        l_scratch = pltpu.VMEM((bq_sz * num_q_heads, 128), jnp.float32)
        m_scratch = l_scratch
        acc_scratch = pltpu.VMEM((bq_sz * num_q_heads, head_dim), jnp.float32)

        scratch_shapes = [
            bkv_double_buf,
            bq_double_bufq,
            bo_double_buf,
            bl_double_buf,
            bm_double_buf,
            pltpu.SemaphoreType.DMA((6, 2)),
            l_scratch,
            m_scratch,
            acc_scratch,
        ]

        scalar_prefetches = (
            kv_lens,
            page_indices,
            cu_q_lens,
            jnp.array([start_seq_idx, end_seq_idx], jnp.int32),
            jnp.zeros((3,), jnp.int32),
            jnp.full((4,), -1, jnp.int32),
            jnp.full((6,), -1, jnp.int32),
        )

        scope_name = f"SWA-{case.symbol}-bq_{bq_sz}-bkvp_{bkv_p}"
        kernel = jax.named_scope(scope_name)(
            pl.pallas_call(
                functools.partial(
                    _mla_sliding_window_ragged_paged_attention_kernel,
                    sm_scale=sm_scale,
                    sliding_window=sliding_window,
                    static_q_len=static_q_len,
                    bq_sz=bq_sz,
                    bkv_p=bkv_p,
                    logical_page_size=logical_page_size,
                    unnormalized_output=unnormalized_output,
                    q_compute_block_size=q_compute_block_size,
                ),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=len(scalar_prefetches),
                    in_specs=in_specs,
                    out_specs=out_specs,
                    grid=grid,
                    scratch_shapes=scratch_shapes,
                ),
                compiler_params=pltpu.CompilerParams(
                    dimension_semantics=("arbitrary",),
                    vmem_limit_bytes=vmem_limit_bytes,
                    disable_bounds_checks=True,
                ),
                out_shape=[
                    jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
                    jax.ShapeDtypeStruct(shape=cache_kv.shape, dtype=cache_kv.dtype),
                    jax.ShapeDtypeStruct(shape=(q.shape[0], num_l_heads), dtype=jnp.float32),
                    jax.ShapeDtypeStruct(shape=(q.shape[0], num_l_heads), dtype=jnp.float32),
                ],
                input_output_aliases={
                    11: 0,
                    10: 1,
                    12: 2,
                    13: 3,
                },
                name=scope_name,
            )
        )
        return kernel(
            *scalar_prefetches,
            attention_sinks,
            q,
            new_kv,
            cache_kv,
            in_output,
            in_l,
            in_m,
        )

    num_l_heads = align_to(num_q_heads, 128)
    if unnormalized_output:
        l = jnp.zeros((q.shape[0], num_l_heads), dtype=jnp.float32)
        m = jnp.full((q.shape[0], num_l_heads), jnp.finfo(jnp.float32).min, dtype=jnp.float32)
        in_output = jnp.zeros_like(q)
    else:
        l = jnp.zeros((q.shape[0], num_l_heads), dtype=jnp.float32)
        m = jnp.zeros((q.shape[0], num_l_heads), dtype=jnp.float32)
        in_output = jnp.zeros_like(q)
        
    output, updated_kv, out_l, out_m = run_mla_kernel(
        q,
        new_kv,
        cache_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        num_kv_pages_per_block=num_kv_pages_per_blocks[0],
        num_queries_per_block=num_queries_per_blocks[0],
        start_seq_idx=jnp.array(0),
        end_seq_idx=distribution[0],
        in_output=in_output,
        in_l=l,
        in_m=m,
        attention_sinks=attention_sinks,
        static_q_len=1,
        unnormalized_output=unnormalized_output,
        case=MlaCase.DECODE,
    )

    if chunk_prefill_size is not None:
        output, updated_kv, out_l, out_m = run_mla_kernel(
            q,
            new_kv,
            updated_kv,
            kv_lens,
            page_indices,
            cu_q_lens,
            num_kv_pages_per_block=num_kv_pages_per_blocks[1],
            num_queries_per_block=num_queries_per_blocks[1],
            start_seq_idx=distribution[0],
            end_seq_idx=distribution[1],
            in_output=output,
            in_l=out_l,
            in_m=out_m,
            attention_sinks=attention_sinks,
            static_q_len=chunk_prefill_size,
            unnormalized_output=unnormalized_output,
            case=MlaCase.PREFILL,
        )

    output, updated_kv, out_l, out_m = run_mla_kernel(
        q,
        new_kv,
        updated_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        num_kv_pages_per_block=num_kv_pages_per_blocks[2],
        num_queries_per_block=num_queries_per_blocks[2],
        start_seq_idx=distribution[1],
        end_seq_idx=distribution[2],
        in_output=output,
        in_l=out_l,
        in_m=out_m,
        attention_sinks=attention_sinks,
        static_q_len=None,
        unnormalized_output=unnormalized_output,
        case=MlaCase.MIXED,
    )
    
    output = prepare_outputs(output, actual_num_q_heads, actual_head_dim)
    out_l = out_l[:, :actual_num_q_heads]
    return output, updated_kv, out_l, out_m

def workload(
    q: jax.Array,
    new_kv: jax.Array,
    cache_kv: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    attention_sinks: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    sm_scale = 1.0
    sliding_window = 128
    logical_page_size = 128
    chunk_prefill_size = None
    num_kv_pages_per_block = 2
    num_queries_per_block = 32
    q_compute_block_size = 2
    vmem_limit_bytes = 100 * 1024 * 1024
    unnormalized_output = True

    if cache_kv.shape[1] != logical_page_size * 2:
        total_pages = cache_kv.shape[0]
        cache_kv = jnp.zeros((total_pages, logical_page_size * 2, 4, 128), dtype=cache_kv.dtype)

    return mla_sliding_window_ragged_paged_attention(
        q,
        new_kv,
        cache_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        attention_sinks,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        logical_page_size=logical_page_size,
        chunk_prefill_size=chunk_prefill_size,
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block,
        q_compute_block_size=q_compute_block_size,
        vmem_limit_bytes=vmem_limit_bytes,
        unnormalized_output=unnormalized_output,
    )

def benchmark(num_warmup=5, num_iters=100):
    """Benchmark and return results dict."""
    inputs_list = create_inputs()
    fn = jax.jit(workload)
    times_list = []
    std_ms_list = []
    output_shape_list = []
    for inputs in inputs_list:
        for _ in range(num_warmup):
            out = fn(*inputs)
            jax.block_until_ready(out)
        times = []
        for _ in range(num_iters):
            t0 = time.perf_counter()
            out = fn(*inputs)
            jax.block_until_ready(out)
            times.append(time.perf_counter() - t0)
        times = np.array(times) * 1000
        avg = float(np.mean(times))
        times_list.append(round(avg, 4))
        std_ms_list.append(round(float(np.std(times)), 4))
        if hasattr(out, 'shape'):
            out_shape = list(out.shape)
        elif isinstance(out, (tuple, list)):
            out_shape = [list(x.shape) if hasattr(x, 'shape') else [] for x in out]
        else:
            out_shape = []
        output_shape_list.append(out_shape)
    return {
        'name': CONFIG['name'],
        'model': CONFIG['model'],
        'operator': CONFIG['operator'],
        'config': {k: v for k, v in CONFIG.items() if k not in ('name', 'model', 'operator', 'atol', 'rtol')},
        'time_ms': times_list,
        'std_ms': std_ms_list,
        'output_shape': output_shape_list,
        'status': 'success',
    }


if __name__ == '__main__':
    import json
    print(json.dumps(benchmark()))
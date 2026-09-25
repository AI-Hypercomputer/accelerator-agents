"""DeepSeek V4 Compressed Sparse Attention (CSA / Sparse MLA) — Pallas TPU Kernel.

Self-contained implementation.
"""
import sys
import types
import jax
import jax.numpy as jnp
import numpy as np

# ==============================================================================
# Inlined csa_gather.py
# ==============================================================================
import functools
from typing import Any

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
import jax.numpy as jnp


def main_kernel(
    nope_in_hbm_ref: Any,
    rope_in_hbm_ref: Any,
    indices_hbm_ref: Any,
    nope_out_hbm_ref: Any,
    rope_out_hbm_ref: Any,
    nope_sem: Any,
    *,
    core_axis_name: str,
    subcore_axis_name: str,
    num_row_subchunks: int,
    num_streams: int,
):
  tpu_info = pltpu.get_tpu_info()
  sc_info = tpu_info.sparse_core
  assert sc_info is not None
  num_simd_lanes = sc_info.num_lanes
  num_cores = jax.lax.axis_size((core_axis_name, subcore_axis_name))
  row_subchunk_size = num_simd_lanes
  row_chunk_size = row_subchunk_size * num_row_subchunks
  block_size = row_chunk_size * num_cores
  num_blocks = pl.cdiv(indices_hbm_ref.shape[0], block_size)

  # Inputs are 8-bit;
  # nope output stays uint8 (4/int32), rope is unpacked to bf16 (2/int32).
  in_bits = jax.dtypes.itemsize_bits(nope_in_hbm_ref.dtype)
  in_packing = 32 // in_bits
  in_mask = (1 << in_bits) - 1  # 0xFF for 8-bit.
  rope_out_bits = jax.dtypes.itemsize_bits(rope_out_hbm_ref.dtype)
  rope_out_packing = 32 // rope_out_bits
  core_index = lax.axis_index((core_axis_name, subcore_axis_name))

  # SparseCore gather 32-bit words
  nope_in_i32 = nope_in_hbm_ref.bitcast(jnp.int32)
  rope_in_i32 = rope_in_hbm_ref.bitcast(jnp.int32)
  nope_out_i32 = nope_out_hbm_ref.bitcast(jnp.int32)
  rope_out_i32 = rope_out_hbm_ref.bitcast(jnp.int32)

  nope_in_cols = nope_in_i32.shape[1]
  rope_in_cols = rope_in_i32.shape[1]
  rope_out_cols = rope_out_hbm_ref.shape[1]

  def process_rope(gather_ref, out_ref, idx_sub, out_row_base=0):
    # one (1, 128) uint8 is one token's rope data, which encodes 64 bf16
    # values. (0, 64) are the high bits for bf16 data, (64, 128) are the low.
    half = rope_out_cols
    col_hi = pl.ds(0, half)
    col_lo = pl.ds(half, half)

    def bf16_bits(k):
      sub = lax.rem(idx_sub[k], in_packing)
      hi = jnp.bitwise_and(
          jnp.bitwise_right_shift(
              gather_ref[pl.ds(k, 1), col_hi], in_bits * sub
          ),
          in_mask,
      )
      lo = jnp.bitwise_and(
          jnp.bitwise_right_shift(
              gather_ref[pl.ds(k, 1), col_lo], in_bits * sub
          ),
          in_mask,
      )
      return jnp.bitwise_or(jnp.left_shift(hi, in_bits), lo)

    for t in range(num_simd_lanes // rope_out_packing):
      packed = jnp.zeros((1, half), dtype=jnp.int32)
      for pk in range(rope_out_packing):
        k = t * rope_out_packing + pk
        packed = jnp.bitwise_or(
            packed, jnp.left_shift(bf16_bits(k), pk * rope_out_bits)
        )
      out_ref[pl.ds(out_row_base + t, 1), pl.ds(0, half)] = packed

  def outer_pipeline(idx_ref):
    b = pl.program_id(0)
    out_row_base = (b * num_cores + core_index) * num_row_subchunks

    # Subchunk handled by stream `s` at inner step `r`. `num_streams`
    # independent `pl.Indirect` gathers run concurrently per step, keeping
    # several gather DMAs in flight to raise effective read bandwidth.
    def subchunk(r, s):
      return r * num_streams + s

    def idx_window(r, s):
      return idx_ref[
          pl.ds(subchunk(r, s) * row_subchunk_size, row_subchunk_size)
      ]

    # int32 rows produced per stream in the rope output (2 tokens per row).
    rope_rows_per_stream = row_subchunk_size // rope_out_packing

    def _body(*refs):
      r = pl.program_id(0)
      nope_g = refs[0 * num_streams : 1 * num_streams]
      rope_g = refs[1 * num_streams : 2 * num_streams]
      rope_o = refs[2 * num_streams]

      # nope needs no vector work at all. The nope output keeps the
      # cache's raw per-token layout. The gathered row *is* the output
      # row.
      #
      # DMA the gather buffer straight to HBM rather than routing it
      # through a pipeline output buffer: staging it there costs a
      # VMEM->VMEM copy.
      nope_copies = []
      for s in range(num_streams):
        out_row = (out_row_base + subchunk(r, s)) * row_subchunk_size
        copy = pltpu.make_async_copy(
            nope_g[s],
            nope_out_i32.at[pl.ds(out_row, row_subchunk_size)],
            nope_sem.at[s],
        )
        copy.start()
        nope_copies.append(copy)

      for s in range(num_streams):
        process_rope(
            gather_ref=rope_g[s],
            out_ref=rope_o,
            idx_sub=idx_window(r, s),
            out_row_base=s * rope_rows_per_stream,
        )

      # Wait for all nope DMAs to complete.
      for copy in nope_copies:
        copy.wait()

    # Have multiple parallel `pl.Indirect` to hide random access read
    # latency. Output contiguous memory access, multiple output parallel
    # DMAs not help with performance.

    # nope: gather int32 row == index (1 int32 row per entry).
    nope_in_specs = tuple(
        pl.BlockSpec(
            (pl.Indirect(row_subchunk_size), nope_in_cols),
            lambda r, s=s: (idx_window(r, s), 0),
        )
        for s in range(num_streams)
    )
    # rope: gather int32 row == index // in_packing (in_packing entries/row).
    rope_in_specs = tuple(
        pl.BlockSpec(
            (pl.Indirect(row_subchunk_size), rope_in_cols),
            lambda r, s=s: (lax.div(idx_window(r, s), in_packing), 0),
        )
        for s in range(num_streams)
    )

    # One merged rope output block, covering all `num_streams` subchunks.
    #
    # nope has no pipeline output block -- `_body` DMAs it to HBM directly (
    # this approach has better performance based on measured performance in
    # microbenchmarks).
    rope_out_spec = pl.BlockSpec(
        (num_streams * rope_rows_per_stream, rope_out_cols),
        lambda r: (out_row_base // num_streams + r, 0),
    )
    pltpu.emit_pipeline(
        _body,
        grid=(num_row_subchunks // num_streams,),
        in_specs=nope_in_specs + rope_in_specs,
        out_specs=(rope_out_spec,),
    )(
        *([nope_in_i32] * num_streams),
        *([rope_in_i32] * num_streams),
        rope_out_i32,
    )

  pltpu.emit_pipeline(
      outer_pipeline,
      grid=(num_blocks,),
      in_specs=pl.BlockSpec(
          (row_chunk_size,),
          lambda b: (b * num_cores + core_index,),
      ),
  )(indices_hbm_ref)


@functools.partial(jax.jit)
def csa_gather(
    nope_cache: jax.Array,
    rope_cache: jax.Array,
    indices: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Fused SparseCore gather of the nope and rope caches.

  Args:
    nope_cache: (total_pages, page_size, 4, 128) uint8. Each (4, 128) uint8 is
      token's nope + nope scales. It encodes 448 fp8 + 7 e8m0 scales + padding.
    rope_cache: (total_pages, page_size // 4, 4, 128) uint8. Each (1, 128) uint8
      is token's rope. It encodes 64 bf16.
    indices: (N,) int32. Token indices into the caches.

  Returns:
    nope_out: (N, 4, 128) uint8.
      Each (4, 128) uint8 is token's nope. It will be flattened to (1, 512)
      downstream.
    rope_out: (N, 64) bf16.
      Each (64) bf16 is token's rope.
  """
  assert indices.ndim == 1, "Indices must be 1D."
  assert nope_cache.dtype == rope_cache.dtype, "Caches must share a dtype."
  assert nope_cache.dtype == jnp.uint8, "Caches must be uint8."
  assert nope_cache.shape[2] == 4
  assert rope_cache.shape[3] == 128

  # Flatten both caches to 128-wide rows and view as raw bytes.
  nope_cache = nope_cache.reshape(-1, nope_cache.shape[3])
  rope_cache = rope_cache.reshape(-1, rope_cache.shape[3])
  sc_info = pltpu.get_tpu_info().sparse_core
  assert sc_info is not None, "SparseCore info is missing."
  out_size = indices.size
  nope_subrows = 4
  nope_out_cols = 128
  rope_out_cols = 64
  num_simd_lanes = sc_info.num_lanes
  num_cores = sc_info.num_cores * sc_info.num_subcores

  # `num_streams` independent `pl.Indirect` gathers are issued per
  # pipeline step to keep multiple gather DMAs in flight.
  # See `outer_pipeline` for details.
  num_streams = 4
  num_row_subchunks = 32
  assert (
      num_row_subchunks % num_streams == 0
  ), f"{num_streams=} must divide {num_row_subchunks=}."
  row_subchunk_size = num_simd_lanes
  row_chunk_size = row_subchunk_size * num_row_subchunks
  block_size = row_chunk_size * num_cores
  out_pad_size = (
      (out_size + block_size - 1) // block_size
  ) * block_size - out_size
  indices = jnp.pad(indices, ((0, out_pad_size)))
  vector_mesh = plsc.VectorSubcoreMesh(
      num_cores=sc_info.num_cores,
      num_subcores=sc_info.num_subcores,
      core_axis_name="core",
      subcore_axis_name="subcore",
  )
  nope_out, rope_out = pl.kernel(
      functools.partial(
          main_kernel,
          core_axis_name=vector_mesh.core_axis_name,
          subcore_axis_name=vector_mesh.subcore_axis_name,
          num_row_subchunks=num_row_subchunks,
          num_streams=num_streams,
      ),
      out_type=(
          jax.ShapeDtypeStruct(
              ((out_size + out_pad_size) * nope_subrows, nope_out_cols),
              jnp.uint8,
          ),
          jax.ShapeDtypeStruct(
              (out_size + out_pad_size, rope_out_cols), jnp.bfloat16
          ),
      ),
      # One DMA semaphore per stream for the direct nope gather-buffer -> HBM
      # copies issued in `main_kernel`.
      scratch_types=(pltpu.SemaphoreType.DMA((num_streams,)),),
      compiler_params=pltpu.CompilerParams(
          use_tc_tiling_on_sc=True,
          needs_layout_passes=True,
          disable_bounds_checks=True,
      ),
      mesh=vector_mesh,
      name="sc_csa_gather",
  )(nope_cache, rope_cache, indices)
  return (
      nope_out.reshape(-1, nope_subrows, nope_out_cols)[:out_size],
      rope_out[:out_size],
  )


_csa_gather_fn = csa_gather
csa_gather = types.SimpleNamespace(csa_gather=_csa_gather_fn)

# ==============================================================================
# Inlined sparse_mla.py
# ==============================================================================
"""TPU-Friendly MLA Ragged Paged Attention kernel."""

import functools

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

# inlined csa_gather

DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024


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


def get_kv_cache_shape(
    total_num_pages,
    page_size,
    kv_dim,
    kv_dtype,
):
  kv_packing = get_dtype_packing(kv_dtype)
  return (
      total_num_pages,
      align_to(page_size, kv_packing) // kv_packing,
      kv_packing,
      align_to(kv_dim, 128),
  )


def _largest_divisor(x: int, cap: int) -> int:
  """Largest divisor of ``x`` that is <= ``cap``."""
  for candidate in range(min(x, cap), 0, -1):
    if x % candidate == 0:
      return candidate
  return 1


_GATHER_PAGE_CHUNK = 128


def _gather_page_ids_kernel(windows_ref, logical_ref, out_ref, *, num_chunks):
  logical = logical_ref[...]  # i32[block_tokens, topk]
  out = jnp.zeros_like(logical)
  for c in range(num_chunks):
    window_chunk = windows_ref[
        :, c * _GATHER_PAGE_CHUNK : (c + 1) * _GATHER_PAGE_CHUNK
    ]  # i32[block_tokens, 128]
    local = logical - c * _GATHER_PAGE_CHUNK
    gathered = jnp.take_along_axis(
        window_chunk, jnp.clip(local, 0, _GATHER_PAGE_CHUNK - 1), axis=1
    )
    out = jnp.where((local >= 0) & (local < _GATHER_PAGE_CHUNK), gathered, out)
  out_ref[...] = out


def gather_page_ids(
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    seq_page_ids: jax.Array,  # i32[num_tokens, topk]  (logical page within seq)
    seq_ids_segment: jax.Array,  # i32[num_tokens]  (token -> seq id)
    max_num_seqs: int,
    *,
    block_tokens: int = 8,
) -> jax.Array:
  """Gathers physical page ids for the CSA top-k tokens."""
  num_tokens, topk = seq_page_ids.shape
  pages_per_seq = page_indices.shape[0] // max_num_seqs
  num_chunks = cdiv(pages_per_seq, _GATHER_PAGE_CHUNK)
  padded_pps = num_chunks * _GATHER_PAGE_CHUNK

  page_table = page_indices.reshape(max_num_seqs, pages_per_seq)
  if padded_pps != pages_per_seq:
    page_table = jnp.pad(page_table, ((0, 0), (0, padded_pps - pages_per_seq)))
  # Per-token page-table window. This is a whole-row gather.
  windows = page_table[seq_ids_segment]  # i32[num_tokens, padded_pps]
  logical = jnp.clip(seq_page_ids, 0, pages_per_seq - 1)

  padded_tokens = align_to(num_tokens, block_tokens)
  if padded_tokens != num_tokens:
    pad = padded_tokens - num_tokens
    windows = jnp.pad(windows, ((0, pad), (0, 0)))
    logical = jnp.pad(logical, ((0, pad), (0, 0)))

  out = pl.pallas_call(
      functools.partial(_gather_page_ids_kernel, num_chunks=num_chunks),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=[
              pl.BlockSpec((block_tokens, padded_pps), lambda t: (t, 0)),
              pl.BlockSpec((block_tokens, topk), lambda t: (t, 0)),
          ],
          out_specs=pl.BlockSpec((block_tokens, topk), lambda t: (t, 0)),
          grid=(padded_tokens // block_tokens,),
      ),
      out_shape=jax.ShapeDtypeStruct((padded_tokens, topk), jnp.int32),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("arbitrary",),
          disable_bounds_checks=True,
      ),
      name="gather_page_ids",
  )(windows, logical)
  return out[:num_tokens]


# DSV4 FP8 nope layout: 448 e4m3 values plus 7 e8m0 scales, one scale per
# 64-wide block of the 448.
def _make_dsv4_fp8_scale_expand_matrix():
  """One-hot [7, 448] matrix mapping each e8m0 scale onto its 64-lane block."""
  scale_id = lax.broadcasted_iota(jnp.int32, (7, 448), 0)
  block_id = lax.broadcasted_iota(jnp.int32, (7, 448), 1) // 64
  return (scale_id == block_id).astype(jnp.bfloat16)


def _dequant_dsv4_fp8(
    bkv_nope: jax.Array, dsv4_fp8_scale_expand_matrix: jax.Array
):
  """Dequantize FP8 values to BF16."""
  nope_fp8 = pltpu.bitcast(bkv_nope[:, :448], jnp.float8_e4m3fn).astype(
      jnp.bfloat16
  )
  # On TPU v6e, hardware does not support unpacking f8E8M0FNU to bf16.
  # Since e8m0 is an unsigned exponent with bias 127 matching bf16 exponent bits,
  # shifting the byte left by 7 and bitcasting uint16 to bf16 is bitwise exact.
  # Mosaic cannot legalize 16-bit vector shifts on v6e, so widen to 32-bit before shifting.
  scale_u32 = bkv_nope[:, 448 : 448 + 7].astype(jnp.uint32) << 7
  scale_u16 = scale_u32.astype(jnp.uint16)
  nope_scales = pltpu.bitcast(scale_u16, jnp.bfloat16)
  # Using on-hot matrix to broadcast each scale across its 64-lane block is
  # more efficient than using
  # `nope_scales = jnp.repeat(nope_scales.T, 64, axis=0).T`
  nope_scales = jnp.dot(
      nope_scales,
      dsv4_fp8_scale_expand_matrix,
      preferred_element_type=jnp.float32,
  ).astype(jnp.bfloat16)
  nope = (nope_fp8 * nope_scales).astype(jnp.bfloat16)
  return nope


def _attention_kernel(
    # Prefetch
    kv_lens_ref,  # [max_num_seqs]
    start_end_seq_idx_ref,  # [3] (start_seq_idx, end_seq_idx, num_valid_tokens)
    sem_ids_ref,  # [2] (bi_sem_idx, bo_sem_idx)
    bo_ids_ref,  # [2] seq_idx_start of the in-flight bo DMA per bo sem
    # Input
    attention_sinks_ref,  # float32[num_q_heads]
    q_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    cache_kv_nope_hbm_ref,  # [total_num_pages, page_size * 4, 128]
    cache_kv_rope_hbm_ref,  # [total_num_pages, page_size, rope_dim]
    swa_accumution_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    swa_l_hbm_ref,  # [max_num_tokens, num_l_heads]
    swa_m_hbm_ref,  # [max_num_tokens, num_l_heads]
    # Output
    o_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    # Scratch
    bkv_nope_x2_ref,  # [2, batch_size, page_size * 4, 128]
    bkv_rope_x2_ref,  # [2, batch_size, page_size, rope_dim]
    bq_x2_ref,  # [2, batch_size, num_q_heads, head_dim]
    bo_x2_ref,  # [2, batch_size, num_q_heads, head_dim]
    bl_x2_ref,  # [2, batch_size, num_l_heads]
    bm_x2_ref,  # [2, batch_size, num_l_heads]
    swa_acc_x2_ref,  # [2, batch_size, num_q_heads, head_dim]
    sems,  # [7, 2]
    *,
    sm_scale: float,
    batch_size: int = 1,
):
  assert q_hbm_ref.shape == o_hbm_ref.shape

  num_tokens, num_q_heads, head_dim = q_hbm_ref.shape
  assert kv_lens_ref.shape[0] == num_tokens
  bkv_sz = cache_kv_rope_hbm_ref.shape[1]

  q_dtype = q_hbm_ref.dtype
  q_packing = get_dtype_packing(q_dtype)
  # Validate against the KV dtype.
  assert o_hbm_ref.dtype == q_dtype

  assert head_dim % 128 == 0
  assert num_q_heads % q_packing == 0

  start_seq_idx = start_end_seq_idx_ref[0]
  end_seq_idx = start_end_seq_idx_ref[1]
  num_valid_tokens = start_end_seq_idx_ref[2]

  batch_start_seq_idx = start_seq_idx + pl.program_id(0) * batch_size
  batch_end_seq_idx = batch_start_seq_idx + batch_size - 1

  def flash_attention_step1_qk_softmax(
      q,  # [bq_sz * num_q_heads, head_dim]
      kv,  # [bkv_sz, head_dim] <- Correspond to data from bkv_*_x2_ref
      kv_len,  # scalar
      swa_m,  # [bq_sz * num_q_heads],
      swa_l,  # [bq_sz * num_q_heads],
      attention_sinks,  # [num_q_heads]
  ):
    assert len(q.shape) == 2
    assert len(kv.shape) == 2
    assert q.shape[0] % num_q_heads == 0
    assert q.shape[1] == head_dim
    assert kv.shape == (bkv_sz, head_dim)

    # Follow FlashAttention-2 forward pass.
    s = jnp.einsum("nd,md->nm", q, kv, preferred_element_type=jnp.float32)
    s *= sm_scale
    k_span = lax.broadcasted_iota(jnp.int32, s.shape, 1)
    s = jnp.where(k_span < kv_len, s, jnp.finfo(s.dtype).min)

    s_rowmax = jnp.max(s, axis=1, keepdims=True)
    m_prev = swa_m
    m_curr = jnp.maximum(m_prev, s_rowmax)
    p = jnp.exp(s - m_curr)
    exp_m_diff = jnp.exp(m_prev - m_curr)
    p_rowsum = jnp.sum(p, axis=1, keepdims=True)
    l_prev = swa_l
    l_curr = exp_m_diff * l_prev + p_rowsum
    exp_attention_sinks = jnp.exp(attention_sinks - m_curr)
    l = l_curr + exp_attention_sinks

    return p, exp_m_diff, l

  def flash_attention_step2_pv(
      p,
      kv,
      exp_m_diff,
      swa_acc,
      l,
  ):
    pv = jnp.einsum("nm,md->nd", p, kv, preferred_element_type=jnp.float32)

    o_prev = swa_acc
    acc = exp_m_diff * o_prev + pv
    out = (
        lax.div(acc, l)
        if q_dtype == jnp.float32
        else (acc * pl.reciprocal(l, approx=True)).astype(q_dtype)
    )
    return out

  def _async_copy(src, dst, sem, wait):
    cp = pltpu.make_async_copy(src, dst, sem)
    if wait:
      cp.wait()
    else:
      cp.start()

  def _fetch_bkv_batch(seq_idx_start, bkv_sem_idx, *, wait=False):
    sem_nope = sems.at[0, bkv_sem_idx]
    sem_rope = sems.at[6, bkv_sem_idx]

    bkv_nope_vmem_ref = bkv_nope_x2_ref.at[bkv_sem_idx, :]
    bkv_rope_vmem_ref = bkv_rope_x2_ref.at[bkv_sem_idx, :]

    # The index into cache_kv_hbm_ref should be relative to the current
    # chunk.
    page_idx_start = seq_idx_start - start_seq_idx
    if not wait:
      _async_copy(
          cache_kv_nope_hbm_ref.at[pl.ds(page_idx_start, batch_size)],
          bkv_nope_vmem_ref,
          sem_nope,
          wait,
      )
      _async_copy(
          cache_kv_rope_hbm_ref.at[pl.ds(page_idx_start, batch_size)],
          bkv_rope_vmem_ref,
          sem_rope,
          wait,
      )
    else:
      dst_nope = bkv_nope_vmem_ref
      _async_copy(src=dst_nope, dst=dst_nope, sem=sem_nope, wait=True)
      dst_rope = bkv_rope_vmem_ref
      _async_copy(src=dst_rope, dst=dst_rope, sem=sem_rope, wait=True)

  def _fetch_bq_batch(seq_idx_start, bq_sem_idx, *, wait=False):
    sem = sems.at[1, bq_sem_idx]
    bq_vmem_ref = bq_x2_ref.at[bq_sem_idx, :]

    if not wait:
      _async_copy(
          q_hbm_ref.at[pl.ds(seq_idx_start, batch_size)],
          bq_vmem_ref,
          sem,
          wait,
      )
    else:
      _async_copy(src=bq_vmem_ref, dst=bq_vmem_ref, sem=sem, wait=True)

  def _send_bo_batch(seq_idx_start, bo_sem_idx, *, wait=False):
    sem = sems.at[2, bo_sem_idx]
    sz = jnp.clip(num_valid_tokens - seq_idx_start, 1, batch_size)
    vmem_ref = bo_x2_ref.at[bo_sem_idx, pl.ds(0, sz)]

    if not wait:
      # Remember where this DMA started so its wait can rebuild the same
      # `sz`; a mismatch here deadlocks on the DMA semaphore.
      bo_ids_ref[bo_sem_idx] = seq_idx_start
      _async_copy(
          vmem_ref,
          o_hbm_ref.at[pl.ds(seq_idx_start, sz)],
          sem,
          wait,
      )
    else:
      _async_copy(src=vmem_ref, dst=vmem_ref, sem=sem, wait=True)

  def _fetch_swa_batch(seq_idx_start, bq_sem_idx, *, wait=False):
    sem_acc = sems.at[3, bq_sem_idx]
    sem_l = sems.at[4, bq_sem_idx]
    sem_m = sems.at[5, bq_sem_idx]

    if not wait:
      _async_copy(
          swa_accumution_hbm_ref.at[pl.ds(seq_idx_start, batch_size)],
          swa_acc_x2_ref.at[bq_sem_idx, :],
          sem_acc,
          wait=False,
      )
      _async_copy(
          swa_l_hbm_ref.at[pl.ds(seq_idx_start, batch_size)],
          bl_x2_ref.at[bq_sem_idx, :],
          sem_l,
          wait=False,
      )
      _async_copy(
          swa_m_hbm_ref.at[pl.ds(seq_idx_start, batch_size)],
          bm_x2_ref.at[bq_sem_idx, :],
          sem_m,
          wait=False,
      )

    else:
      dst_acc = swa_acc_x2_ref.at[bq_sem_idx, :]
      _async_copy(src=dst_acc, dst=dst_acc, sem=sem_acc, wait=True)

      dst_l = bl_x2_ref.at[bq_sem_idx, :]
      _async_copy(src=dst_l, dst=dst_l, sem=sem_l, wait=True)

      dst_m = bm_x2_ref.at[bq_sem_idx, :]
      _async_copy(src=dst_m, dst=dst_m, sem=sem_m, wait=True)

  def start_fetch_bkv_batch(seq_idx_start, bkv_sem_idx):
    return _fetch_bkv_batch(seq_idx_start, bkv_sem_idx)

  def wait_fetch_bkv_batch(seq_idx_start, bkv_sem_idx):
    return _fetch_bkv_batch(seq_idx_start, bkv_sem_idx, wait=True)

  def start_fetch_bq_batch(seq_idx_start, bq_sem_idx):
    return _fetch_bq_batch(seq_idx_start, bq_sem_idx)

  def wait_fetch_bq_batch(seq_idx_start, bq_sem_idx):
    return _fetch_bq_batch(seq_idx_start, bq_sem_idx, wait=True)

  def start_fetch_swa_batch(seq_idx_start, bq_sem_idx):
    return _fetch_swa_batch(seq_idx_start, bq_sem_idx)

  def wait_fetch_swa_batch(seq_idx_start, bq_sem_idx):
    return _fetch_swa_batch(seq_idx_start, bq_sem_idx, wait=True)

  def start_send_bo_batch(seq_idx_start, bo_sem_idx):
    return _send_bo_batch(seq_idx_start, bo_sem_idx)

  def wait_send_bo_batch(bo_sem_idx):
    # `sz` must match the started DMA exactly, so recover its
    # seq_idx_start rather than using the current step's.
    old_seq_idx_start = bo_ids_ref[bo_sem_idx]

    @pl.when(old_seq_idx_start >= 0)
    def _():
      _send_bo_batch(old_seq_idx_start, bo_sem_idx, wait=True)

  def load_bq(bq_sem_idx, batch_idx):
    q = bq_x2_ref.at[bq_sem_idx, batch_idx][...]
    return q

  def load_bkv(bkv_sem_idx, batch_idx, dsv4_fp8_scale_expand_matrix):
    bkv_nope = bkv_nope_x2_ref.at[bkv_sem_idx, batch_idx][...]
    # The gather nope is (4, 128) u8, reshape it to (1, 512) u8.
    bkv_nope = bkv_nope.reshape(bkv_sz, -1)
    bkv_nope = _dequant_dsv4_fp8(bkv_nope, dsv4_fp8_scale_expand_matrix)

    bkv_rope = bkv_rope_x2_ref.at[bkv_sem_idx, batch_idx][...]
    bkv = jnp.concatenate([bkv_nope, bkv_rope], axis=-1)

    # In vLLM, multiple caches may overlay on the same KV Tensor. For example,
    # compressor state cache write data in bfloat16 / float32 format, certain
    # byte pattern are interpreted as NaN in FP8, e.g. float8_e8m0fnu byte 0xFF
    # decodes to NaN.
    # We need to mask out the data by the actual kv_len to avoid NaN propagting
    # to the downstream computation.
    kv_len = kv_lens_ref[batch_start_seq_idx + batch_idx]
    k_span = lax.broadcasted_iota(jnp.int32, bkv.shape, 0)
    bkv = jnp.where(k_span < kv_len, bkv, 0)
    return bkv

  def load_swa_output(bq_sem_idx, batch_idx):
    swa_acc = swa_acc_x2_ref[bq_sem_idx, batch_idx, ...]
    swa_l = bl_x2_ref[bq_sem_idx, batch_idx, :num_q_heads][..., None]
    swa_m = bm_x2_ref[bq_sem_idx, batch_idx, :num_q_heads][..., None]
    return swa_acc, swa_l, swa_m

  def process():

    def get_next_seq_ids(seq_idx, bi_sem_idx):
      next_seq_idx = seq_idx + batch_size
      next_bi_sem_idx = lax.select(bi_sem_idx == 0, 1, 0)
      return next_seq_idx, next_bi_sem_idx

    bi_sem_idx = sem_ids_ref[0]
    next_seq_idx, next_bi_sem_idx = get_next_seq_ids(
        batch_start_seq_idx, bi_sem_idx
    )

    # Prefetch next seq
    @pl.when(next_seq_idx < end_seq_idx)
    def prefetch_next_seq():
      sem_ids_ref[0] = next_bi_sem_idx
      start_fetch_bq_batch(next_seq_idx, next_bi_sem_idx)
      start_fetch_swa_batch(next_seq_idx, next_bi_sem_idx)
      start_fetch_bkv_batch(next_seq_idx, next_bi_sem_idx)

    bo_sem_idx = sem_ids_ref[1]
    sem_ids_ref[1] = lax.select(bo_sem_idx == 0, 1, 0)
    attention_sinks = attention_sinks_ref[...][..., None]
    dsv4_fp8_scale_expand_matrix = _make_dsv4_fp8_scale_expand_matrix()

    prev_p = None
    prev_bkv = None
    prev_exp_m_diff = None
    prev_l = None
    prev_swa_acc = None
    prev_out = None

    # Wait for cur blocks if not ready yet
    wait_fetch_bq_batch(batch_start_seq_idx, bi_sem_idx)
    wait_fetch_swa_batch(batch_start_seq_idx, bi_sem_idx)
    wait_fetch_bkv_batch(batch_start_seq_idx, bi_sem_idx)

    @pl.when(pl.program_id(0) >= 2)
    def _wait_send():
      wait_send_bo_batch(bo_sem_idx)

    for batch_idx in range(batch_size):
      bkv = load_bkv(bi_sem_idx, batch_idx, dsv4_fp8_scale_expand_matrix)
      bq = load_bq(bi_sem_idx, batch_idx)

      if prev_out is not None:
        # Artificial dependency to force MXU/VPU interleaving by limiting LLO's QK runahead
        # We use jnp.where to prevent XLA from optimizing the dependency away.
        # prev_out won't be inf in practice.
        bq = jnp.where(prev_out == jnp.inf, prev_out, bq)

      swa_acc, swa_l, swa_m = load_swa_output(bi_sem_idx, batch_idx)

      p, exp_m_diff, l = flash_attention_step1_qk_softmax(
          bq,
          bkv,
          kv_lens_ref[batch_start_seq_idx + batch_idx],
          swa_m,
          swa_l,
          attention_sinks,
      )

      if prev_p is not None:
        assert prev_bkv is not None
        assert prev_exp_m_diff is not None
        assert prev_l is not None
        out = flash_attention_step2_pv(
            prev_p,
            prev_bkv,
            prev_exp_m_diff,
            prev_swa_acc,
            prev_l,
        )

        # Store output from acc to bo.
        bo_x2_ref.at[bo_sem_idx, batch_idx - 1][...] = out
        prev_out = out

      prev_p = p
      prev_bkv = bkv
      prev_exp_m_diff = exp_m_diff
      prev_l = l
      prev_swa_acc = swa_acc

    # end of pipelining loop
    out = flash_attention_step2_pv(
        prev_p, prev_bkv, prev_exp_m_diff, prev_swa_acc, prev_l
    )
    bo_x2_ref.at[bo_sem_idx, batch_size - 1][...] = out
    start_send_bo_batch(batch_start_seq_idx, bo_sem_idx)

  ### ------- Kernel start ------- ###

  @pl.when(batch_start_seq_idx == start_seq_idx)
  def prologue():
    start_fetch_bq_batch(batch_start_seq_idx, 0)
    start_fetch_swa_batch(batch_start_seq_idx, 0)
    start_fetch_bkv_batch(batch_start_seq_idx, 0)

  process()

  @pl.when(batch_end_seq_idx == end_seq_idx - 1)
  def epilogue():
    wait_send_bo_batch(0)

    @pl.when(pl.num_programs(0) >= 2)
    def _wait_1():
      wait_send_bo_batch(1)

  ### ------- Kernel end ------- ###


def prepare_q_inputs(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim],
):
  _, actual_num_q_heads, actual_head_dim = q.shape
  q_packing = get_dtype_packing(q.dtype)
  num_q_heads = align_to(actual_num_q_heads, q_packing)
  head_dim = align_to(actual_head_dim, 128)
  q = jnp.pad(
      q,
      (
          (0, 0),
          (0, num_q_heads - actual_num_q_heads),
          (0, head_dim - actual_head_dim),
      ),
      constant_values=0,
  )
  return q


def prepare_swa_inputs(
    swa_accumution: jax.Array,  # [max_num_tokens, num_q_heads, head_dim]
    swa_l: jax.Array,  # [max_num_tokens, num_q_heads]
    swa_m: jax.Array,  # [max_num_tokens, num_q_heads]
):
  _, actual_num_q_heads, actual_head_dim = swa_accumution.shape
  swa_packing = get_dtype_packing(swa_accumution.dtype)
  num_q_heads = align_to(actual_num_q_heads, swa_packing)
  head_dim = align_to(actual_head_dim, 128)
  swa_accumution = jnp.pad(
      swa_accumution,
      (
          (0, 0),
          (0, num_q_heads - actual_num_q_heads),
          (0, head_dim - actual_head_dim),
      ),
      constant_values=0,
  )
  num_l_heads = align_to(num_q_heads, 128)
  swa_l = jnp.pad(
      swa_l,
      (
          (0, 0),
          (0, num_l_heads - actual_num_q_heads),
      ),
      constant_values=0,
  )
  swa_m = jnp.pad(
      swa_m,
      (
          (0, 0),
          (0, num_l_heads - actual_num_q_heads),
      ),
      constant_values=0,
  )
  return swa_accumution, swa_l, swa_m


def prepare_outputs(
    out,  # [max_num_tokens, num_q_heads, head_dim]
    actual_num_q_heads: int,
    actual_head_dim: int,
):
  return out[:, :actual_num_q_heads, :actual_head_dim]


# Main Attention kernel for DeepSeek V4 CSA (gather and attention)
# Note that the compressed kv tokens of current batch (current forward pass)
# have been written to the `cache_kv` by the compressor module before calling
# this function, `kv_lens` reflects the length after compressed kv cache write.
@functools.partial(
    jax.jit,
    static_argnames=(
        "sm_scale",
        "attention_kernel_batch_size",
        "gather_and_attention_chunk_size",
        "vmem_limit_bytes",
    ),
)
def sparse_ragged_paged_attention(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, head_dim]
    cache_kv_nope: jax.Array,  # [total_num_pages, page_size, 4, 128]
    cache_kv_rope: jax.Array,  # [total_num_pages, page_size // 4, 4, 128]
    topk_indices: jax.Array,  # i32[max_num_tokens, csa_topk]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    attention_sinks: jax.Array,  # float32[actual_num_q_heads]
    swa_accumution: jax.Array,  # bf16[max_num_tokens, num_q_heads, head_dim]
    swa_l: jax.Array,  # float32[max_num_tokens, num_q_heads]
    swa_m: jax.Array,  # float32[max_num_tokens, num_q_heads]
    *,
    sm_scale: float = 1.0,
    # Kernel optimization params.
    gather_and_attention_chunk_size: int = 64,
    attention_kernel_batch_size: int = 16,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
) -> jax.Array:
  """MLA Ragged paged attention that supports mixed prefill and decode.

  Args:
    q: concatenated all sequences' queries.
    cache_kv_nope: the current kv cache for nope.
    cache_kv_rope: the current kv cache for rope.
    topk_indices: for each query token, the indices of the top k key tokens to
      attend to.
    page_indices: flattened page indices look-up table by (seq_id, page_id).
    cu_q_lens: the cumulative sum of the effective query lengths. Similar to
      kv_lens, only the first num_seqs+1 values are valid.
    distribution: (i, j, k) represents that sequences[0:i] are decode-only,
      sequences[i:j] are chunked-prefill-only, and sequences[j:k] are mixed. The
      k is also the total number of sequences.
    sm_scale: the softmax scale which will be applied to the Q@K^T.
    vmem_limit_bytes: the vmem limit for the pallas kernel.

  Returns:
    The output of attention.
  """
  # The cache is DSV4 FP8 format.
  # nope_cache contains 448 fp8 + 7 fp8 scales,
  # rope_cache contains 64 bf16
  assert cache_kv_nope.dtype == jnp.uint8
  assert cache_kv_rope.dtype == jnp.uint8

  _, actual_num_q_heads, actual_head_dim = q.shape

  q = prepare_q_inputs(q)  # [max_num_tokens, num_q_heads, head_dim]
  head_dim = q.shape[-1]
  attention_sinks = jnp.pad(
      attention_sinks,
      (0, q.shape[1] - actual_num_q_heads),
      constant_values=jnp.finfo(attention_sinks.dtype).min,
  )
  assert swa_accumution.dtype == q.dtype
  swa_accumution, swa_l, swa_m = prepare_swa_inputs(
      swa_accumution, swa_l, swa_m
  )

  _, page_size, _, _ = cache_kv_nope.shape

  _, num_q_heads, _ = q.shape
  max_num_seqs = cu_q_lens.shape[0] - 1
  num_page_indices = page_indices.shape[0]
  assert num_page_indices % max_num_seqs == 0

  def run_mla_kernel(
      q: jax.Array,  # [max_num_tokens, num_q_heads, head_dim]
      cache_kv_nope: jax.Array,  # [total_num_pages, page_size, nope_dim]
      cache_kv_rope: jax.Array,  # [total_num_pages, page_size, rope_dim]
      kv_lens: jax.Array,  # i32[max_num_seqs]
      attention_sinks: jax.Array,  # float32[num_q_heads]
      swa_accumution: jax.Array,  # bf16[max_num_tokens, num_q_heads, head_dim]
      swa_l: jax.Array,  # float32[max_num_tokens, num_l_heads]
      swa_m: jax.Array,  # float32[max_num_tokens, num_l_heads]
      start_seq_idx: jax.Array,  # i32
      end_seq_idx: jax.Array,  # i32
      num_valid_tokens: jax.Array,  # i32
      kernel_batch_size: int,
  ):
    batch_size = kernel_batch_size
    end_seq_idx = jnp.maximum(start_seq_idx, end_seq_idx)
    grid = (cdiv(end_seq_idx - start_seq_idx, batch_size),)
    in_specs = [
        pl.BlockSpec(memory_space=pltpu.VMEM),  # attention_sinks
        pl.BlockSpec(memory_space=pltpu.HBM),  # q
        pl.BlockSpec(memory_space=pltpu.HBM),  # cache_kv_nope
        pl.BlockSpec(memory_space=pltpu.HBM),  # cache_kv_rope
        pl.BlockSpec(memory_space=pltpu.HBM),  # swa_accumution
        pl.BlockSpec(memory_space=pltpu.HBM),  # swa_l
        pl.BlockSpec(memory_space=pltpu.HBM),  # swa_m
    ]

    out_specs = pl.BlockSpec(memory_space=pltpu.HBM)  # o

    # One batch entry's worth of gathered top-k rows, per cache.
    bkv_nope_double_buf = pltpu.VMEM(
        (2, batch_size, *cache_kv_nope.shape[1:]),
        cache_kv_nope.dtype,
    )
    bkv_rope_double_buf = pltpu.VMEM(
        (2, batch_size, *cache_kv_rope.shape[1:]),
        cache_kv_rope.dtype,
    )

    bq_double_bufq = pltpu.VMEM(
        (2, batch_size, num_q_heads, head_dim),
        q.dtype,
    )

    bo_double_buf = bq_double_bufq

    num_l_heads = align_to(num_q_heads, 128)
    bl_double_buf = pltpu.VMEM(
        (2, batch_size, num_l_heads),
        jnp.float32,
    )
    bm_double_buf = bl_double_buf

    swa_acc_double_buf = pltpu.VMEM(
        (2, batch_size, num_q_heads, head_dim),
        q.dtype,
    )

    scratch_shapes = [
        bkv_nope_double_buf,
        bkv_rope_double_buf,
        bq_double_bufq,
        bo_double_buf,  # Double buffering for output block.
        bl_double_buf,  # Double buffering for l output.
        bm_double_buf,  # Double buffering for m output.
        swa_acc_double_buf,  # Buffer for swa_accumution.
        # Semaphores for double buffering of bkv_nope, bq, bo, swa_acc, swa_l, swa_m, bkv_rope
        pltpu.SemaphoreType.DMA((7, 2)),
    ]

    scalar_prefetches = (
        kv_lens,
        jnp.array([start_seq_idx, end_seq_idx, num_valid_tokens], jnp.int32),
        # (bi_sem_idx, bo_sem_idx)
        jnp.zeros((2,), jnp.int32),
        # seq_idx_start of the in-flight bo DMA per bo sem; -1 = none.
        jnp.full((2,), -1, jnp.int32),
    )

    scope_name = f"SparseMLA-p_{cache_kv_rope.shape[1]}-bz_{batch_size}-gcz_{cache_kv_nope.shape[0]}"
    kernel = jax.named_scope(scope_name)(
        pl.pallas_call(
            functools.partial(
                _attention_kernel,
                sm_scale=sm_scale,
                batch_size=batch_size,
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
            out_shape=jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
            input_output_aliases={
                5: 0,  # Alias output activation with q
            },
            name=scope_name,
        )
    )
    return kernel(
        *scalar_prefetches,
        attention_sinks,
        q,
        cache_kv_nope,
        cache_kv_rope,
        swa_accumution,
        swa_l,
        swa_m,
    )

  tokens_per_seq = cu_q_lens[1:] - cu_q_lens[:-1]
  seq_ids_segment = jnp.repeat(
      jnp.arange(max_num_seqs), tokens_per_seq, total_repeat_length=q.shape[0]
  )
  assert topk_indices is not None
  # TODO: skip gather for padding tokens in topk_indices.
  kv_lens = jnp.sum(topk_indices != -1, axis=-1)

  seq_page_ids = topk_indices // page_size
  token_offset = topk_indices % page_size
  topk = topk_indices.shape[-1]
  page_ids = gather_page_ids(
      page_indices, seq_page_ids, seq_ids_segment, max_num_seqs
  )

  # For the "-1" padding elements in topk_indices, we scatter the corresponding
  # page_ids and token_offset to avoid gather memory access hotspotting.
  is_padding = topk_indices == -1
  total_num_pages = cache_kv_nope.shape[0]
  flat_element_index = jnp.arange(q.shape[0] * topk, dtype=jnp.int32).reshape(
      q.shape[0], topk
  )
  # 104729 and 15485863 are randomly chosen large prime numbers.
  scattered_page_ids = (flat_element_index * 104729) % total_num_pages
  scattered_token_offset = (flat_element_index * 15485863) % page_size
  page_ids = jnp.where(is_padding, scattered_page_ids, page_ids)
  token_offset = jnp.where(
      is_padding,
      scattered_token_offset,
      token_offset,
  )

  assert page_ids.shape == (q.shape[0], topk)
  num_chunks = cdiv(q.shape[0], gather_and_attention_chunk_size)

  for i in range(num_chunks):
    start_pos = i * gather_and_attention_chunk_size
    end_pos = min(start_pos + gather_and_attention_chunk_size, q.shape[0])
    chunk_size = end_pos - start_pos
    indices = (
        page_ids[start_pos:end_pos, ...] * page_size
        + token_offset[start_pos:end_pos, ...]
    ).reshape(-1)

    # For prefilling of short sequences (or early in the sequence), there are
    # very few number of KVs in the sequence, so different qs' selected topk
    # would have large overlap. This causes gather read hotspotting. We've seen
    # 30%+ performance degradation compared to the no-duplicate-indices case.
    #
    # TODO: we could consider let the caller (tpu-runner) to sort the sequences
    # based on their lengths. For the sequences-segment below certain length,
    # we use a different kernel (dense attention and mask), for the rest of
    # sequences, we use this gather-and-attention kernel.
    gathered_nope_buffer, gathered_rope_buffer = csa_gather.csa_gather(
        cache_kv_nope,
        cache_kv_rope,
        indices,
    )
    gathered_nope_buffer = gathered_nope_buffer.reshape(chunk_size, -1, 128)
    gathered_rope_buffer = gathered_rope_buffer.reshape(chunk_size, topk, -1)
    # We treat each query token as a one independent sequence, attend to their
    # respective gathered kv tokens in the `gathered_kv_buffer`.
    # -1 in topk_indices is padded elements at the end of each row.
    # Batching
    kernel_batch_size = _largest_divisor(
        chunk_size, attention_kernel_batch_size
    )
    assert chunk_size % kernel_batch_size == 0
    # The kernel grid walks [start_pos, batch_end) in `kernel_batch_size`
    # steps, so `batch_end - start_pos` MUST be a multiple of
    # `kernel_batch_size`.
    batch_end = start_pos + (
        cdiv(
            jnp.minimum(
                cu_q_lens[distribution[2]],
                end_pos,
            )
            - start_pos,
            kernel_batch_size,
        )
        * kernel_batch_size
    )
    q = run_mla_kernel(
        q,
        gathered_nope_buffer,
        gathered_rope_buffer,
        kv_lens,
        attention_sinks,
        swa_accumution,
        swa_l,
        swa_m,
        start_seq_idx=start_pos,
        end_seq_idx=batch_end,
        num_valid_tokens=cu_q_lens[distribution[2]],
        kernel_batch_size=kernel_batch_size,
    )
  return prepare_outputs(
      q, actual_num_q_heads, actual_head_dim
  )  # [max_num_tokens, actual_num_q_heads, actual_head_dim]



# ==============================================================================
# Benchmark Harness
# ==============================================================================
CONFIGS = {
    'decode_small': {
        'name': 'decode_small',
        'model': 'DeepSeek-V4',
        'operator': 'compressed_sparse_attention',
        'batch_size': 16,
        'q_len': 1,
        'kv_len': 1024,
        'num_q_heads': 128,
        'head_dim': 512,
        'page_size': 256,
        'csa_topk': 256,
        'sm_scale': 1.0,
        'gather_and_attention_chunk_size': 128,
        'attention_kernel_batch_size': 4,
    },
    'dsv4_pro_decode': {
        'name': 'dsv4_pro_decode',
        'model': 'DeepSeek-V4-Pro',
        'operator': 'compressed_sparse_attention',
        'batch_size': 256,
        'q_len': 1,
        'kv_len': 9216,
        'num_q_heads': 128,
        'head_dim': 512,
        'page_size': 1024,
        'csa_topk': 1024,
        'sm_scale': 1.0,
        'gather_and_attention_chunk_size': 128,
        'attention_kernel_batch_size': 4,
    },
    'dsv4_pro_prefill_chunk_first': {
        'name': 'dsv4_pro_prefill_chunk_first',
        'model': 'DeepSeek-V4-Pro',
        'operator': 'compressed_sparse_attention',
        'batch_size': 1,
        'q_len': 1024,
        'kv_len': 1024,
        'num_q_heads': 128,
        'head_dim': 512,
        'page_size': 1024,
        'csa_topk': 1024,
        'sm_scale': 1.0,
        'gather_and_attention_chunk_size': 128,
        'attention_kernel_batch_size': 4,
    },
    'dsv4_pro_prefill_chunk_last': {
        'name': 'dsv4_pro_prefill_chunk_last',
        'model': 'DeepSeek-V4-Pro',
        'operator': 'compressed_sparse_attention',
        'batch_size': 1,
        'q_len': 1024,
        'kv_len': 8192,
        'num_q_heads': 128,
        'head_dim': 512,
        'page_size': 1024,
        'csa_topk': 1024,
        'sm_scale': 1.0,
        'gather_and_attention_chunk_size': 128,
        'attention_kernel_batch_size': 4,
    },
    'dsv4_flash_decode': {
        'name': 'dsv4_flash_decode',
        'model': 'DeepSeek-V4-Flash',
        'operator': 'compressed_sparse_attention',
        'batch_size': 256,
        'q_len': 1,
        'kv_len': 9216,
        'num_q_heads': 64,
        'head_dim': 512,
        'page_size': 1024,
        'csa_topk': 512,
        'sm_scale': 1.0,
        'gather_and_attention_chunk_size': 128,
        'attention_kernel_batch_size': 4,
    },
    'dsv4_flash_prefill_chunk_first': {
        'name': 'dsv4_flash_prefill_chunk_first',
        'model': 'DeepSeek-V4-Flash',
        'operator': 'compressed_sparse_attention',
        'batch_size': 1,
        'q_len': 1024,
        'kv_len': 1024,
        'num_q_heads': 64,
        'head_dim': 512,
        'page_size': 1024,
        'csa_topk': 512,
        'sm_scale': 1.0,
        'gather_and_attention_chunk_size': 128,
        'attention_kernel_batch_size': 4,
    },
    'dsv4_flash_prefill_chunk_last': {
        'name': 'dsv4_flash_prefill_chunk_last',
        'model': 'DeepSeek-V4-Flash',
        'operator': 'compressed_sparse_attention',
        'batch_size': 1,
        'q_len': 1024,
        'kv_len': 8192,
        'num_q_heads': 64,
        'head_dim': 512,
        'page_size': 1024,
        'csa_topk': 512,
        'sm_scale': 1.0,
        'gather_and_attention_chunk_size': 128,
        'attention_kernel_batch_size': 4,
    },
}
CONFIG = CONFIGS['decode_small']

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype('float32')).max)


def create_inputs(dtype=jnp.bfloat16, config=None):
  """Returns inputs tuple for CSA."""
  cfg = (
      CONFIG
      if config is None
      else (CONFIGS[config] if isinstance(config, str) else config)
  )

  key = jax.random.key(42)
  k1, k2, k3, k4, k5, k6, k7, k8, k9, k10 = jax.random.split(key, 10)

  B = cfg['batch_size']
  q_len = cfg['q_len']
  num_tokens = B * q_len
  kv_len = cfg['kv_len']
  num_heads = cfg['num_q_heads']
  head_dim = cfg['head_dim']
  page_size = cfg['page_size']
  topk = cfg['csa_topk']

  pages_per_seq = (kv_len + page_size - 1) // page_size + 2
  total_pages = B * pages_per_seq

  q = jax.random.normal(k1, (num_tokens, num_heads, head_dim), dtype=dtype)
  cache_kv_nope = jax.random.randint(
      k8, (total_pages, page_size, 4, 128), 1, 126, dtype=jnp.uint8
  )
  rope_bf16 = jax.random.normal(
      k9, (total_pages, page_size, 64), dtype=jnp.bfloat16
  )
  rope_u16 = jax.lax.bitcast_convert_type(rope_bf16, jnp.uint16)
  hi = (rope_u16 >> 8).astype(jnp.uint8)
  lo = (rope_u16 & 0xFF).astype(jnp.uint8)
  cache_kv_rope = jnp.concatenate([hi, lo], axis=-1).reshape(
      total_pages, page_size // 4, 4, 128
  )

  topk_indices = jax.random.randint(
      k4, (num_tokens, topk), 0, kv_len, dtype=jnp.int32
  )
  page_indices = jnp.arange(total_pages, dtype=jnp.int32)
  cu_q_lens = jnp.arange(0, num_tokens + 1, q_len, dtype=jnp.int32)
  distribution = jnp.array([B, B, B], dtype=jnp.int32)

  attention_sinks = jax.random.uniform(k5, (num_heads,), dtype=jnp.float32)
  swa_accumulation = jax.random.normal(
      k6, (num_tokens, num_heads, head_dim), dtype=dtype
  )
  swa_l = jax.random.uniform(k7, (num_tokens, num_heads), dtype=jnp.float32)
  swa_m = jax.random.uniform(k10, (num_tokens, num_heads), dtype=jnp.float32)

  return (
      q,
      cache_kv_nope,
      cache_kv_rope,
      topk_indices,
      page_indices,
      cu_q_lens,
      distribution,
      attention_sinks,
      swa_accumulation,
      swa_l,
      swa_m,
  )


def get_inputs(dtype=jnp.bfloat16):
  """Returns list of inputs across all defined configurations."""
  return [
      (list(create_inputs(dtype=dtype, config=cfg)), [])
      for cfg in CONFIGS.values()
  ]


def computation(
    q,
    cache_kv_nope,
    cache_kv_rope,
    topk_indices,
    page_indices,
    cu_q_lens,
    distribution,
    attention_sinks,
    swa_accumulation,
    swa_l,
    swa_m,
):
  """Pallas kernel computation execution."""
  num_tokens, num_heads, head_dim = q.shape
  B = len(cu_q_lens) - 1
  q_len = num_tokens // B
  topk = topk_indices.shape[-1]
  sm_scale = 1.0

  return sparse_ragged_paged_attention(
      q,
      cache_kv_nope,
      cache_kv_rope,
      topk_indices,
      page_indices,
      cu_q_lens,
      distribution,
      attention_sinks,
      swa_accumulation,
      swa_l,
      swa_m,
      sm_scale=sm_scale,
      gather_and_attention_chunk_size=128,
      attention_kernel_batch_size=4,
      vmem_limit_bytes=100 * 1024 * 1024,
  )

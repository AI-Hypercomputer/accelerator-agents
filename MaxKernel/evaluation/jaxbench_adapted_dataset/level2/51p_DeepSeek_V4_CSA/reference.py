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
from jax.experimental.pallas import tpu_sc as plsc

# Initialization
def get_inputs():
    import jax
    import jax.numpy as jnp

    configs = [
        ("decode_large_batch", 256, 1, 9216, 1024, 1024),
        ("decode_medium_batch", 128, 1, 9216, 1024, 1024),
        ("decode_small_pages", 128, 1, 9216, 256, 1024),
        ("decode_random_access", 256, 1, 9216, 64, 512),
        ("prefill_full", 1, 1024, 1024, 1024, 1024),
        ("prefill_short", 1, 256, 1024, 1024, 1024),
        ("prefill_short_small_pages", 1, 256, 1024, 256, 1024),
        ("prefill_mid_chunk_512", 1, 512, 4096, 1024, 1024),
        ("prefill_mid_chunk_1024", 1, 1024, 8192, 1024, 1024),
        ("prefill_mid_chunk_small_pages", 1, 1024, 8192, 64, 512),
    ]

    outputs = []
    k_cfg = jax.random.PRNGKey(0)

    # --- DSV4 CSA geometry -------------------------------------------------
    NOPE_DIM = 448
    NUM_SCALES = NOPE_DIM // 64  # 7
    ROPE_DIM = 64
    TOKEN_BYTES = 512  # (4, 128) uint8 per token in the nope cache
    HEAD_DIM = NOPE_DIM + ROPE_DIM  # 512
    NUM_Q_HEADS = 8
    Q_DTYPE = jnp.bfloat16
    SM_SCALE = float(HEAD_DIM ** -0.5)
    ATTN_BATCH = 16
    VMEM_LIMIT_BYTES = 100 * 1024 * 1024

    # --- Cache contents ----------------------------------------------------
    CACHE_TILE = 8192
    k_nope, k_scale, k_rope, k_cfg = jax.random.split(k_cfg, 4)

    nope_f8 = jax.random.normal(
        k_nope, (CACHE_TILE, NOPE_DIM), jnp.float32
    ).astype(jnp.float8_e4m3fn)
    nope_bytes = jax.lax.bitcast_convert_type(nope_f8, jnp.uint8)
    
    scale_bytes = jax.random.randint(
        k_scale, (CACHE_TILE, NUM_SCALES), 125, 130, dtype=jnp.int32
    ).astype(jnp.uint8)
    
    pad_bytes = jnp.zeros(
        (CACHE_TILE, TOKEN_BYTES - NOPE_DIM - NUM_SCALES), jnp.uint8
    )
    
    nope_tile = jnp.concatenate(
        [nope_bytes, scale_bytes, pad_bytes], axis=1
    )

    rope_bits = jax.lax.bitcast_convert_type(
        jax.random.normal(
            k_rope, (CACHE_TILE, ROPE_DIM), jnp.float32
        ).astype(jnp.bfloat16),
        jnp.uint16,
    ).astype(jnp.uint32)
    rope_tile = jnp.concatenate(
        [
            (rope_bits >> 8).astype(jnp.uint8),
            (rope_bits & 0xFF).astype(jnp.uint8),
        ],
        axis=1,
    )

    def fill(tile, num_slots):
        reps = -(-num_slots // tile.shape[0])
        return jnp.tile(tile, (reps, 1))[:num_slots]

    keys = jax.random.split(k_cfg, len(configs))
    for cfg, key in zip(configs, keys):
        name, batch_size, q_len, kv_len, page_size, csa_topk = cfg
        k_page, k_topk, k_q, k_sink, k_acc, k_l, k_m = jax.random.split(key, 7)

        num_tokens = batch_size * q_len
        pages_per_seq = -(-kv_len // page_size)
        total_num_pages = batch_size * pages_per_seq
        assert num_tokens % ATTN_BATCH == 0

        q = jax.random.normal(
            k_q, (num_tokens, NUM_Q_HEADS, HEAD_DIM), jnp.float32
        ).astype(Q_DTYPE)

        num_slots = total_num_pages * page_size
        cache_kv_nope = fill(nope_tile, num_slots).reshape(
            total_num_pages, page_size, 4, 128
        )
        cache_kv_rope = fill(rope_tile, num_slots).reshape(
            total_num_pages, page_size // 4, 4, 128
        )

        page_indices = jax.random.permutation(
            k_page, total_num_pages
        ).astype(jnp.int32)

        cu_q_lens = (jnp.arange(batch_size + 1, dtype=jnp.int32) * q_len)
        num_decode = batch_size if q_len == 1 else 0
        distribution = jnp.array(
            [num_decode, batch_size, batch_size], jnp.int32
        )

        pos_in_seq = jnp.arange(num_tokens, dtype=jnp.int32) % q_len
        causal_len = (kv_len - q_len + pos_in_seq + 1)[:, None]
        slot = jnp.arange(csa_topk, dtype=jnp.int32)[None, :]
        scattered = jax.random.randint(
            k_topk, (num_tokens, csa_topk), 0, 1 << 30, dtype=jnp.int32
        ) % causal_len
        topk_indices = jnp.where(
            slot < causal_len,
            jnp.where(causal_len < csa_topk, slot, scattered),
            -1,
        ).astype(jnp.int32)

        attention_sinks = jax.random.normal(
            k_sink, (NUM_Q_HEADS,), jnp.float32
        )
        swa_accumution = jax.random.normal(
            k_acc, (num_tokens, NUM_Q_HEADS, HEAD_DIM), jnp.float32
        ).astype(Q_DTYPE)
        swa_l = jax.random.uniform(
            k_l, (num_tokens, NUM_Q_HEADS), jnp.float32, 1.0, 64.0
        )
        swa_m = jax.random.normal(k_m, (num_tokens, NUM_Q_HEADS), jnp.float32)

        dynamic_args = [
            q,
            cache_kv_nope,
            cache_kv_rope,
            topk_indices,
            page_indices,
            cu_q_lens,
            distribution,
            attention_sinks,
            swa_accumution,
            swa_l,
            swa_m,
        ]
        outputs.append((dynamic_args, []))

    return outputs

# Computation

def main_kernel(
    nope_in_hbm_ref: jax.Ref,
    rope_in_hbm_ref: jax.Ref,
    indices_hbm_ref: jax.Ref,
    nope_out_hbm_ref: jax.Ref,
    rope_out_hbm_ref: jax.Ref,
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
  nope_out_packing = 32 // jax.dtypes.itemsize_bits(nope_out_hbm_ref.dtype)
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
  nope_out_cols = nope_out_hbm_ref.shape[1]
  rope_out_cols = rope_out_hbm_ref.shape[1]

  def _delta_swap(x, y, shift, swap_mask):
    """Exchanges selected bits between two words.

    Swaps x's bits at positions ``swap_mask << shift`` with y's bits at
    positions ``swap_mask``; all other bits are left untouched. Worked
    example with ``shift=8, swap_mask=0x00FF00FF`` on byte-quads
    ``x = [x0 x1 x2 x3]`` and ``y = [y0 y1 y2 y3]`` (byte 0 = least
    significant)::

        swap_mask << 8 = 0xFF00FF00  -> x's bytes 1 and 3
        swap_mask      = 0x00FF00FF  -> y's bytes 0 and 2
        result:  x = [x0 y0 x2 y2],  y = [x1 y1 x3 y3]

    i.e. x's odd bytes trade places with y's even bytes. The identity
    ``t = ((x >> s) ^ y) & mask;  x ^= t << s;  y ^= t`` does this in 6 ops
    with only `t` as scratch. The arithmetic right shift is safe: its
    sign-extended high bits are dropped by the `& swap_mask`.
    """
    t = jnp.bitwise_and(
        jnp.bitwise_xor(jnp.bitwise_right_shift(x, shift), y), swap_mask
    )
    x = jnp.bitwise_xor(x, jnp.left_shift(t, shift))
    y = jnp.bitwise_xor(y, t)
    return x, y

  def process_nope(gather_ref, out_ref, out_row_base=0):
    # A more efficient implementation of `process_nope_reference`, with fewer
    # ALU ops. We've seen that SparseCore's Integer ALU FLOPs being the
    # bottleneck of the kernel, this is the preferred implementation.
    col_slice = pl.ds(0, 128)
    low_16 = jnp.int32(0x0000FFFF)  # one 16-bit half of each word
    low_byte_of_each_half = jnp.int32(0x00FF00FF)  # bytes 0 and 2
    for i in range(num_simd_lanes // nope_out_packing):
      d = [
          gather_ref[pl.ds(i * nope_out_packing + j, 1), col_slice]
          for j in range(nope_out_packing)
      ]
      # Round 1: transpose the four 2x2 byte blocks -- exchange the high
      # 16 bits of each word with the low 16 bits of its distance-2 peer.
      d[0], d[2] = _delta_swap(d[0], d[2], 16, low_16)
      d[1], d[3] = _delta_swap(d[1], d[3], 16, low_16)
      # Round 2: transpose within each 2x2 block -- exchange the odd bytes
      # of each word with the even bytes of its adjacent peer.
      d[0], d[1] = _delta_swap(d[0], d[1], 8, low_byte_of_each_half)
      d[2], d[3] = _delta_swap(d[2], d[3], 8, low_byte_of_each_half)
      # d[m] now holds byte lane m of all 4 inputs -> output sub-row m.s
      for m in range(nope_out_packing):
        out_ref[pl.ds(out_row_base + i, 1), pl.ds(m * 128, 128)] = d[m]

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

    # Rows produced per stream in each output (nope packs 4/int32, rope 2).
    nope_rows_per_stream = row_subchunk_size // nope_out_packing
    rope_rows_per_stream = row_subchunk_size // rope_out_packing

    def _body(*refs):
      r = pl.program_id(0)
      nope_g = refs[0 * num_streams : 1 * num_streams]
      rope_g = refs[1 * num_streams : 2 * num_streams]
      nope_o = refs[2 * num_streams]
      rope_o = refs[2 * num_streams + 1]
      for s in range(num_streams):
        process_nope(
            gather_ref=nope_g[s],
            out_ref=nope_o,
            out_row_base=s * nope_rows_per_stream,
        )
        process_rope(
            gather_ref=rope_g[s],
            out_ref=rope_o,
            idx_sub=idx_window(r, s),
            out_row_base=s * rope_rows_per_stream,
        )

    # Have multiple parallel `pl.Indirect` to hide random access read latency.
    # Output contiguous memory access, multiple output parallel DMAs not help
    # with performance.

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
    # One merged output block per cache, covering all `num_streams`
    # subchunks.
    nope_out_spec = pl.BlockSpec(
        (num_streams * nope_rows_per_stream, nope_out_cols),
        lambda r: (out_row_base // num_streams + r, 0),
    )
    rope_out_spec = pl.BlockSpec(
        (num_streams * rope_rows_per_stream, rope_out_cols),
        lambda r: (out_row_base // num_streams + r, 0),
    )
    pltpu.emit_pipeline(
        _body,
        grid=(num_row_subchunks // num_streams,),
        in_specs=nope_in_specs + rope_in_specs,
        out_specs=(nope_out_spec, rope_out_spec),
    )(
        *([nope_in_i32] * num_streams),
        *([rope_in_i32] * num_streams),
        nope_out_i32,
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
    nope_out: (N, 512) uint8.
      Each (512) uint8 is token's nope.
    rope_out: (N, 64) bf16.
      Each (64) bf16 is token's rope.
  """
  assert indices.ndim == 1, "Indices must be 1D."
  assert nope_cache.dtype == rope_cache.dtype, "Caches must share a dtype."
  assert nope_cache.dtype == jnp.uint8, "Caches must be uint8."

  # Flatten both caches to 128-wide rows and view as raw bytes.
  nope_cache = nope_cache.reshape(-1, nope_cache.shape[3])
  rope_cache = rope_cache.reshape(-1, rope_cache.shape[3])
  sc_info = pltpu.get_tpu_info().sparse_core
  assert sc_info is not None, "SparseCore info is missing."
  out_size = indices.size
  nope_out_cols = 512
  # rope: each 128-byte entry encodes 64 bf16 (high bytes [0:64], low [64:128]).
  rope_out_cols = 64
  num_simd_lanes = sc_info.num_lanes
  num_cores = sc_info.num_cores * sc_info.num_subcores

  # `num_streams` independent `pl.Indirect` gathers are issued per
  # pipeline step to keep multiple gather DMAs in flight.
  # See `outer_pipeline` for details.
  num_streams = 2
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
              (out_size + out_pad_size, nope_out_cols), jnp.uint8
          ),
          jax.ShapeDtypeStruct(
              (out_size + out_pad_size, rope_out_cols), jnp.bfloat16
          ),
      ),
      compiler_params=pltpu.CompilerParams(
          use_tc_tiling_on_sc=True,
          needs_layout_passes=True,
          disable_bounds_checks=True,
      ),
      mesh=vector_mesh,
      name="sc_csa_gather",
  )(nope_cache, rope_cache, indices)
  return (
      nope_out[:out_size],
      rope_out[:out_size],
  )



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


def _dequant_dsv4_fp8(bkv_nope: jax.Array):
  """Dequantize FP8 values to BF16."""
  nope_fp8 = pltpu.bitcast(bkv_nope[:, :448], jnp.float8_e4m3fn).astype(
      jnp.bfloat16
  )
  nope_scales = pltpu.bitcast(
      bkv_nope[:, 448 : 448 + 7], jnp.float8_e8m0fnu
  ).astype(jnp.bfloat16)
  nope_scales = jnp.repeat(nope_scales.T, 64, axis=0).T
  nope = (nope_fp8 * nope_scales).astype(jnp.bfloat16)
  return nope


def _attention_kernel(
    # Prefetch
    kv_lens_ref,  # [max_num_seqs]
    start_end_seq_idx_ref,  # [2] (start_seq_idx, end_seq_idx)
    sem_ids_ref,  # [2] (bi_sem_idx, bo_sem_idx)
    bo_ids_ref,  # [2, batch_size] (bo_sem_0_seq_idx, bo_sem_1_seq_idx)
    # Input
    attention_sinks_ref,  # float32[num_q_heads]
    q_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    cache_kv_nope_hbm_ref,  # [total_num_pages, page_size, nope_dim]
    cache_kv_rope_hbm_ref,  # [total_num_pages, page_size, rope_dim]
    swa_accumution_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    swa_l_hbm_ref,  # [max_num_tokens, num_l_heads]
    swa_m_hbm_ref,  # [max_num_tokens, num_l_heads]
    # Output
    o_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    # Scratch
    bkv_nope_x2_ref,  # [2, batch_size, page_size, nope_dim]
    bkv_rope_x2_ref,  # [2, batch_size, page_size, rope_dim]
    bq_x2_ref,  # [2, batch_size, num_q_heads, head_dim]
    bo_x2_ref,  # [2, batch_size, num_q_heads, head_dim]
    bl_x2_ref,  # [2, batch_size, num_l_heads]
    bm_x2_ref,  # [2, batch_size, num_l_heads]
    swa_acc_x2_ref,  # [2, batch_size, num_q_heads, head_dim]
    sems,  # [7, 2, batch_size]
    *,
    sm_scale: float,
    batch_size: int = 1,
):
  assert q_hbm_ref.shape == o_hbm_ref.shape

  num_tokens, num_q_heads, head_dim = q_hbm_ref.shape
  _, page_size, _ = cache_kv_nope_hbm_ref.shape
  assert kv_lens_ref.shape[0] == num_tokens
  bkv_sz = page_size

  q_dtype = q_hbm_ref.dtype
  q_packing = get_dtype_packing(q_dtype)
  # Validate against the KV dtype.
  assert o_hbm_ref.dtype == q_dtype

  assert head_dim % 128 == 0
  assert num_q_heads % q_packing == 0

  start_seq_idx = start_end_seq_idx_ref[0]
  end_seq_idx = start_end_seq_idx_ref[1]

  batch_start_seq_idx = start_seq_idx + pl.program_id(0) * batch_size
  batch_end_seq_idx = batch_start_seq_idx + batch_size - 1

  def flash_attention_step1_qk_softmax(
      q,  # [bq_sz * num_q_heads, head_dim]
      kv,  # [bkv_sz, head_dim] <- Correspond to data from bkv_*_x2_ref
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

  def _fetch_bkv(seq_idx, bkv_sem_idx, batch_idx, *, wait=False):
    sem_nope = sems.at[0, bkv_sem_idx, batch_idx]
    sem_rope = sems.at[6, bkv_sem_idx, batch_idx]

    bkv_nope_vmem_ref = bkv_nope_x2_ref.at[bkv_sem_idx, batch_idx]
    bkv_rope_vmem_ref = bkv_rope_x2_ref.at[bkv_sem_idx, batch_idx]

    # The index into cache_kv_hbm_ref should be relative to the current
    # chunk.
    page_idx = seq_idx - start_seq_idx
    if not wait:
      _async_copy(
          cache_kv_nope_hbm_ref.at[page_idx],
          bkv_nope_vmem_ref,
          sem_nope,
          wait,
      )
      _async_copy(
          cache_kv_rope_hbm_ref.at[page_idx],
          bkv_rope_vmem_ref,
          sem_rope,
          wait,
      )
    else:
      # When we wait, we can use a dummy copy to wait for DMAs to complete where
      # src == dst. However, the dma size must be correct.
      dst_nope = bkv_nope_vmem_ref
      _async_copy(
          src=dst_nope,
          dst=dst_nope,
          sem=sem_nope,
          wait=True,
      )
      dst_rope = bkv_rope_vmem_ref
      _async_copy(
          src=dst_rope,
          dst=dst_rope,
          sem=sem_rope,
          wait=True,
      )

  def _fetch_bq(seq_idx, bq_sem_idx, batch_idx, *, wait=False):
    sem = sems.at[1, bq_sem_idx, batch_idx]
    bq_vmem_ref = bq_x2_ref.at[bq_sem_idx, batch_idx]

    _async_copy(
        q_hbm_ref.at[seq_idx],
        bq_vmem_ref,
        sem,
        wait,
    )

  def _send_bo(seq_idx, bo_sem_idx, batch_idx, *, wait=False):
    sem = sems.at[2, bo_sem_idx, batch_idx]
    vmem_ref = bo_x2_ref.at[bo_sem_idx, batch_idx]

    _async_copy(
        vmem_ref,
        o_hbm_ref.at[seq_idx],
        sem,
        wait,
    )

  def _fetch_swa(seq_idx, bq_sem_idx, batch_idx, *, wait=False):
    sem_acc = sems.at[3, bq_sem_idx, batch_idx]
    sem_l = sems.at[4, bq_sem_idx, batch_idx]
    sem_m = sems.at[5, bq_sem_idx, batch_idx]

    if not wait:
      _async_copy(
          swa_accumution_hbm_ref.at[seq_idx],
          swa_acc_x2_ref.at[bq_sem_idx, batch_idx],
          sem_acc,
          wait=False,
      )
      _async_copy(
          swa_l_hbm_ref.at[seq_idx],
          bl_x2_ref.at[bq_sem_idx, batch_idx],
          sem_l,
          wait=False,
      )
      _async_copy(
          swa_m_hbm_ref.at[seq_idx],
          bm_x2_ref.at[bq_sem_idx, batch_idx],
          sem_m,
          wait=False,
      )

    else:
      dst_acc = swa_acc_x2_ref.at[bq_sem_idx, batch_idx]
      _async_copy(src=dst_acc, dst=dst_acc, sem=sem_acc, wait=True)

      dst_l = bl_x2_ref.at[bq_sem_idx, batch_idx]
      _async_copy(src=dst_l, dst=dst_l, sem=sem_l, wait=True)

      dst_m = bm_x2_ref.at[bq_sem_idx, batch_idx]
      _async_copy(src=dst_m, dst=dst_m, sem=sem_m, wait=True)

  def start_fetch_bkv(seq_idx, bkv_sem_idx, batch_idx):
    return _fetch_bkv(seq_idx, bkv_sem_idx, batch_idx)

  def wait_fetch_bkv(seq_idx, bkv_sem_idx, batch_idx):
    return _fetch_bkv(seq_idx, bkv_sem_idx, batch_idx, wait=True)

  def start_fetch_bq(seq_idx, bq_sem_idx, batch_idx):
    return _fetch_bq(seq_idx, bq_sem_idx, batch_idx)

  def wait_fetch_bq(seq_idx, bq_sem_idx, batch_idx):
    return _fetch_bq(seq_idx, bq_sem_idx, batch_idx, wait=True)

  def start_fetch_swa(seq_idx, bq_sem_idx, batch_idx):
    return _fetch_swa(seq_idx, bq_sem_idx, batch_idx)

  def wait_fetch_swa(seq_idx, bq_sem_idx, batch_idx):
    return _fetch_swa(seq_idx, bq_sem_idx, batch_idx, wait=True)

  def start_send_bo(seq_idx, bo_sem_idx, batch_idx):
    bo_ids_ref[bo_sem_idx, batch_idx] = seq_idx
    _send_bo(seq_idx, bo_sem_idx, batch_idx)

  def wait_send_bo(bo_sem_idx, batch_idx):
    old_seq_idx = bo_ids_ref[bo_sem_idx, batch_idx]

    @pl.when(0 <= old_seq_idx)
    def _():
      _send_bo(old_seq_idx, bo_sem_idx, batch_idx, wait=True)

  def load_bq(bq_sem_idx, batch_idx):
    q = bq_x2_ref.at[bq_sem_idx, batch_idx][...]
    return q

  def load_bkv(bkv_sem_idx, batch_idx):
    bkv_nope = bkv_nope_x2_ref.at[bkv_sem_idx, batch_idx][...]
    bkv_nope = _dequant_dsv4_fp8(bkv_nope)

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
      for batch_idx in range(batch_size):
        start_fetch_bq(next_seq_idx + batch_idx, next_bi_sem_idx, batch_idx)
        start_fetch_swa(next_seq_idx + batch_idx, next_bi_sem_idx, batch_idx)
        start_fetch_bkv(next_seq_idx + batch_idx, next_bi_sem_idx, batch_idx)

    bo_sem_idx = sem_ids_ref[1]
    sem_ids_ref[1] = lax.select(bo_sem_idx == 0, 1, 0)
    attention_sinks = attention_sinks_ref[...][..., None]

    prev_p = None
    prev_bkv = None
    prev_exp_m_diff = None
    prev_l = None
    prev_swa_acc = None

    for batch_idx in range(batch_size):

      # Wait for cur blocks if not ready yet
      wait_fetch_bq(batch_start_seq_idx + batch_idx, bi_sem_idx, batch_idx)
      wait_fetch_swa(batch_start_seq_idx + batch_idx, bi_sem_idx, batch_idx)
      wait_fetch_bkv(batch_start_seq_idx + batch_idx, bi_sem_idx, batch_idx)

      bkv = load_bkv(bi_sem_idx, batch_idx)
      bq = load_bq(bi_sem_idx, batch_idx)
      swa_acc, swa_l, swa_m = load_swa_output(bi_sem_idx, batch_idx)

      p, exp_m_diff, l = flash_attention_step1_qk_softmax(
          bq,
          bkv,
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

        # Wait for previous bo to be fully sent before storing new bo.
        wait_send_bo(bo_sem_idx, batch_idx - 1)
        # Store output from acc to bo.
        bo_x2_ref.at[bo_sem_idx, batch_idx - 1][...] = out
        # Send cur bo
        start_send_bo(
            batch_start_seq_idx + batch_idx - 1, bo_sem_idx, batch_idx - 1
        )

      prev_p = p
      prev_bkv = bkv
      prev_exp_m_diff = exp_m_diff
      prev_l = l
      prev_swa_acc = swa_acc

    # end of pipelining loop
    assert prev_p is not None
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

    # Wait for previous bo to be fully sent before storing new bo.
    wait_send_bo(bo_sem_idx, batch_size - 1)
    # Store output from acc to bo.
    bo_x2_ref.at[bo_sem_idx, batch_size - 1][...] = out
    # Send cur bo
    start_send_bo(
        batch_start_seq_idx + batch_size - 1, bo_sem_idx, batch_size - 1
    )

  ### ------- Kernel start ------- ###

  @pl.when(batch_start_seq_idx == start_seq_idx)
  def prologue():
    for batch_idx in range(batch_size):
      start_fetch_bq(batch_start_seq_idx + batch_idx, 0, batch_idx)
      start_fetch_swa(batch_start_seq_idx + batch_idx, 0, batch_idx)
      start_fetch_bkv(batch_start_seq_idx + batch_idx, 0, batch_idx)

  process()

  @pl.when(batch_end_seq_idx == end_seq_idx - 1)
  def epilogue():
    for i in range(2):
      for batch_idx in range(batch_size):
        wait_send_bo(i, batch_idx)

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
    gather_and_attention_chunk_size: int | None = None,
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
  if gather_and_attention_chunk_size is None:
    gather_and_attention_chunk_size = q.shape[0]

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

    page_size = cache_kv_nope.shape[1]
    bkv_nope_double_buf = pltpu.VMEM(
        (2, batch_size, page_size, *cache_kv_nope.shape[2:]),
        cache_kv_nope.dtype,
    )
    bkv_rope_double_buf = pltpu.VMEM(
        (2, batch_size, page_size, *cache_kv_rope.shape[2:]),
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
        pltpu.SemaphoreType.DMA((7, 2, batch_size)),
    ]

    scalar_prefetches = (
        kv_lens,
        jnp.array([start_seq_idx, end_seq_idx], jnp.int32),
        # (bi_sem_idx, bo_sem_idx)
        jnp.zeros((2,), jnp.int32),
        # (bo_sem_0_seq_idx, bo_sem_1_seq_idx)
        jnp.full((2, batch_size), -1, jnp.int32),
    )

    scope_name = f"MLA-p_{page_size}"
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

  # TODO: handle the case where q.shape[0] is not divisible by
  # gather_and_attention_chunk_size.
  assert q.shape[0] % gather_and_attention_chunk_size == 0
  num_chunks = q.shape[0] // gather_and_attention_chunk_size

  for i in range(num_chunks):
    start_pos = i * gather_and_attention_chunk_size
    end_pos = start_pos + gather_and_attention_chunk_size
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
    gathered_nope_buffer, gathered_rope_buffer = csa_gather(
        cache_kv_nope,
        cache_kv_rope,
        indices,
    )
    gathered_nope_buffer = gathered_nope_buffer.reshape(
        gather_and_attention_chunk_size, topk, -1
    )
    gathered_rope_buffer = gathered_rope_buffer.reshape(
        gather_and_attention_chunk_size, topk, -1
    )
    # We treat each query token as a one independent sequence, attend to their
    # respective gathered kv tokens in the `gathered_kv_buffer`.
    # -1 in topk_indices is padded elements at the end of each row.
    # Batching
    assert gather_and_attention_chunk_size % attention_kernel_batch_size == 0
    batch_end = (
        cdiv(
            jnp.minimum(
                cu_q_lens[distribution[2]],
                start_pos + gather_and_attention_chunk_size,
            ),
            attention_kernel_batch_size,
        )
        * attention_kernel_batch_size
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
        kernel_batch_size=attention_kernel_batch_size,
    )
  return prepare_outputs(
      q, actual_num_q_heads, actual_head_dim
  )  # [max_num_tokens, actual_num_q_heads, actual_head_dim]

def computation(
    q: jax.Array,
    cache_kv_nope: jax.Array,
    cache_kv_rope: jax.Array,
    topk_indices: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    attention_sinks: jax.Array,
    swa_accumution: jax.Array,
    swa_l: jax.Array,
    swa_m: jax.Array,
):
    sm_scale = float(512 ** -0.5)
    gather_and_attention_chunk_size = None
    attention_kernel_batch_size = 16
    vmem_limit_bytes = 100 * 1024 * 1024

    return sparse_ragged_paged_attention(
        q,
        cache_kv_nope,
        cache_kv_rope,
        topk_indices,
        page_indices,
        cu_q_lens,
        distribution,
        attention_sinks,
        swa_accumution,
        swa_l,
        swa_m,
        sm_scale=sm_scale,
        gather_and_attention_chunk_size=gather_and_attention_chunk_size,
        attention_kernel_batch_size=attention_kernel_batch_size,
        vmem_limit_bytes=vmem_limit_bytes,
    )
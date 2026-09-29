---
title: 'Module: python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention_v3`/
symbols:
  ragged_paged_attention: ragged_paged_attention().
  get_default_block_sizes: get_default_block_sizes().
  RpaCase: RpaCase#
  ragged_paged_attention.run_rpa_kernel: ragged_paged_attention().run_rpa_kernel().
  _ragged_paged_attention_kernel_loop.process.compute_with_bq: _ragged_paged_attention_kernel_loop().process().compute_with_bq().
  get_vmem_estimate_bytes: get_vmem_estimate_bytes().
  _ragged_paged_attention_kernel_loop: _ragged_paged_attention_kernel_loop().
  RpaCase.symbol: RpaCase#symbol().
  RpaCase.get_range: RpaCase#get_range().
  _ragged_paged_attention_kernel_loop.process.compute_with_bq.compute_with_bkv.attention_loop: _ragged_paged_attention_kernel_loop().process().compute_with_bq().compute_with_bkv().attention_loop().
  _ragged_paged_attention_kernel_loop._fetch_bkv: _ragged_paged_attention_kernel_loop()._fetch_bkv().
  _ragged_paged_attention_kernel_loop._update_kv_cache: _ragged_paged_attention_kernel_loop()._update_kv_cache().
  _ragged_paged_attention_kernel_loop._async_copy: _ragged_paged_attention_kernel_loop()._async_copy().
  _ragged_paged_attention_kernel_loop.flash_attention_step1_qk_softmax: _ragged_paged_attention_kernel_loop().flash_attention_step1_qk_softmax().
  prepare_inputs: prepare_inputs().
  ragged_paged_attention._prepare_block_sizes: ragged_paged_attention()._prepare_block_sizes().
  _ragged_paged_attention_kernel_loop.process.compute_with_bq.compute_with_bkv: _ragged_paged_attention_kernel_loop().process().compute_with_bq().compute_with_bkv().
  _ragged_paged_attention_kernel_loop.prologue: _ragged_paged_attention_kernel_loop().prologue().
  RpaCase.DECODE: RpaCase#DECODE.
  RpaCase.MIXED: RpaCase#MIXED.
  _ragged_paged_attention_kernel_loop.strided_load: _ragged_paged_attention_kernel_loop().strided_load().
  merge_kv: merge_kv().
  static_validate_inputs: static_validate_inputs().
  get_vmem_limit: get_vmem_limit().
  _ragged_paged_attention_kernel_loop.broadcast_minor: _ragged_paged_attention_kernel_loop().broadcast_minor().
  prepare_kv_cache_fused: prepare_kv_cache_fused().
  get_smem_estimate_bytes: get_smem_estimate_bytes().
  get_kv_cache_shape: get_kv_cache_shape().
  _ragged_paged_attention_kernel_loop.process.compute_with_bq.compute_with_bkv.prefetch_next_bkv: _ragged_paged_attention_kernel_loop().process().compute_with_bq().compute_with_bkv().prefetch_next_bkv().
  _ragged_paged_attention_kernel_loop.epilogue: _ragged_paged_attention_kernel_loop().epilogue().
  dynamic_validate_inputs: dynamic_validate_inputs().
  _ragged_paged_attention_kernel_loop._fetch_mask: _ragged_paged_attention_kernel_loop()._fetch_mask().
  _ragged_paged_attention_kernel_loop._fetch_bq: _ragged_paged_attention_kernel_loop()._fetch_bq().
  _ragged_paged_attention_kernel_loop._send_bo: _ragged_paged_attention_kernel_loop()._send_bo().
  _ragged_paged_attention_kernel_loop.start_fetch_mask: _ragged_paged_attention_kernel_loop().start_fetch_mask().
  _ragged_paged_attention_kernel_loop.start_fetch_bkv: _ragged_paged_attention_kernel_loop().start_fetch_bkv().
  _ragged_paged_attention_kernel_loop.start_fetch_bq: _ragged_paged_attention_kernel_loop().start_fetch_bq().
  _ragged_paged_attention_kernel_loop.wait_send_bo: _ragged_paged_attention_kernel_loop().wait_send_bo().
  _ragged_paged_attention_kernel_loop.wait_update_kv_cache: _ragged_paged_attention_kernel_loop().wait_update_kv_cache().
  RpaCase.PREFILL: RpaCase#PREFILL.
  _ragged_paged_attention_kernel_loop.mask_and: _ragged_paged_attention_kernel_loop().mask_and().
  _semaphore_kwargs: _semaphore_kwargs().
  _ragged_paged_attention_kernel: _ragged_paged_attention_kernel().
  _ragged_paged_attention_kernel_loop.flash_attention_step2_pv: _ragged_paged_attention_kernel_loop().flash_attention_step2_pv().
  _ragged_paged_attention_kernel_loop._fetch_mask.loop_body: _ragged_paged_attention_kernel_loop()._fetch_mask().loop_body().
  _ragged_paged_attention_kernel_loop._update_kv_cache.loop_body: _ragged_paged_attention_kernel_loop()._update_kv_cache().loop_body().
  _ragged_paged_attention_kernel_loop.wait_fetch_mask: _ragged_paged_attention_kernel_loop().wait_fetch_mask().
  _ragged_paged_attention_kernel_loop.wait_fetch_bkv: _ragged_paged_attention_kernel_loop().wait_fetch_bkv().
  _ragged_paged_attention_kernel_loop.wait_fetch_bq: _ragged_paged_attention_kernel_loop().wait_fetch_bq().
  _ragged_paged_attention_kernel_loop.start_send_bo: _ragged_paged_attention_kernel_loop().start_send_bo().
  _ragged_paged_attention_kernel_loop.start_update_kv_cache: _ragged_paged_attention_kernel_loop().start_update_kv_cache().
  _ragged_paged_attention_kernel_loop.strided_store: _ragged_paged_attention_kernel_loop().strided_store().
  _ragged_paged_attention_kernel_loop.load_bq: _ragged_paged_attention_kernel_loop().load_bq().
  _ragged_paged_attention_kernel_loop.load_bkv: _ragged_paged_attention_kernel_loop().load_bkv().
  _ragged_paged_attention_kernel_loop.process: _ragged_paged_attention_kernel_loop().process().
  DEFAULT_MASK_VALUE: DEFAULT_MASK_VALUE.
  static_validate_inputs._validate_block_sizes: static_validate_inputs()._validate_block_sizes().
  ref_ragged_paged_attention: ref_ragged_paged_attention().
  _ragged_paged_attention_kernel_loop.process.compute_with_bq.prefetch_next_bq: _ragged_paged_attention_kernel_loop().process().compute_with_bq().prefetch_next_bq().
  _ragged_paged_attention_kernel_loop.process.compute_with_bq.compute_with_bkv.wait_cur_bq: _ragged_paged_attention_kernel_loop().process().compute_with_bq().compute_with_bkv().wait_cur_bq().
  _ragged_paged_attention_kernel_loop.process.compute_with_bq.compute_with_bkv.update_cur_bkv_to_cache: _ragged_paged_attention_kernel_loop().process().compute_with_bq().compute_with_bkv().update_cur_bkv_to_cache().
  _ragged_paged_attention_kernel_loop.pipeline: _ragged_paged_attention_kernel_loop().pipeline().
  has_bank_conflicts: has_bank_conflicts().
  ragged_paged_attention.run_rpa_kernel.run: ragged_paged_attention().run_rpa_kernel().run().
  DEFAULT_VMEM_LIMIT_BYTES: DEFAULT_VMEM_LIMIT_BYTES.
  logger: logger.
  _COMPILER_PARAMS_SUPPORTS_SEMAPHORE: _COMPILER_PARAMS_SUPPORTS_SEMAPHORE.
  _ragged_paged_attention_kernel_loop.process.get_next_bq_ids: _ragged_paged_attention_kernel_loop().process().get_next_bq_ids().
  _ragged_paged_attention_kernel_loop.process.get_next_bkv_ids: _ragged_paged_attention_kernel_loop().process().get_next_bkv_ids().
  prepare_outputs: prepare_outputs().
  prepare_updated_kv_cache_fused: prepare_updated_kv_cache_fused().
  _ragged_paged_attention_kernel_loop.debug_print: _ragged_paged_attention_kernel_loop().debug_print().
---
# Module: [`python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py)

## Classes
### `RpaCase`  ·  implements/extends Enum
- def: [`python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py:61`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L61)
- doc: Represents the different cases for Ragged Paged Attention.
- signature: `class RpaCase(Enum):`
- members:
  - `get_range(self, distribution)` — [`L81`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L81)
  - `symbol(self)` — [`L74`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L74)
  - `DECODE` — [`L69`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L69)
  - `MIXED` — [`L71`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L71)
  - `PREFILL` — [`L70`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L70)
- used by: [`ragged_paged_attention`](ragged_paged_attention_v3.md#ragged_paged_attention), [`get_default_block_sizes`](ragged_paged_attention_v3.md#get_default_block_sizes), [`run_rpa_kernel`](ragged_paged_attention_v3.md#ragged_paged_attention.run_rpa_kernel), [`_ragged_paged_attention_kernel_loop`](ragged_paged_attention_v3.md#_ragged_paged_attention_kernel_loop)

## Functions
- `_async_copy(src, dst, sem, wait)` — [`L558`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L558)
- `_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, *, wait=False)` — [`L615`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L615)
- `_fetch_bq(seq_idx, bq_idx, bq_sem_idx, *, wait=False)` — [`L722`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L722)
- `_fetch_mask(seq_idx, bq_idx, bkvmask_idx, bkvmask_sem_idx, *, wait=False)` — [`L567`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L567)
- `_prepare_block_sizes(block_sizes, case)` — [`L2013`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L2013)
- `_ragged_paged_attention_kernel(*args, **kwargs)` — [`L314`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L314)
- `_ragged_paged_attention_kernel_loop(seq_idx, kv_lens_ref, page_indices_ref, cu_q_lens_ref, cu_kv_lens_ref, cu_seq_mask_lens, distribution_ref, sem_ids_ref, bo_ids_ref, bkv_update_ids_ref, q_hbm_ref, kv_hbm_ref, kv_cache_hbm_ref, custom_mask_ref, zero_mask_ref, attention_sink_ref, o_hbm_ref, updated_kv_cache_hbm_ref, bkvmask_ref, bkv_x2_ref, bq_x2_ref, bo_x2_ref, sems, l_ref, m_ref, acc_ref, *, causal: bool = True, sm_scale: float, sliding_window: int | None = None, soft_cap: float | None = None, mask_value: float | None = None, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, xai_temperature_len: float | None = None, softmax_dtype: jnp.dtype | None = None, static_q_len: int | None = None, bq_sz, bkv_sz, bq_csz, bkv_csz, case: RpaCase = RpaCase.MIXED, skip_kv_mask: bool = False, tpu_version: int = 6, debug_mode: bool = False, mask_aligned_to_cu_kv: bool = False)` — [`L329`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L329)
- `_semaphore_kwargs(disable_semaphore_checks: bool)` — [`L55`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L55)
- `_send_bo(seq_idx, bo_idx, bo_sem_idx, *, wait=False)` — [`L736`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L736)
- `_update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz, *, wait=False)` — [`L671`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L671)
- `_validate_block_sizes(block_sizes, prefix)` — [`L1406`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L1406)
- `attention_loop(idx)` — [`L1030`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L1030)
- `broadcast_minor(src, shape)` — [`L865`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L865)
- `compute_with_bkv(bkv_idx)` — [`L977`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L977)
- `compute_with_bq(bq_idx)` — [`L918`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L918)
- `debug_print(msg, *args)` — [`L436`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L436)
- `dynamic_validate_inputs(queries, keys, values, kv_cache_fused, kv_lens, page_indices, cu_q_lens, cu_kv_lens, distribution, **kwargs)` — [`L1443`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L1443) — Runtime validation of dynamic tensor values.
- `epilogue()` — [`L1139`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L1139)
- `flash_attention_step1_qk_softmax(q, k, v, l_ref, m_ref, *, processed_q_len, processed_kv_len, effective_kv_len, xai_temperature_reg=None, custom_mask_data=None)` — [`L440`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L440)
- `flash_attention_step2_pv(p, v, exp_m_diff, o_ref)` — [`L539`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L539)
- `get_default_block_sizes(q_dtype, kv_dtype, actual_num_q_heads, actual_num_kv_heads, head_dim, page_size, max_num_tokens, max_num_seqs, pages_per_seq, *, case: RpaCase = RpaCase.MIXED, vmem_limit_bytes: int | None = None, use_custom_mask: bool = False, sliding_window: int | None = None)` — [`L1492`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L1492) — Get (bq_sz, bkv_sz, bq_csz, bkv_csz) by some heuristic formulas.
- `get_kv_cache_shape(total_num_pages, page_size, actual_num_kv_heads, actual_head_dim, kv_dtype)` — [`L297`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L297)
- `get_next_bkv_ids(seq_idx, bq_idx, bkv_idx, bkv_sem_idx, *, num_bkv)` — [`L898`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L898)
- `get_next_bq_ids(seq_idx, bq_idx, bq_sem_idx)` — [`L890`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L890)
- `get_smem_estimate_bytes(max_num_seqs, pages_per_seq)` — [`L206`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L206)
- `get_vmem_estimate_bytes(actual_num_kv_heads, actual_num_q_heads_per_kv_head, actual_head_dim, bq_sz, bkv_sz, q_dtype, kv_dtype, use_custom_mask=False, bkv_csz=None)` — [`L238`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L238)
- `get_vmem_limit()` — [`L1624`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L1624)
- `has_bank_conflicts(stride, distance=24, num_banks=32)` — [`L1147`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L1147)
- `load_bkv(bkv_sem_idx, kv_head_idx, start, sz)` — [`L839`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L839)
- `load_bq(bq_sem_idx, kv_head_idx, start, sz)` — [`L829`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L829)
- `loop_body(i, _)` — [`L596`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L596)
- `loop_body(i, states)` — [`L688`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L688)
- `mask_and(mask, new_mask)` — [`L875`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L875)
- `merge_kv(k: jax.Array, v: jax.Array)` — [`L1157`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L1157)
- `pipeline()` — [`L1135`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L1135)
- `prefetch_next_bkv()` — [`L989`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L989)
- `prefetch_next_bq()` — [`L972`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L972)
- `prepare_inputs(q: jax.Array, k: jax.Array, v: jax.Array, attention_sink: jax.Array | float | None = None)` — [`L1188`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L1188)
- `prepare_kv_cache_fused(kv_cache_fused: jax.Array)` — [`L1270`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L1270)
- `prepare_outputs(out, actual_num_q_heads_per_kv_head: int, actual_head_dim: int)` — [`L1245`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L1245)
- `prepare_updated_kv_cache_fused(kv_cache_fused, actual_num_kv_heads: int, actual_head_dim: int)` — [`L1297`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L1297) — Extract actual KV cache from processed fused format.
- `process(static_q_len=None)` — [`L880`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L880)
- `prologue()` — [`L1122`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L1122)
- `ragged_paged_attention(queries: jax.Array, keys: jax.Array, values: jax.Array, kv_cache_fused: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, cu_kv_lens: jax.Array, distribution: jax.Array, custom_mask: jax.Array | None, attention_sink: jax.Array | None = None, *, causal: int = 1, sm_scale: float = 1, sliding_window: int | None = None, soft_cap: float | None = None, mask_value: float | None = DEFAULT_MASK_VALUE, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, xai_temperature_len: float | None = None, softmax_dtype: jnp.dtype | None = None, chunk_prefill_size: int | None = None, d_block_sizes: tuple[int, int, int, int] | None = None, p_block_sizes: tuple[int, int, int, int] | None = None, m_block_sizes: tuple[int, int, int, int] | None = None, vmem_limit_bytes: int | None = None, out_dtype=None, skip_kv_mask: bool = False, disable_semaphore_checks: bool = True, debug_mode: bool = False, mask_aligned_to_cu_kv: bool = False)` — [`L1665`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L1665) — Ragged paged attention with fused KV cache.
- `ref_ragged_paged_attention(queries: jax.Array, k_pages: jax.Array, v_pages: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, num_seqs: jax.Array, *, custom_mask: jax.Array = None, causal: bool = True, sm_scale: float = 1, sliding_window: int | None = None, soft_cap: float | None = None, mask_value: float | None = DEFAULT_MASK_VALUE, k_scale: float | None = None, v_scale: float | None = None, xai_temperature_len: float | None = None, attention_sink: jax.Array | float | None = None, softmax_dtype: jnp.dtype | None = None)` — [`L93`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L93) — Reference implementation for ragged paged attention.
- `run(scalar_prefetches, q, kv, kv_cache)` — [`L1983`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L1983)
- `run_rpa_kernel(q, kv_cache, *, bq_sz, bkv_sz, bq_csz, bkv_csz, static_q_len=None, case: RpaCase = RpaCase.MIXED)` — [`L1816`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L1816)
- `start_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)` — [`L756`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L756)
- `start_fetch_bq(seq_idx, bq_idx, bq_sem_idx)` — [`L762`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L762)
- `start_fetch_mask(seq_idx, bq_idx, bkvmask_idx, bkvmask_sem_idx)` — [`L750`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L750)
- `start_send_bo(seq_idx, bo_idx, bo_sem_idx)` — [`L768`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L768)
- `start_update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz)` — [`L781`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L781)
- `static_validate_inputs(queries, keys, values, kv_cache_fused, kv_lens, page_indices, cu_q_lens, cu_kv_lens, distribution, *, causal: int = 1, sm_scale: float = 1, sliding_window: int | None = None, soft_cap: float | None = None, mask_value: float | None = None, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, xai_temperature_len: float | None = None, chunk_prefill_size: int | None = None, d_block_sizes: tuple[int, int, int, int] | None = None, p_block_sizes: tuple[int, int, int, int] | None = None, m_block_sizes: tuple[int, int, int, int] | None = None, vmem_limit_bytes: int | None = None, skip_kv_mask: bool = False, attention_sink=None)` — [`L1315`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L1315) — Validate inputs to the RPA kernel statically.
- `strided_load(ref, start, sz, step, *, dtype=None)` — [`L797`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L797)
- `strided_store(ref, start, sz, step, val)` — [`L813`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L813)
- `update_cur_bkv_to_cache()` — [`L1010`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L1010)
- `wait_cur_bq()` — [`L997`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L997)
- `wait_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)` — [`L759`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L759)
- `wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx)` — [`L765`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L765)
- `wait_fetch_mask(seq_idx, bq_idx, bkvmask_idx, bkvmask_sem_idx)` — [`L753`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L753)
- `wait_send_bo(bo_sem_idx)` — [`L773`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L773)
- `wait_update_kv_cache(bkv_sem_idx)` — [`L787`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L787)

## Module values
- `DEFAULT_MASK_VALUE` — [`L46`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L46)
- `DEFAULT_VMEM_LIMIT_BYTES` — [`L47`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L47)
- `_COMPILER_PARAMS_SUPPORTS_SEMAPHORE` — [`L50`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L50)
- `logger` — [`L48`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py#L48)


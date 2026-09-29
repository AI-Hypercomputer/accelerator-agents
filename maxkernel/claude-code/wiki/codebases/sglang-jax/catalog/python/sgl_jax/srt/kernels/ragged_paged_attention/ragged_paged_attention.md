---
title: 'Module: python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention`/
symbols:
  ragged_paged_attention: ragged_paged_attention().
  _ragged_paged_attention_kernel.process.compute_with_bq.compute_with_bkv: _ragged_paged_attention_kernel().process().compute_with_bq().compute_with_bkv().
  _ragged_paged_attention_kernel.process.compute_with_bq: _ragged_paged_attention_kernel().process().compute_with_bq().
  _ragged_paged_attention_kernel._fetch_bkv: _ragged_paged_attention_kernel()._fetch_bkv().
  _ragged_paged_attention_kernel._update_kv_cache: _ragged_paged_attention_kernel()._update_kv_cache().
  _ragged_paged_attention_kernel._async_copy: _ragged_paged_attention_kernel()._async_copy().
  prepare_inputs: prepare_inputs().
  _ragged_paged_attention_kernel.strided_load_bkv._convert_to_target_bitwidth: _ragged_paged_attention_kernel().strided_load_bkv()._convert_to_target_bitwidth().
  _ragged_paged_attention_kernel.process: _ragged_paged_attention_kernel().process().
  get_vmem_estimate_bytes: get_vmem_estimate_bytes().
  _ragged_paged_attention_kernel.prologue: _ragged_paged_attention_kernel().prologue().
  DEFAULT_MASK_VALUE: DEFAULT_MASK_VALUE.
  _ragged_paged_attention_kernel.flash_attention_step2_pv: _ragged_paged_attention_kernel().flash_attention_step2_pv().
  _ragged_paged_attention_kernel: _ragged_paged_attention_kernel().
  _ragged_paged_attention_kernel.flash_attention_step1_qk_softmax: _ragged_paged_attention_kernel().flash_attention_step1_qk_softmax().
  _ragged_paged_attention_kernel.strided_load_bkv: _ragged_paged_attention_kernel().strided_load_bkv().
  merge_kv: merge_kv().
  static_validate_inputs_fused: static_validate_inputs_fused().
  prepare_kv_cache_fused: prepare_kv_cache_fused().
  _ragged_paged_attention_kernel.strided_load: _ragged_paged_attention_kernel().strided_load().
  _ragged_paged_attention_kernel.broadcast_minor: _ragged_paged_attention_kernel().broadcast_minor().
  get_smem_estimate_bytes: get_smem_estimate_bytes().
  get_kv_cache_shape: get_kv_cache_shape().
  _ragged_paged_attention_kernel.process.compute_with_bq.compute_with_bkv.prefetch_next_bkv: _ragged_paged_attention_kernel().process().compute_with_bq().compute_with_bkv().prefetch_next_bkv().
  _ragged_paged_attention_kernel.epilogue: _ragged_paged_attention_kernel().epilogue().
  prepare_kv: prepare_kv().
  prepare_kv_cache: prepare_kv_cache().
  _ragged_paged_attention_kernel._fetch_mask: _ragged_paged_attention_kernel()._fetch_mask().
  _ragged_paged_attention_kernel._fetch_bq: _ragged_paged_attention_kernel()._fetch_bq().
  _ragged_paged_attention_kernel._send_bo: _ragged_paged_attention_kernel()._send_bo().
  _ragged_paged_attention_kernel.start_fetch_mask: _ragged_paged_attention_kernel().start_fetch_mask().
  _ragged_paged_attention_kernel.start_fetch_bkv: _ragged_paged_attention_kernel().start_fetch_bkv().
  _ragged_paged_attention_kernel.start_fetch_bq: _ragged_paged_attention_kernel().start_fetch_bq().
  _ragged_paged_attention_kernel.wait_send_bo: _ragged_paged_attention_kernel().wait_send_bo().
  _ragged_paged_attention_kernel.wait_update_kv_cache: _ragged_paged_attention_kernel().wait_update_kv_cache().
  _ragged_paged_attention_kernel._fetch_mask.loop_body: _ragged_paged_attention_kernel()._fetch_mask().loop_body().
  _ragged_paged_attention_kernel._fetch_bkv.loop_body: _ragged_paged_attention_kernel()._fetch_bkv().loop_body().
  _ragged_paged_attention_kernel._update_kv_cache.loop_body: _ragged_paged_attention_kernel()._update_kv_cache().loop_body().
  _ragged_paged_attention_kernel.wait_fetch_mask: _ragged_paged_attention_kernel().wait_fetch_mask().
  _ragged_paged_attention_kernel.wait_fetch_bkv: _ragged_paged_attention_kernel().wait_fetch_bkv().
  _ragged_paged_attention_kernel.wait_fetch_bq: _ragged_paged_attention_kernel().wait_fetch_bq().
  _ragged_paged_attention_kernel.start_send_bo: _ragged_paged_attention_kernel().start_send_bo().
  _ragged_paged_attention_kernel.start_update_kv_cache: _ragged_paged_attention_kernel().start_update_kv_cache().
  ref_ragged_paged_attention_fused: ref_ragged_paged_attention_fused().
  ref_ragged_paged_attention: ref_ragged_paged_attention().
  _ragged_paged_attention_kernel.process.compute_with_bq.prefetch_next_bq: _ragged_paged_attention_kernel().process().compute_with_bq().prefetch_next_bq().
  _ragged_paged_attention_kernel.process.compute_with_bq.compute_with_bkv.wait_cur_bq: _ragged_paged_attention_kernel().process().compute_with_bq().compute_with_bkv().wait_cur_bq().
  _ragged_paged_attention_kernel.process.compute_with_bq.compute_with_bkv.update_cur_bkv_to_cache: _ragged_paged_attention_kernel().process().compute_with_bq().compute_with_bkv().update_cur_bkv_to_cache().
  _ragged_paged_attention_kernel.process_decode: _ragged_paged_attention_kernel().process_decode().
  _ragged_paged_attention_kernel.process_prefill: _ragged_paged_attention_kernel().process_prefill().
  _ragged_paged_attention_kernel.process_mixed: _ragged_paged_attention_kernel().process_mixed().
  _ragged_paged_attention_kernel.flash_attention_step1_qk_softmax.load_with_init: _ragged_paged_attention_kernel().flash_attention_step1_qk_softmax().load_with_init().
  DEFAULT_VMEM_LIMIT_BYTES: DEFAULT_VMEM_LIMIT_BYTES.
  logger: logger.
  _ragged_paged_attention_kernel.flash_attention_step2_pv.load_with_init: _ragged_paged_attention_kernel().flash_attention_step2_pv().load_with_init().
  _ragged_paged_attention_kernel.load_bq: _ragged_paged_attention_kernel().load_bq().
  _ragged_paged_attention_kernel.process.get_next_bq_ids: _ragged_paged_attention_kernel().process().get_next_bq_ids().
  _ragged_paged_attention_kernel.process.get_next_bkv_ids: _ragged_paged_attention_kernel().process().get_next_bkv_ids().
  _ragged_paged_attention_kernel.process.compute_with_bq.compute_with_bkv.load_mask: _ragged_paged_attention_kernel().process().compute_with_bq().compute_with_bkv().load_mask().
  prepare_outputs: prepare_outputs().
  get_kernel_scope_name: get_kernel_scope_name().
  prepare_updated_kv_cache_fused: prepare_updated_kv_cache_fused().
  prepare_updated_kv_cache: prepare_updated_kv_cache().
---
# Module: [`python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py)

## Functions
- `_async_copy(src, dst, sem, wait)` — [`L488`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L488)
- `_convert_to_target_bitwidth(val, target_bitwidth: int)` — [`L789`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L789)
- `_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, *, wait=False)` — [`L541`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L541)
- `_fetch_bq(seq_idx, bq_idx, bq_sem_idx, *, wait=False)` — [`L665`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L665)
- `_fetch_mask(seq_idx, bq_idx, bkvmask_idx, bkvmask_sem_idx, *, wait=False)` — [`L495`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L495)
- `_ragged_paged_attention_kernel(kv_lens_ref, page_indices_ref, cu_q_lens_ref, cu_kv_lens_ref, cu_seq_mask_lens, distribution_ref, sem_ids_ref, bo_ids_ref, bkv_update_ids_ref, q_hbm_ref, kv_hbm_ref, kv_cache_fused_hbm_ref, custom_mask_ref, zero_mask_ref, o_hbm_ref, updated_kv_cache_fused_hbm_ref, bkvmask_ref, bkv_fused_x2_ref, bq_x2_ref, bo_x2_ref, sems, l_ref, m_ref, acc_ref, *, sm_scale: float, sliding_window: int | None = None, soft_cap: float | None = None, mask_value: float = DEFAULT_MASK_VALUE, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, xai_temperature_len: float | None = None, softmax_dtype: jnp.dtype | None = None, chunk_prefill_size: int | None = None, bkv_p, bq_sz)` — [`L302`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L302)
- `_send_bo(seq_idx, bo_idx, bo_sem_idx, *, wait=False)` — [`L679`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L679)
- `_update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz, *, wait=False)` — [`L613`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L613)
- `broadcast_minor(src, shape)` — [`L825`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L825)
- `compute_with_bkv(bkv_idx, _)` — [`L910`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L910)
- `compute_with_bq(bq_idx, _)` — [`L879`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L879)
- `epilogue()` — [`L1097`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L1097)
- `flash_attention_step1_qk_softmax(q, k, v, *, bkv_idx, kv_head_idx, q_span, k_span, mask, xai_temperature_reg=None)` — [`L394`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L394)
- `flash_attention_step2_pv(q_shape_0, v, p, exp_m_diff, *, bkv_idx, kv_head_idx)` — [`L467`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L467)
- `get_kernel_scope_name(bq_size, bkv_p, page_size)` — [`L1370`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L1370)
- `get_kv_cache_shape(total_num_pages, page_size, actual_num_kv_heads, actual_head_dim, kv_dtype)` — [`L285`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L285)
- `get_next_bkv_ids(seq_idx, bq_idx, bkv_idx, bkv_sem_idx)` — [`L853`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L853)
- `get_next_bq_ids(seq_idx, bq_idx, bq_sem_idx)` — [`L845`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L845)
- `get_smem_estimate_bytes(max_num_seqs, pages_per_seq)` — [`L226`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L226)
- `get_vmem_estimate_bytes(actual_num_kv_heads, actual_num_q_heads_per_kv_head, actual_head_dim, bq_sz, bkv_sz, q_dtype, kv_dtype)` — [`L252`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L252)
- `load_bq(bq_sem_idx, kv_head_idx, *, actual_bq_sz=bq_sz)` — [`L740`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L740)
- `load_mask(q_span, k_span)` — [`L947`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L947)
- `load_with_init(ref, init_val)` — [`L414`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L414)
- `load_with_init(ref, init_val)` — [`L478`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L478)
- `loop_body(i, _)` — [`L518`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L518)
- `loop_body(i, offset)` — [`L571`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L571)
- `loop_body(i, states)` — [`L633`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L633)
- `merge_kv(k: jax.Array, v: jax.Array)` — [`L1105`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L1105)
- `prefetch_next_bkv()` — [`L922`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L922)
- `prefetch_next_bq()` — [`L906`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L906)
- `prepare_inputs(q: jax.Array, k: jax.Array, v: jax.Array)` — [`L1179`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L1179)
- `prepare_kv(k: jax.Array, v: jax.Array)` — [`L1137`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L1137)
- `prepare_kv_cache(k_cache: jax.Array, v_cache: jax.Array)` — [`L1790`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L1790)
- `prepare_kv_cache_fused(kv_cache_fused: jax.Array)` — [`L1737`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L1737)
- `prepare_outputs(out, actual_num_q_heads_per_kv_head: int, actual_head_dim: int)` — [`L1220`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L1220)
- `prepare_updated_kv_cache(kv_cache, actual_num_kv_heads: int, actual_head_dim: int)` — [`L1715`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L1715) — return [total_num_pages, page_size , actual_num_kv_heads, actual_head_dim]
- `prepare_updated_kv_cache_fused(kv_cache_fused, actual_num_kv_heads: int, actual_head_dim: int)` — [`L1768`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L1768) — Extract actual KV cache from processed fused format.
- `process(static_q_len=None)` — [`L836`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L836)
- `process_decode()` — [`L1085`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L1085)
- `process_mixed()` — [`L1093`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L1093)
- `process_prefill()` — [`L1089`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L1089)
- `prologue()` — [`L1069`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L1069)
- `ragged_paged_attention(queries: jax.Array, keys: jax.Array, values: jax.Array, kv_cache_fused: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, cu_kv_lens: jax.Array, distribution: jax.Array, custom_mask: jax.Array, *, causal: int = 1, sm_scale: float = 1, sliding_window: int | None = None, soft_cap: float | None = None, mask_value: float | None = DEFAULT_MASK_VALUE, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, xai_temperature_len: float | None = None, softmax_dtype: jnp.dtype | None = None, chunk_prefill_size: int | None = None, num_kv_pages_per_block: int | None = None, num_queries_per_block: int | None = None, vmem_limit_bytes: int | None = None)` — [`L1394`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L1394) — Ragged paged attention that supports mixed prefill and decode with fused KV cache.
- `ref_ragged_paged_attention(queries: jax.Array, k_pages: jax.Array, v_pages: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, num_seqs: jax.Array, *, custom_mask: jax.Array = None, causal: bool = True, sm_scale: float = 1, sliding_window: int | None = None, soft_cap: float | None = None, mask_value: float | None = DEFAULT_MASK_VALUE, k_scale: float | None = None, v_scale: float | None = None, xai_temperature_len: float | None = None, softmax_dtype: jnp.dtype | None = None)` — [`L122`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L122)
- `ref_ragged_paged_attention_fused(queries: jax.Array, kv_pages_fused: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, num_seqs: jax.Array, *, sm_scale: float = 1, sliding_window: int | None = None, soft_cap: float | None = None, mask_value: float | None = DEFAULT_MASK_VALUE, k_scale: float | None = None, v_scale: float | None = None, xai_temperature_len: float | None = None, softmax_dtype: jnp.dtype | None = None)` — [`L35`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L35)
- `start_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)` — [`L699`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L699)
- `start_fetch_bq(seq_idx, bq_idx, bq_sem_idx)` — [`L705`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L705)
- `start_fetch_mask(seq_idx, bq_idx, bkvmask_idx, bkvmask_sem_idx)` — [`L693`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L693)
- `start_send_bo(seq_idx, bo_idx, bo_sem_idx)` — [`L711`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L711)
- `start_update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz)` — [`L724`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L724)
- `static_validate_inputs_fused(queries: jax.Array, keys: jax.Array, values: jax.Array, kv_cache_fused: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, cu_kv_lens: jax.Array, distribution: jax.Array, *, sm_scale: float = 1, sliding_window: int | None = None, soft_cap: float | None = None, mask_value: float | None = DEFAULT_MASK_VALUE, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, xai_temperature_len: float | None = None, softmax_dtype: jnp.dtype | None = None, chunk_prefill_size: int | None = None, num_kv_pages_per_block: int | None = None, num_queries_per_block: int | None = None, vmem_limit_bytes: int | None = None)` — [`L1246`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L1246) — Validate inputs to the RPA kernel statically with fused KV cache.
- `strided_load(ref, start, step)` — [`L749`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L749)
- `strided_load_bkv(bkv_sem_idx, start, step)` — [`L761`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L761)
- `update_cur_bkv_to_cache()` — [`L944`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L944)
- `wait_cur_bq()` — [`L931`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L931)
- `wait_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)` — [`L702`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L702)
- `wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx)` — [`L708`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L708)
- `wait_fetch_mask(seq_idx, bq_idx, bkvmask_idx, bkvmask_sem_idx)` — [`L696`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L696)
- `wait_send_bo(bo_sem_idx)` — [`L716`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L716)
- `wait_update_kv_cache(bkv_sem_idx)` — [`L730`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L730)

## Module values
- `DEFAULT_MASK_VALUE` — [`L30`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L30)
- `DEFAULT_VMEM_LIMIT_BYTES` — [`L31`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L31)
- `logger` — [`L32`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention.py#L32)


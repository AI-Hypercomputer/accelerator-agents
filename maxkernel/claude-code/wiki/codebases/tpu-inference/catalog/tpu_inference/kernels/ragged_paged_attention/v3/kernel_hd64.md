---
title: 'Module: tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.ragged_paged_attention.v3.kernel_hd64`/
symbols:
  _ragged_paged_attention_kernel.debug_print: _ragged_paged_attention_kernel().debug_print().
  ragged_paged_attention_hd64: ragged_paged_attention_hd64().
  _ragged_paged_attention_kernel.process.compute_with_bq.compute_with_bkv: _ragged_paged_attention_kernel().process().compute_with_bq().compute_with_bkv().
  ref_ragged_paged_attention_hd64: ref_ragged_paged_attention_hd64().
  _ragged_paged_attention_kernel.process.compute_with_bq: _ragged_paged_attention_kernel().process().compute_with_bq().
  _ragged_paged_attention_kernel._update_kv_cache: _ragged_paged_attention_kernel()._update_kv_cache().
  _ragged_paged_attention_kernel._fetch_bkv: _ragged_paged_attention_kernel()._fetch_bkv().
  static_validate_inputs: static_validate_inputs().
  _ragged_paged_attention_kernel: _ragged_paged_attention_kernel().
  prepare_inputs: prepare_inputs().
  dynamic_validate_inputs: dynamic_validate_inputs().
  _ragged_paged_attention_kernel.process: _ragged_paged_attention_kernel().process().
  get_vmem_estimate_bytes: get_vmem_estimate_bytes().
  get_kv_cache_shape: get_kv_cache_shape().
  _ragged_paged_attention_kernel.flash_attention_step2_pv: _ragged_paged_attention_kernel().flash_attention_step2_pv().
  _ragged_paged_attention_kernel._async_copy: _ragged_paged_attention_kernel()._async_copy().
  _ragged_paged_attention_kernel._fetch_bq: _ragged_paged_attention_kernel()._fetch_bq().
  _ragged_paged_attention_kernel._send_bo: _ragged_paged_attention_kernel()._send_bo().
  merge_kv: merge_kv().
  DEFAULT_MASK_VALUE: DEFAULT_MASK_VALUE.
  _ragged_paged_attention_kernel.flash_attention_step1_qk_softmax: _ragged_paged_attention_kernel().flash_attention_step1_qk_softmax().
  _ragged_paged_attention_kernel._update_kv_cache.loop_body: _ragged_paged_attention_kernel()._update_kv_cache().loop_body().
  _ragged_paged_attention_kernel.broadcast_minor: _ragged_paged_attention_kernel().broadcast_minor().
  get_smem_estimate_bytes: get_smem_estimate_bytes().
  _ragged_paged_attention_kernel.prologue: _ragged_paged_attention_kernel().prologue().
  _ragged_paged_attention_kernel.epilogue: _ragged_paged_attention_kernel().epilogue().
  _ragged_paged_attention_kernel.start_fetch_bkv: _ragged_paged_attention_kernel().start_fetch_bkv().
  _ragged_paged_attention_kernel.start_fetch_bq: _ragged_paged_attention_kernel().start_fetch_bq().
  _ragged_paged_attention_kernel.wait_send_bo: _ragged_paged_attention_kernel().wait_send_bo().
  _ragged_paged_attention_kernel.wait_update_kv_cache: _ragged_paged_attention_kernel().wait_update_kv_cache().
  _ragged_paged_attention_kernel.wait_fetch_bkv: _ragged_paged_attention_kernel().wait_fetch_bkv().
  _ragged_paged_attention_kernel.wait_fetch_bq: _ragged_paged_attention_kernel().wait_fetch_bq().
  _ragged_paged_attention_kernel.start_send_bo: _ragged_paged_attention_kernel().start_send_bo().
  _ragged_paged_attention_kernel.start_update_kv_cache: _ragged_paged_attention_kernel().start_update_kv_cache().
  _ragged_paged_attention_kernel.strided_load: _ragged_paged_attention_kernel().strided_load().
  _ragged_paged_attention_kernel.strided_load_bkv: _ragged_paged_attention_kernel().strided_load_bkv().
  _ragged_paged_attention_kernel.process.compute_with_bq.prefetch_next_bq: _ragged_paged_attention_kernel().process().compute_with_bq().prefetch_next_bq().
  _ragged_paged_attention_kernel.process.compute_with_bq.compute_with_bkv.prefetch_next_bkv: _ragged_paged_attention_kernel().process().compute_with_bq().compute_with_bkv().prefetch_next_bkv().
  _ragged_paged_attention_kernel.process.compute_with_bq.compute_with_bkv.wait_cur_bq: _ragged_paged_attention_kernel().process().compute_with_bq().compute_with_bkv().wait_cur_bq().
  _ragged_paged_attention_kernel.process.compute_with_bq.compute_with_bkv.update_cur_bkv_to_cache: _ragged_paged_attention_kernel().process().compute_with_bq().compute_with_bkv().update_cur_bkv_to_cache().
  _ragged_paged_attention_kernel.process_decode: _ragged_paged_attention_kernel().process_decode().
  _ragged_paged_attention_kernel.process_prefill: _ragged_paged_attention_kernel().process_prefill().
  _ragged_paged_attention_kernel.process_mixed: _ragged_paged_attention_kernel().process_mixed().
  _ragged_paged_attention_kernel.flash_attention_step1_qk_softmax.load_with_init: _ragged_paged_attention_kernel().flash_attention_step1_qk_softmax().load_with_init().
  DEFAULT_VMEM_LIMIT_BYTES: DEFAULT_VMEM_LIMIT_BYTES.
  _ragged_paged_attention_kernel.flash_attention_step2_pv.load_with_init: _ragged_paged_attention_kernel().flash_attention_step2_pv().load_with_init().
  _ragged_paged_attention_kernel.load_bq: _ragged_paged_attention_kernel().load_bq().
  _ragged_paged_attention_kernel.process.get_next_bq_ids: _ragged_paged_attention_kernel().process().get_next_bq_ids().
  _ragged_paged_attention_kernel.process.get_next_bkv_ids: _ragged_paged_attention_kernel().process().get_next_bkv_ids().
  prepare_outputs: prepare_outputs().
  get_kernel_scope_name: get_kernel_scope_name().
---
# Module: [`tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py)

## Functions
- `_async_copy(src, dst, sem, wait)` — [`L466`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L466)
- `_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, *, wait=False)` — [`L476`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L476)
- `_fetch_bq(seq_idx, bq_idx, bq_sem_idx, *, wait=False)` — [`L624`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L624)
- `_ragged_paged_attention_kernel(kv_lens_ref, page_indices_ref, cu_q_lens_ref, distribution_ref, sem_ids_ref, bo_ids_ref, bkv_update_ids_ref, q_hbm_ref, kv_hbm_ref, kv_cache_hbm_ref, attention_sink_ref, o_hbm_ref, updated_kv_cache_hbm_ref, bkv_x2_ref, bq_x2_ref, bo_x2_ref, sems, l_ref, m_ref, acc_ref, *, sm_scale: float, sliding_window: int | None = None, soft_cap: float | None = None, mask_value: float = DEFAULT_MASK_VALUE, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, chunk_prefill_size: int | None = None, bkv_p, bq_sz, debug_mode: bool = False)` — [`L254`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L254)
- `_send_bo(seq_idx, bo_idx, bo_sem_idx, *, wait=False)` — [`L648`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L648)
- `_update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz, *, wait=False)` — [`L554`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L554)
- `broadcast_minor(src, shape)` — [`L752`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L752)
- `compute_with_bkv(bkv_idx, _)` — [`L817`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L817)
- `compute_with_bq(bq_idx, _)` — [`L806`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L806)
- `debug_print(msg, *args)` — [`L349`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L349)
- `dynamic_validate_inputs(queries: jax.Array, keys: jax.Array, values: jax.Array, kv_cache: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, distribution: jax.Array, attention_sink: jax.Array | None = None, *, sm_scale: float = 1, sliding_window: int | None = None, soft_cap: float | None = None, mask_value: float | None = DEFAULT_MASK_VALUE, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, chunk_prefill_size: int | None = None, num_kv_pages_per_block: int | None = None, num_queries_per_block: int | None = None, vmem_limit_bytes: int | None = None, debug_mode: bool = False)` — [`L1084`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L1084)
- `epilogue()` — [`L973`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L973)
- `flash_attention_step1_qk_softmax(q, kv, *, bq_idx, bkv_idx, kv_head_idx)` — [`L368`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L368)
- `flash_attention_step2_pv(kv, p, exp_m_diff, *, bkv_idx, kv_head_idx)` — [`L439`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L439)
- `get_kernel_scope_name(bq_size, bkv_p, page_size, sliding_window)` — [`L1318`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L1318)
- `get_kv_cache_shape(total_num_pages, page_size, actual_num_kv_heads, actual_head_dim, kv_dtype)` — [`L236`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L236)
- `get_next_bkv_ids(seq_idx, bq_idx, bkv_idx, bkv_sem_idx)` — [`L780`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L780)
- `get_next_bq_ids(seq_idx, bq_idx, bq_sem_idx)` — [`L772`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L772)
- `get_smem_estimate_bytes(max_num_seqs, pages_per_seq)` — [`L185`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L185)
- `get_vmem_estimate_bytes(actual_num_kv_heads, actual_num_q_heads_per_kv_head, actual_head_dim, bq_sz, bkv_sz, q_dtype, kv_dtype)` — [`L204`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L204)
- `load_bq(bq_sem_idx, kv_head_idx, *, actual_bq_sz=bq_sz)` — [`L717`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L717)
- `load_with_init(ref, init_val)` — [`L383`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L383)
- `load_with_init(ref, init_val)` — [`L454`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L454)
- `loop_body(i, states)` — [`L590`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L590)
- `merge_kv(k: jax.Array, v: jax.Array)` — [`L981`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L981)
- `prefetch_next_bkv()` — [`L828`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L828)
- `prefetch_next_bq()` — [`L813`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L813)
- `prepare_inputs(q: jax.Array, k: jax.Array, v: jax.Array, attention_sink: jax.Array | None = None)` — [`L1009`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L1009)
- `prepare_outputs(out, actual_num_q_heads_per_kv_head: int, actual_head_dim: int)` — [`L1060`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L1060)
- `process(static_q_len=None)` — [`L763`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L763)
- `process_decode()` — [`L961`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L961)
- `process_mixed()` — [`L969`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L969)
- `process_prefill()` — [`L965`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L965)
- `prologue()` — [`L946`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L946)
- `ragged_paged_attention_hd64(queries: jax.Array, keys: jax.Array, values: jax.Array, kv_cache: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, distribution: jax.Array, attention_sink: jax.Array | None = None, *, sm_scale: float = 1, sliding_window: int | None = None, soft_cap: float | None = None, mask_value: float | None = DEFAULT_MASK_VALUE, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, chunk_prefill_size: int | None = None, num_kv_pages_per_block: int | None = None, num_queries_per_block: int | None = None, vmem_limit_bytes: int | None = None, debug_mode: bool = False)` — [`L1342`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L1342) — A variant of ragged paged attention for head_dim=64.
- `ref_ragged_paged_attention_hd64(queries: jax.Array, keys: jax.Array, values: jax.Array, kv_cache: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, distribution: jax.Array, attention_sink: jax.Array | None = None, *, sm_scale: float = 1, sliding_window: int | None = None, soft_cap: float | None = None, mask_value: float | None = DEFAULT_MASK_VALUE, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None)` — [`L39`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L39)
- `start_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)` — [`L672`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L672)
- `start_fetch_bq(seq_idx, bq_idx, bq_sem_idx)` — [`L678`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L678)
- `start_send_bo(seq_idx, bo_idx, bo_sem_idx)` — [`L684`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L684)
- `start_update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz)` — [`L697`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L697)
- `static_validate_inputs(queries: jax.Array, keys: jax.Array, values: jax.Array, kv_cache: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, distribution: jax.Array, attention_sink: jax.Array | None = None, *, sm_scale: float = 1, sliding_window: int | None = None, soft_cap: float | None = None, mask_value: float | None = DEFAULT_MASK_VALUE, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, chunk_prefill_size: int | None = None, num_kv_pages_per_block: int | None = None, num_queries_per_block: int | None = None, vmem_limit_bytes: int | None = None, debug_mode: bool = False)` — [`L1176`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L1176) — Validate inputs to the RPA kernel statically.
- `strided_load(ref, start, step)` — [`L726`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L726)
- `strided_load_bkv(bkv_sem_idx, start, step)` — [`L734`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L734)
- `update_cur_bkv_to_cache()` — [`L845`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L845)
- `wait_cur_bq()` — [`L835`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L835)
- `wait_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)` — [`L675`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L675)
- `wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx)` — [`L681`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L681)
- `wait_send_bo(bo_sem_idx)` — [`L689`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L689)
- `wait_update_kv_cache(bkv_sem_idx)` — [`L703`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L703)

## Module values
- `DEFAULT_MASK_VALUE` — [`L32`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L32)
- `DEFAULT_VMEM_LIMIT_BYTES` — [`L34`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py#L34)


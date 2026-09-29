---
title: 'Module: ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.ragged_page_attention_v3._pallas_impl_fwd_h64`/
symbols:
  ragged_paged_attention: ragged_paged_attention().
  ref_ragged_paged_attention_hd64: ref_ragged_paged_attention_hd64().
  _ragged_paged_attention_kernel.process.compute_with_bq: _ragged_paged_attention_kernel().process().compute_with_bq().
  _ragged_paged_attention_kernel.process.compute_with_bq.compute_with_bkv: _ragged_paged_attention_kernel().process().compute_with_bq().compute_with_bkv().
  get_vmem_estimate_bytes: get_vmem_estimate_bytes().
  _ragged_paged_attention_kernel._fetch_bkv: _ragged_paged_attention_kernel()._fetch_bkv().
  static_validate_inputs: static_validate_inputs().
  _ragged_paged_attention_kernel: _ragged_paged_attention_kernel().
  prepare_inputs: prepare_inputs().
  dynamic_validate_inputs: dynamic_validate_inputs().
  _ragged_paged_attention_kernel.process: _ragged_paged_attention_kernel().process().
  _ragged_paged_attention_kernel._update_kv_cache: _ragged_paged_attention_kernel()._update_kv_cache().
  merge_kv: merge_kv().
  DEFAULT_MASK_VALUE: DEFAULT_MASK_VALUE.
  _clamp_block_sizes_to_vmem: _clamp_block_sizes_to_vmem().
  _ragged_paged_attention_kernel.flash_attention: _ragged_paged_attention_kernel().flash_attention().
  _ragged_paged_attention_kernel.broadcast_minor: _ragged_paged_attention_kernel().broadcast_minor().
  get_smem_estimate_bytes: get_smem_estimate_bytes().
  get_kv_cache_shape: get_kv_cache_shape().
  _ragged_paged_attention_kernel._async_copy: _ragged_paged_attention_kernel()._async_copy().
  _ragged_paged_attention_kernel.prologue: _ragged_paged_attention_kernel().prologue().
  _ragged_paged_attention_kernel.epilogue: _ragged_paged_attention_kernel().epilogue().
  _ragged_paged_attention_kernel._fetch_bq: _ragged_paged_attention_kernel()._fetch_bq().
  _ragged_paged_attention_kernel._send_bo: _ragged_paged_attention_kernel()._send_bo().
  _ragged_paged_attention_kernel.start_fetch_bkv: _ragged_paged_attention_kernel().start_fetch_bkv().
  _ragged_paged_attention_kernel.start_fetch_bq: _ragged_paged_attention_kernel().start_fetch_bq().
  _ragged_paged_attention_kernel.wait_send_bo: _ragged_paged_attention_kernel().wait_send_bo().
  _ragged_paged_attention_kernel.wait_update_kv_cache: _ragged_paged_attention_kernel().wait_update_kv_cache().
  _ragged_paged_attention_kernel._fetch_bkv.loop_body: _ragged_paged_attention_kernel()._fetch_bkv().loop_body().
  _ragged_paged_attention_kernel._update_kv_cache.loop_body: _ragged_paged_attention_kernel()._update_kv_cache().loop_body().
  _ragged_paged_attention_kernel.wait_fetch_bkv: _ragged_paged_attention_kernel().wait_fetch_bkv().
  _ragged_paged_attention_kernel.wait_fetch_bq: _ragged_paged_attention_kernel().wait_fetch_bq().
  _ragged_paged_attention_kernel.start_send_bo: _ragged_paged_attention_kernel().start_send_bo().
  _ragged_paged_attention_kernel.start_update_kv_cache: _ragged_paged_attention_kernel().start_update_kv_cache().
  _ragged_paged_attention_kernel.strided_load: _ragged_paged_attention_kernel().strided_load().
  _ragged_paged_attention_kernel.strided_load_bkv: _ragged_paged_attention_kernel().strided_load_bkv().
  _ragged_paged_attention_kernel.flash_attention.load_with_init: _ragged_paged_attention_kernel().flash_attention().load_with_init().
  _ragged_paged_attention_kernel._fetch_bkv._fetch_bkv_from_new_kv: _ragged_paged_attention_kernel()._fetch_bkv()._fetch_bkv_from_new_kv().
  _ragged_paged_attention_kernel.process.compute_with_bq.prefetch_next_bq: _ragged_paged_attention_kernel().process().compute_with_bq().prefetch_next_bq().
  _ragged_paged_attention_kernel.process.compute_with_bq.compute_with_bkv.prefetch_next_bkv: _ragged_paged_attention_kernel().process().compute_with_bq().compute_with_bkv().prefetch_next_bkv().
  _ragged_paged_attention_kernel.process.compute_with_bq.compute_with_bkv.wait_cur_bq: _ragged_paged_attention_kernel().process().compute_with_bq().compute_with_bkv().wait_cur_bq().
  _ragged_paged_attention_kernel.process.compute_with_bq.compute_with_bkv.update_cur_bkv_to_cache: _ragged_paged_attention_kernel().process().compute_with_bq().compute_with_bkv().update_cur_bkv_to_cache().
  _ragged_paged_attention_kernel.process_decode: _ragged_paged_attention_kernel().process_decode().
  _ragged_paged_attention_kernel.process_prefill: _ragged_paged_attention_kernel().process_prefill().
  _ragged_paged_attention_kernel.process_mixed: _ragged_paged_attention_kernel().process_mixed().
  DEFAULT_SCOPED_VMEM_LIMIT_BYTES: DEFAULT_SCOPED_VMEM_LIMIT_BYTES.
  _ragged_paged_attention_kernel.get_next_bkv_idx_start: _ragged_paged_attention_kernel().get_next_bkv_idx_start().
  _ragged_paged_attention_kernel.load_bq: _ragged_paged_attention_kernel().load_bq().
  _ragged_paged_attention_kernel.process.get_next_bq_ids: _ragged_paged_attention_kernel().process().get_next_bq_ids().
  _ragged_paged_attention_kernel.process.get_next_bkv_ids: _ragged_paged_attention_kernel().process().get_next_bkv_ids().
  prepare_outputs: prepare_outputs().
  DEFAULT_VMEM_LIMIT_BYTES: DEFAULT_VMEM_LIMIT_BYTES.
---
# Module: [`ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py)

## Functions
- `_async_copy(src, dst, sem, wait)` — [`L631`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L631)
- `_clamp_block_sizes_to_vmem(*, actual_num_kv_heads, actual_num_q_heads_per_kv_head, actual_head_dim, page_size, q_dtype, kv_dtype, bkv_p, bq_sz, vmem_limit_bytes)` — [`L342`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L342) — Clamp TPU block sizes until scratch buffers fit within scoped VMEM.
- `_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, *, wait=False)` — [`L638`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L638)
- `_fetch_bkv_from_new_kv()` — [`L679`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L679)
- `_fetch_bq(seq_idx, bq_idx, bq_sem_idx, *, wait=False)` — [`L728`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L728)
- `_ragged_paged_attention_kernel(kv_lens_ref, page_indices_ref, cu_q_lens_ref, distribution_ref, sem_ids_ref, bo_ids_ref, bkv_update_ids_ref, q_hbm_ref, kv_hbm_ref, kv_cache_hbm_ref, attention_sink_ref, o_hbm_ref, updated_kv_cache_hbm_ref, bkv_x2_ref, bq_x2_ref, bo_x2_ref, sems, l_ref, m_ref, acc_ref, *, softmax_scale: float, sliding_window: int | None = None, logits_soft_cap: float | None = None, mask_value: float = DEFAULT_MASK_VALUE, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, chunk_prefill_size: int | None = None, bkv_p, bq_sz)` — [`L431`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L431) — Pallas TPU kernel for ragged paged attention V3 with head_dim=64.
- `_send_bo(seq_idx, bo_idx, bo_sem_idx, *, wait=False)` — [`L742`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L742)
- `_update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz, *, wait=False)` — [`L690`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L690)
- `broadcast_minor(src, shape)` — [`L830`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L830)
- `compute_with_bkv(bkv_idx, _)` — [`L881`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L881)
- `compute_with_bq(bq_idx, _)` — [`L872`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L872)
- `dynamic_validate_inputs(queries: jax.Array, keys: jax.Array, values: jax.Array, kv_cache: jax.Array, kv_lens: jax.Array, block_tables: jax.Array, query_start_loc: jax.Array, distribution: jax.Array, softmax_aux: jax.Array | None = None, *, softmax_scale: float = 1, sliding_window: int | None = None, logits_soft_cap: float | None = None, mask_value: float | None = DEFAULT_MASK_VALUE, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, chunk_prefill_size: int | None = None, num_kv_pages_per_block: int | None = None, num_queries_per_block: int | None = None, vmem_limit_bytes: int | None = None)` — [`L1115`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L1115) — Validate inputs with both static and dynamic checks for head_dim=64.
- `epilogue()` — [`L965`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L965)
- `flash_attention(q, kv, *, bq_idx, bkv_idx, kv_head_idx)` — [`L560`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L560)
- `get_kv_cache_shape(total_num_pages, page_size, actual_num_kv_heads, actual_head_dim, kv_dtype)` — [`L396`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L396) — Compute the KV cache shape for head_dim=64 with K/V concatenated layout.
- `get_next_bkv_ids(seq_idx, bq_idx, bkv_idx, bkv_sem_idx)` — [`L856`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L856)
- `get_next_bkv_idx_start()` — [`L554`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L554)
- `get_next_bq_ids(seq_idx, bq_idx, bq_sem_idx)` — [`L848`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L848)
- `get_smem_estimate_bytes(max_num_seqs, pages_per_seq)` — [`L269`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L269) — Estimate scalar memory (SMEM) usage for the head_dim=64 kernel.
- `get_vmem_estimate_bytes(actual_num_kv_heads, actual_num_q_heads_per_kv_head, actual_head_dim, bq_sz, bkv_sz, q_dtype, kv_dtype)` — [`L295`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L295) — Estimate VMEM usage for the head_dim=64 kernel scratch buffers.
- `load_bq(bq_sem_idx, kv_head_idx, *, actual_bq_sz=bq_sz)` — [`L797`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L797)
- `load_with_init(ref, init_val)` — [`L576`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L576)
- `loop_body(i, offset)` — [`L660`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L660)
- `loop_body(i, states)` — [`L703`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L703)
- `merge_kv(k: jax.Array, v: jax.Array)` — [`L971`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L971) — Concatenate key and value tensors along head dimension for head_dim=64.
- `prefetch_next_bkv()` — [`L891`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L891)
- `prefetch_next_bq()` — [`L877`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L877)
- `prepare_inputs(q: jax.Array, k: jax.Array, v: jax.Array, softmax_aux: jax.Array | None = None)` — [`L1012`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L1012) — Transform inputs into TPU-optimized layout for head_dim=64.
- `prepare_outputs(out, actual_num_q_heads_per_kv_head: int, actual_head_dim: int)` — [`L1076`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L1076) — Transform kernel output back to standard shape for head_dim=64.
- `process(static_q_len=None)` — [`L839`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L839)
- `process_decode()` — [`L953`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L953)
- `process_mixed()` — [`L961`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L961)
- `process_prefill()` — [`L957`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L957)
- `prologue()` — [`L948`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L948)
- `ragged_paged_attention(queries: jax.Array, keys: jax.Array, values: jax.Array, kv_cache: jax.Array, kv_lens: jax.Array, block_tables: jax.Array, query_start_loc: jax.Array, distribution: jax.Array, softmax_aux: jax.Array | None = None, *, softmax_scale: float = 1, sliding_window: int | None = None, logits_soft_cap: float | None = None, mask_value: float | None = DEFAULT_MASK_VALUE, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, chunk_prefill_size: int | None = None, num_kv_pages_per_block: int | None = None, num_queries_per_block: int | None = None, vmem_limit_bytes: int | None = None)` — [`L1386`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L1386) — Ragged paged attention V3 for head_dim=64 with mixed prefill and decode.
- `ref_ragged_paged_attention_hd64(queries: jax.Array, keys: jax.Array, values: jax.Array, kv_cache: jax.Array, kv_lens: jax.Array, block_tables: jax.Array, query_start_loc: jax.Array, distribution: jax.Array, softmax_aux: jax.Array | None = None, *, softmax_scale: float = 1, sliding_window: int | None = None, logits_soft_cap: float | None = None, mask_value: float | None = DEFAULT_MASK_VALUE, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None)` — [`L88`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L88) — Reference implementation of V3 ragged paged attention for head_dim=64.
- `start_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)` — [`L756`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L756)
- `start_fetch_bq(seq_idx, bq_idx, bq_sem_idx)` — [`L762`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L762)
- `start_send_bo(seq_idx, bo_idx, bo_sem_idx)` — [`L768`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L768)
- `start_update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz)` — [`L781`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L781)
- `static_validate_inputs(queries: jax.Array, keys: jax.Array, values: jax.Array, kv_cache: jax.Array, kv_lens: jax.Array, block_tables: jax.Array, query_start_loc: jax.Array, distribution: jax.Array, softmax_aux: jax.Array | None = None, *, softmax_scale: float = 1, sliding_window: int | None = None, logits_soft_cap: float | None = None, mask_value: float | None = DEFAULT_MASK_VALUE, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, chunk_prefill_size: int | None = None, num_kv_pages_per_block: int | None = None, num_queries_per_block: int | None = None, vmem_limit_bytes: int | None = None)` — [`L1227`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L1227) — Validate static shape and dtype constraints for the head_dim=64 kernel.
- `strided_load(ref, start, step)` — [`L805`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L805)
- `strided_load_bkv(bkv_sem_idx, start, step, *, bkv_mask)` — [`L813`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L813)
- `update_cur_bkv_to_cache()` — [`L902`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L902)
- `wait_cur_bq()` — [`L896`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L896)
- `wait_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)` — [`L759`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L759)
- `wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx)` — [`L765`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L765)
- `wait_send_bo(bo_sem_idx)` — [`L773`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L773)
- `wait_update_kv_cache(bkv_sem_idx)` — [`L787`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L787)

## Module values
- `DEFAULT_MASK_VALUE` — [`L83`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L83)
- `DEFAULT_SCOPED_VMEM_LIMIT_BYTES` — [`L339`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L339)
- `DEFAULT_VMEM_LIMIT_BYTES` — [`L85`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd_h64.py#L85)


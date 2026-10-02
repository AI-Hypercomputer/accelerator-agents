---
title: 'Module: ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.multi_latent_ragged_page_attention._pallas_impl_fwd`/
symbols:
  mla_ragged_paged_attention: mla_ragged_paged_attention().
  _mla_ragged_paged_attention_kernel.process.compute_with_bq: _mla_ragged_paged_attention_kernel().process().compute_with_bq().
  _mla_ragged_paged_attention_kernel.process.compute_with_bq.compute_with_bkv: _mla_ragged_paged_attention_kernel().process().compute_with_bq().compute_with_bkv().
  ref_mla_ragged_paged_attention: ref_mla_ragged_paged_attention().
  _mla_ragged_paged_attention_kernel.prologue: _mla_ragged_paged_attention_kernel().prologue().
  static_validate_inputs: static_validate_inputs().
  dynamic_validate_inputs: dynamic_validate_inputs().
  _mla_ragged_paged_attention_kernel: _mla_ragged_paged_attention_kernel().
  DEFAULT_MASK_VALUE: DEFAULT_MASK_VALUE.
  _mla_ragged_paged_attention_kernel.process: _mla_ragged_paged_attention_kernel().process().
  _mla_ragged_paged_attention_kernel._fetch_bkv: _mla_ragged_paged_attention_kernel()._fetch_bkv().
  _mla_ragged_paged_attention_kernel.wait_send_bo: _mla_ragged_paged_attention_kernel().wait_send_bo().
  prepare_q_inputs: prepare_q_inputs().
  prepare_kv_inputs: prepare_kv_inputs().
  _CTRL_SEM_OFF: _CTRL_SEM_OFF.
  _CTRL_BO_OFF: _CTRL_BO_OFF.
  update_kv_cache: update_kv_cache().
  _mla_ragged_paged_attention_kernel.flash_attention: _mla_ragged_paged_attention_kernel().flash_attention().
  _mla_ragged_paged_attention_kernel.start_send_bo: _mla_ragged_paged_attention_kernel().start_send_bo().
  _mla_ragged_paged_attention_kernel.start_fetch_bkv: _mla_ragged_paged_attention_kernel().start_fetch_bkv().
  _mla_ragged_paged_attention_kernel.broadcast_minor: _mla_ragged_paged_attention_kernel().broadcast_minor().
  _CTRL_DIST_OFF: _CTRL_DIST_OFF.
  get_kv_cache_shape: get_kv_cache_shape().
  _mla_ragged_paged_attention_kernel._async_copy: _mla_ragged_paged_attention_kernel()._async_copy().
  _mla_ragged_paged_attention_kernel.process.compute_with_bq.prefetch_next_bq: _mla_ragged_paged_attention_kernel().process().compute_with_bq().prefetch_next_bq().
  _mla_ragged_paged_attention_kernel.process.compute_with_bq.compute_with_bkv.prefetch_next_bkv: _mla_ragged_paged_attention_kernel().process().compute_with_bq().compute_with_bkv().prefetch_next_bkv().
  _mla_ragged_paged_attention_kernel._fetch_bq: _mla_ragged_paged_attention_kernel()._fetch_bq().
  _mla_ragged_paged_attention_kernel._send_bo: _mla_ragged_paged_attention_kernel()._send_bo().
  _mla_ragged_paged_attention_kernel.start_fetch_bq: _mla_ragged_paged_attention_kernel().start_fetch_bq().
  update_kv_cache.seq_loop_body: update_kv_cache().seq_loop_body().
  _mla_ragged_paged_attention_kernel._fetch_bkv.loop_body: _mla_ragged_paged_attention_kernel()._fetch_bkv().loop_body().
  _mla_ragged_paged_attention_kernel.wait_fetch_bkv: _mla_ragged_paged_attention_kernel().wait_fetch_bkv().
  _mla_ragged_paged_attention_kernel.wait_fetch_bq: _mla_ragged_paged_attention_kernel().wait_fetch_bq().
  _mla_ragged_paged_attention_kernel.process.get_next_bkv_ids: _mla_ragged_paged_attention_kernel().process().get_next_bkv_ids().
  _mla_ragged_paged_attention_kernel.prologue._update_seq_cache: _mla_ragged_paged_attention_kernel().prologue()._update_seq_cache().
  _mla_ragged_paged_attention_kernel.flash_attention.load_with_init: _mla_ragged_paged_attention_kernel().flash_attention().load_with_init().
  _mla_ragged_paged_attention_kernel.process.compute_with_bq.compute_with_bkv.wait_cur_bq: _mla_ragged_paged_attention_kernel().process().compute_with_bq().compute_with_bkv().wait_cur_bq().
  _mla_ragged_paged_attention_kernel.process_decode: _mla_ragged_paged_attention_kernel().process_decode().
  _mla_ragged_paged_attention_kernel.process_prefill: _mla_ragged_paged_attention_kernel().process_prefill().
  _mla_ragged_paged_attention_kernel.process_mixed: _mla_ragged_paged_attention_kernel().process_mixed().
  _mla_ragged_paged_attention_kernel.epilogue: _mla_ragged_paged_attention_kernel().epilogue().
  _CTRL_BKV_OFF: _CTRL_BKV_OFF.
  _mla_ragged_paged_attention_kernel._bkv_start_for_seq: _mla_ragged_paged_attention_kernel()._bkv_start_for_seq().
  _CTRL_SIZE: _CTRL_SIZE.
  update_kv_cache.seq_loop_body.token_loop_body: update_kv_cache().seq_loop_body().token_loop_body().
  _mla_ragged_paged_attention_kernel.load_bq: _mla_ragged_paged_attention_kernel().load_bq().
  _mla_ragged_paged_attention_kernel.load_bkv: _mla_ragged_paged_attention_kernel().load_bkv().
  _mla_ragged_paged_attention_kernel.process.get_next_bq_ids: _mla_ragged_paged_attention_kernel().process().get_next_bq_ids().
  _mla_ragged_paged_attention_kernel.prologue._update_seq_cache._write_token: _mla_ragged_paged_attention_kernel().prologue()._update_seq_cache()._write_token().
  prepare_outputs: prepare_outputs().
  DEFAULT_VMEM_LIMIT_BYTES: DEFAULT_VMEM_LIMIT_BYTES.
---
# Module: [`ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py)

## Functions
- `_async_copy(src, dst, sem, wait)` — [`L620`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L620)
- `_bkv_start_for_seq(si)` — [`L780`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L780) — Compute the first KV block index worth attending when sliding_window is set.
- `_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, *, wait=False)` — [`L627`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L627)
- `_fetch_bq(seq_idx, bq_idx, bq_sem_idx, *, wait=False)` — [`L684`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L684)
- `_mla_ragged_paged_attention_kernel(kv_lens_ref, page_indices_ref, cu_q_lens_ref, ctrl_ref, ql_nope_hbm_ref, q_pe_hbm_ref, new_kv_c_hbm_ref, new_k_pe_hbm_ref, cache_kv_hbm_ref, o_hbm_ref, updated_cache_kv_hbm_ref, bkvc_x2_ref, bkpe_x2_ref, bq_nope_x2_ref, bq_rope_x2_ref, bo_x2_ref, sems, l_ref, m_ref, acc_ref, kv_upd_cache_ref, kv_upd_kvc_ref, kv_upd_kpe_ref, *, sm_scale: float, sliding_window: int | None = None, soft_cap: float | None = None, mask_value: float = DEFAULT_MASK_VALUE, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, chunk_prefill_size: int | None = None, bkv_p, bq_sz)` — [`L484`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L484)
- `_send_bo(seq_idx, bo_idx, bo_sem_idx, *, wait=False)` — [`L707`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L707)
- `_update_seq_cache(i, _)` — [`L899`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L899)
- `_write_token(j, _)` — [`L906`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L906)
- `broadcast_minor(src, shape)` — [`L772`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L772)
- `compute_with_bkv(bkv_idx, _)` — [`L839`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L839)
- `compute_with_bq(bq_idx, _)` — [`L830`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L830)
- `dynamic_validate_inputs(ql_nope: jax.Array, q_pe: jax.Array, new_kv_c: jax.Array, new_k_pe: jax.Array, cache_kv: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, distribution: jax.Array, *, sm_scale: float = 1, sliding_window: int | None = None, soft_cap: float | None = None, mask_value: float | None = DEFAULT_MASK_VALUE, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, chunk_prefill_size: int | None = None, num_kv_pages_per_block: int | None = None, num_queries_per_block: int | None = None, vmem_limit_bytes: int | None = None)` — [`L288`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L288) — Validate inputs to the MLA RPA kernel dynamically.
- `epilogue()` — [`L996`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L996)
- `flash_attention(ql_nope, q_pe, kv_c, k_pe, *, bq_idx, bkv_idx)` — [`L556`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L556)
- `get_kv_cache_shape(total_num_pages, page_size, kv_dim, kv_dtype)` — [`L37`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L37) — Compute the canonical 4-D MLA paged KV cache tensor shape.
- `get_next_bkv_ids(seq_idx, bq_idx, bkv_idx, bkv_sem_idx)` — [`L810`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L810)
- `get_next_bq_ids(seq_idx, bq_idx, bq_sem_idx)` — [`L802`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L802)
- `load_bkv(bkv_sem_idx, *, bkvc_mask, bkpe_mask)` — [`L761`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L761)
- `load_bq(bq_sem_idx, *, actual_bq_sz=bq_sz)` — [`L746`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L746)
- `load_with_init(ref, init_val)` — [`L580`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L580)
- `loop_body(i, _)` — [`L643`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L643)
- `mla_ragged_paged_attention(queries_nope: jax.Array, queries_pe: jax.Array, keys_values: jax.Array, keys_pe: jax.Array, kv_cache: jax.Array, kv_lens: jax.Array, block_tables: jax.Array, query_start_loc: jax.Array, distribution: jax.Array, *, softmax_scale: float = 1, sliding_window: int | None = None, logits_soft_cap: float | None = None, mask_value: float | None = DEFAULT_MASK_VALUE, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, chunk_prefill_size: int | None = None, num_kv_pages_per_block: int | None = None, num_queries_per_block: int | None = None, vmem_limit_bytes: int | None = None)` — [`L1118`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L1118) — MLA ragged paged attention with cache update (TPU Pallas core impl).
- `prefetch_next_bkv()` — [`L851`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L851)
- `prefetch_next_bq()` — [`L835`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L835)
- `prepare_kv_inputs(kv: jax.Array)` — [`L1042`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L1042) — Re-shape and pad a KV vector tensor into the packed layout expected by the kernel.
- `prepare_outputs(out, actual_num_q_heads: int, actual_head_dim: int)` — [`L1069`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L1069) — Unpack the kernel output back to the canonical ``[tokens, heads, dim]`` layout.
- `prepare_q_inputs(q: jax.Array)` — [`L1001`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L1001) — Re-shape and pad a query tensor into the packed layout expected by the kernel.
- `process(static_q_len=None)` — [`L787`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L787)
- `process_decode()` — [`L984`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L984)
- `process_mixed()` — [`L992`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L992)
- `process_prefill()` — [`L988`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L988)
- `prologue()` — [`L892`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L892)
- `ref_mla_ragged_paged_attention(ql_nope: jax.Array, q_pe: jax.Array, new_kv_c: jax.Array, new_k_pe: jax.Array, cache_kv: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, distribution: jax.Array, *, sm_scale: float = 1, sliding_window: int | None = None, soft_cap: float | None = None, mask_value: float | None = DEFAULT_MASK_VALUE, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None)` — [`L124`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L124) — Reference (non-Pallas) implementation of MLA ragged paged attention.
- `seq_loop_body(i, cache_kv)` — [`L100`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L100)
- `start_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)` — [`L721`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L721)
- `start_fetch_bq(seq_idx, bq_idx, bq_sem_idx)` — [`L727`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L727)
- `start_send_bo(seq_idx, bo_idx, bo_sem_idx)` — [`L733`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L733)
- `static_validate_inputs(ql_nope: jax.Array, q_pe: jax.Array, new_kv_c: jax.Array, new_k_pe: jax.Array, cache_kv: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, distribution: jax.Array, *, sm_scale: float = 1, sliding_window: int | None = None, soft_cap: float | None = None, mask_value: float | None = DEFAULT_MASK_VALUE, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, chunk_prefill_size: int | None = None, num_kv_pages_per_block: int | None = None, num_queries_per_block: int | None = None, vmem_limit_bytes: int | None = None)` — [`L371`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L371) — Validate inputs to the MLA RPA kernel statically.
- `token_loop_body(j, cache_kv_)` — [`L105`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L105)
- `update_kv_cache(new_kv_c: jax.Array, new_k_pe: jax.Array, cache_kv: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, distribution: jax.Array)` — [`L73`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L73) — Update KV cache with new tokens.
- `wait_cur_bq()` — [`L856`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L856)
- `wait_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)` — [`L724`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L724)
- `wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx)` — [`L730`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L730)
- `wait_send_bo(bo_sem_idx)` — [`L738`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L738)

## Module values
- `DEFAULT_MASK_VALUE` — [`L26`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L26)
- `DEFAULT_VMEM_LIMIT_BYTES` — [`L28`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L28)
- `_CTRL_BKV_OFF` — [`L33`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L33)
- `_CTRL_BO_OFF` — [`L32`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L32)
- `_CTRL_DIST_OFF` — [`L30`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L30)
- `_CTRL_SEM_OFF` — [`L31`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L31)
- `_CTRL_SIZE` — [`L34`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/multi_latent_ragged_page_attention/_pallas_impl_fwd.py#L34)


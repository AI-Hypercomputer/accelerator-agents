---
title: 'Module: tpu_inference/kernels/mla/v1/kernel.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/mla/v1/kernel.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.mla.v1.kernel`/
symbols:
  DEFAULT_MASK_VALUE: DEFAULT_MASK_VALUE.
  DEFAULT_VMEM_LIMIT_BYTES: DEFAULT_VMEM_LIMIT_BYTES.
  get_kv_cache_shape: get_kv_cache_shape().
  update_kv_cache: update_kv_cache().
  ref_mla_ragged_paged_attention: ref_mla_ragged_paged_attention().
  dynamic_validate_inputs: dynamic_validate_inputs().
  static_validate_inputs: static_validate_inputs().
  _mla_ragged_paged_attention_kernel: _mla_ragged_paged_attention_kernel().
  prepare_q_inputs: prepare_q_inputs().
  prepare_kv_inputs: prepare_kv_inputs().
  prepare_outputs: prepare_outputs().
  mla_ragged_paged_attention: mla_ragged_paged_attention().
---
# Module: [`tpu_inference/kernels/mla/v1/kernel.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v1/kernel.py)

## Functions
- `_mla_ragged_paged_attention_kernel(kv_lens_ref, page_indices_ref, cu_q_lens_ref, distribution_ref, sem_ids_ref, bo_ids_ref, bkv_update_ids_ref, ql_nope_hbm_ref, q_pe_hbm_ref, new_kv_c_hbm_ref, new_k_pe_hbm_ref, cache_kv_hbm_ref, o_hbm_ref, updated_cache_kv_hbm_ref, bkvc_x2_ref, bkpe_x2_ref, bq_nope_x2_ref, bq_rope_x2_ref, bo_x2_ref, sems, l_ref, m_ref, acc_ref, *, sm_scale: float, sliding_window: int | None=None, soft_cap: float | None=None, mask_value: float=DEFAULT_MASK_VALUE, q_scale: float | None=None, k_scale: float | None=None, v_scale: float | None=None, chunk_prefill_size: int | None=None, bkv_p, bq_sz, debug_mode: bool=False)` — [`L489`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v1/kernel.py#L489)
- `dynamic_validate_inputs(ql_nope: jax.Array, q_pe: jax.Array, new_kv_c: jax.Array, new_k_pe: jax.Array, cache_kv: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, distribution: jax.Array, *, sm_scale: float=1.0, sliding_window: int | None=None, soft_cap: float | None=None, mask_value: float | None=DEFAULT_MASK_VALUE, q_scale: float | None=None, k_scale: float | None=None, v_scale: float | None=None, chunk_prefill_size: int | None=None, num_kv_pages_per_block: int | None=None, num_queries_per_block: int | None=None, vmem_limit_bytes: int | None=None, debug_mode: bool=False)` — [`L259`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v1/kernel.py#L259) — Validate inputs to the MLA RPA kernel dynamically.
- `get_kv_cache_shape(total_num_pages, page_size, kv_dim, kv_dtype)` — [`L32`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v1/kernel.py#L32)
- `mla_ragged_paged_attention(ql_nope: jax.Array, q_pe: jax.Array, new_kv_c: jax.Array, new_k_pe: jax.Array, cache_kv: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, distribution: jax.Array, *, sm_scale: float=1.0, sliding_window: int | None=None, soft_cap: float | None=None, mask_value: float | None=DEFAULT_MASK_VALUE, q_scale: float | None=None, k_scale: float | None=None, v_scale: float | None=None, chunk_prefill_size: int | None=None, num_kv_pages_per_block: int | None=None, num_queries_per_block: int | None=None, vmem_limit_bytes: int | None=None, debug_mode: bool=False)` — [`L1092`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v1/kernel.py#L1092) — MLA Ragged paged attention that supports mixed prefill and decode.
- `prepare_kv_inputs(kv: jax.Array)` — [`L1042`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v1/kernel.py#L1042)
- `prepare_outputs(out, actual_num_q_heads: int, actual_head_dim: int)` — [`L1057`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v1/kernel.py#L1057)
- `prepare_q_inputs(q: jax.Array)` — [`L1014`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v1/kernel.py#L1014)
- `ref_mla_ragged_paged_attention(ql_nope: jax.Array, q_pe: jax.Array, new_kv_c: jax.Array, new_k_pe: jax.Array, cache_kv: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, distribution: jax.Array, *, sm_scale: float=1.0, sliding_window: int | None=None, soft_cap: float | None=None, mask_value: float | None=DEFAULT_MASK_VALUE, q_scale: float | None=None, k_scale: float | None=None, v_scale: float | None=None)` — [`L104`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v1/kernel.py#L104)
- `static_validate_inputs(ql_nope: jax.Array, q_pe: jax.Array, new_kv_c: jax.Array, new_k_pe: jax.Array, cache_kv: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, distribution: jax.Array, *, sm_scale: float=1.0, sliding_window: int | None=None, soft_cap: float | None=None, mask_value: float | None=DEFAULT_MASK_VALUE, q_scale: float | None=None, k_scale: float | None=None, v_scale: float | None=None, chunk_prefill_size: int | None=None, num_kv_pages_per_block: int | None=None, num_queries_per_block: int | None=None, vmem_limit_bytes: int | None=None, debug_mode: bool=False)` — [`L350`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v1/kernel.py#L350) — Validate inputs to the MLA RPA kernel statically.
- `update_kv_cache(new_kv_c: jax.Array, new_k_pe: jax.Array, cache_kv: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, distribution: jax.Array)` — [`L48`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v1/kernel.py#L48) — Update KV cache with new tokens.

## Module values
- `DEFAULT_MASK_VALUE` — [`L27`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v1/kernel.py#L27)
- `DEFAULT_VMEM_LIMIT_BYTES` — [`L29`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v1/kernel.py#L29)


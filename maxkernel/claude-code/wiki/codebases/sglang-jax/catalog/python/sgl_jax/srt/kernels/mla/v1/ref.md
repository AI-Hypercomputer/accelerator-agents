---
title: 'Module: python/sgl_jax/srt/kernels/mla/v1/ref.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/kernels/mla/v1/ref.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.kernels.mla.v1.ref`/
symbols:
  ref_mla_ragged_paged_attention: ref_mla_ragged_paged_attention().
  static_validate_inputs: static_validate_inputs().
  update_kv_cache: update_kv_cache().
  dynamic_validate_inputs: dynamic_validate_inputs().
  get_kv_cache_shape: get_kv_cache_shape().
  DEFAULT_MASK_VALUE: DEFAULT_MASK_VALUE.
  update_kv_cache.seq_loop_body: update_kv_cache().seq_loop_body().
  update_kv_cache.seq_loop_body.token_loop_body: update_kv_cache().seq_loop_body().token_loop_body().
---
# Module: [`python/sgl_jax/srt/kernels/mla/v1/ref.py`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v1/ref.py)

## Functions
- `dynamic_validate_inputs(ql_nope: jax.Array, q_pe: jax.Array, new_kv_c: jax.Array, new_k_pe: jax.Array, cache_kv: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, distribution: jax.Array, *, sm_scale: float = 1, sliding_window: int | None = None, soft_cap: float | None = None, mask_value: float | None = DEFAULT_MASK_VALUE, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, chunk_prefill_size: int | None = None, num_kv_pages_per_block: int | None = None, num_queries_per_block: int | None = None, vmem_limit_bytes: int | None = None, debug_mode: bool = False)` — [`L252`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v1/ref.py#L252) — Validate inputs to the MLA RPA kernel dynamically.
- `get_kv_cache_shape(total_num_pages, page_size, kv_dim, kv_dtype)` — [`L41`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v1/ref.py#L41)
- `ref_mla_ragged_paged_attention(ql_nope: jax.Array, q_pe: jax.Array, new_kv_c: jax.Array, new_k_pe: jax.Array, cache_kv: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, distribution: jax.Array, *, sm_scale: float = 1, sliding_window: int | None = None, soft_cap: float | None = None, mask_value: float | None = DEFAULT_MASK_VALUE, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None)` — [`L108`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v1/ref.py#L108)
- `seq_loop_body(i, cache_kv)` — [`L84`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v1/ref.py#L84)
- `static_validate_inputs(ql_nope: jax.Array, q_pe: jax.Array, new_kv_c: jax.Array, new_k_pe: jax.Array, cache_kv: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, distribution: jax.Array, *, sm_scale: float = 1, sliding_window: int | None = None, soft_cap: float | None = None, mask_value: float | None = DEFAULT_MASK_VALUE, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, chunk_prefill_size: int | None = None, num_kv_pages_per_block: int | None = None, num_queries_per_block: int | None = None, vmem_limit_bytes: int | None = None, debug_mode: bool = False)` — [`L305`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v1/ref.py#L305) — Validate inputs to the MLA RPA kernel statically.
- `token_loop_body(j, cache_kv_)` — [`L89`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v1/ref.py#L89)
- `update_kv_cache(new_kv_c: jax.Array, new_k_pe: jax.Array, cache_kv: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, distribution: jax.Array)` — [`L57`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v1/ref.py#L57) — Update KV cache with new tokens.

## Module values
- `DEFAULT_MASK_VALUE` — [`L38`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v1/ref.py#L38)


---
title: 'Module: python/sgl_jax/srt/kernels/paged_attention/paged_attention.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/kernels/paged_attention/paged_attention.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.kernels.paged_attention.paged_attention`/
symbols:
  paged_attention: paged_attention().
  paged_attention_kernel._compute: paged_attention_kernel()._compute().
  paged_attention_unbatched._local_impl: paged_attention_unbatched()._local_impl().
  paged_attention_kernel: paged_attention_kernel().
  paged_attention_unbatched: paged_attention_unbatched().
  paged_attention_reference: paged_attention_reference().
  DEFAULT_MASK_VALUE: DEFAULT_MASK_VALUE.
  paged_attention_kernel._compute.body: paged_attention_kernel()._compute().body().
---
# Module: [`python/sgl_jax/srt/kernels/paged_attention/paged_attention.py`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/paged_attention/paged_attention.py)

## Functions
- `_compute(start_page_idx, end_page_idx, o, m_i, l_i)` — [`L65`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/paged_attention/paged_attention.py#L65)
- `_local_impl(q_loc, k_pages_loc, v_pages_loc, block_tables_loc, lengths_loc, k_scales_loc, v_scales_loc)` — [`L192`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/paged_attention/paged_attention.py#L192)
- `body(start_k, carry)` — [`L82`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/paged_attention/paged_attention.py#L82)
- `paged_attention(q: jax.Array, k_pages: jax.Array, v_pages: jax.Array, block_tables: jax.Array, lengths: jax.Array | None, k_scales_pages: jax.Array | None = None, v_scales_pages: jax.Array | None = None, *, block_h: int = 16, pages_per_compute_block: int = 8, k_splits: int = 16, num_warps: int = 8, num_stages: int = 2, interpret: bool = False, debug: bool = False, mask_value: float = DEFAULT_MASK_VALUE, sm_scale: float = 1, xai_temperature_len: float | None = None, attn_logits_soft_cap: float | None = None, mesh: Mesh | None = None, sharding_axis: str = "tensor")` — [`L359`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/paged_attention/paged_attention.py#L359) — Paged grouped query attention.
- `paged_attention_kernel(q_ref, k_pages_ref, k_scales_pages_ref, v_pages_ref, v_scales_pages_ref, block_tables_ref, lengths_ref, o_ref: Any, *residual_refs: Any, pages_per_compute_block: int, mask_value: float, sm_scale: float = 1, attn_logits_soft_cap: float | None, xai_temperature_len: float | None = None)` — [`L41`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/paged_attention/paged_attention.py#L41)
- `paged_attention_reference(q: jax.Array, k: jax.Array, v: jax.Array, lengths: jax.Array, *, mask_value: float = DEFAULT_MASK_VALUE, sm_scale: float = 1, attn_logits_soft_cap: float | None = None)` — [`L468`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/paged_attention/paged_attention.py#L468) — Grouped query attention reference implementation.
- `paged_attention_unbatched(q: jax.Array, k_pages: jax.Array, v_pages: jax.Array, block_table: jax.Array, length: jax.Array | None, k_scales_pages: jax.Array | None = None, v_scales_pages: jax.Array | None = None, *, block_h: int, pages_per_compute_block: int, k_splits: int, num_warps: int, num_stages: int, interpret: bool, debug: bool, mask_value: float, sm_scale: float = 1, xai_temperature_len: float | None = None, attn_logits_soft_cap: float | None = None, mesh: Mesh | None = None, sharding_axis: str = "tensor")` — [`L168`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/paged_attention/paged_attention.py#L168)

## Module values
- `DEFAULT_MASK_VALUE` — [`L38`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/paged_attention/paged_attention.py#L38)


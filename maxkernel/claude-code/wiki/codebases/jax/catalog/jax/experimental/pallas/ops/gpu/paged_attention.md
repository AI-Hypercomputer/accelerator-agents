---
title: 'Module: jax/experimental/pallas/ops/gpu/paged_attention.py'
type: catalog
provenance: extracted
module: jax/experimental/pallas/ops/gpu/paged_attention.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax.experimental.pallas.ops.gpu.paged_attention`/
symbols:
  paged_attention_unbatched: paged_attention_unbatched().
  paged_attention_reference: paged_attention_reference().
  paged_attention_kernel._compute.body: paged_attention_kernel()._compute().body().
  paged_attention_kernel: paged_attention_kernel().
  paged_attention: paged_attention().
  paged_attention_kernel._compute: paged_attention_kernel()._compute().
  DEFAULT_MASK_VALUE: DEFAULT_MASK_VALUE.
---
# Module: [`jax/experimental/pallas/ops/gpu/paged_attention.py`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/paged_attention.py)

## Functions
- `_compute(start_page_idx, end_page_idx, o, m_i, l_i)` — [`L55`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/paged_attention.py#L55)
- `body(start_k, carry)` — [`L61`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/paged_attention.py#L61)
- `paged_attention(q: jax.Array, k_pages: jax.Array, v_pages: jax.Array, block_tables: jax.Array, lengths: jax.Array | None, k_scales_pages: jax.Array | None = None, v_scales_pages: jax.Array | None = None, *, block_h: int = 16, pages_per_compute_block: int = 8, k_splits: int = 16, num_warps: int = 8, num_stages: int = 2, interpret: bool = False, debug: bool = False, mask_value: float = DEFAULT_MASK_VALUE, attn_logits_soft_cap: float | None = None)` — [`L298`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/paged_attention.py#L298) — Paged grouped query attention.
- `paged_attention_kernel(q_ref, k_pages_ref, k_scales_pages_ref, v_pages_ref, v_scales_pages_ref, block_tables_ref, lengths_ref, o_ref: Any, *residual_refs: Any, num_heads: int, pages_per_compute_block: int, mask_value: float, attn_logits_soft_cap: float | None)` — [`L32`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/paged_attention.py#L32)
- `paged_attention_reference(q: jax.Array, k: jax.Array, v: jax.Array, lengths: jax.Array, *, mask_value: float = DEFAULT_MASK_VALUE, attn_logits_soft_cap: float | None = None)` — [`L401`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/paged_attention.py#L401) — Grouped query attention reference implementation.
- `paged_attention_unbatched(q: jax.Array, k_pages: jax.Array, v_pages: jax.Array, block_tables: jax.Array, lengths: jax.Array | None, k_scales_pages: jax.Array | None = None, v_scales_pages: jax.Array | None = None, *, block_h: int, pages_per_compute_block: int, k_splits: int, num_warps: int, num_stages: int, interpret: bool, debug: bool, mask_value: float, attn_logits_soft_cap: float | None)` — [`L150`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/paged_attention.py#L150)

## Module values
- `DEFAULT_MASK_VALUE` — [`L29`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/paged_attention.py#L29)


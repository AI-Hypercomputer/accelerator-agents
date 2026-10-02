---
title: 'Module: ejkernel/kernels/_pallas/tpu/prefill_page_attention/_interface.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/prefill_page_attention/_interface.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.prefill_page_attention._interface`/
symbols:
  _HBM_ANY: _HBM_ANY.
  prefill_page_attention: prefill_page_attention().
---
# Module: [`ejkernel/kernels/_pallas/tpu/prefill_page_attention/_interface.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/prefill_page_attention/_interface.py)

## Functions
- `prefill_page_attention(query: Float[Array, "chunk_size num_heads head_dim"], key_cache: Float[Array, "num_kv_heads total_num_pages page_size head_dim"], value_cache: Float[Array, "num_kv_heads total_num_pages page_size head_dim"], context_len: Int[Array, 1], page_indices: Int[Array, num_pages], *, softmax_scale: float | None = None, mask_value: float = DEFAULT_MASK_VALUE, attn_logits_soft_cap: float | None = None, sliding_window: int | None = None, block_k: int | None = None, num_warps: int | None = None, num_stages: int | None = None)` — [`L105`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/prefill_page_attention/_interface.py#L105) — Chunked prefill attention with paged KV cache for TPU.

## Module values
- `_HBM_ANY` — [`L90`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/prefill_page_attention/_interface.py#L90)


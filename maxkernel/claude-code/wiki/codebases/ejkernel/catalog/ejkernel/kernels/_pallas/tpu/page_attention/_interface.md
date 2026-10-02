---
title: 'Module: ejkernel/kernels/_pallas/tpu/page_attention/_interface.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/page_attention/_interface.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.page_attention._interface`/
symbols:
  _HBM_ANY: _HBM_ANY.
  page_attention: page_attention().
---
# Module: [`ejkernel/kernels/_pallas/tpu/page_attention/_interface.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/page_attention/_interface.py)

## Functions
- `page_attention(query: Float[Array, "num_seqs num_heads head_dim"], key_cache: Float[Array, "num_kv_heads total_num_pages page_size head_dim"], value_cache: Float[Array, "num_kv_heads total_num_pages page_size head_dim"], context_lens: Int[Array, num_seqs], block_tables: Int[Array, "num_seqs max_blocks"], attn_scale: float | None = None, max_context_len: int | None = None, num_splits: int = 0, *, mask_value: float = DEFAULT_MASK_VALUE, attn_logits_soft_cap: float | None = None, pages_per_compute_block: int | None = None, megacore_mode: str | None = None, inline_seq_dim: bool = True, sliding_window: int | None = None, num_warps: int = 4, num_stages: int = 3)` — [`L115`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/page_attention/_interface.py#L115) — Paged grouped query attention.

## Module values
- `_HBM_ANY` — [`L93`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/page_attention/_interface.py#L93)


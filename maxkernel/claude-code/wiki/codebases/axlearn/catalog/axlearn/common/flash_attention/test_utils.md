---
title: 'Module: axlearn/common/flash_attention/test_utils.py'
type: catalog
provenance: extracted
module: axlearn/common/flash_attention/test_utils.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.flash_attention.test_utils`/generate_
symbols:
  generate_attention_data: attention_data().
  generate_paged_attention_data: paged_attention_data().
---
# Module: [`axlearn/common/flash_attention/test_utils.py`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/test_utils.py)

## Functions
- `generate_attention_data(batch_size: int, query_len: int, kv_len: int, num_heads: int, per_head_dim: int, num_kv_heads: Optional[int] = None, mask_fn: Optional[MaskFn] = None, sliding_window_sz: Optional[int] = None, attention_bias_type: Literal[None, "2d", "4d"] = None, with_segment_ids: bool = False, dtype=jnp.bfloat16, kv_dtype=None, query_offset: Tensor | int = 0)` — [`L73`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/test_utils.py#L73) — Generates QKV and Bias for unit test purposes.
- `generate_paged_attention_data(*, batch_size: int, query_len: int, kv_len: int, num_heads: int, per_head_dim: int, num_kv_heads: int, page_size: int, mask_fn: Optional[MaskFn] = None, sliding_window_sz: Optional[int] = None, attention_bias_type: Literal[None, "2d", "4d"] = None, with_segment_ids: bool = False, dtype=jnp.bfloat16, query_offset: Tensor | int = 0)` — [`L24`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/test_utils.py#L24) — Generates query, key value pages, and page tables for paged attention testing.


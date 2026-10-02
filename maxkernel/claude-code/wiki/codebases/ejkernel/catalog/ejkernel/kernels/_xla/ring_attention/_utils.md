---
title: 'Module: ejkernel/kernels/_xla/ring_attention/_utils.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_xla/ring_attention/_utils.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._xla.ring_attention._utils`/
symbols:
  _chunk_attention_bias: _chunk_attention_bias().
  below_or_on_diag: below_or_on_diag().
---
# Module: [`ejkernel/kernels/_xla/ring_attention/_utils.py`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ring_attention/_utils.py)

## Functions
- `_chunk_attention_bias(query_chunk_size: int, key_chunk_size: int, bias: chex.Array | None, q_segment_ids: chex.Array | None, kv_segment_ids: chex.Array | None, q_position_ids: chex.Array | None, kv_position_ids: chex.Array | None, deterministic: bool, attn_dropout: chex.Array | None, pdrop: float, causal_block_size: int | None, dtype: DTypeLike, query_chunk_idx: int, key_chunk_idx: int, sliding_window: int | tuple[int, int] | None = None, attention_sink_size: int = 0)` — [`L28`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ring_attention/_utils.py#L28) — Compute the additive attention bias for one query-chunk × KV-chunk pair.
- `below_or_on_diag(r: int, r_blk_size: int, c: int, c_blk_size: int, causal_block_size: int)` — [`L215`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ring_attention/_utils.py#L215) — Check whether a query-chunk/KV-chunk pair overlaps or lies below the causal diagonal.


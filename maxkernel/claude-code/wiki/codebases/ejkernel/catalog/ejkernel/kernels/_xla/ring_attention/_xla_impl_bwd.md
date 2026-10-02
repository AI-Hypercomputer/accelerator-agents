---
title: 'Module: ejkernel/kernels/_xla/ring_attention/_xla_impl_bwd.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_xla/ring_attention/_xla_impl_bwd.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._xla.ring_attention._xla_impl_bwd`/_
symbols:
  _blockwise_attention_bwd: blockwise_attention_bwd().
  _blockwise_attention_bwd.scan_attention.skip_upper_half: blockwise_attention_bwd().scan_attention().skip_upper_half().
  _ring_attention_bwd.scan_kv_block: ring_attention_bwd().scan_kv_block().
  _ring_attention_bwd.scan_kv_block._ppermute_or_none: ring_attention_bwd().scan_kv_block()._ppermute_or_none().
  _ring_attention_bwd: ring_attention_bwd().
  _blockwise_attention_bwd.scan_attention: blockwise_attention_bwd().scan_attention().
  _blockwise_attention_bwd.scan_attention.scan_kv_block: blockwise_attention_bwd().scan_attention().scan_kv_block().
---
# Module: [`ejkernel/kernels/_xla/ring_attention/_xla_impl_bwd.py`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ring_attention/_xla_impl_bwd.py)

## Functions
- `_blockwise_attention_bwd(query: chex.Array, key: chex.Array, value: chex.Array, g: chex.Array, carry, q_chunk_idx_start: int, k_chunk_idx_start: int, bias: chex.Array | None, q_segment_ids: chex.Array | None, kv_segment_ids: chex.Array | None, q_position_ids: chex.Array | None, kv_position_ids: chex.Array | None, softmax_aux: chex.Array | None, softmax_scale: float | None, causal_block_size: int | None, query_chunk_size: int, key_chunk_size: int, deterministic: bool, dropout_rng: PRNGKeyArray | None, pdrop: float, dtype: DTypeLike, policy, precision: lax.PrecisionLike, prevent_cse: bool, sliding_window: int | tuple[int, int] | None = None, logits_soft_cap: float | None = None, attention_sink_size: int = 0, causal: bool = False)` — [`L63`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ring_attention/_xla_impl_bwd.py#L63) — Backward pass for blockwise attention over one KV shard.
- `_ppermute_or_none(x)` — [`L477`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ring_attention/_xla_impl_bwd.py#L477) — Rotate a tensor to the next device in the ring, or pass through.
- `_ring_attention_bwd(axis_name: str | None, float32_logits: bool, softmax_scale: float | None, query_chunk_size: int, key_chunk_size: int, causal_block_size: int | None, deterministic: bool, dropout_rng: PRNGKeyArray | None, pdrop: float, dtype: DTypeLike, policy, precision: lax.PrecisionLike, prevent_cse: bool, sliding_window: int | tuple[int, int] | None, logits_soft_cap: float | None, attention_sink_size: int, causal: bool, res, g: chex.Array)` — [`L336`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ring_attention/_xla_impl_bwd.py#L336) — Backward pass for ring attention (XLA custom-VJP backward rule).
- `scan_attention(carry, scan)` — [`L190`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ring_attention/_xla_impl_bwd.py#L190) — Process backward for one query chunk against all KV chunks.
- `scan_kv_block(carry, scan)` — [`L218`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ring_attention/_xla_impl_bwd.py#L218) — Compute backward gradients for one query-KV chunk pair.
- `scan_kv_block(carry, idx)` — [`L424`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ring_attention/_xla_impl_bwd.py#L424) — Process one ring step backward: compute local gradients and rotate.
- `skip_upper_half(carry, args)` — [`L263`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ring_attention/_xla_impl_bwd.py#L263) — Conditionally skip KV blocks above the causal diagonal in backward.


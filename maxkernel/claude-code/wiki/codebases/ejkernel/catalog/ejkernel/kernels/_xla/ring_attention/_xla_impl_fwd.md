---
title: 'Module: ejkernel/kernels/_xla/ring_attention/_xla_impl_fwd.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_xla/ring_attention/_xla_impl_fwd.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._xla.ring_attention._xla_impl_fwd`/_
symbols:
  _ring_attention_fwd: ring_attention_fwd().
  _blockwise_attention_fwd: blockwise_attention_fwd().
  _blockwise_attention_fwd.scan_attention.skip_upper_half: blockwise_attention_fwd().scan_attention().skip_upper_half().
  _ring_attention_fwd.scan_kv_block: ring_attention_fwd().scan_kv_block().
  _ring_attention_fwd.scan_kv_block._ppermute_or_none: ring_attention_fwd().scan_kv_block()._ppermute_or_none().
  _blockwise_attention_fwd.scan_attention: blockwise_attention_fwd().scan_attention().
  _blockwise_attention_fwd.scan_attention.scan_kv_block: blockwise_attention_fwd().scan_attention().scan_kv_block().
---
# Module: [`ejkernel/kernels/_xla/ring_attention/_xla_impl_fwd.py`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ring_attention/_xla_impl_fwd.py)

## Functions
- `_blockwise_attention_fwd(query: chex.Array, key: chex.Array, value: chex.Array, carry, q_chunk_idx_start: int, k_chunk_idx_start: int, bias: chex.Array | None, q_segment_ids: chex.Array | None, kv_segment_ids: chex.Array | None, q_position_ids: chex.Array | None, kv_position_ids: chex.Array | None, softmax_aux: chex.Array | None, softmax_scale: float | None, causal_block_size: int | None, query_chunk_size: int, key_chunk_size: int, deterministic: bool, dropout_rng: PRNGKeyArray | None, pdrop: float, dtype: DTypeLike, policy, precision: lax.PrecisionLike, prevent_cse: bool, sliding_window: int | tuple[int, int] | None = None, logits_soft_cap: float | None = None, attention_sink_size: int = 0, causal: bool = False)` — [`L65`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ring_attention/_xla_impl_fwd.py#L65) — Forward pass for blockwise attention over one KV shard.
- `_ppermute_or_none(x)` — [`L524`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ring_attention/_xla_impl_fwd.py#L524) — Rotate a tensor to the next device in the ring, or pass through.
- `_ring_attention_fwd(query: chex.Array, key: chex.Array, value: chex.Array, bias: chex.Array | None, q_segment_ids: chex.Array | None, kv_segment_ids: chex.Array | None, q_position_ids: chex.Array | None, kv_position_ids: chex.Array | None, softmax_aux: chex.Array | None, axis_name: str | None, float32_logits: bool, softmax_scale: float | None, query_chunk_size: int, key_chunk_size: int, causal_block_size: int | None, deterministic: bool, dropout_rng: PRNGKeyArray | None, pdrop: float, dtype: DTypeLike, policy, precision: lax.PrecisionLike, prevent_cse: bool, sliding_window: int | tuple[int, int] | None = None, logits_soft_cap: float | None = None, attention_sink_size: int = 0, causal: bool = False)` — [`L360`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ring_attention/_xla_impl_fwd.py#L360) — Forward pass for ring attention (XLA custom-VJP forward rule).
- `scan_attention(_, scan)` — [`L193`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ring_attention/_xla_impl_fwd.py#L193) — Process one query chunk against all KV chunks.
- `scan_kv_block(carry, scan)` — [`L210`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ring_attention/_xla_impl_fwd.py#L210) — Process one KV chunk and update the online softmax state.
- `scan_kv_block(carry, idx)` — [`L472`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ring_attention/_xla_impl_fwd.py#L472) — Process one ring step: compute local attention and rotate KV blocks.
- `skip_upper_half(carry, args)` — [`L303`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ring_attention/_xla_impl_fwd.py#L303) — Conditionally skip KV blocks above the causal diagonal.


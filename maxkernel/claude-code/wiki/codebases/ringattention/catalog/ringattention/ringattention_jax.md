---
title: 'Module: ringattention/ringattention_jax.py'
type: catalog
provenance: extracted
module: ringattention/ringattention_jax.py
status: fresh
symbol_base: scip-python python ringattention 0.0.0 `ringattention.ringattention_jax`/
symbols:
  below_or_on_diag: below_or_on_diag().
  _blockwise_attention_fwd: _blockwise_attention_fwd().
  _blockwise_attention_fwd.scan_attention.skip_upper_half: _blockwise_attention_fwd().scan_attention().skip_upper_half().
  _blockwise_attention_bwd: _blockwise_attention_bwd().
  _blockwise_attention_bwd.scan_attention.skip_upper_half: _blockwise_attention_bwd().scan_attention().skip_upper_half().
  _ring_attention_fwd: _ring_attention_fwd().
  ring_attention: ring_attention().
  _ring_attention_fwd.scan_kv_block: _ring_attention_fwd().scan_kv_block().
  _ring_attention_bwd: _ring_attention_bwd().
  _ring_attention_bwd.scan_kv_block: _ring_attention_bwd().scan_kv_block().
  _blockwise_attention_fwd.scan_attention: _blockwise_attention_fwd().scan_attention().
  _blockwise_attention_bwd.scan_attention: _blockwise_attention_bwd().scan_attention().
  _chunk_attention_bias: _chunk_attention_bias().
  _blockwise_attention_fwd.scan_attention.scan_kv_block: _blockwise_attention_fwd().scan_attention().scan_kv_block().
  _blockwise_attention_bwd.scan_attention.scan_kv_block: _blockwise_attention_bwd().scan_attention().scan_kv_block().
---
# Module: [`ringattention/ringattention_jax.py`](../../../../../raw/code/ringattention/ringattention/ringattention_jax.py)

## Functions
- `_blockwise_attention_bwd(q, k, v, g, carry, q_chunk_idx_start, k_chunk_idx_start, bias, segment_ids, causal_block_size, query_chunk_size, key_chunk_size, deterministic, dropout_rng, attn_pdrop, dtype, policy, precision, prevent_cse, cache_idx)` — [`L179`](../../../../../raw/code/ringattention/ringattention/ringattention_jax.py#L179)
- `_blockwise_attention_fwd(q, k, v, carry, q_chunk_idx_start, k_chunk_idx_start, bias, segment_ids, causal_block_size, query_chunk_size, key_chunk_size, deterministic, dropout_rng, attn_pdrop, dtype, policy, precision, prevent_cse, cache_idx)` — [`L97`](../../../../../raw/code/ringattention/ringattention/ringattention_jax.py#L97) — documented in [ringattention-ringattention_jax](../../concepts/ringattention-ringattention_jax.md)
- `_chunk_attention_bias(query_chunk_size, key_chunk_size, bias, segment_ids, deterministic, attn_dropout, attn_pdrop, causal_block_size, dtype, query_chunk_idx, key_chunk_idx)` — [`L270`](../../../../../raw/code/ringattention/ringattention/ringattention_jax.py#L270) — documented in [ringattention-ringattention_jax](../../concepts/ringattention-ringattention_jax.md)
- `_ring_attention_bwd(axis_name, float32_logits, blockwise_kwargs, res, g)` — [`L53`](../../../../../raw/code/ringattention/ringattention/ringattention_jax.py#L53)
- `_ring_attention_fwd(q, k, v, attn_bias, segment_ids, cache_idx, axis_name, float32_logits, blockwise_kwargs)` — [`L15`](../../../../../raw/code/ringattention/ringattention/ringattention_jax.py#L15)
- `below_or_on_diag(r, r_blk_size, c, c_blk_size, causal_block_size)` — [`L322`](../../../../../raw/code/ringattention/ringattention/ringattention_jax.py#L322) — documented in [ringattention-ringattention_jax](../../concepts/ringattention-ringattention_jax.md)
- `ring_attention(q, k, v, attn_bias, segment_ids, cache_idx, axis_name, float32_logits, blockwise_kwargs)` — [`L91`](../../../../../raw/code/ringattention/ringattention/ringattention_jax.py#L91)
- `scan_attention(_, scan)` — [`L126`](../../../../../raw/code/ringattention/ringattention/ringattention_jax.py#L126) — documented in [ringattention-ringattention_jax](../../concepts/ringattention-ringattention_jax.md)
- `scan_attention(carry, scan)` — [`L213`](../../../../../raw/code/ringattention/ringattention/ringattention_jax.py#L213) — documented in [ringattention-ringattention_jax](../../concepts/ringattention-ringattention_jax.md)
- `scan_kv_block(carry, idx)` — [`L27`](../../../../../raw/code/ringattention/ringattention/ringattention_jax.py#L27) — documented in [ringattention-ringattention_jax](../../concepts/ringattention-ringattention_jax.md)
- `scan_kv_block(carry, idx)` — [`L65`](../../../../../raw/code/ringattention/ringattention/ringattention_jax.py#L65)
- `scan_kv_block(carry, scan)` — [`L129`](../../../../../raw/code/ringattention/ringattention/ringattention_jax.py#L129) — documented in [ringattention-ringattention_jax](../../concepts/ringattention-ringattention_jax.md)
- `scan_kv_block(carry, scan)` — [`L218`](../../../../../raw/code/ringattention/ringattention/ringattention_jax.py#L218) — documented in [ringattention-ringattention_jax](../../concepts/ringattention-ringattention_jax.md)
- `skip_upper_half(carry, args)` — [`L145`](../../../../../raw/code/ringattention/ringattention/ringattention_jax.py#L145) — documented in [ringattention-ringattention_jax](../../concepts/ringattention-ringattention_jax.md)
- `skip_upper_half(carry, args)` — [`L233`](../../../../../raw/code/ringattention/ringattention/ringattention_jax.py#L233) — documented in [ringattention-ringattention_jax](../../concepts/ringattention-ringattention_jax.md)


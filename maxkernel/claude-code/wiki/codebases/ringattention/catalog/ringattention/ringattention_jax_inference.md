---
title: 'Module: ringattention/ringattention_jax_inference.py'
type: catalog
provenance: extracted
module: ringattention/ringattention_jax_inference.py
status: fresh
symbol_base: scip-python python ringattention 0.0.0 `ringattention.ringattention_jax_inference`/
symbols:
  _ring_attention_inference_fwd: _ring_attention_inference_fwd().
  _ring_attention_inference_bwd: _ring_attention_inference_bwd().
  ring_attention_inference: ring_attention_inference().
  _ring_attention_inference_fwd.scan_kv_block: _ring_attention_inference_fwd().scan_kv_block().
  _ring_attention_inference_bwd.scan_kv_block: _ring_attention_inference_bwd().scan_kv_block().
---
# Module: [`ringattention/ringattention_jax_inference.py`](../../../../../raw/code/ringattention/ringattention/ringattention_jax_inference.py)

## Functions
- `_ring_attention_inference_bwd(axis_name, float32_logits, res, g)` — [`L38`](../../../../../raw/code/ringattention/ringattention/ringattention_jax_inference.py#L38)
- `_ring_attention_inference_fwd(q, k, v, attn_mask, axis_name, float32_logits)` — [`L8`](../../../../../raw/code/ringattention/ringattention/ringattention_jax_inference.py#L8)
- `ring_attention_inference(q, k, v, attn_mask, axis_name, float32_logits=True)` — [`L68`](../../../../../raw/code/ringattention/ringattention/ringattention_jax_inference.py#L68)
- `scan_kv_block(carry, idx)` — [`L17`](../../../../../raw/code/ringattention/ringattention/ringattention_jax_inference.py#L17)
- `scan_kv_block(carry, idx)` — [`L47`](../../../../../raw/code/ringattention/ringattention/ringattention_jax_inference.py#L47)


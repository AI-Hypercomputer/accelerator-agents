---
title: 'Module: python/sgl_jax/srt/layers/radix_lightning_attention.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/layers/radix_lightning_attention.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.layers.radix_lightning_attention`/RadixLightningAttention#
symbols:
  RadixLightningAttention.__call__: __call__().
  RadixLightningAttention.layer_id: layer_id.
  RadixLightningAttention: ''
  RadixLightningAttention.__init__: __init__().
  RadixLightningAttention.num_heads: num_heads.
  RadixLightningAttention.head_dim: head_dim.
---
# Module: [`python/sgl_jax/srt/layers/radix_lightning_attention.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/radix_lightning_attention.py)

## Classes
### `RadixLightningAttention`  ·  implements/extends Module
- def: [`python/sgl_jax/srt/layers/radix_lightning_attention.py:14`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/radix_lightning_attention.py#L14)
- doc: Layer dispatcher for Lightning / GLA-family linear attention.
- signature: `class RadixLightningAttention(nnx.Module):`
- members:
  - `head_dim` — [`L31`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/radix_lightning_attention.py#L31)
  - `layer_id` — [`L29`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/radix_lightning_attention.py#L29)
  - `num_heads` — [`L30`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/radix_lightning_attention.py#L30)
- protocol/private: `__call__`[`L33`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/radix_lightning_attention.py#L33), `__init__`[`L22`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/radix_lightning_attention.py#L22)
- uses (calls/refs, reference-scoped): [`ForwardBatch`](../model_executor/forward_batch_info.md#ForwardBatch), [`attn_backend`](../model_executor/forward_batch_info.md#ForwardBatch.attn_backend)
- used by: [`__call__`](attention/linear/lightning_backend.md#LightningAttnBackend.__call__), [`logger`](attention/linear/lightning_backend.md#logger)


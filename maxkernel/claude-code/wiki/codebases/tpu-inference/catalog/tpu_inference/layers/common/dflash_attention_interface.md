---
title: 'Module: tpu_inference/layers/common/dflash_attention_interface.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/common/dflash_attention_interface.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.common.dflash_attention_interface`/dflash_concat_attention().
symbols:
  dflash_concat_attention: ''
  dflash_concat_attention._body: _body().
---
# Module: [`tpu_inference/layers/common/dflash_attention_interface.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/dflash_attention_interface.py)

## Functions
- `_body(i: int, current: jax.Array)` — [`L91`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/dflash_attention_interface.py#L91)
- `dflash_concat_attention(q: jax.Array, k_ctx: jax.Array, k_noise: jax.Array, v_ctx: jax.Array, v_noise: jax.Array, attention_metadata: AttentionMetadata, *, max_query_len: int, sm_scale: float)` — [`L28`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/dflash_attention_interface.py#L28) — Computes DFlash concat attention outputs for query tokens.


---
title: 'Module: src/maxtext/models/gemma2.py'
type: catalog
provenance: extracted
module: src/maxtext/models/gemma2.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.models.gemma2`/Gemma2DecoderLayer
symbols:
  Gemma2DecoderLayer.__call__: '#__call__().'
  Gemma2DecoderLayer.activation_axis_names: '#activation_axis_names.'
  Gemma2DecoderLayer.rngs: '#rngs.'
  Gemma2DecoderLayer.self_attention_local: '#self_attention_local.'
  Gemma2DecoderLayer.mlp_local: '#mlp_local.'
  Gemma2DecoderLayer.mlp_global: '#mlp_global.'
  Gemma2DecoderLayer.__init__: '#__init__().'
  Gemma2DecoderLayer.self_attention_global: '#self_attention_global.'
  Gemma2DecoderLayer.config: '#config.'
  Gemma2DecoderLayer.mesh: '#mesh.'
  Gemma2DecoderLayer.quant: '#quant.'
  Gemma2DecoderLayer.dropout: '#dropout.'
  Gemma2DecoderLayer.model_mode: '#model_mode.'
  Gemma2DecoderLayer.pre_self_attention_norm_local: '#pre_self_attention_norm_local.'
  Gemma2DecoderLayer.post_self_attention_norm_local: '#post_self_attention_norm_local.'
  Gemma2DecoderLayer.pre_ffw_norm_local: '#pre_ffw_norm_local.'
  Gemma2DecoderLayer.post_ffw_norm_local: '#post_ffw_norm_local.'
  Gemma2DecoderLayer.pre_self_attention_norm_global: '#pre_self_attention_norm_global.'
  Gemma2DecoderLayer.post_self_attention_norm_global: '#post_self_attention_norm_global.'
  Gemma2DecoderLayer.pre_ffw_norm_global: '#pre_ffw_norm_global.'
  Gemma2DecoderLayer.post_ffw_norm_global: '#post_ffw_norm_global.'
  Gemma2DecoderLayerToLinen: ToLinen.
  Gemma2DecoderLayer: '#'
---
# Module: [`src/maxtext/models/gemma2.py`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma2.py)

## Classes
### `Gemma2DecoderLayer`  ·  implements/extends Module
- def: [`src/maxtext/models/gemma2.py:38`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma2.py#L38)
- doc: Transformer decoder layer that attends to the encoder.
- signature: `class Gemma2DecoderLayer(nnx.Module):`
- members:
  - `activation_axis_names` — [`L212`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma2.py#L212)
  - `config` — [`L41`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma2.py#L41)
  - `dropout` — [`L137`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma2.py#L137)
  - `mesh` — [`L42`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma2.py#L42)
  - `mlp_global` — [`L188`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma2.py#L188)
  - `mlp_local` — [`L114`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma2.py#L114)
  - `model_mode` — [`L43`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma2.py#L43)
  - `post_ffw_norm_global` — [`L203`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma2.py#L203)
  - `post_ffw_norm_local` — [`L129`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma2.py#L129)
  - `post_self_attention_norm_global` — [`L172`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma2.py#L172)
  - `post_self_attention_norm_local` — [`L98`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma2.py#L98)
  - `pre_ffw_norm_global` — [`L180`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma2.py#L180)
  - `pre_ffw_norm_local` — [`L106`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma2.py#L106)
  - `pre_self_attention_norm_global` — [`L139`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma2.py#L139)
  - `pre_self_attention_norm_local` — [`L64`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma2.py#L64)
  - `quant` — [`L44`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma2.py#L44)
  - `rngs` — [`L59`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma2.py#L59)
  - `self_attention_global` — [`L147`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma2.py#L147)
  - `self_attention_local` — [`L72`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma2.py#L72)
- protocol/private: `__call__`[`L216`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma2.py#L216), `__init__`[`L46`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma2.py#L46)
- used by: [`Gemma2DecoderLayerToLinen`](gemma2.md#Gemma2DecoderLayerToLinen)

## Module values
- `Gemma2DecoderLayerToLinen` — [`L324`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma2.py#L324)


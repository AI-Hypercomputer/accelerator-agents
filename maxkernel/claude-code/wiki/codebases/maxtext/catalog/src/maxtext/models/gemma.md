---
title: 'Module: src/maxtext/models/gemma.py'
type: catalog
provenance: extracted
module: src/maxtext/models/gemma.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.models.gemma`/GemmaDecoderLayer
symbols:
  GemmaDecoderLayer.__call__: '#__call__().'
  GemmaDecoderLayer.self_attention: '#self_attention.'
  GemmaDecoderLayer.mlp: '#mlp.'
  GemmaDecoderLayer.__init__: '#__init__().'
  GemmaDecoderLayer.rngs: '#rngs.'
  GemmaDecoderLayer.activation_axis_names: '#activation_axis_names.'
  GemmaDecoderLayer.pre_self_attention_norm: '#pre_self_attention_norm.'
  GemmaDecoderLayer.pre_ffw_norm: '#pre_ffw_norm.'
  GemmaDecoderLayer.dropout: '#dropout.'
  GemmaDecoderLayer.config: '#config.'
  GemmaDecoderLayer.mesh: '#mesh.'
  GemmaDecoderLayer.model_mode: '#model_mode.'
  GemmaDecoderLayer.quant: '#quant.'
  GemmaDecoderLayerToLinen: ToLinen.
  GemmaDecoderLayer: '#'
---
# Module: [`src/maxtext/models/gemma.py`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma.py)

## Classes
### `GemmaDecoderLayer`  ·  implements/extends Module
- def: [`src/maxtext/models/gemma.py:37`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma.py#L37)
- doc: Transformer decoder layer that attends to the encoder.
- signature: `class GemmaDecoderLayer(nnx.Module):`
- members:
  - `activation_axis_names` — [`L119`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma.py#L119)
  - `config` — [`L40`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma.py#L40)
  - `dropout` — [`L117`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma.py#L117)
  - `mesh` — [`L41`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma.py#L41)
  - `mlp` — [`L103`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma.py#L103)
  - `model_mode` — [`L42`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma.py#L42)
  - `pre_ffw_norm` — [`L95`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma.py#L95)
  - `pre_self_attention_norm` — [`L63`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma.py#L63)
  - `quant` — [`L43`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma.py#L43)
  - `rngs` — [`L58`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma.py#L58)
  - `self_attention` — [`L71`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma.py#L71)
- protocol/private: `__call__`[`L121`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma.py#L121), `__init__`[`L45`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma.py#L45)
- used by: [`GemmaDecoderLayerToLinen`](gemma.md#GemmaDecoderLayerToLinen)

## Module values
- `GemmaDecoderLayerToLinen` — [`L188`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma.py#L188)


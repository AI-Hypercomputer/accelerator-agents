---
title: 'Module: src/maxtext/models/mistral.py'
type: catalog
provenance: extracted
module: src/maxtext/models/mistral.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.models.mistral`/MistralDecoderLayer
symbols:
  MistralDecoderLayer.__call__: '#__call__().'
  MistralDecoderLayer.mlp: '#mlp.'
  MistralDecoderLayer.self_attention: '#self_attention.'
  MistralDecoderLayer.activation_axis_names: '#activation_axis_names.'
  MistralDecoderLayer.rngs: '#rngs.'
  MistralDecoderLayer.pre_self_attention_layer_norm: '#pre_self_attention_layer_norm.'
  MistralDecoderLayer.post_self_attention_layer_norm: '#post_self_attention_layer_norm.'
  MistralDecoderLayer.dropout: '#dropout.'
  MistralDecoderLayerToLinen: ToLinen.
  MistralDecoderLayer.quant: '#quant.'
  MistralDecoderLayer: '#'
  MistralDecoderLayer.config: '#config.'
  MistralDecoderLayer.mesh: '#mesh.'
  MistralDecoderLayer.update_cache: '#update_cache().'
  MistralDecoderLayer.__init__: '#__init__().'
---
# Module: [`src/maxtext/models/mistral.py`](../../../../../../../raw/code/maxtext/src/maxtext/models/mistral.py)

## Classes
### `MistralDecoderLayer`  ·  implements/extends Module
- def: [`src/maxtext/models/mistral.py:39`](../../../../../../../raw/code/maxtext/src/maxtext/models/mistral.py#L39)
- doc: Transformer decoder layer that attends to the encoder.
- signature: `class MistralDecoderLayer(nnx.Module):`
- members:
  - `update_cache(cache, val)` — [`L190`](../../../../../../../raw/code/maxtext/src/maxtext/models/mistral.py#L190)
  - `activation_axis_names` — [`L121`](../../../../../../../raw/code/maxtext/src/maxtext/models/mistral.py#L121)
  - `config` — [`L51`](../../../../../../../raw/code/maxtext/src/maxtext/models/mistral.py#L51)
  - `dropout` — [`L119`](../../../../../../../raw/code/maxtext/src/maxtext/models/mistral.py#L119)
  - `mesh` — [`L52`](../../../../../../../raw/code/maxtext/src/maxtext/models/mistral.py#L52)
  - `mlp` — [`L105`](../../../../../../../raw/code/maxtext/src/maxtext/models/mistral.py#L105)
  - `post_self_attention_layer_norm` — [`L96`](../../../../../../../raw/code/maxtext/src/maxtext/models/mistral.py#L96)
  - `pre_self_attention_layer_norm` — [`L59`](../../../../../../../raw/code/maxtext/src/maxtext/models/mistral.py#L59)
  - `quant` — [`L53`](../../../../../../../raw/code/maxtext/src/maxtext/models/mistral.py#L53)
  - `rngs` — [`L54`](../../../../../../../raw/code/maxtext/src/maxtext/models/mistral.py#L54)
  - `self_attention` — [`L68`](../../../../../../../raw/code/maxtext/src/maxtext/models/mistral.py#L68)
- protocol/private: `__call__`[`L123`](../../../../../../../raw/code/maxtext/src/maxtext/models/mistral.py#L123), `__init__`[`L42`](../../../../../../../raw/code/maxtext/src/maxtext/models/mistral.py#L42)
- used by: [`MistralDecoderLayerToLinen`](mistral.md#MistralDecoderLayerToLinen)

## Module values
- `MistralDecoderLayerToLinen` — [`L203`](../../../../../../../raw/code/maxtext/src/maxtext/models/mistral.py#L203)


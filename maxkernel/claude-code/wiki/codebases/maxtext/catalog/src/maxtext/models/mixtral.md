---
title: 'Module: src/maxtext/models/mixtral.py'
type: catalog
provenance: extracted
module: src/maxtext/models/mixtral.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.models.mixtral`/MixtralDecoderLayer
symbols:
  MixtralDecoderLayer.__call__: '#__call__().'
  MixtralDecoderLayer.self_attention: '#self_attention.'
  MixtralDecoderLayer.MoeBlock_0: '#MoeBlock_0.'
  MixtralDecoderLayer.activation_axis_names: '#activation_axis_names.'
  MixtralDecoderLayer.rngs: '#rngs.'
  MixtralDecoderLayer.pre_self_attention_layer_norm: '#pre_self_attention_layer_norm.'
  MixtralDecoderLayer.post_self_attention_layer_norm: '#post_self_attention_layer_norm.'
  MixtralDecoderLayer.config: '#config.'
  MixtralDecoderLayerToLinen: ToLinen.
  MixtralDecoderLayer.quant: '#quant.'
  MixtralDecoderLayer: '#'
  MixtralDecoderLayer.dropout: '#dropout.'
  MixtralDecoderLayer.__init__: '#__init__().'
  MixtralDecoderLayer.mesh: '#mesh.'
  MixtralDecoderLayer.model_mode: '#model_mode.'
---
# Module: [`src/maxtext/models/mixtral.py`](../../../../../../../raw/code/maxtext/src/maxtext/models/mixtral.py)

## Classes
### `MixtralDecoderLayer`  ·  implements/extends Module
- def: [`src/maxtext/models/mixtral.py:40`](../../../../../../../raw/code/maxtext/src/maxtext/models/mixtral.py#L40)
- doc: Transformer decoder layer that attends to the encoder.
- signature: `class MixtralDecoderLayer(nnx.Module):`
- members:
  - `MoeBlock_0` — [`L108`](../../../../../../../raw/code/maxtext/src/maxtext/models/mixtral.py#L108)
  - `activation_axis_names` — [`L124`](../../../../../../../raw/code/maxtext/src/maxtext/models/mixtral.py#L124)
  - `config` — [`L53`](../../../../../../../raw/code/maxtext/src/maxtext/models/mixtral.py#L53)
  - `dropout` — [`L122`](../../../../../../../raw/code/maxtext/src/maxtext/models/mixtral.py#L122)
  - `mesh` — [`L54`](../../../../../../../raw/code/maxtext/src/maxtext/models/mixtral.py#L54)
  - `model_mode` — [`L55`](../../../../../../../raw/code/maxtext/src/maxtext/models/mixtral.py#L55)
  - `post_self_attention_layer_norm` — [`L99`](../../../../../../../raw/code/maxtext/src/maxtext/models/mixtral.py#L99)
  - `pre_self_attention_layer_norm` — [`L62`](../../../../../../../raw/code/maxtext/src/maxtext/models/mixtral.py#L62)
  - `quant` — [`L56`](../../../../../../../raw/code/maxtext/src/maxtext/models/mixtral.py#L56)
  - `rngs` — [`L57`](../../../../../../../raw/code/maxtext/src/maxtext/models/mixtral.py#L57)
  - `self_attention` — [`L71`](../../../../../../../raw/code/maxtext/src/maxtext/models/mixtral.py#L71)
- protocol/private: `__call__`[`L126`](../../../../../../../raw/code/maxtext/src/maxtext/models/mixtral.py#L126), `__init__`[`L44`](../../../../../../../raw/code/maxtext/src/maxtext/models/mixtral.py#L44)
- used by: [`MixtralDecoderLayerToLinen`](mixtral.md#MixtralDecoderLayerToLinen)

## Module values
- `MixtralDecoderLayerToLinen` — [`L196`](../../../../../../../raw/code/maxtext/src/maxtext/models/mixtral.py#L196)


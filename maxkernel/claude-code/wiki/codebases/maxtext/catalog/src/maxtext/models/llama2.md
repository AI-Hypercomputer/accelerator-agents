---
title: 'Module: src/maxtext/models/llama2.py'
type: catalog
provenance: extracted
module: src/maxtext/models/llama2.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.models.llama2`/LlamaDecoderLayer
symbols:
  LlamaDecoderLayer.__call__: '#__call__().'
  LlamaDecoderLayer.activation_axis_names: '#activation_axis_names.'
  LlamaDecoderLayer._maybe_shard_with_logical: '#_maybe_shard_with_logical.'
  LlamaDecoderLayerToLinen: ToLinen.
  LlamaDecoderLayer.self_attention: '#self_attention.'
  LlamaDecoderLayer.mlp: '#mlp.'
  LlamaDecoderLayer.mesh: '#mesh.'
  LlamaDecoderLayer.quant: '#quant.'
  LlamaDecoderLayer: '#'
  LlamaDecoderLayer.config: '#config.'
  LlamaDecoderLayer.pre_self_attention_layer_norm: '#pre_self_attention_layer_norm.'
  LlamaDecoderLayer.post_self_attention_layer_norm: '#post_self_attention_layer_norm.'
  LlamaDecoderLayer.dropout: '#dropout.'
  LlamaDecoderLayer.update_cache: '#update_cache().'
  LlamaDecoderLayer.__init__: '#__init__().'
---
# Module: [`src/maxtext/models/llama2.py`](../../../../../../../raw/code/maxtext/src/maxtext/models/llama2.py)

## Classes
### `LlamaDecoderLayer`  ·  implements/extends Module
- def: [`src/maxtext/models/llama2.py:43`](../../../../../../../raw/code/maxtext/src/maxtext/models/llama2.py#L43)
- doc: Transformer decoder layer that attends to the encoder.
- signature: `class LlamaDecoderLayer(nnx.Module):`
- members:
  - `update_cache(cache, val)` — [`L216`](../../../../../../../raw/code/maxtext/src/maxtext/models/llama2.py#L216)
  - `activation_axis_names` — [`L59`](../../../../../../../raw/code/maxtext/src/maxtext/models/llama2.py#L59)
  - `config` — [`L54`](../../../../../../../raw/code/maxtext/src/maxtext/models/llama2.py#L54)
  - `dropout` — [`L130`](../../../../../../../raw/code/maxtext/src/maxtext/models/llama2.py#L130)
  - `mesh` — [`L55`](../../../../../../../raw/code/maxtext/src/maxtext/models/llama2.py#L55)
  - `mlp` — [`L116`](../../../../../../../raw/code/maxtext/src/maxtext/models/llama2.py#L116)
  - `post_self_attention_layer_norm` — [`L106`](../../../../../../../raw/code/maxtext/src/maxtext/models/llama2.py#L106)
  - `pre_self_attention_layer_norm` — [`L66`](../../../../../../../raw/code/maxtext/src/maxtext/models/llama2.py#L66)
  - `quant` — [`L56`](../../../../../../../raw/code/maxtext/src/maxtext/models/llama2.py#L56)
  - `self_attention` — [`L77`](../../../../../../../raw/code/maxtext/src/maxtext/models/llama2.py#L77)
- protocol/private: `__call__`[`L140`](../../../../../../../raw/code/maxtext/src/maxtext/models/llama2.py#L140), `__init__`[`L46`](../../../../../../../raw/code/maxtext/src/maxtext/models/llama2.py#L46), `_maybe_shard_with_logical`[`L132`](../../../../../../../raw/code/maxtext/src/maxtext/models/llama2.py#L132)
- used by: [`LlamaDecoderLayerToLinen`](llama2.md#LlamaDecoderLayerToLinen)

## Module values
- `LlamaDecoderLayerToLinen` — [`L229`](../../../../../../../raw/code/maxtext/src/maxtext/models/llama2.py#L229)


---
title: 'Module: src/maxtext/models/olmo3.py'
type: catalog
provenance: extracted
module: src/maxtext/models/olmo3.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.models.olmo3`/
symbols:
  Olmo3DecoderLayer.__call__: Olmo3DecoderLayer#__call__().
  Olmo3ScannableBlock.quant: Olmo3ScannableBlock#quant.
  Olmo3DecoderLayer.attention: Olmo3DecoderLayer#attention.
  get_attention_type: get_attention_type().
  Olmo3DecoderLayerToLinen: Olmo3DecoderLayerToLinen.
  Olmo3ScannableBlock.__call__: Olmo3ScannableBlock#__call__().
  Olmo3ScannableBlockToLinen: Olmo3ScannableBlockToLinen.
  OLMO3_ATTENTION_PATTERN: OLMO3_ATTENTION_PATTERN.
  Olmo3DecoderLayer: Olmo3DecoderLayer#
  Olmo3DecoderLayer.config: Olmo3DecoderLayer#config.
  Olmo3DecoderLayer.attention_type: Olmo3DecoderLayer#attention_type.
  Olmo3DecoderLayer.quant: Olmo3DecoderLayer#quant.
  Olmo3DecoderLayer.post_self_attention_layer_norm: Olmo3DecoderLayer#post_self_attention_layer_norm.
  Olmo3DecoderLayer.post_mlp_layer_norm: Olmo3DecoderLayer#post_mlp_layer_norm.
  Olmo3DecoderLayer.mlp: Olmo3DecoderLayer#mlp.
  Olmo3DecoderLayer.dropout: Olmo3DecoderLayer#dropout.
  Olmo3DecoderLayer.update_cache: Olmo3DecoderLayer#update_cache().
  Olmo3ScannableBlock: Olmo3ScannableBlock#
  Olmo3ScannableBlock.config: Olmo3ScannableBlock#config.
  Olmo3DecoderLayer.__init__: Olmo3DecoderLayer#__init__().
  Olmo3DecoderLayer.mesh: Olmo3DecoderLayer#mesh.
  Olmo3DecoderLayer.model_mode: Olmo3DecoderLayer#model_mode.
  Olmo3ScannableBlock.__init__: Olmo3ScannableBlock#__init__().
  Olmo3ScannableBlock.mesh: Olmo3ScannableBlock#mesh.
  Olmo3ScannableBlock.model_mode: Olmo3ScannableBlock#model_mode.
---
# Module: [`src/maxtext/models/olmo3.py`](../../../../../../../raw/code/maxtext/src/maxtext/models/olmo3.py)

## Classes
### `Olmo3DecoderLayer`  ·  implements/extends Module
- def: [`src/maxtext/models/olmo3.py:61`](../../../../../../../raw/code/maxtext/src/maxtext/models/olmo3.py#L61)
- doc: Transformer decoder layer that attends to the encoder.
- signature: `class Olmo3DecoderLayer(nnx.Module):`
- members:
  - `update_cache(cache, val)` — [`L225`](../../../../../../../raw/code/maxtext/src/maxtext/models/olmo3.py#L225)
  - `attention` — [`L107`](../../../../../../../raw/code/maxtext/src/maxtext/models/olmo3.py#L107)
  - `attention_type` — [`L76`](../../../../../../../raw/code/maxtext/src/maxtext/models/olmo3.py#L76)
  - `config` — [`L73`](../../../../../../../raw/code/maxtext/src/maxtext/models/olmo3.py#L73)
  - `dropout` — [`L146`](../../../../../../../raw/code/maxtext/src/maxtext/models/olmo3.py#L146)
  - `mesh` — [`L74`](../../../../../../../raw/code/maxtext/src/maxtext/models/olmo3.py#L74)
  - `mlp` — [`L133`](../../../../../../../raw/code/maxtext/src/maxtext/models/olmo3.py#L133)
  - `model_mode` — [`L75`](../../../../../../../raw/code/maxtext/src/maxtext/models/olmo3.py#L75)
  - `post_mlp_layer_norm` — [`L91`](../../../../../../../raw/code/maxtext/src/maxtext/models/olmo3.py#L91)
  - `post_self_attention_layer_norm` — [`L82`](../../../../../../../raw/code/maxtext/src/maxtext/models/olmo3.py#L82)
  - `quant` — [`L77`](../../../../../../../raw/code/maxtext/src/maxtext/models/olmo3.py#L77)
- protocol/private: `__call__`[`L148`](../../../../../../../raw/code/maxtext/src/maxtext/models/olmo3.py#L148), `__init__`[`L64`](../../../../../../../raw/code/maxtext/src/maxtext/models/olmo3.py#L64)
- used by: [`quant`](olmo3.md#Olmo3ScannableBlock.quant), [`Olmo3DecoderLayerToLinen`](olmo3.md#Olmo3DecoderLayerToLinen)

### `Olmo3ScannableBlock`  ·  implements/extends Module
- def: [`src/maxtext/models/olmo3.py:244`](../../../../../../../raw/code/maxtext/src/maxtext/models/olmo3.py#L244)
- doc: A repeatable block of Olmo 3 decoder layers.
- signature: `class Olmo3ScannableBlock(nnx.Module):`
- members:
  - `config` — [`L266`](../../../../../../../raw/code/maxtext/src/maxtext/models/olmo3.py#L266)
  - `mesh` — [`L267`](../../../../../../../raw/code/maxtext/src/maxtext/models/olmo3.py#L267)
  - `model_mode` — [`L268`](../../../../../../../raw/code/maxtext/src/maxtext/models/olmo3.py#L268)
  - `quant` — [`L269`](../../../../../../../raw/code/maxtext/src/maxtext/models/olmo3.py#L269)
- protocol/private: `__call__`[`L283`](../../../../../../../raw/code/maxtext/src/maxtext/models/olmo3.py#L283), `__init__`[`L258`](../../../../../../../raw/code/maxtext/src/maxtext/models/olmo3.py#L258)
- uses (calls/refs, reference-scoped): [`get_attention_type`](olmo3.md#get_attention_type), [`Olmo3DecoderLayer`](olmo3.md#Olmo3DecoderLayer)
- used by: [`Olmo3ScannableBlockToLinen`](olmo3.md#Olmo3ScannableBlockToLinen)

## Functions
- `get_attention_type(layer_id)` — [`L55`](../../../../../../../raw/code/maxtext/src/maxtext/models/olmo3.py#L55) — Get attention type based on layer ID.

## Module values
- `OLMO3_ATTENTION_PATTERN` — [`L47`](../../../../../../../raw/code/maxtext/src/maxtext/models/olmo3.py#L47)
- `Olmo3DecoderLayerToLinen` — [`L238`](../../../../../../../raw/code/maxtext/src/maxtext/models/olmo3.py#L238)
- `Olmo3ScannableBlockToLinen` — [`L323`](../../../../../../../raw/code/maxtext/src/maxtext/models/olmo3.py#L323)


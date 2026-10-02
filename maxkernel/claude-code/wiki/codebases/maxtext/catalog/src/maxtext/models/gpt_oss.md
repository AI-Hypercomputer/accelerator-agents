---
title: 'Module: src/maxtext/models/gpt_oss.py'
type: catalog
provenance: extracted
module: src/maxtext/models/gpt_oss.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.models.gpt_oss`/
symbols:
  GptOssDecoderLayer.__call__: GptOssDecoderLayer#__call__().
  GptOssScannableBlock.quant: GptOssScannableBlock#quant.
  GptOssDecoderLayer.GptOssAttention: GptOssDecoderLayer#GptOssAttention.
  get_attention_type: get_attention_type().
  GptOssDecoderLayer.GptOssMlp: GptOssDecoderLayer#GptOssMlp.
  GptOssDecoderLayerToLinen: GptOssDecoderLayerToLinen.
  GptOssScannableBlock.__call__: GptOssScannableBlock#__call__().
  GptOssScannableBlockToLinen: GptOssScannableBlockToLinen.
  GPT_OSS_ATTENTION_PATTERN: GPT_OSS_ATTENTION_PATTERN.
  GptOssDecoderLayer: GptOssDecoderLayer#
  GptOssDecoderLayer.quant: GptOssDecoderLayer#quant.
  GptOssDecoderLayer.config: GptOssDecoderLayer#config.
  GptOssDecoderLayer.attention_type: GptOssDecoderLayer#attention_type.
  GptOssDecoderLayer.pre_self_attention_layer_norm: GptOssDecoderLayer#pre_self_attention_layer_norm.
  GptOssDecoderLayer.post_self_attention_layer_norm: GptOssDecoderLayer#post_self_attention_layer_norm.
  GptOssDecoderLayer.dropout: GptOssDecoderLayer#dropout.
  GptOssDecoderLayer.update_cache: GptOssDecoderLayer#update_cache().
  GptOssScannableBlock: GptOssScannableBlock#
  GptOssScannableBlock.config: GptOssScannableBlock#config.
  GptOssDecoderLayer.__init__: GptOssDecoderLayer#__init__().
  GptOssDecoderLayer.mesh: GptOssDecoderLayer#mesh.
  GptOssDecoderLayer.model_mode: GptOssDecoderLayer#model_mode.
  GptOssScannableBlock.__init__: GptOssScannableBlock#__init__().
  GptOssScannableBlock.mesh: GptOssScannableBlock#mesh.
  GptOssScannableBlock.model_mode: GptOssScannableBlock#model_mode.
---
# Module: [`src/maxtext/models/gpt_oss.py`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt_oss.py)

## Classes
### `GptOssDecoderLayer`  ·  implements/extends Module
- def: [`src/maxtext/models/gpt_oss.py:57`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt_oss.py#L57)
- doc: Transformer decoder layer that attends to the encoder.
- signature: `class GptOssDecoderLayer(nnx.Module):`
- members:
  - `update_cache(cache, val)` — [`L214`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt_oss.py#L214)
  - `GptOssAttention` — [`L97`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt_oss.py#L97)
  - `GptOssMlp` — [`L121`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt_oss.py#L121)
  - `attention_type` — [`L72`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt_oss.py#L72)
  - `config` — [`L69`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt_oss.py#L69)
  - `dropout` — [`L135`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt_oss.py#L135)
  - `mesh` — [`L70`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt_oss.py#L70)
  - `model_mode` — [`L71`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt_oss.py#L71)
  - `post_self_attention_layer_norm` — [`L87`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt_oss.py#L87)
  - `pre_self_attention_layer_norm` — [`L78`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt_oss.py#L78)
  - `quant` — [`L73`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt_oss.py#L73)
- protocol/private: `__call__`[`L137`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt_oss.py#L137), `__init__`[`L60`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt_oss.py#L60)
- used by: [`quant`](gpt_oss.md#GptOssScannableBlock.quant), [`GptOssDecoderLayerToLinen`](gpt_oss.md#GptOssDecoderLayerToLinen)

### `GptOssScannableBlock`  ·  implements/extends Module
- def: [`src/maxtext/models/gpt_oss.py:233`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt_oss.py#L233)
- doc: A repeatable block of GPT OSS decoder layers.
- signature: `class GptOssScannableBlock(nnx.Module):`
- members:
  - `config` — [`L255`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt_oss.py#L255)
  - `mesh` — [`L256`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt_oss.py#L256)
  - `model_mode` — [`L257`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt_oss.py#L257)
  - `quant` — [`L258`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt_oss.py#L258)
- protocol/private: `__call__`[`L272`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt_oss.py#L272), `__init__`[`L247`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt_oss.py#L247)
- uses (calls/refs, reference-scoped): [`get_attention_type`](gpt_oss.md#get_attention_type), [`GptOssDecoderLayer`](gpt_oss.md#GptOssDecoderLayer)
- used by: [`GptOssScannableBlockToLinen`](gpt_oss.md#GptOssScannableBlockToLinen)

## Functions
- `get_attention_type(layer_id)` — [`L51`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt_oss.py#L51) — Get attention type based on layer ID.

## Module values
- `GPT_OSS_ATTENTION_PATTERN` — [`L45`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt_oss.py#L45)
- `GptOssDecoderLayerToLinen` — [`L227`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt_oss.py#L227)
- `GptOssScannableBlockToLinen` — [`L306`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt_oss.py#L306)


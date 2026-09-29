---
title: 'Module: src/maxtext/models/qwen3_5.py'
type: catalog
provenance: extracted
module: src/maxtext/models/qwen3_5.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.models.qwen3_5`/Qwen3_5
symbols:
  Qwen3_5DecoderLayer.__call__: DecoderLayer#__call__().
  Qwen3_5DecoderLayer.attention: DecoderLayer#attention.
  Qwen3_5ScannableBlock.rngs: ScannableBlock#rngs.
  Qwen3_5DecoderLayer.mlp: DecoderLayer#mlp.
  Qwen3_5DecoderLayer.quant: DecoderLayer#quant.
  Qwen3_5DecoderLayer.activation_axis_names: DecoderLayer#activation_axis_names.
  Qwen3_5DecoderLayer.input_layernorm: DecoderLayer#input_layernorm.
  Qwen3_5FullAttention: FullAttention#
  Qwen3_5ScannableBlock.config: ScannableBlock#config.
  Qwen3_5DecoderLayer.mesh: DecoderLayer#mesh.
  Qwen3_5ScannableBlock.__call__: ScannableBlock#__call__().
  Qwen3_5DecoderLayerToLinen: DecoderLayerToLinen.
  Qwen3_5ScannableBlockToLinen: ScannableBlockToLinen.
  Qwen3_5GatedDeltaNet: GatedDeltaNet#
  Qwen3_5DecoderLayer.config: DecoderLayer#config.
  Qwen3_5DecoderLayer.layer_idx: DecoderLayer#layer_idx.
  Qwen3_5DecoderLayer: DecoderLayer#
  Qwen3_5SparseMoEBlock: SparseMoEBlock#
  Qwen3_5ScannableBlock: ScannableBlock#
  Qwen3_5ScannableBlock.mesh: ScannableBlock#mesh.
  Qwen3_5ScannableBlock.model_mode: ScannableBlock#model_mode.
  Qwen3_5ScannableBlock.quant: ScannableBlock#quant.
  Qwen3_5DecoderLayer.post_attention_layernorm: DecoderLayer#post_attention_layernorm.
  Qwen3_5ScannableBlock.__init__: ScannableBlock#__init__().
  Qwen3_5DecoderLayer.__init__: DecoderLayer#__init__().
  Qwen3_5DecoderLayer.model_mode: DecoderLayer#model_mode.
---
# Module: [`src/maxtext/models/qwen3_5.py`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen3_5.py)

## Classes
### `Qwen3_5DecoderLayer`
- def: [`src/maxtext/models/qwen3_5.py:111`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen3_5.py#L111)
- members:
  - `activation_axis_names` — [`L134`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen3_5.py#L134)
  - `attention` — [`L150`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen3_5.py#L150)
  - `config` — [`L128`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen3_5.py#L128)
  - `input_layernorm` — [`L137`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen3_5.py#L137)
  - `layer_idx` — [`L131`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen3_5.py#L131)
  - `mesh` — [`L129`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen3_5.py#L129)
  - `mlp` — [`L175`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen3_5.py#L175)
  - `model_mode` — [`L130`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen3_5.py#L130)
  - `post_attention_layernorm` — [`L166`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen3_5.py#L166)
  - `quant` — [`L132`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen3_5.py#L132)
- protocol/private: `__call__`[`L177`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen3_5.py#L177), `__init__`[`L125`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen3_5.py#L125)
- uses (calls/refs, reference-scoped): [`Qwen3_5FullAttention`](qwen3_5.md#Qwen3_5FullAttention), [`Qwen3_5GatedDeltaNet`](qwen3_5.md#Qwen3_5GatedDeltaNet), [`Qwen3_5SparseMoEBlock`](qwen3_5.md#Qwen3_5SparseMoEBlock)
- used by: [`rngs`](qwen3_5.md#Qwen3_5ScannableBlock.rngs), [`Qwen3_5DecoderLayerToLinen`](qwen3_5.md#Qwen3_5DecoderLayerToLinen)

### `Qwen3_5FullAttention`  ·  implements/extends Qwen3NextFullAttention
- def: [`src/maxtext/models/qwen3_5.py:50`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen3_5.py#L50)
- doc: Qwen3.5 Gated Attention layer that is identical to Qwen3-Next
- signature: `class Qwen3_5FullAttention(Qwen3NextFullAttention):`
- used by: [`__call__`](qwen3_5.md#Qwen3_5DecoderLayer.__call__), [`attention`](qwen3_5.md#Qwen3_5DecoderLayer.attention)

### `Qwen3_5GatedDeltaNet`  ·  implements/extends Qwen3NextGatedDeltaNet
- def: [`src/maxtext/models/qwen3_5.py:46`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen3_5.py#L46)
- doc: Qwen3.5 GatedDeltaNet layer that is identical to Qwen3-Next GatedDeltaNet
- signature: `class Qwen3_5GatedDeltaNet(Qwen3NextGatedDeltaNet):`
- used by: [`__call__`](qwen3_5.md#Qwen3_5DecoderLayer.__call__), [`attention`](qwen3_5.md#Qwen3_5DecoderLayer.attention)

### `Qwen3_5ScannableBlock`  ·  implements/extends Module
- def: [`src/maxtext/models/qwen3_5.py:58`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen3_5.py#L58)
- doc: Scanned Structure for Text-only Architecture, explicitly invoking Qwen3_5 layers.
- signature: `class Qwen3_5ScannableBlock(nnx.Module):`
- members:
  - `config` — [`L62`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen3_5.py#L62)
  - `mesh` — [`L63`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen3_5.py#L63)
  - `model_mode` — [`L64`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen3_5.py#L64)
  - `quant` — [`L65`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen3_5.py#L65)
  - `rngs` — [`L66`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen3_5.py#L66)
- protocol/private: `__call__`[`L83`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen3_5.py#L83), `__init__`[`L61`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen3_5.py#L61)
- uses (calls/refs, reference-scoped): [`Qwen3_5DecoderLayer`](qwen3_5.md#Qwen3_5DecoderLayer)
- used by: [`Qwen3_5ScannableBlockToLinen`](qwen3_5.md#Qwen3_5ScannableBlockToLinen)

### `Qwen3_5SparseMoEBlock`  ·  implements/extends Qwen3NextSparseMoeBlock
- def: [`src/maxtext/models/qwen3_5.py:54`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen3_5.py#L54)
- doc: Shares same MoE code as Qwen3-Next
- signature: `class Qwen3_5SparseMoEBlock(Qwen3NextSparseMoeBlock):`
- used by: [`mlp`](qwen3_5.md#Qwen3_5DecoderLayer.mlp)

## Module values
- `Qwen3_5DecoderLayerToLinen` — [`L246`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen3_5.py#L246)
- `Qwen3_5ScannableBlockToLinen` — [`L252`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen3_5.py#L252)


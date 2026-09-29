---
title: 'Module: src/maxtext/models/gemma4.py'
type: catalog
provenance: extracted
module: src/maxtext/models/gemma4.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.models.gemma4`/
symbols:
  Gemma4DecoderLayer.__call__: Gemma4DecoderLayer#__call__().
  Gemma4ScannableBlock.num_of_layers: Gemma4ScannableBlock#num_of_layers.
  Gemma4MoE.config: Gemma4MoE#config.
  Gemma4MoE.__call__: Gemma4MoE#__call__().
  Gemma4DecoderLayer.mlp: Gemma4DecoderLayer#mlp.
  Gemma4DecoderLayer.self_attention: Gemma4DecoderLayer#self_attention.
  Gemma4DecoderLayer.activation_axis_names: Gemma4DecoderLayer#activation_axis_names.
  Gemma4MoE.moe_block: Gemma4MoE#moe_block.
  Gemma4DecoderLayer.rngs: Gemma4DecoderLayer#rngs.
  Gemma4DecoderLayer.post_self_attention_norm: Gemma4DecoderLayer#post_self_attention_norm.
  Gemma4DecoderLayer.post_ffw_norm: Gemma4DecoderLayer#post_ffw_norm.
  Gemma4MoE.pre_feedforward_layernorm_2: Gemma4MoE#pre_feedforward_layernorm_2.
  Gemma4MoE.post_feedforward_layernorm_1: Gemma4MoE#post_feedforward_layernorm_1.
  Gemma4MoE.post_feedforward_layernorm_2: Gemma4MoE#post_feedforward_layernorm_2.
  Gemma4MoE.gate_norm: Gemma4MoE#gate_norm.
  Gemma4MoE.rngs: Gemma4MoE#rngs.
  Gemma4DecoderLayer.config: Gemma4DecoderLayer#config.
  Gemma4ScannableBlock.__call__: Gemma4ScannableBlock#__call__().
  get_attention_type: get_attention_type().
  Gemma4MoE.pre_forward_scale_2: Gemma4MoE#pre_forward_scale_2.
  Gemma4DecoderLayer.pre_self_attention_norm: Gemma4DecoderLayer#pre_self_attention_norm.
  Gemma4DecoderLayer.pre_ffw_norm: Gemma4DecoderLayer#pre_ffw_norm.
  Gemma4DecoderLayer.quant: Gemma4DecoderLayer#quant.
  Gemma4DecoderLayerToLinen: Gemma4DecoderLayerToLinen.
  Gemma4ScannableBlockToLinen: Gemma4ScannableBlockToLinen.
  GEMMA4_ATTENTION_PATTERN: GEMMA4_ATTENTION_PATTERN.
  Gemma4DecoderLayer: Gemma4DecoderLayer#
  Gemma4DecoderLayer.attention_type: Gemma4DecoderLayer#attention_type.
  Gemma4ScannableBlock.config: Gemma4ScannableBlock#config.
  Gemma4MoE: Gemma4MoE#
  Gemma4MoE.quant: Gemma4MoE#quant.
  Gemma4DecoderLayer.layer_scalar: Gemma4DecoderLayer#layer_scalar.
  Gemma4DecoderLayer.update_cache: Gemma4DecoderLayer#update_cache().
  Gemma4ScannableBlock: Gemma4ScannableBlock#
  Gemma4ScannableBlock.mesh: Gemma4ScannableBlock#mesh.
  Gemma4ScannableBlock.model_mode: Gemma4ScannableBlock#model_mode.
  Gemma4ScannableBlock.quant: Gemma4ScannableBlock#quant.
  Gemma4ScannableBlock.rngs: Gemma4ScannableBlock#rngs.
  Gemma4MoE.__init__: Gemma4MoE#__init__().
  Gemma4MoE.mesh: Gemma4MoE#mesh.
  Gemma4DecoderLayer.__init__: Gemma4DecoderLayer#__init__().
  Gemma4DecoderLayer.mesh: Gemma4DecoderLayer#mesh.
  Gemma4DecoderLayer.layer_idx: Gemma4DecoderLayer#layer_idx.
  Gemma4ScannableBlock.__init__: Gemma4ScannableBlock#__init__().
---
# Module: [`src/maxtext/models/gemma4.py`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4.py)

## Classes
### `Gemma4DecoderLayer`  ·  implements/extends Module
- def: [`src/maxtext/models/gemma4.py:148`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4.py#L148)
- doc: Transformer decoder layer for Gemma4.
- signature: `class Gemma4DecoderLayer(nnx.Module):`
- members:
  - `__init__(self, config: Config, mesh: Mesh, model_mode: str, rngs: nnx.Rngs, quant: None | Quant = None, attention_type: AttentionType = AttentionType.LOCAL_SLIDING, layer_idx: int = 0)` — [`L151`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4.py#L151) — Initializes the instance.
  - `update_cache(cache, val)` — [`L394`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4.py#L394)
  - `activation_axis_names` — [`L305`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4.py#L305)
  - `attention_type` — [`L177`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4.py#L177)
  - `config` — [`L173`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4.py#L173)
  - `layer_idx` — [`L178`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4.py#L178)
  - `layer_scalar` — [`L302`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4.py#L302)
  - `mesh` — [`L174`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4.py#L174)
  - `mlp` — [`L270`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4.py#L270)
  - `post_ffw_norm` — [`L292`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4.py#L292)
  - `post_self_attention_norm` — [`L251`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4.py#L251)
  - `pre_ffw_norm` — [`L261`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4.py#L261)
  - `pre_self_attention_norm` — [`L183`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4.py#L183)
  - `quant` — [`L175`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4.py#L175)
  - `rngs` — [`L176`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4.py#L176)
  - `self_attention` — [`L219`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4.py#L219)
- protocol/private: `__call__`[`L309`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4.py#L309)
- uses (calls/refs, reference-scoped): [`Gemma4MoE`](gemma4.md#Gemma4MoE)
- used by: [`num_of_layers`](gemma4.md#Gemma4ScannableBlock.num_of_layers), [`Gemma4DecoderLayerToLinen`](gemma4.md#Gemma4DecoderLayerToLinen)

### `Gemma4MoE`  ·  implements/extends Module
- def: [`src/maxtext/models/gemma4.py:55`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4.py#L55)
- doc: Gemma4 specific MoE block containing layer norms and a generic MoE block.
- signature: `class Gemma4MoE(nnx.Module):`
- members:
  - `config` — [`L65`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4.py#L65)
  - `gate_norm` — [`L106`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4.py#L106)
  - `mesh` — [`L66`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4.py#L66)
  - `moe_block` — [`L70`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4.py#L70)
  - `post_feedforward_layernorm_1` — [`L92`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4.py#L92)
  - `post_feedforward_layernorm_2` — [`L99`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4.py#L99)
  - `pre_feedforward_layernorm_2` — [`L85`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4.py#L85)
  - `pre_forward_scale_2` — [`L81`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4.py#L81)
  - `quant` — [`L68`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4.py#L68)
  - `rngs` — [`L67`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4.py#L67)
- protocol/private: `__call__`[`L116`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4.py#L116), `__init__`[`L58`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4.py#L58)
- used by: [`mlp`](gemma4.md#Gemma4DecoderLayer.mlp)

### `Gemma4ScannableBlock`  ·  implements/extends Module
- def: [`src/maxtext/models/gemma4.py:411`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4.py#L411)
- doc: A repeatable block of Gemma4 decoder layers.
- signature: `class Gemma4ScannableBlock(nnx.Module):`
- members:
  - `__init__(self, config: Config, mesh: Mesh, model_mode: str, rngs: nnx.Rngs, quant: None | Quant = None, num_of_layers: int = 1)` — [`L414`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4.py#L414) — Initializes the instance.
  - `config` — [`L433`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4.py#L433)
  - `mesh` — [`L434`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4.py#L434)
  - `model_mode` — [`L435`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4.py#L435)
  - `num_of_layers` — [`L438`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4.py#L438)
  - `quant` — [`L436`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4.py#L436)
  - `rngs` — [`L437`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4.py#L437)
- protocol/private: `__call__`[`L454`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4.py#L454)
- uses (calls/refs, reference-scoped): [`get_attention_type`](gemma4.md#get_attention_type), [`Gemma4DecoderLayer`](gemma4.md#Gemma4DecoderLayer)
- used by: [`Gemma4ScannableBlockToLinen`](gemma4.md#Gemma4ScannableBlockToLinen)

## Functions
- `get_attention_type(layer_id)` — [`L50`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4.py#L50)

## Module values
- `GEMMA4_ATTENTION_PATTERN` — [`L40`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4.py#L40)
- `Gemma4DecoderLayerToLinen` — [`L405`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4.py#L405)
- `Gemma4ScannableBlockToLinen` — [`L499`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4.py#L499)


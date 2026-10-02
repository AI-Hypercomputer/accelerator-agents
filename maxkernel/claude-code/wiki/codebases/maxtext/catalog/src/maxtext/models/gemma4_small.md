---
title: 'Module: src/maxtext/models/gemma4_small.py'
type: catalog
provenance: extracted
module: src/maxtext/models/gemma4_small.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.models.gemma4_small`/
symbols:
  Gemma4SmallDecoderLayer.__call__: Gemma4SmallDecoderLayer#__call__().
  Gemma4SmallPLE.__call__: Gemma4SmallPLE#__call__().
  Gemma4SmallDecoderLayer.post_per_layer_input_norm: Gemma4SmallDecoderLayer#post_per_layer_input_norm.
  Gemma4SmallDecoderLayer.activation_axis_names: Gemma4SmallDecoderLayer#activation_axis_names.
  Gemma4SmallDecoderLayer.compute_shared_kv: Gemma4SmallDecoderLayer#compute_shared_kv().
  kv_donor_layer_idx: kv_donor_layer_idx().
  get_attention_pattern: get_attention_pattern().
  Gemma4SmallDecoderLayer.is_shared: Gemma4SmallDecoderLayer#is_shared.
  is_kv_shared_layer: is_kv_shared_layer().
  Gemma4SmallDecoderLayer.self_attention: Gemma4SmallDecoderLayer#self_attention.
  get_attention_type: get_attention_type().
  Gemma4SmallDecoderLayer.per_layer_input_gate: Gemma4SmallDecoderLayer#per_layer_input_gate.
  build_layer_types: build_layer_types().
  kv_cache_slot_map: kv_cache_slot_map().
  is_kv_donor_layer: is_kv_donor_layer().
  PLEToLinen: PLEToLinen.
  Gemma4SmallDecoderLayerToLinen: Gemma4SmallDecoderLayerToLinen.
  first_kv_shared_layer_idx: first_kv_shared_layer_idx().
  Gemma4SmallPLE._ple_dim: Gemma4SmallPLE#_ple_dim.
  Gemma4SmallPLE._num_layers: Gemma4SmallPLE#_num_layers.
  Gemma4SmallDecoderLayer.pre_self_attention_norm: Gemma4SmallDecoderLayer#pre_self_attention_norm.
  Gemma4SmallDecoderLayer.per_layer_projection: Gemma4SmallDecoderLayer#per_layer_projection.
  GEMMA4_E2B_ATTENTION_PATTERN: GEMMA4_E2B_ATTENTION_PATTERN.
  GEMMA4_E4B_ATTENTION_PATTERN: GEMMA4_E4B_ATTENTION_PATTERN.
  Gemma4SmallPLE: Gemma4SmallPLE#
  Gemma4SmallPLE.config: Gemma4SmallPLE#config.
  Gemma4SmallPLE.embed_tokens_per_layer: Gemma4SmallPLE#embed_tokens_per_layer.
  Gemma4SmallPLE.per_layer_model_projection: Gemma4SmallPLE#per_layer_model_projection.
  Gemma4SmallPLE.per_layer_projection_norm: Gemma4SmallPLE#per_layer_projection_norm.
  Gemma4SmallPLE._embed_scale_value: Gemma4SmallPLE#_embed_scale_value.
  Gemma4SmallPLE._proj_scale_value: Gemma4SmallPLE#_proj_scale_value.
  Gemma4SmallDecoderLayer: Gemma4SmallDecoderLayer#
  Gemma4SmallDecoderLayer.config: Gemma4SmallDecoderLayer#config.
  Gemma4SmallDecoderLayer.attention_type: Gemma4SmallDecoderLayer#attention_type.
  Gemma4SmallDecoderLayer.post_self_attention_norm: Gemma4SmallDecoderLayer#post_self_attention_norm.
  Gemma4SmallDecoderLayer.pre_ffw_norm: Gemma4SmallDecoderLayer#pre_ffw_norm.
  Gemma4SmallDecoderLayer.post_ffw_norm: Gemma4SmallDecoderLayer#post_ffw_norm.
  Gemma4SmallDecoderLayer.mlp: Gemma4SmallDecoderLayer#mlp.
  Gemma4SmallDecoderLayer.layer_scalar: Gemma4SmallDecoderLayer#layer_scalar.
  Gemma4SmallPLE.__init__: Gemma4SmallPLE#__init__().
  Gemma4SmallPLE.mesh: Gemma4SmallPLE#mesh.
  Gemma4SmallDecoderLayer.__init__: Gemma4SmallDecoderLayer#__init__().
  Gemma4SmallDecoderLayer.mesh: Gemma4SmallDecoderLayer#mesh.
  Gemma4SmallDecoderLayer.quant: Gemma4SmallDecoderLayer#quant.
  Gemma4SmallDecoderLayer.layer_idx: Gemma4SmallDecoderLayer#layer_idx.
  Gemma4SmallDecoderLayer.num_kv_heads: Gemma4SmallDecoderLayer#num_kv_heads.
  Gemma4SmallDecoderLayer.head_dim: Gemma4SmallDecoderLayer#head_dim.
---
# Module: [`src/maxtext/models/gemma4_small.py`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_small.py)

## Classes
### `Gemma4SmallDecoderLayer`  ·  implements/extends Module
- def: [`src/maxtext/models/gemma4_small.py:227`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_small.py#L227)
- doc: Transformer decoder layer for Gemma 4 small (E2B / E4B).
- signature: `class Gemma4SmallDecoderLayer(nnx.Module):`
- members:
  - `compute_shared_kv(self, inputs: jax.Array, decoder_positions: jax.Array)` — [`L395`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_small.py#L395) — Returns the rotated, normed K / V for this (non-shared) layer.
  - `activation_axis_names` — [`L391`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_small.py#L391)
  - `attention_type` — [`L243`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_small.py#L243)
  - `config` — [`L240`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_small.py#L240)
  - `head_dim` — [`L268`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_small.py#L268)
  - `is_shared` — [`L247`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_small.py#L247)
  - `layer_idx` — [`L244`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_small.py#L244)
  - `layer_scalar` — [`L388`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_small.py#L388)
  - `mesh` — [`L241`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_small.py#L241)
  - `mlp` — [`L339`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_small.py#L339)
  - `num_kv_heads` — [`L267`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_small.py#L267)
  - `per_layer_input_gate` — [`L355`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_small.py#L355)
  - `per_layer_projection` — [`L365`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_small.py#L365)
  - `post_ffw_norm` — [`L291`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_small.py#L291)
  - `post_per_layer_input_norm` — [`L375`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_small.py#L375)
  - `post_self_attention_norm` — [`L277`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_small.py#L277)
  - `pre_ffw_norm` — [`L284`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_small.py#L284)
  - `pre_self_attention_norm` — [`L270`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_small.py#L270)
  - `quant` — [`L242`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_small.py#L242)
  - `self_attention` — [`L302`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_small.py#L302)
- protocol/private: `__call__`[`L408`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_small.py#L408), `__init__`[`L230`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_small.py#L230)
- uses (calls/refs, reference-scoped): [`is_kv_shared_layer`](gemma4_small.md#is_kv_shared_layer)
- used by: [`Gemma4SmallDecoderLayerToLinen`](gemma4_small.md#Gemma4SmallDecoderLayerToLinen)

### `Gemma4SmallPLE`  ·  implements/extends Module
- def: [`src/maxtext/models/gemma4_small.py:147`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_small.py#L147)
- doc: Builds the `[B, S, num_layers, D_ple]` per-layer-input tensor.
- signature: `class Gemma4SmallPLE(nnx.Module):`
- members:
  - `__call__(self, input_ids: jax.Array, inputs_embeds: jax.Array)` — [`L200`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_small.py#L200) — Returns ``per_layer_inputs`` of shape ``[B, S, L, D_ple]``.
  - `config` — [`L157`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_small.py#L157)
  - `embed_tokens_per_layer` — [`L164`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_small.py#L164)
  - `mesh` — [`L158`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_small.py#L158)
  - `per_layer_model_projection` — [`L173`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_small.py#L173)
  - `per_layer_projection_norm` — [`L184`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_small.py#L184)
- protocol/private: `__init__`[`L150`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_small.py#L150), `_embed_scale_value`[`L197`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_small.py#L197), `_num_layers`[`L196`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_small.py#L196), `_ple_dim`[`L195`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_small.py#L195), `_proj_scale_value`[`L198`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_small.py#L198)
- used by: [`PLEToLinen`](gemma4_small.md#PLEToLinen)

## Functions
- `build_layer_types(num_layers, model_name)` — [`L68`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_small.py#L68) — Returns the per-layer attention-type tuple for the full decoder stack.
- `first_kv_shared_layer_idx(num_layers: int, num_kv_shared_layers: int)` — [`L73`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_small.py#L73) — Index of the first KV-shared layer, or ``num_layers`` if none.
- `get_attention_pattern(model_name)` — [`L55`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_small.py#L55) — Returns the repeating sliding/global attention pattern for a small variant.
- `get_attention_type(layer_id, model_name=None)` — [`L62`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_small.py#L62) — Returns the attention type for ``layer_id`` under the variant's pattern.
- `is_kv_donor_layer(layer_idx: int, layer_types: tuple[AttentionType, ...], num_kv_shared_layers: int)` — [`L130`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_small.py#L130) — Returns True iff this layer's K/V are reused by some shared layer.
- `is_kv_shared_layer(layer_idx: int, num_layers: int, num_kv_shared_layers: int)` — [`L80`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_small.py#L80) — Returns True iff layer ``layer_idx`` reuses K/V from an earlier layer.
- `kv_cache_slot_map(layer_types: tuple[AttentionType, ...], num_kv_shared_layers: int)` — [`L109`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_small.py#L109) — Maps decoder layer index -> KV-cache slot.
- `kv_donor_layer_idx(layer_idx: int, layer_types: tuple[AttentionType, ...], num_kv_shared_layers: int)` — [`L88`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_small.py#L88) — Index of the layer that owns the K/V used by ``layer_idx``.

## Module values
- `GEMMA4_E2B_ATTENTION_PATTERN` — [`L38`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_small.py#L38)
- `GEMMA4_E4B_ATTENTION_PATTERN` — [`L45`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_small.py#L45)
- `Gemma4SmallDecoderLayerToLinen` — [`L481`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_small.py#L481)
- `PLEToLinen` — [`L221`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_small.py#L221)


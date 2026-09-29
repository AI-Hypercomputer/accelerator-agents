---
title: 'Module: src/maxtext/models/qwen2.py'
type: catalog
provenance: extracted
module: src/maxtext/models/qwen2.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.models.qwen2`/
symbols:
  AttentionWithNorm.apply_attention_with_norm: AttentionWithNorm#apply_attention_with_norm().
  Qwen2DecoderLayer.__call__: Qwen2DecoderLayer#__call__().
  AttentionWithNorm.activation_axis_names: AttentionWithNorm#activation_axis_names.
  AttentionWithNorm: AttentionWithNorm#
  Qwen2DecoderLayer: Qwen2DecoderLayer#
  Qwen2DecoderLayer.__init__: Qwen2DecoderLayer#__init__().
  Qwen2DecoderLayerToLinen: Qwen2DecoderLayerToLinen.
  AttentionWithNorm.__init__: AttentionWithNorm#__init__().
  AttentionWithNorm.config: AttentionWithNorm#config.
  AttentionWithNorm.pre_self_attention_layer_norm: AttentionWithNorm#pre_self_attention_layer_norm.
  AttentionWithNorm.self_attention: AttentionWithNorm#self_attention.
  AttentionWithNorm.post_self_attention_layer_norm: AttentionWithNorm#post_self_attention_layer_norm.
  Qwen2DecoderLayer.mlp: Qwen2DecoderLayer#mlp.
  AttentionWithNorm.mesh: AttentionWithNorm#mesh.
  AttentionWithNorm.quant: AttentionWithNorm#quant.
---
# Module: [`src/maxtext/models/qwen2.py`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen2.py)

## Classes
### `AttentionWithNorm`  ·  implements/extends Module
- def: [`src/maxtext/models/qwen2.py:42`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen2.py#L42)
- doc: Base class with shared common components: self-attention block with normalization.
- signature: `class AttentionWithNorm(nnx.Module):`
- members:
  - `apply_attention_with_norm(self, inputs: jnp.ndarray, decoder_segment_ids: None | jnp.ndarray, decoder_positions: None | jnp.ndarray, deterministic: bool, model_mode: str, kv_cache: None | jnp.ndarray = None, attention_metadata: None | dict[str, Any] = None)` — [`L112`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen2.py#L112) — Applies self-attention with pre and post-layer normalization.
  - `activation_axis_names` — [`L59`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen2.py#L59)
  - `config` — [`L53`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen2.py#L53)
  - `mesh` — [`L54`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen2.py#L54)
  - `post_self_attention_layer_norm` — [`L103`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen2.py#L103)
  - `pre_self_attention_layer_norm` — [`L62`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen2.py#L62)
  - `quant` — [`L55`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen2.py#L55)
  - `self_attention` — [`L73`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen2.py#L73)
- protocol/private: `__init__`[`L45`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen2.py#L45)
- uses (calls/refs, reference-scoped): [`Qwen2DecoderLayer`](qwen2.md#Qwen2DecoderLayer)
- used by: [`__call__`](qwen2.md#Qwen2DecoderLayer.__call__), [`Qwen2DecoderLayer`](qwen2.md#Qwen2DecoderLayer), [`__init__`](qwen2.md#Qwen2DecoderLayer.__init__)

### `Qwen2DecoderLayer`  ·  implements/extends AttentionWithNorm
- def: [`src/maxtext/models/qwen2.py:151`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen2.py#L151)
- doc: Qwen2 Transformer decoder layer (dense).
- signature: `class Qwen2DecoderLayer(AttentionWithNorm):`
- members:
  - `mlp` — [`L163`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen2.py#L163)
- protocol/private: `__call__`[`L177`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen2.py#L177), `__init__`[`L154`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen2.py#L154)
- uses (calls/refs, reference-scoped): [`apply_attention_with_norm`](qwen2.md#AttentionWithNorm.apply_attention_with_norm), [`activation_axis_names`](qwen2.md#AttentionWithNorm.activation_axis_names), [`AttentionWithNorm`](qwen2.md#AttentionWithNorm), [`__init__`](qwen2.md#AttentionWithNorm.__init__), [`config`](qwen2.md#AttentionWithNorm.config)
- used by: [`AttentionWithNorm`](qwen2.md#AttentionWithNorm), [`Qwen2DecoderLayerToLinen`](qwen2.md#Qwen2DecoderLayerToLinen)

## Module values
- `Qwen2DecoderLayerToLinen` — [`L214`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen2.py#L214)


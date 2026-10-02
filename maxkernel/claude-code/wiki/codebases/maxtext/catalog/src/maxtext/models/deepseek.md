---
title: 'Module: src/maxtext/models/deepseek.py'
type: catalog
provenance: extracted
module: src/maxtext/models/deepseek.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.models.deepseek`/DeepSeek
symbols:
  DeepSeekGenericLayer.config: GenericLayer#config.
  DeepSeekMoELayer.__call__: MoELayer#__call__().
  DeepSeekDenseLayer.__call__: DenseLayer#__call__().
  DeepSeekGenericLayer.self_attention_with_norm_op: GenericLayer#self_attention_with_norm_op().
  DeepSeekGenericLayer.mesh: GenericLayer#mesh.
  DeepSeekGenericLayer.with_logical_constraint: GenericLayer#with_logical_constraint().
  DeepSeekGenericLayer.engram: GenericLayer#engram.
  DeepSeekGenericLayer.engram_op: GenericLayer#engram_op().
  DeepSeekGenericLayer.out_sharding: GenericLayer#out_sharding.
  DeepSeekGenericLayer.attention_op: GenericLayer#attention_op().
  DeepSeekDenseLayer.mlp_op: DenseLayer#mlp_op().
  DeepSeekMoELayer.mlp_op: MoELayer#mlp_op().
  DeepSeekGenericLayer.mlp_intermediate_sharding: GenericLayer#mlp_intermediate_sharding.
  DeepSeekGenericLayer.mhc_mlp: GenericLayer#mhc_mlp.
  DeepSeekDenseLayer.mlp: DenseLayer#mlp.
  DeepSeekGenericLayer.dummy_inputs_shape: GenericLayer#dummy_inputs_shape.
  DeepSeekGenericLayer.post_self_attention_layer_norm: GenericLayer#post_self_attention_layer_norm.
  DeepSeekGenericLayer.dropout: GenericLayer#dropout.
  DeepSeekGenericLayer.mhc_attention: GenericLayer#mhc_attention.
  DeepSeekGenericLayer.is_engram_enabled: GenericLayer#is_engram_enabled.
  DeepSeekGenericLayer.post_attention_norm_op: GenericLayer#post_attention_norm_op().
  DeepSeekGenericLayer: GenericLayer#
  DeepSeekGenericLayer.engram_layer_norm: GenericLayer#engram_layer_norm.
  DeepSeekGenericLayer.self_attention: GenericLayer#self_attention.
  DeepSeekGenericLayer.dropout_op: GenericLayer#dropout_op().
  DeepSeekGenericLayer.pre_attention_norm_op: GenericLayer#pre_attention_norm_op().
  DeepSeekMoELayer.DeepSeekMoeBlock_0: MoELayer#DeepSeekMoeBlock_0.
  DeepSeekGenericLayer.pre_self_attention_layer_norm: GenericLayer#pre_self_attention_layer_norm.
  DeepSeekMoELayer.extract_fn: MoELayer#extract_fn().
  DeepSeekGenericLayer.model_mode: GenericLayer#model_mode.
  DeepSeekGenericLayer.rngs: GenericLayer#rngs.
  DeepSeekGenericLayer.mlp_op: GenericLayer#mlp_op().
  DeepSeekGenericLayer.logical_axis_names: GenericLayer#logical_axis_names().
  DeepSeekGenericLayer.post_process: GenericLayer#post_process().
  DeepSeekGenericLayer.is_mhc_enabled: GenericLayer#is_mhc_enabled.
  DeepSeekGenericLayer.mlp_logical_axis_names: GenericLayer#mlp_logical_axis_names().
  DeepSeekDenseLayer: DenseLayer#
  DeepSeekMoELayer: MoELayer#
  DeepSeekDenseLayer.__init__: DenseLayer#__init__().
  DeepSeekDenseLayerToLinen: DenseLayerToLinen.
  DeepSeekMoELayer.__init__: MoELayer#__init__().
  DeepSeekMoELayerToLinen: MoELayerToLinen.
  DeepSeekGenericLayer.__init__: GenericLayer#__init__().
  DeepSeekGenericLayer.ngram_hash_mapping: GenericLayer#ngram_hash_mapping.
  DeepSeekGenericLayer.quant: GenericLayer#quant.
  DeepSeekGenericLayer.layer_idx: GenericLayer#layer_idx.
---
# Module: [`src/maxtext/models/deepseek.py`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek.py)

## Classes
### `DeepSeekDenseLayer`  ·  implements/extends DeepSeekGenericLayer
- def: [`src/maxtext/models/deepseek.py:311`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek.py#L311)
- doc: DeepSeek-style dense layer with Multi-Head Latent Attention.
- signature: `class DeepSeekDenseLayer(DeepSeekGenericLayer):`
- members:
  - `mlp_op(self, x, deterministic, *args, **kwargs)` — [`L338`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek.py#L338) — documented in [maxtext-models-deepseek](../../../../concepts/maxtext-models-deepseek.md)
  - `mlp` — [`L324`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek.py#L324) — documented in [maxtext-models-deepseek](../../../../concepts/maxtext-models-deepseek.md)
- protocol/private: `__call__`[`L342`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek.py#L342), `__init__`[`L314`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek.py#L314)
- uses (calls/refs, reference-scoped): [`config`](deepseek.md#DeepSeekGenericLayer.config), [`self_attention_with_norm_op`](deepseek.md#DeepSeekGenericLayer.self_attention_with_norm_op), [`with_logical_constraint`](deepseek.md#DeepSeekGenericLayer.with_logical_constraint), [`engram_op`](deepseek.md#DeepSeekGenericLayer.engram_op), [`out_sharding`](deepseek.md#DeepSeekGenericLayer.out_sharding), [`mhc_mlp`](deepseek.md#DeepSeekGenericLayer.mhc_mlp), [`mlp_intermediate_sharding`](deepseek.md#DeepSeekGenericLayer.mlp_intermediate_sharding), [`dummy_inputs_shape`](deepseek.md#DeepSeekGenericLayer.dummy_inputs_shape), [`is_engram_enabled`](deepseek.md#DeepSeekGenericLayer.is_engram_enabled), [`post_attention_norm_op`](deepseek.md#DeepSeekGenericLayer.post_attention_norm_op), [`DeepSeekGenericLayer`](deepseek.md#DeepSeekGenericLayer), [`dropout_op`](deepseek.md#DeepSeekGenericLayer.dropout_op), [`rngs`](deepseek.md#DeepSeekGenericLayer.rngs), [`post_process`](deepseek.md#DeepSeekGenericLayer.post_process), [`is_mhc_enabled`](deepseek.md#DeepSeekGenericLayer.is_mhc_enabled), [`__init__`](deepseek.md#DeepSeekGenericLayer.__init__)
- used by: [`DeepSeekGenericLayer`](deepseek.md#DeepSeekGenericLayer), [`mlp_op`](deepseek.md#DeepSeekGenericLayer.mlp_op), [`DeepSeekDenseLayerToLinen`](deepseek.md#DeepSeekDenseLayerToLinen)

### `DeepSeekGenericLayer`  ·  implements/extends Module
- def: [`src/maxtext/models/deepseek.py:53`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek.py#L53)
- doc: Generic DeepSeek layer with Multi-Head Latent Attention.
- signature: `class DeepSeekGenericLayer(nnx.Module):`
- members:
  - `attention_op(self, x, decoder_segment_ids, decoder_positions, deterministic, previous_chunk=None, slot: None | int = None)` — [`L207`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek.py#L207) — Executes the attention layer. — documented in [maxtext-models-deepseek](../../../../concepts/maxtext-models-deepseek.md)
  - `dropout_op(self, x, deterministic)` — [`L195`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek.py#L195) — documented in [maxtext-models-deepseek](../../../../concepts/maxtext-models-deepseek.md)
  - `engram_op(self, x, decoder_input_tokens)` — [`L305`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek.py#L305) — documented in [maxtext-models-deepseek](../../../../concepts/maxtext-models-deepseek.md)
  - `logical_axis_names(self)` — [`L231`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek.py#L231) — Generate logical names for activations generally. — documented in [maxtext-models-deepseek](../../../../concepts/maxtext-models-deepseek.md)
  - `mlp_logical_axis_names(self)` — [`L238`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek.py#L238) — Generate logical names for activations in MLP. — documented in [maxtext-models-deepseek](../../../../concepts/maxtext-models-deepseek.md)
  - `mlp_op(self, x, deterministic, *args, **kwargs)` — [`L180`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek.py#L180) — Executes the MLP operation. To be implemented by subclasses.
  - `post_attention_norm_op(self, x)` — [`L203`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek.py#L203) — documented in [maxtext-models-deepseek](../../../../concepts/maxtext-models-deepseek.md)
  - `post_process(self, layer_output, load_balance_loss, moe_bias_updates, kv_cache=None)` — [`L244`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek.py#L244) — postprocessing. — documented in [maxtext-models-deepseek](../../../../concepts/maxtext-models-deepseek.md)
  - `pre_attention_norm_op(self, x)` — [`L199`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek.py#L199) — documented in [maxtext-models-deepseek](../../../../concepts/maxtext-models-deepseek.md)
  - `self_attention_with_norm_op(self, inputs, decoder_segment_ids, decoder_positions, deterministic, previous_chunk=None, slot: None | int = None)` — [`L266`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek.py#L266) — self-attention with normalization — documented in [maxtext-models-deepseek](../../../../concepts/maxtext-models-deepseek.md)
  - `with_logical_constraint(self, x)` — [`L184`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek.py#L184) — documented in [maxtext-models-deepseek](../../../../concepts/maxtext-models-deepseek.md)
  - `config` — [`L69`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek.py#L69) — documented in [maxtext-models-deepseek](../../../../concepts/maxtext-models-deepseek.md)
  - `dropout` — [`L175`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek.py#L175) — documented in [maxtext-models-deepseek](../../../../concepts/maxtext-models-deepseek.md)
  - `dummy_inputs_shape` — [`L79`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek.py#L79) — documented in [maxtext-models-deepseek](../../../../concepts/maxtext-models-deepseek.md)
  - `engram` — [`L125`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek.py#L125) — documented in [maxtext-models-deepseek](../../../../concepts/maxtext-models-deepseek.md)
  - `engram_layer_norm` — [`L105`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek.py#L105) — documented in [maxtext-models-deepseek](../../../../concepts/maxtext-models-deepseek.md)
  - `is_engram_enabled` — [`L76`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek.py#L76) — documented in [maxtext-models-deepseek](../../../../concepts/maxtext-models-deepseek.md)
  - `is_mhc_enabled` — [`L74`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek.py#L74) — documented in [maxtext-models-deepseek](../../../../concepts/maxtext-models-deepseek.md)
  - `layer_idx` — [`L75`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek.py#L75) — documented in [maxtext-models-deepseek](../../../../concepts/maxtext-models-deepseek.md)
  - `mesh` — [`L71`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek.py#L71) — documented in [maxtext-models-deepseek](../../../../concepts/maxtext-models-deepseek.md)
  - `mhc_attention` — [`L177`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek.py#L177) — documented in [maxtext-models-deepseek](../../../../concepts/maxtext-models-deepseek.md)
  - `mhc_mlp` — [`L178`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek.py#L178) — documented in [maxtext-models-deepseek](../../../../concepts/maxtext-models-deepseek.md)
  - `mlp_intermediate_sharding` — [`L82`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek.py#L82) — documented in [maxtext-models-deepseek](../../../../concepts/maxtext-models-deepseek.md)
  - `model_mode` — [`L70`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek.py#L70) — documented in [maxtext-models-deepseek](../../../../concepts/maxtext-models-deepseek.md)
  - `ngram_hash_mapping` — [`L116`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek.py#L116) — documented in [maxtext-models-deepseek](../../../../concepts/maxtext-models-deepseek.md)
  - `out_sharding` — [`L81`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek.py#L81) — documented in [maxtext-models-deepseek](../../../../concepts/maxtext-models-deepseek.md)
  - `post_self_attention_layer_norm` — [`L95`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek.py#L95) — documented in [maxtext-models-deepseek](../../../../concepts/maxtext-models-deepseek.md)
  - `pre_self_attention_layer_norm` — [`L86`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek.py#L86) — documented in [maxtext-models-deepseek](../../../../concepts/maxtext-models-deepseek.md)
  - `quant` — [`L72`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek.py#L72) — documented in [maxtext-models-deepseek](../../../../concepts/maxtext-models-deepseek.md)
  - `rngs` — [`L73`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek.py#L73) — documented in [maxtext-models-deepseek](../../../../concepts/maxtext-models-deepseek.md)
  - `self_attention` — [`L143`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek.py#L143) — documented in [maxtext-models-deepseek](../../../../concepts/maxtext-models-deepseek.md)
- protocol/private: `__init__`[`L60`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek.py#L60)
- uses (calls/refs, reference-scoped): [`mlp_op`](deepseek.md#DeepSeekDenseLayer.mlp_op), [`mlp_op`](deepseek.md#DeepSeekMoELayer.mlp_op), [`DeepSeekDenseLayer`](deepseek.md#DeepSeekDenseLayer), [`DeepSeekMoELayer`](deepseek.md#DeepSeekMoELayer)
- used by: [`__call__`](deepseek.md#DeepSeekMoELayer.__call__), [`__call__`](deepseek.md#DeepSeekDenseLayer.__call__), [`mlp_op`](deepseek.md#DeepSeekDenseLayer.mlp_op), [`mlp_op`](deepseek.md#DeepSeekMoELayer.mlp_op), [`mlp`](deepseek.md#DeepSeekDenseLayer.mlp), [`DeepSeekMoeBlock_0`](deepseek.md#DeepSeekMoELayer.DeepSeekMoeBlock_0), [`extract_fn`](deepseek.md#DeepSeekMoELayer.extract_fn), [`DeepSeekDenseLayer`](deepseek.md#DeepSeekDenseLayer), [`DeepSeekMoELayer`](deepseek.md#DeepSeekMoELayer), [`__init__`](deepseek.md#DeepSeekDenseLayer.__init__), [`__init__`](deepseek.md#DeepSeekMoELayer.__init__)

### `DeepSeekMoELayer`  ·  implements/extends DeepSeekGenericLayer
- def: [`src/maxtext/models/deepseek.py:396`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek.py#L396)
- doc: DeepSeek-style MoE layer with Multi-Head Latent Attention.
- signature: `class DeepSeekMoELayer(DeepSeekGenericLayer):`
- members:
  - `extract_fn(x)` — [`L522`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek.py#L522)
  - `mlp_op(self, x, deterministic, *args, **kwargs)` — [`L603`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek.py#L603) — documented in [maxtext-models-deepseek](../../../../concepts/maxtext-models-deepseek.md)
  - `DeepSeekMoeBlock_0` — [`L413`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek.py#L413) — documented in [maxtext-models-deepseek](../../../../concepts/maxtext-models-deepseek.md)
- protocol/private: `__call__`[`L424`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek.py#L424), `__init__`[`L403`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek.py#L403)
- uses (calls/refs, reference-scoped): [`config`](deepseek.md#DeepSeekGenericLayer.config), [`self_attention_with_norm_op`](deepseek.md#DeepSeekGenericLayer.self_attention_with_norm_op), [`mesh`](deepseek.md#DeepSeekGenericLayer.mesh), [`with_logical_constraint`](deepseek.md#DeepSeekGenericLayer.with_logical_constraint), [`engram_op`](deepseek.md#DeepSeekGenericLayer.engram_op), [`out_sharding`](deepseek.md#DeepSeekGenericLayer.out_sharding), [`mhc_mlp`](deepseek.md#DeepSeekGenericLayer.mhc_mlp), [`mlp_intermediate_sharding`](deepseek.md#DeepSeekGenericLayer.mlp_intermediate_sharding), [`is_engram_enabled`](deepseek.md#DeepSeekGenericLayer.is_engram_enabled), [`post_attention_norm_op`](deepseek.md#DeepSeekGenericLayer.post_attention_norm_op), [`DeepSeekGenericLayer`](deepseek.md#DeepSeekGenericLayer), [`dropout_op`](deepseek.md#DeepSeekGenericLayer.dropout_op), [`rngs`](deepseek.md#DeepSeekGenericLayer.rngs), [`post_process`](deepseek.md#DeepSeekGenericLayer.post_process), [`is_mhc_enabled`](deepseek.md#DeepSeekGenericLayer.is_mhc_enabled), [`__init__`](deepseek.md#DeepSeekGenericLayer.__init__), [`quant`](deepseek.md#DeepSeekGenericLayer.quant)
- used by: [`DeepSeekGenericLayer`](deepseek.md#DeepSeekGenericLayer), [`mlp_op`](deepseek.md#DeepSeekGenericLayer.mlp_op), [`DeepSeekMoELayerToLinen`](deepseek.md#DeepSeekMoELayerToLinen)

## Module values
- `DeepSeekDenseLayerToLinen` — [`L390`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek.py#L390)
- `DeepSeekMoELayerToLinen` — [`L610`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek.py#L610)


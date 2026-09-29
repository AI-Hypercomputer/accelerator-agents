---
title: 'Module: tpu_inference/models/jax/llama_eagle3.py'
type: catalog
provenance: extracted
module: tpu_inference/models/jax/llama_eagle3.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.models.jax.llama_eagle3`/
symbols:
  Eagle3LlamaDecoderLayer.config: Eagle3LlamaDecoderLayer#config.
  EagleLlama3WeightLoader.load_weights: EagleLlama3WeightLoader#load_weights().
  Eagle3LlamaDecoderLayer.__call__: Eagle3LlamaDecoderLayer#__call__().
  EagleLlama3ForCausalLM.load_weights: EagleLlama3ForCausalLM#load_weights().
  EagleLlama3ForCausalLM.model: EagleLlama3ForCausalLM#model.
  Eagle3LlamaModel.__call__: Eagle3LlamaModel#__call__().
  Eagle3LlamaDecoderLayer.hidden_norm: Eagle3LlamaDecoderLayer#hidden_norm.
  init_fn: init_fn.
  Eagle3LlamaDecoderLayer.input_layernorm: Eagle3LlamaDecoderLayer#input_layernorm.
  Eagle3LlamaModel.embed_tokens: Eagle3LlamaModel#embed_tokens.
  EagleLlama3ForCausalLM.compute_logits: EagleLlama3ForCausalLM#compute_logits().
  update_reshape_map_for_eagle3: update_reshape_map_for_eagle3().
  EagleLlama3ForCausalLM.lm_head: EagleLlama3ForCausalLM#lm_head.
  EagleLlama3ForCausalLM.vllm_config: EagleLlama3ForCausalLM#vllm_config.
  EagleLlama3ForCausalLM.__call__: EagleLlama3ForCausalLM#__call__().
  EagleLlama3ForCausalLM.combine_hidden_states: EagleLlama3ForCausalLM#combine_hidden_states().
  Eagle3LlamaModel.layers: Eagle3LlamaModel#layers.
  Eagle3LlamaDecoderLayer: Eagle3LlamaDecoderLayer#
  Eagle3LlamaDecoderLayer._norm_before_residual: Eagle3LlamaDecoderLayer#_norm_before_residual().
  Eagle3LlamaDecoderLayer._norm_after_residual: Eagle3LlamaDecoderLayer#_norm_after_residual().
  Eagle3LlamaModel.fc: Eagle3LlamaModel#fc.
  Eagle3LlamaModel.norm: Eagle3LlamaModel#norm.
  EagleLlama3WeightLoader: EagleLlama3WeightLoader#
  EagleLlama3ForCausalLM.WeightLoader: EagleLlama3ForCausalLM#WeightLoader.
  EagleLlama3WeightLoader.vllm_config: EagleLlama3WeightLoader#vllm_config.
  EagleLlama3ForCausalLM.rng: EagleLlama3ForCausalLM#rng.
  EagleLlama3ForCausalLM: EagleLlama3ForCausalLM#
  logger: logger.
  Eagle3LlamaDecoderLayer.__init__: Eagle3LlamaDecoderLayer#__init__().
  EagleLlama3WeightLoader.__init__: EagleLlama3WeightLoader#__init__().
  EagleLlama3WeightLoader.mesh: EagleLlama3WeightLoader#mesh.
  Eagle3LlamaModel: Eagle3LlamaModel#
  EagleLlama3ForCausalLM.mesh: EagleLlama3ForCausalLM#mesh.
  EagleLlama3ForCausalLM.draft_id_to_target_id: EagleLlama3ForCausalLM#draft_id_to_target_id.
  Eagle3LlamaModel.__init__: Eagle3LlamaModel#__init__().
  EagleLlama3ForCausalLM.__init__: EagleLlama3ForCausalLM#__init__().
---
# Module: [`tpu_inference/models/jax/llama_eagle3.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_eagle3.py)

## Classes
### `Eagle3LlamaDecoderLayer`  ·  implements/extends LlamaDecoderLayer
- def: [`tpu_inference/models/jax/llama_eagle3.py:37`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_eagle3.py#L37)
- signature: `class Eagle3LlamaDecoderLayer(LlamaDecoderLayer):`
- members:
  - `config` — [`L46`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_eagle3.py#L46)
  - `hidden_norm` — [`L84`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_eagle3.py#L84)
  - `input_layernorm` — [`L76`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_eagle3.py#L76)
- protocol/private: `__call__`[`L104`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_eagle3.py#L104), `__init__`[`L39`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_eagle3.py#L39), `_norm_after_residual`[`L98`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_eagle3.py#L98), `_norm_before_residual`[`L92`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_eagle3.py#L92)
- uses (calls/refs, reference-scoped): [`AttentionMetadata`](../../layers/common/attention_metadata.md#AttentionMetadata), [`head_dim`](llama3.md#LlamaAttention.head_dim), [`k_proj`](llama3.md#LlamaAttention.k_proj), [`q_proj`](llama3.md#LlamaAttention.q_proj), [`self_attn`](llama3.md#LlamaDecoderLayer.self_attn), [`v_proj`](llama3.md#LlamaAttention.v_proj), [`init_fn`](llama_eagle3.md#init_fn), [`hidden_size`](llama3.md#LlamaAttention.hidden_size), [`num_heads`](llama3.md#LlamaAttention.num_heads), [`num_kv_heads`](llama3.md#LlamaAttention.num_kv_heads), [`__init__`](llama3.md#LlamaDecoderLayer.__init__), [`LlamaDecoderLayer`](llama3.md#LlamaDecoderLayer), `hidden_size`, [`mlp`](llama3.md#LlamaDecoderLayer.mlp), [`post_attention_layernorm`](llama3.md#LlamaDecoderLayer.post_attention_layernorm), `rms_norm_eps`
- used by: [`LlamaDecoderLayer`](llama3.md#LlamaDecoderLayer), [`layers`](llama_eagle3.md#Eagle3LlamaModel.layers)

### `Eagle3LlamaModel`  ·  implements/extends Module
- def: [`tpu_inference/models/jax/llama_eagle3.py:135`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_eagle3.py#L135)
- signature: `class Eagle3LlamaModel(nnx.Module):`
- members:
  - `embed_tokens` — [`L142`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_eagle3.py#L142)
  - `fc` — [`L165`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_eagle3.py#L165)
  - `layers` — [`L150`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_eagle3.py#L150)
  - `norm` — [`L174`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_eagle3.py#L174)
- protocol/private: `__call__`[`L182`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_eagle3.py#L182), `__init__`[`L137`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_eagle3.py#L137)
- uses (calls/refs, reference-scoped): [`AttentionMetadata`](../../layers/common/attention_metadata.md#AttentionMetadata), [`init_fn`](llama_eagle3.md#init_fn), [`Eagle3LlamaDecoderLayer`](llama_eagle3.md#Eagle3LlamaDecoderLayer)
- used by: [`load_weights`](llama_eagle3.md#EagleLlama3WeightLoader.load_weights), [`model`](llama_eagle3.md#EagleLlama3ForCausalLM.model), [`combine_hidden_states`](llama_eagle3.md#EagleLlama3ForCausalLM.combine_hidden_states)

### `EagleLlama3ForCausalLM`
- def: [`tpu_inference/models/jax/llama_eagle3.py:263`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_eagle3.py#L263)
- members:
  - `combine_hidden_states(self, hidden_states: jax.Array)` — [`L332`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_eagle3.py#L332)
  - `compute_logits(self, hidden_states: jax.Array)` — [`L315`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_eagle3.py#L315)
  - `load_weights(self, rng_key: jax.Array)` — [`L335`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_eagle3.py#L335)
  - `WeightLoader` — [`L264`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_eagle3.py#L264)
  - `draft_id_to_target_id` — [`L295`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_eagle3.py#L295)
  - `lm_head` — [`L286`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_eagle3.py#L286)
  - `mesh` — [`L271`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_eagle3.py#L271)
  - `model` — [`L280`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_eagle3.py#L280)
  - `rng` — [`L270`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_eagle3.py#L270)
  - `vllm_config` — [`L269`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_eagle3.py#L269)
- protocol/private: `__call__`[`L299`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_eagle3.py#L299), `__init__`[`L266`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_eagle3.py#L266)
- uses (calls/refs, reference-scoped): [`AttentionMetadata`](../../layers/common/attention_metadata.md#AttentionMetadata), [`load_weights`](llama_eagle3.md#EagleLlama3WeightLoader.load_weights), [`init_fn`](llama_eagle3.md#init_fn), [`EagleLlama3WeightLoader`](llama_eagle3.md#EagleLlama3WeightLoader), [`fc`](llama_eagle3.md#Eagle3LlamaModel.fc), [`Eagle3LlamaModel`](llama_eagle3.md#Eagle3LlamaModel)
- used by: [`_get_model_architecture`](../common/model_loader.md#_get_model_architecture), [`load_weights`](llama_eagle3.md#EagleLlama3WeightLoader.load_weights)

### `EagleLlama3WeightLoader`  ·  implements/extends BaseWeightLoader
- def: [`tpu_inference/models/jax/llama_eagle3.py:227`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_eagle3.py#L227)
- signature: `class EagleLlama3WeightLoader(BaseWeightLoader):`
- members:
  - `load_weights(self, model: EagleLlama3ForCausalLM, mappings: dict)` — [`L234`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_eagle3.py#L234)
  - `mesh` — [`L232`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_eagle3.py#L232)
  - `vllm_config` — [`L231`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_eagle3.py#L231)
- protocol/private: `__init__`[`L229`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_eagle3.py#L229)
- uses (calls/refs, reference-scoped): [`get_default_maps`](utils/weight_utils.md#get_default_maps), [`BaseWeightLoader`](utils/weight_utils.md#BaseWeightLoader), [`load_hf_weights`](utils/weight_utils.md#load_hf_weights), [`model`](llama_eagle3.md#EagleLlama3ForCausalLM.model), [`embed_tokens`](llama_eagle3.md#Eagle3LlamaModel.embed_tokens), `embedding`, [`update_reshape_map_for_eagle3`](llama_eagle3.md#update_reshape_map_for_eagle3), [`__init__`](utils/weight_utils.md#BaseWeightLoader.__init__), [`EagleLlama3ForCausalLM`](llama_eagle3.md#EagleLlama3ForCausalLM)
- used by: [`BaseWeightLoader`](utils/weight_utils.md#BaseWeightLoader), [`load_weights`](llama_eagle3.md#EagleLlama3ForCausalLM.load_weights), [`WeightLoader`](llama_eagle3.md#EagleLlama3ForCausalLM.WeightLoader)

## Functions
- `update_reshape_map_for_eagle3(vllm_config: VllmConfig, metadata_map: MetadataMap)` — [`L210`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_eagle3.py#L210)

## Module values
- `init_fn` — [`L34`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_eagle3.py#L34)
- `logger` — [`L32`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_eagle3.py#L32)


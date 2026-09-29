---
title: 'Module: tpu_inference/models/jax/llama_guard_4.py'
type: catalog
provenance: extracted
module: tpu_inference/models/jax/llama_guard_4.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.models.jax.llama_guard_4`/
symbols:
  LlamaGuard4ForCausalLM.end_layer: LlamaGuard4ForCausalLM#end_layer.
  LlamaGuard4WeightLoader.pp_missing_layers: LlamaGuard4WeightLoader#pp_missing_layers.
  LlamaGuard4ForCausalLM.lm_head: LlamaGuard4ForCausalLM#lm_head.
  LlamaGuard4ForCausalLM.embedder: LlamaGuard4ForCausalLM#embedder.
  LlamaGuard4ForCausalLM.final_norm: LlamaGuard4ForCausalLM#final_norm.
  LlamaGuard4ForCausalLM.load_weights: LlamaGuard4ForCausalLM#load_weights().
  LlamaGuard4ForCausalLM.__call__: LlamaGuard4ForCausalLM#__call__().
  LlamaGuard4ForCausalLM.vision_rope: LlamaGuard4ForCausalLM#vision_rope.
  LlamaGuard4ForCausalLM.vision_model: LlamaGuard4ForCausalLM#vision_model.
  LlamaGuard4ForCausalLM.is_last_rank: LlamaGuard4ForCausalLM#is_last_rank.
  LlamaGuard4ForCausalLM.text_config: LlamaGuard4ForCausalLM#text_config.
  LlamaGuard4ForCausalLM.multi_modal_projector: LlamaGuard4ForCausalLM#multi_modal_projector.
  LlamaGuard4ForCausalLM._print_model_architecture: LlamaGuard4ForCausalLM#_print_model_architecture().
  LlamaGuard4ForCausalLM.hidden_size: LlamaGuard4ForCausalLM#hidden_size.
  LlamaGuard4ForCausalLM.dtype: LlamaGuard4ForCausalLM#dtype.
  LlamaGuard4ForCausalLM.rng: LlamaGuard4ForCausalLM#rng.
  LlamaGuard4ForCausalLM.embed_input_ids: LlamaGuard4ForCausalLM#embed_input_ids().
  LlamaGuard4ForCausalLM.vision_config: LlamaGuard4ForCausalLM#vision_config.
  logger: logger.
  LlamaGuard4ForCausalLM.is_first_rank: LlamaGuard4ForCausalLM#is_first_rank.
  LlamaGuard4ForCausalLM.vllm_config: LlamaGuard4ForCausalLM#vllm_config.
  LlamaGuard4ForCausalLM.rope_scaling: LlamaGuard4ForCausalLM#rope_scaling.
  LlamaGuard4ForCausalLM.model_config: LlamaGuard4ForCausalLM#model_config.
  LlamaGuard4ForCausalLM.projector_config_dict: LlamaGuard4ForCausalLM#projector_config_dict.
  LlamaGuard4ForCausalLM.use_qk_norm: LlamaGuard4ForCausalLM#use_qk_norm.
  LlamaGuard4ForCausalLM._compute_vision_features_jit: LlamaGuard4ForCausalLM#_compute_vision_features_jit().
  LlamaGuard4ForCausalLM.layers: LlamaGuard4ForCausalLM#layers.
  LlamaGuard4ForCausalLM.precompile_vision_encoder: LlamaGuard4ForCausalLM#precompile_vision_encoder().
  LlamaGuard4ForCausalLM.compute_logits: LlamaGuard4ForCausalLM#compute_logits().
  LlamaGuard4ForCausalLM.embed_multimodal: LlamaGuard4ForCausalLM#embed_multimodal().
  LlamaGuard4ForCausalLM.num_layers: LlamaGuard4ForCausalLM#num_layers.
  LlamaGuard4ForCausalLM.num_attention_heads: LlamaGuard4ForCausalLM#num_attention_heads.
  LlamaGuard4ForCausalLM.num_key_value_heads: LlamaGuard4ForCausalLM#num_key_value_heads.
  LlamaGuard4ForCausalLM.head_dim: LlamaGuard4ForCausalLM#head_dim.
  LlamaGuard4WeightLoader: LlamaGuard4WeightLoader#
  LlamaGuard4WeightLoader.map_loaded_to_standardized_name: LlamaGuard4WeightLoader#map_loaded_to_standardized_name().
  LlamaGuard4ForCausalLM.is_verbose: LlamaGuard4ForCausalLM#is_verbose.
  LlamaGuard4ForCausalLM.rope_theta_text: LlamaGuard4ForCausalLM#rope_theta_text.
  LlamaGuard4ForCausalLM.image_token_id: LlamaGuard4ForCausalLM#image_token_id.
  LlamaGuard4ForCausalLM.mesh: LlamaGuard4ForCausalLM#mesh.
  LlamaGuard4ForCausalLM.start_layer: LlamaGuard4ForCausalLM#start_layer.
  LlamaGuard4WeightLoader.__init__: LlamaGuard4WeightLoader#__init__().
  LlamaGuard4WeightLoader.is_verbose: LlamaGuard4WeightLoader#is_verbose.
  LlamaGuard4WeightLoader._loaded_to_standardized_keys: LlamaGuard4WeightLoader#_loaded_to_standardized_keys.
  LlamaGuard4ForCausalLM: LlamaGuard4ForCausalLM#
  LlamaGuard4WeightLoader.is_runai_streamer: LlamaGuard4WeightLoader#is_runai_streamer.
  LlamaGuard4WeightLoader._language_transpose_map: LlamaGuard4WeightLoader#_language_transpose_map.
  LlamaGuard4WeightLoader._vision_transpose_map: LlamaGuard4WeightLoader#_vision_transpose_map.
  LlamaGuard4WeightLoader._projector_transpose_map: LlamaGuard4WeightLoader#_projector_transpose_map.
  LlamaGuard4WeightLoader._weight_shape_map: LlamaGuard4WeightLoader#_weight_shape_map.
  LlamaGuard4WeightLoader._vision_weight_shape_map: LlamaGuard4WeightLoader#_vision_weight_shape_map.
  LlamaGuard4WeightLoader.load_weights: LlamaGuard4WeightLoader#load_weights().
  LlamaGuard4ForCausalLM.__init__: LlamaGuard4ForCausalLM#__init__().
---
# Module: [`tpu_inference/models/jax/llama_guard_4.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py)

## Classes
### `LlamaGuard4ForCausalLM`  ·  implements/extends Module
- def: [`tpu_inference/models/jax/llama_guard_4.py:331`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L331)
- signature: `class LlamaGuard4ForCausalLM(nnx.Module):`
- members:
  - `compute_logits(self, hidden_states: jax.Array)` — [`L628`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L628)
  - `embed_input_ids(self, input_ids: jax.Array, multimodal_embeddings: jax.Array | None = None, *, is_multimodal: jax.Array | None = None)` — [`L633`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L633)
  - `embed_multimodal(self, **kwargs)` — [`L655`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L655)
  - `load_weights(self, rng: jax.Array, cache_dir: Optional[str] = None)` — [`L576`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L576)
  - `precompile_vision_encoder(self, run_compilation_fn: Callable)` — [`L557`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L557)
  - `dtype` — [`L360`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L360)
  - `embedder` — [`L389`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L389)
  - `end_layer` — [`L426`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L426)
  - `final_norm` — [`L518`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L518)
  - `head_dim` — [`L370`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L370)
  - `hidden_size` — [`L358`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L358)
  - `image_token_id` — [`L379`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L379)
  - `is_first_rank` — [`L384`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L384)
  - `is_last_rank` — [`L385`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L385)
  - `is_verbose` — [`L352`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L352)
  - `layers` — [`L425`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L425)
  - `lm_head` — [`L527`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L527)
  - `mesh` — [`L351`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L351)
  - `model_config` — [`L342`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L342)
  - `multi_modal_projector` — [`L418`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L418)
  - `num_attention_heads` — [`L366`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L366)
  - `num_key_value_heads` — [`L368`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L368)
  - `num_layers` — [`L362`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L362)
  - `projector_config_dict` — [`L346`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L346)
  - `rng` — [`L382`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L382)
  - `rope_scaling` — [`L377`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L377)
  - `rope_theta_text` — [`L375`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L375)
  - `start_layer` — [`L426`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L426)
  - `text_config` — [`L343`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L343)
  - `use_qk_norm` — [`L355`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L355)
  - `vision_config` — [`L344`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L344)
  - `vision_model` — [`L409`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L409)
  - `vision_rope` — [`L400`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L400)
  - `vllm_config` — [`L340`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L340)
- protocol/private: `__call__`[`L590`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L590), `__init__`[`L333`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L333), `_compute_vision_features_jit`[`L552`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L552), `_print_model_architecture`[`L540`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L540)
- uses (calls/refs, reference-scoped): [`AttentionMetadata`](../../layers/common/attention_metadata.md#AttentionMetadata), [`JaxIntermediateTensors`](jax_intermediate_tensor.md#JaxIntermediateTensors), [`PPMissingLayer`](../../layers/jax/pp_utils.md#PPMissingLayer), [`get_pp_group`](../../distributed/jax_parallel_state.md#get_pp_group), [`encode`](../../layers/jax/layers.md#Embedder.encode), [`is_first_rank`](../../distributed/jax_parallel_state.md#GroupCoordinator.is_first_rank), [`is_last_rank`](../../distributed/jax_parallel_state.md#GroupCoordinator.is_last_rank), [`__post_init__`](../../layers/jax/rope.md#Llama4VisionRotaryEmbedding.__post_init__), [`RMSNorm`](../../layers/jax/layers.md#RMSNorm), [`dtype`](../../layers/jax/layers.md#RMSNorm.dtype), [`activation_ffw_td`](../../layers/jax/layers.md#RMSNorm.activation_ffw_td), [`dtype`](../../layers/jax/layers.md#Embedder.dtype), [`dims`](../../layers/jax/layers.md#RMSNorm.dims), [`epsilon`](../../layers/jax/layers.md#RMSNorm.epsilon), [`random_init`](../../layers/jax/layers.md#RMSNorm.random_init), [`Embedder`](../../layers/jax/layers.md#Embedder), [`keyvalue_skh`](../../layers/jax/attention/attention.md#Attention.keyvalue_skh), [`DenseFFW`](../../layers/jax/layers.md#DenseFFW), [`dtype`](../../layers/jax/attention/attention.md#Attention.dtype), [`hidden_size`](../../layers/jax/layers.md#Embedder.hidden_size), [`rngs`](../../layers/jax/layers.md#RMSNorm.rngs), [`LMhead`](../../layers/jax/layers.md#LMhead), [`merge_multimodal_embeddings`](utils/multi_modal_utils.md#merge_multimodal_embeddings), [`random_init`](../../layers/jax/layers.md#Embedder.random_init), [`vocab_size`](../../layers/jax/layers.md#Embedder.vocab_size), [`TransformerBlock`](../../layers/jax/transformer_block.md#TransformerBlock), [`logger`](llama_guard_4.md#logger), [`query_tnh`](../../layers/jax/attention/attention.md#Attention.query_tnh), [`custom_module`](../../layers/jax/transformer_block.md#TransformerBlock.custom_module), [`dtype`](../../layers/jax/layers.md#DenseFFW.dtype), [`mesh`](../../layers/jax/attention/attention.md#Attention.mesh), [`Llama4Attention`](../../layers/jax/attention/llama4_attention.md#Llama4Attention), [`activation_q_td`](../../layers/jax/attention/attention.md#Attention.activation_q_td), [`KVCacheType`](../../layers/jax/constants.md#KVCacheType.KVCacheType), [`random_init`](../../layers/jax/attention/attention.md#Attention.random_init), [`random_init`](../../layers/jax/layers.md#DenseFFW.random_init), [`rank_in_group`](../../distributed/jax_parallel_state.md#GroupCoordinator.rank_in_group), [`rngs`](../../layers/jax/layers.md#Embedder.rngs), [`rope_input_ordering`](../../layers/jax/attention/attention.md#Attention.rope_input_ordering), [`vd_sharding`](../../layers/jax/layers.md#Embedder.vd_sharding)  (+47 more)
- used by: [`_get_model_architecture`](../common/model_loader.md#_get_model_architecture)

### `LlamaGuard4WeightLoader`  ·  implements/extends BaseWeightLoader
- def: [`tpu_inference/models/jax/llama_guard_4.py:58`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L58)
- signature: `class LlamaGuard4WeightLoader(BaseWeightLoader):`
- members:
  - `load_weights(self, model_for_loading: nnx.Module)` — [`L246`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L246)
  - `map_loaded_to_standardized_name(self, loaded_key: str)` — [`L221`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L221)
  - `is_runai_streamer` — [`L69`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L69)
  - `is_verbose` — [`L72`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L72)
  - `pp_missing_layers` — [`L249`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L249)
- protocol/private: `__init__`[`L60`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L60), `_language_transpose_map`[`L76`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L76), `_loaded_to_standardized_keys`[`L129`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L129), `_projector_transpose_map`[`L102`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L102), `_vision_transpose_map`[`L88`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L88), `_vision_weight_shape_map`[`L119`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L119), `_weight_shape_map`[`L107`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L107)
- uses (calls/refs, reference-scoped): [`PPMissingLayer`](../../layers/jax/pp_utils.md#PPMissingLayer), [`BaseWeightLoader`](utils/weight_utils.md#BaseWeightLoader), [`get_param`](utils/weight_utils.md#get_param), [`print_param_info`](utils/weight_utils.md#print_param_info), [`_is_pp_missing_layer`](utils/weight_utils.md#_is_pp_missing_layer), [`get_weights_iterator`](utils/weight_utils.md#BaseWeightLoader.get_weights_iterator), [`shard_put`](../../layers/jax/misc.md#shard_put), [`transpose_params`](utils/weight_utils.md#transpose_params), [`__init__`](utils/weight_utils.md#BaseWeightLoader.__init__), [`reshape_params`](utils/weight_utils.md#reshape_params)
- used by: [`load_weights`](llama_guard_4.md#LlamaGuard4ForCausalLM.load_weights), [`BaseWeightLoader`](utils/weight_utils.md#BaseWeightLoader)

## Module values
- `logger` — [`L55`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama_guard_4.py#L55)


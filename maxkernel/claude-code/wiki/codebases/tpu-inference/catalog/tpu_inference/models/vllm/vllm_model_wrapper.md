---
title: 'Module: tpu_inference/models/vllm/vllm_model_wrapper.py'
type: catalog
provenance: extracted
module: tpu_inference/models/vllm/vllm_model_wrapper.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.models.vllm.vllm_model_wrapper`/
symbols:
  VllmModelWrapper.vllm_config: VllmModelWrapper#vllm_config.
  VllmModelWrapper._pooler: VllmModelWrapper#_pooler.
  VllmModelWrapper.step_fun_impl: VllmModelWrapper#step_fun_impl().
  VllmModelWrapper.load_weights: VllmModelWrapper#load_weights().
  VllmModelWrapper.model: VllmModelWrapper#model.
  VllmModelWrapper.draft_step_fun_impl: VllmModelWrapper#draft_step_fun_impl().
  VllmModelWrapper.embed_multimodal_func_torch: VllmModelWrapper#embed_multimodal_func_torch().
  VllmModelWrapper.wrap_precompile_vision_encoder_fn: VllmModelWrapper#wrap_precompile_vision_encoder_fn().
  VllmModelWrapper._mm_encoder_jit_manager: VllmModelWrapper#_mm_encoder_jit_manager.
  VllmModelWrapper.jit_step_func: VllmModelWrapper#jit_step_func().
  VllmModelWrapper.compute_logits_func: VllmModelWrapper#compute_logits_func().
  logger: logger.
  _VllmRunner.vllm_model: _VllmRunner#vllm_model.
  VllmModelWrapper.mesh: VllmModelWrapper#mesh.
  VllmModelWrapper.combine_hidden_states_func: VllmModelWrapper#combine_hidden_states_func().
  _get_sc_allreduce_allgather_offload_min_size_bytes: _get_sc_allreduce_allgather_offload_min_size_bytes().
  load_lora_model: load_lora_model().
  _VllmRunner.forward: _VllmRunner#forward().
  VllmModelWrapper.wrap_embed_multimodal_func: VllmModelWrapper#wrap_embed_multimodal_func().
  VllmModelWrapper.wrap_embed_input_ids_func: VllmModelWrapper#wrap_embed_input_ids_func().
  VllmModelWrapper.build_pooler_func: VllmModelWrapper#build_pooler_func().
  replace_set_lora: replace_set_lora().
  _VllmRunner: _VllmRunner#
  VllmModelWrapper.is_draft_model: VllmModelWrapper#is_draft_model.
  _maybe_patch_for_deepseek_v4: _maybe_patch_for_deepseek_v4().
  _disable_ds_v4_mtp_buffer: _disable_ds_v4_mtp_buffer().
  _VllmRunner.compute_hidden_state: _VllmRunner#compute_hidden_state().
  _VllmRunner.compute_logits: _VllmRunner#compute_logits().
  VllmModelWrapper._apply_pp_patch: VllmModelWrapper#_apply_pp_patch().
  VllmModelWrapper.step_fn_no_options: VllmModelWrapper#step_fn_no_options.
  VllmModelWrapper.embed_multimodal_func_jax: VllmModelWrapper#embed_multimodal_func_jax().
  VllmModelWrapper.move: VllmModelWrapper#move().
  VllmModelWrapper.embed_input_ids_func: VllmModelWrapper#embed_input_ids_func().
  VllmModelWrapper.jit_compute_logits_func: VllmModelWrapper#jit_compute_logits_func().
  VllmModelWrapper.jit_combine_hidden_states_func: VllmModelWrapper#jit_combine_hidden_states_func().
  VllmModelWrapper.compute_pooler_output: VllmModelWrapper#compute_pooler_output().
  VllmModelWrapper: VllmModelWrapper#
  _disable_ds_v4_mtp_buffer._NoOpBuffer: _disable_ds_v4_mtp_buffer()._NoOpBuffer#
  _VllmRunner.pooler: _VllmRunner#pooler.
  VllmModelWrapper.rng: VllmModelWrapper#rng.
  replace_set_lora._tpu_set_lora: replace_set_lora()._tpu_set_lora().
  replace_set_lora._tpu_reset_lora: replace_set_lora()._tpu_reset_lora().
  _disable_ds_v4_mtp_buffer._NoOpBuffer.__getitem__: _disable_ds_v4_mtp_buffer()._NoOpBuffer#__getitem__().
  _disable_ds_v4_mtp_buffer._NoOpBuffer.copy_: _disable_ds_v4_mtp_buffer()._NoOpBuffer#copy_().
  _VllmRunner.__init__: _VllmRunner#__init__().
  VllmModelWrapper.__init__: VllmModelWrapper#__init__().
---
# Module: [`tpu_inference/models/vllm/vllm_model_wrapper.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper.py)

## Classes
### `VllmModelWrapper`
- def: [`tpu_inference/models/vllm/vllm_model_wrapper.py:200`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper.py#L200)
- doc: Wraps a vLLM Pytorch model and let it run on the JAX engine.
- signature: `class VllmModelWrapper:`
- members:
  - `build_pooler_func(self)` — [`L739`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper.py#L739)
  - `combine_hidden_states_func(params_and_buffers: Any, hidden_states: jax.Array)` — [`L720`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper.py#L720)
  - `compute_logits_func(params_and_buffers: Any, hidden_states: jax.Array, lora_metadata)` — [`L690`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper.py#L690)
  - `compute_pooler_output(hidden_states: jax.Array, pooling_metadata: PoolingMetadata, seq_lens: np.ndarray, num_scheduled_tokens: np.ndarray | None = None)` — [`L741`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper.py#L741)
  - `draft_step_fun_impl(params_and_buffers, kv_caches: List[jax.Array], input_ids: jax.Array, hidden_states: jax.Array, attn_metadata: AttentionMetadata, layer_name_to_kvcache_index: Sequence[Tuple[str, int]], spec_step_idx: int = 0)` — [`L476`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper.py#L476)
  - `embed_input_ids_func(params_and_buffers: Any, input_ids: jax.Array, mm_embeds: list[jax.Array] | jax.Array | None = None, *, is_multimodal: jax.Array | None = None)` — [`L648`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper.py#L648)
  - `embed_multimodal_func_jax(params_and_buffers: Any, **kwargs)` — [`L585`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper.py#L585)
  - `embed_multimodal_func_torch(params_and_buffers: Any, modality: str | None = None, **kwargs)` — [`L607`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper.py#L607)
  - `jit_combine_hidden_states_func(self)` — [`L714`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper.py#L714)
  - `jit_compute_logits_func(self)` — [`L682`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper.py#L682)
  - `jit_step_func(self)` — [`L390`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper.py#L390)
  - `load_weights(self, shared_params: Optional[dict[str, jax.Array]] = None)` — [`L237`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper.py#L237)
  - `move(v: torch.Tensor)` — [`L624`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper.py#L624)
  - `step_fun_impl(params_and_buffers, kv_caches: List[jax.Array], input_ids: jax.Array, attn_metadata: AttentionMetadata, input_embeds: jax.Array, input_positions: jax.Array, layer_name_to_kvcache_index: Sequence[Tuple[str, int]], lora_metadata, intermediate_tensors: JaxIntermediateTensors = None, is_first_rank: bool = True, is_last_rank: bool = True, *args)` — [`L409`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper.py#L409)
  - `wrap_embed_input_ids_func(self)` — [`L643`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper.py#L643)
  - `wrap_embed_multimodal_func(self)` — [`L581`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper.py#L581)
  - `wrap_precompile_vision_encoder_fn(self, params: Any)` — [`L562`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper.py#L562) — Return a precompile function for the vision encoder, or None.
  - `is_draft_model` — [`L215`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper.py#L215)
  - `mesh` — [`L204`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper.py#L204)
  - `model` — [`L205`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper.py#L205)
  - `rng` — [`L203`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper.py#L203)
  - `step_fn_no_options` — [`L556`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper.py#L556)
  - `vllm_config` — [`L212`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper.py#L212)
- protocol/private: `__init__`[`L207`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper.py#L207), `_apply_pp_patch`[`L222`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper.py#L222), `_mm_encoder_jit_manager`[`L216`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper.py#L216), `_pooler`[`L355`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper.py#L355)
- uses (calls/refs, reference-scoped): [`ShardingAxisName`](../../layers/common/sharding.md#ShardingAxisName), [`AttentionMetadata`](../../layers/common/attention_metadata.md#AttentionMetadata), [`JaxIntermediateTensors`](../jax/jax_intermediate_tensor.md#JaxIntermediateTensors), [`get_pp_group`](../../distributed/jax_parallel_state.md#get_pp_group), [`get_tpu_quantization_config`](../../layers/vllm/quantization/__init__.md#get_tpu_quantization_config), [`input_positions`](../../layers/common/attention_metadata.md#AttentionMetadata.input_positions), [`set_vllm_model_wrapper_context`](vllm_model_wrapper_context.md#set_vllm_model_wrapper_context), [`patch_mm_model`](experimental/model_patcher.md#patch_mm_model), [`get_vllm_model_wrapper_context`](vllm_model_wrapper_context.md#get_vllm_model_wrapper_context), [`logger`](vllm_model_wrapper.md#logger), [`shard_model_to_tpu`](../../layers/vllm/process_weights/cleanup_sharding.md#shard_model_to_tpu), [`vllm_model`](vllm_model_wrapper.md#_VllmRunner.vllm_model), [`maybe_precompile_vision_encoder_fn`](experimental/vision_tower_jit.md#maybe_precompile_vision_encoder_fn), [`kv_caches`](vllm_model_wrapper_context.md#VllmModelWrapperContext.kv_caches), [`_get_sc_allreduce_allgather_offload_min_size_bytes`](vllm_model_wrapper.md#_get_sc_allreduce_allgather_offload_min_size_bytes), [`load_lora_model`](vllm_model_wrapper.md#load_lora_model), [`apply_model_specific_patches`](experimental/model_patcher.md#apply_model_specific_patches), [`maybe_prepare_for_jit`](experimental/vision_tower_jit.md#maybe_prepare_for_jit), [`maybe_create_mm_encoder_jit_manager`](../../runner/mm_encoder_jit_manager.md#maybe_create_mm_encoder_jit_manager), [`replace_set_lora`](vllm_model_wrapper.md#replace_set_lora), [`_VllmRunner`](vllm_model_wrapper.md#_VllmRunner), [`replace_lora_metadata`](../../runner/lora_utils.md#replace_lora_metadata), [`maybe_jit_embed_multimodal_func`](experimental/vision_tower_jit.md#maybe_jit_embed_multimodal_func), [`precompile_vision_encoder`](../../runner/mm_encoder_jit_manager.md#MMEncoderJITManager.precompile_vision_encoder), [`MMEncoderJITManager`](../../runner/mm_encoder_jit_manager.md#MMEncoderJITManager), [`_disable_ds_v4_mtp_buffer`](vllm_model_wrapper.md#_disable_ds_v4_mtp_buffer), [`_maybe_patch_for_deepseek_v4`](vllm_model_wrapper.md#_maybe_patch_for_deepseek_v4), [`to_torch`](../jax/jax_intermediate_tensor.md#JaxIntermediateTensors.to_torch), [`PoolerFunc`](../common/interface.md#PoolerFunc), [`execute`](../../runner/mm_encoder_jit_manager.md#MMEncoderJITManager.execute), [`JITTED_MM_MODULE_KEYS`](../../envs.md#JITTED_MM_MODULE_KEYS), [`REGISTER_MM_MODULE_CUSTOM_PYTREE_CLASSES`](../../envs.md#REGISTER_MM_MODULE_CUSTOM_PYTREE_CLASSES), [`from_torch`](../jax/jax_intermediate_tensor.md#JaxIntermediateTensors.from_torch), [`pooler`](vllm_model_wrapper.md#_VllmRunner.pooler)
- used by: [`get_vllm_model`](../common/model_loader.md#get_vllm_model)

### `_NoOpBuffer`
- def: [`tpu_inference/models/vllm/vllm_model_wrapper.py:148`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper.py#L148)
- doc: Sentinel that absorbs `buf[:n].copy_(x)` as a no-op.
- signature: `class _NoOpBuffer:`
- members:
  - `copy_(self, *args, **kwargs)` — [`L155`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper.py#L155)
- protocol/private: `__getitem__`[`L152`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper.py#L152)
- used by: [`_disable_ds_v4_mtp_buffer`](vllm_model_wrapper.md#_disable_ds_v4_mtp_buffer)

### `_VllmRunner`  ·  implements/extends Module
- def: [`tpu_inference/models/vllm/vllm_model_wrapper.py:173`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper.py#L173)
- signature: `class _VllmRunner(torch.nn.Module):`
- members:
  - `compute_hidden_state(self, kwargs: dict)` — [`L193`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper.py#L193)
  - `compute_logits(self, hidden_state: torch.Tensor)` — [`L196`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper.py#L196)
  - `forward(self, **kwargs)` — [`L181`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper.py#L181)
  - `pooler` — [`L179`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper.py#L179)
  - `vllm_model` — [`L177`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper.py#L177)
- protocol/private: `__init__`[`L175`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper.py#L175)
- used by: [`get_vllm_model`](../common/model_loader.md#get_vllm_model), [`_pooler`](vllm_model_wrapper.md#VllmModelWrapper._pooler), [`load_weights`](vllm_model_wrapper.md#VllmModelWrapper.load_weights), [`model`](vllm_model_wrapper.md#VllmModelWrapper.model), [`embed_multimodal_func_torch`](vllm_model_wrapper.md#VllmModelWrapper.embed_multimodal_func_torch), [`patch_mm_model`](experimental/model_patcher.md#patch_mm_model), [`wrap_precompile_vision_encoder_fn`](vllm_model_wrapper.md#VllmModelWrapper.wrap_precompile_vision_encoder_fn)

## Functions
- `_disable_ds_v4_mtp_buffer(vllm_config: VllmConfig, vllm_model: torch.nn.Module)` — [`L145`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper.py#L145)
- `_get_sc_allreduce_allgather_offload_min_size_bytes()` — [`L75`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper.py#L75) — Returns the SparseCore all-reduce/all-gather offload minimum size in bytes.
- `_maybe_patch_for_deepseek_v4(vllm_config: VllmConfig)` — [`L97`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper.py#L97)
- `_tpu_reset_lora(self, index: int)` — [`L817`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper.py#L817)
- `_tpu_set_lora(self, index: int, lora_a: torch.Tensor, lora_b: torch.Tensor)` — [`L807`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper.py#L807)
- `load_lora_model(model: torch.nn.Module, vllm_config: VllmConfig, device: str)` — [`L774`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper.py#L774)
- `replace_set_lora(model)` — [`L805`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper.py#L805)

## Module values
- `logger` — [`L72`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper.py#L72)


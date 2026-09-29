---
title: 'Module: easydel/inference/esurge/runners/executors/model_executor.py'
type: catalog
provenance: extracted
module: easydel/inference/esurge/runners/executors/model_executor.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.esurge.runners.executors.model_executor`/ModelStepExecutor#
symbols:
  ModelStepExecutor._backbone_step: _backbone_step().
  ModelStepExecutor.compile_backbone: compile_backbone().
  ModelStepExecutor.compile_lm_head: compile_lm_head().
  ModelStepExecutor._build_backbone_fn: _build_backbone_fn().
  ModelStepExecutor._lm_head_step: _lm_head_step().
  ModelStepExecutor.__init__: __init__().
  ModelStepExecutor.get_compiled: get_compiled().
  ModelStepExecutor.compile: compile().
  ModelStepExecutor._empty_sharding: _empty_sharding.
  ModelStepExecutor.use_aot_forward: use_aot_forward.
  ModelStepExecutor._backbone_cache: _backbone_cache.
  ModelStepExecutor._lm_head_cache: _lm_head_cache.
  ModelStepExecutor.clear_cache: clear_cache().
  ModelStepExecutor.has: has().
  ModelStepExecutor.get_backbone: get_backbone().
  ModelStepExecutor.get_lm_head: get_lm_head().
  ModelStepExecutor._cache_store: _cache_store().
  ModelStepExecutor.cache_keys: cache_keys().
  ModelStepExecutor.has_backbone: has_backbone().
  ModelStepExecutor.has_lm_head: has_lm_head().
  ModelStepExecutor.wrapped_backbone: wrapped_backbone().
  ModelStepExecutor.wrapped_lm_head: wrapped_lm_head().
  ModelStepExecutor.graphdef: graphdef.
  ModelStepExecutor.mesh: mesh.
  ModelStepExecutor.model: model.
  ModelStepExecutor._backbone_fn: _backbone_fn.
  ModelStepExecutor._lm_head_fn: _lm_head_fn.
  ModelStepExecutor._cache_lookup: _cache_lookup().
  ModelStepExecutor._cache_put: _cache_put().
  ModelStepExecutor._cache_get: _cache_get().
  ModelStepExecutor._metadata_version: _metadata_version.
  ModelStepExecutor._use_slot_mapping: _use_slot_mapping.
  ModelStepExecutor._combined: _combined().
  ModelStepExecutor._bound_backbone: _bound_backbone().
  ModelStepExecutor._bound_lm_head: _bound_lm_head().
  ModelStepExecutor._build_lm_head_fn: _build_lm_head_fn().
  ModelStepExecutor: ''
  ModelStepExecutor.metadata: metadata.
  ModelStepExecutor._cache: _cache.
  ModelStepExecutor.bind_graphstate_for_aot: bind_graphstate_for_aot.
  ModelStepExecutor.max_num_reqs: max_num_reqs.
  ModelStepExecutor._wrapped_bound_backbone: _wrapped_bound_backbone().
  ModelStepExecutor._wrapped_bound_lm_head: _wrapped_bound_lm_head().
---
# Module: [`easydel/inference/esurge/runners/executors/model_executor.py`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/model_executor.py)

## Classes
### `ModelStepExecutor`
- def: [`easydel/inference/esurge/runners/executors/model_executor.py:94`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/model_executor.py#L94)
- doc: Compile, cache, and execute the model forward step.
- signature: `class ModelStepExecutor:`
- members:
  - `__init__(self, *, model: EasyDeLBaseModule, mesh: tp.Any, metadata: RaggedPagesCacheConfig | UnifiedAttentionCacheConfig, kv_pages_template: HybridCache | RaggedPagesCache | UnifiedAttentionCache, graphstate_template: tp.Any, graphother_template: tp.Any, max_num_reqs: int, graphdef: tp.Any, empty_sharding: jax.sharding.Sharding, use_aot_forward: bool, bind_graphstate_for_aot: bool = False, cache_capacity: int = 64)` — [`L135`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/model_executor.py#L135) — Initialize the ModelStepExecutor.
  - `_build_backbone_fn(self, *, kv_pages_template: HybridCache | RaggedPagesCache | UnifiedAttentionCache, graphstate_template: tp.Any, graphother_template: tp.Any)` — [`L409`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/model_executor.py#L409) — Build the JIT-compiled backbone function (forward pass without lm_head).
  - `_build_lm_head_fn(self, *, graphstate_template: tp.Any, graphother_template: tp.Any)` — [`L544`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/model_executor.py#L544) — Build the JIT-compiled lm_head function.
  - `cache_keys(self)` — [`L223`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/model_executor.py#L223) — Get all keys currently in backbone + lm_head caches.
  - `clear_cache(self)` — [`L200`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/model_executor.py#L200) — Clear all cached compiled functions.
  - `compile(self, *, num_tokens: int, padded_num_reqs: int, graphdef: tp.Any, graphstate: tp.Any, graphother: tp.Any, inputs: StepFunctionInputs)` — [`L277`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/model_executor.py#L277) — Compile backbone and lm_head for the given dimensions.
  - `compile_backbone(self, *, num_tokens: int, graphdef: tp.Any, graphstate: tp.Any, graphother: tp.Any, inputs: StepFunctionInputs)` — [`L309`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/model_executor.py#L309) — Compile the backbone (transformer forward) for a token bucket.
  - `compile_lm_head(self, *, padded_num_reqs: int, graphdef: tp.Any, graphstate: tp.Any, graphother: tp.Any, inputs: StepFunctionInputs, hidden_dim: int | None = None, dtype: tp.Any = None)` — [`L355`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/model_executor.py#L355) — Compile the lm_head for a request bucket.
  - `get_backbone(self, *, num_tokens: int)` — [`L267`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/model_executor.py#L267) — Retrieve a pre-compiled backbone function.
  - `get_compiled(self, *, num_tokens: int, padded_num_reqs: int)` — [`L242`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/model_executor.py#L242) — Retrieve pre-compiled backbone + lm_head as a combined callable.
  - `get_lm_head(self, *, padded_num_reqs: int)` — [`L272`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/model_executor.py#L272) — Retrieve a pre-compiled lm_head function.
  - `has(self, key: tuple[int, int, str, str])` — [`L227`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/model_executor.py#L227) — Check if a (num_tokens, padded_num_reqs) pair is fully compiled.
  - `has_backbone(self, num_tokens: int)` — [`L234`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/model_executor.py#L234)
  - `has_lm_head(self, padded_num_reqs: int)` — [`L238`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/model_executor.py#L238)
  - `wrapped_backbone(graphstate_, graphother_, kv_pages_, metadata_)` — [`L348`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/model_executor.py#L348)
  - `wrapped_lm_head(graphstate_, graphother_, hs_)` — [`L403`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/model_executor.py#L403)
  - `bind_graphstate_for_aot` — [`L181`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/model_executor.py#L181)
  - `graphdef` — [`L176`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/model_executor.py#L176)
  - `max_num_reqs` — [`L175`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/model_executor.py#L175)
  - `mesh` — [`L173`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/model_executor.py#L173)
  - `metadata` — [`L174`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/model_executor.py#L174)
  - `model` — [`L172`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/model_executor.py#L172)
  - `use_aot_forward` — [`L180`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/model_executor.py#L180)
- protocol/private: `_backbone_cache`[`L194`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/model_executor.py#L194), `_backbone_fn`[`L184`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/model_executor.py#L184), `_backbone_step`[`L464`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/model_executor.py#L464), `_bound_backbone`[`L331`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/model_executor.py#L331), `_bound_lm_head`[`L386`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/model_executor.py#L386), `_cache`[`L198`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/model_executor.py#L198), `_cache_get`[`L220`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/model_executor.py#L220), `_cache_lookup`[`L212`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/model_executor.py#L212), `_cache_put`[`L217`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/model_executor.py#L217), `_cache_store`[`L207`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/model_executor.py#L207), `_combined`[`L253`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/model_executor.py#L253), `_empty_sharding`[`L179`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/model_executor.py#L179), `_lm_head_cache`[`L196`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/model_executor.py#L196), `_lm_head_fn`[`L189`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/model_executor.py#L189), `_lm_head_step`[`L578`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/model_executor.py#L578), `_metadata_version`[`L177`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/model_executor.py#L177), `_use_slot_mapping`[`L178`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/model_executor.py#L178), `_wrapped_bound_backbone`[`L336`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/model_executor.py#L336), `_wrapped_bound_lm_head`[`L391`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/model_executor.py#L391)
- uses (calls/refs, reference-scoped): [`config`](../../../../infra/base_module.md#EasyDeLBaseModule.config), [`EasyDeLBaseModule`](../../../../infra/base_module.md#EasyDeLBaseModule), [`RaggedPagesMetadata`](../../../../caching/ragged_page/cache.md#RaggedPagesMetadata), [`HybridCache`](../../../../caching/hybrid/cache.md#HybridCache), [`RaggedPagesCache`](../../../../caching/ragged_page/cache.md#RaggedPagesCache), [`RaggedPagesCacheConfig`](../../../../caching/ragged_page/cache.md#RaggedPagesCacheConfig), [`mesh`](../../../../infra/base_module.md#EasyDeLBaseModule.mesh), [`compute_embedding`](../../../../infra/base_module.md#EasyDeLBaseModule.compute_embedding), [`UnifiedAttentionCacheConfig`](../../../../caching/unified_attention/cache.md#UnifiedAttentionCacheConfig), [`dtype`](../../../../infra/base_module.md#EasyDeLBaseModule.dtype), [`batch_metadata`](../execution_types.md#StepFunctionInputs.batch_metadata), [`UnifiedAttentionCache`](../../../../caching/unified_attention/cache.md#UnifiedAttentionCache), [`apply_lm_head`](../../../../infra/base_module.md#EasyDeLBaseModule.apply_lm_head), [`kv_pages`](../execution_types.md#StepFunctionInputs.kv_pages), [`get_max_num_seqs`](../../../../caching/ragged_page/cache.md#RaggedPagesCacheConfig.get_max_num_seqs), [`BatchMetadata`](../execution_types.md#BatchMetadata), [`page_size`](../../../../caching/ragged_page/cache.md#RaggedPagesCacheConfig.page_size), [`set_inference_mode`](../../../../utils/inference_mode.md#set_inference_mode), [`StepFunctionInputs`](../execution_types.md#StepFunctionInputs), [`version`](../../../../caching/ragged_page/cache.md#RaggedPagesCacheConfig.version), [`BackboneOutputs`](../execution_types.md#BackboneOutputs), [`num_slices_per_kv_cache_update_page`](../../../../caching/ragged_page/cache.md#RaggedPagesCacheConfig.num_slices_per_kv_cache_update_page), [`ModelStepOutputs`](../execution_types.md#ModelStepOutputs), [`num_requests`](../execution_types.md#BatchMetadata.num_requests), [`query_start_loc`](../execution_types.md#BatchMetadata.query_start_loc), [`request_distribution`](../execution_types.md#BatchMetadata.request_distribution), [`seq_lens`](../execution_types.md#BatchMetadata.seq_lens), [`pixel_values`](../execution_types.md#BatchMetadata.pixel_values), [`pixel_values_videos`](../execution_types.md#BatchMetadata.pixel_values_videos), [`image_grid_thw`](../execution_types.md#BatchMetadata.image_grid_thw), [`input_ids_buf`](../execution_types.md#BatchMetadata.input_ids_buf), [`num_kv_update_slices`](../execution_types.md#BatchMetadata.num_kv_update_slices), [`slot_mapping`](../execution_types.md#BatchMetadata.slot_mapping), [`video_grid_thw`](../execution_types.md#BatchMetadata.video_grid_thw), [`deepstack_visual_embeds`](../execution_types.md#BatchMetadata.deepstack_visual_embeds), [`mrope_position_ids`](../execution_types.md#BatchMetadata.mrope_position_ids), [`pages_tables`](../execution_types.md#BatchMetadata.pages_tables), [`position_ids_buf`](../execution_types.md#BatchMetadata.position_ids_buf), [`prefill_embeds`](../execution_types.md#BatchMetadata.prefill_embeds), [`prefill_embeds_mask`](../execution_types.md#BatchMetadata.prefill_embeds_mask)  (+1 more)
- used by: [`_model_executor`](../execution_manager.md#ExecutionManager._model_executor), [`update_graphs`](../execution_manager.md#ExecutionManager.update_graphs), [`_compile_backbone_variant`](../execution_manager.md#ExecutionManager._compile_backbone_variant), [`release_model_state`](../model_runner.md#eSurgeRunner.release_model_state), [`get_compiled_key`](../execution_manager.md#ExecutionManager.get_compiled_key), [`execute_model`](../execution_manager.md#ExecutionManager.execute_model), [`_compile_lm_head_variant`](../execution_manager.md#ExecutionManager._compile_lm_head_variant), [`clear_cache`](../execution_manager.md#ExecutionManager.clear_cache), [`has_compiled_variants`](../execution_manager.md#ExecutionManager.has_compiled_variants)


---
title: 'Module: easydel/inference/esurge/runners/execution_types.py'
type: catalog
provenance: extracted
module: easydel/inference/esurge/runners/execution_types.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.esurge.runners.execution_types`/
symbols:
  StepFunctionInputs.print_status: StepFunctionInputs#print_status().
  StepFunctionInputs.batch_metadata: StepFunctionInputs#batch_metadata.
  StepFunctionInputs.kv_pages: StepFunctionInputs#kv_pages.
  BatchMetadata: BatchMetadata#
  StepFunctionInputs: StepFunctionInputs#
  BackboneOutputs.kv_pages: BackboneOutputs#kv_pages.
  ModelStepOutputs.kv_pages: ModelStepOutputs#kv_pages.
  StepFunctionOutputs.kv_pages: StepFunctionOutputs#kv_pages.
  BatchMetadata.packed_f32_padded: BatchMetadata#packed_f32_padded.
  BackboneOutputs: BackboneOutputs#
  ModelStepOutputs: ModelStepOutputs#
  BatchMetadata.query_start_loc: BatchMetadata#query_start_loc().
  BatchMetadata.seq_lens: BatchMetadata#seq_lens().
  BatchMetadata.num_requests: BatchMetadata#num_requests().
  BatchMetadata.request_distribution: BatchMetadata#request_distribution().
  BatchMetadata.pixel_values: BatchMetadata#pixel_values.
  BatchMetadata.pixel_values_videos: BatchMetadata#pixel_values_videos.
  StepFunctionInputs.req_num_tokens_full: StepFunctionInputs#req_num_tokens_full.
  BatchMetadata.scheduled: BatchMetadata#scheduled().
  BatchMetadata.logits_indices: BatchMetadata#logits_indices().
  BatchMetadata.top_k: BatchMetadata#top_k().
  BatchMetadata.temperature: BatchMetadata#temperature().
  BatchMetadata.top_p: BatchMetadata#top_p().
  BatchMetadata.min_p: BatchMetadata#min_p().
  BatchMetadata.padded_num_reqs: BatchMetadata#padded_num_reqs().
  BatchMetadata.packed_i32_padded: BatchMetadata#packed_i32_padded.
  BatchMetadata.packed_misc_i32: BatchMetadata#packed_misc_i32.
  BatchMetadata.input_ids_buf: BatchMetadata#input_ids_buf.
  BatchMetadata.slot_mapping: BatchMetadata#slot_mapping.
  BatchMetadata.num_kv_update_slices: BatchMetadata#num_kv_update_slices.
  BatchMetadata.image_grid_thw: BatchMetadata#image_grid_thw.
  BatchMetadata.video_grid_thw: BatchMetadata#video_grid_thw.
  ModelStepOutputs.logits: ModelStepOutputs#logits.
  StepFunctionInputs.rng_key: StepFunctionInputs#rng_key.
  BatchMetadata.frequency_penalties: BatchMetadata#frequency_penalties().
  BatchMetadata.presence_penalties: BatchMetadata#presence_penalties().
  BatchMetadata.repetition_penalties: BatchMetadata#repetition_penalties().
  StepFunctionOutputs.device_state: StepFunctionOutputs#device_state.
  BatchMetadata.packed_qsl_seqlens: BatchMetadata#packed_qsl_seqlens.
  BatchMetadata.pages_tables: BatchMetadata#pages_tables.
  BatchMetadata.position_ids_buf: BatchMetadata#position_ids_buf.
  BatchMetadata.mrope_position_ids: BatchMetadata#mrope_position_ids.
  BatchMetadata.prefill_embeds: BatchMetadata#prefill_embeds.
  BatchMetadata.prefill_embeds_mask: BatchMetadata#prefill_embeds_mask.
  BatchMetadata.visual_pos_masks: BatchMetadata#visual_pos_masks.
  BatchMetadata.deepstack_visual_embeds: BatchMetadata#deepstack_visual_embeds.
  MinimalDeviceState: MinimalDeviceState#
  ModelStepOutputs.hidden_states: ModelStepOutputs#hidden_states.
  StepFunctionInputs.scheduled_full: StepFunctionInputs#scheduled_full.
  StepFunctionInputs.active_mask_full: StepFunctionInputs#active_mask_full.
  MinimalDeviceState.token_ids: MinimalDeviceState#token_ids.
  MinimalDeviceState.num_tokens: MinimalDeviceState#num_tokens.
  BatchMetadata.num_tokens: BatchMetadata#num_tokens.
  BackboneOutputs.hidden_states: BackboneOutputs#hidden_states.
  StepFunctionOutputs: StepFunctionOutputs#
  StepFunctionOutputs.input_ids_buf: StepFunctionOutputs#input_ids_buf.
  StepFunctionOutputs.position_ids_buf: StepFunctionOutputs#position_ids_buf.
  StepFunctionOutputs.query_start_loc: StepFunctionOutputs#query_start_loc.
  StepFunctionOutputs.seq_lens: StepFunctionOutputs#seq_lens.
  StepFunctionOutputs.pages_tables: StepFunctionOutputs#pages_tables.
  StepFunctionOutputs.rng_key: StepFunctionOutputs#rng_key.
  StepFunctionOutputs.out_tokens: StepFunctionOutputs#out_tokens.
  StepFunctionOutputs.valid_mask: StepFunctionOutputs#valid_mask.
  StepFunctionOutputs.hidden_states: StepFunctionOutputs#hidden_states.
  StepFunctionOutputs.logits: StepFunctionOutputs#logits.
---
# Module: [`easydel/inference/esurge/runners/execution_types.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py)

## Classes
### `BackboneOutputs`
- def: [`easydel/inference/esurge/runners/execution_types.py:321`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L321)
- doc: Outputs from the transformer backbone (forward pass without lm_head).
- signature: `class BackboneOutputs:`
- members:
  - `hidden_states` — [`L335`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L335)
  - `kv_pages` — [`L334`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L334)
- uses (calls/refs, reference-scoped): [`HybridCache`](../../../caching/hybrid/cache.md#HybridCache), [`RaggedPagesCache`](../../../caching/ragged_page/cache.md#RaggedPagesCache), [`UnifiedAttentionCache`](../../../caching/unified_attention/cache.md#UnifiedAttentionCache)
- used by: [`_backbone_step`](executors/model_executor.md#ModelStepExecutor._backbone_step), [`compile_backbone`](executors/model_executor.md#ModelStepExecutor.compile_backbone), [`_build_backbone_fn`](executors/model_executor.md#ModelStepExecutor._build_backbone_fn), [`_compile_backbone_variant`](execution_manager.md#ExecutionManager._compile_backbone_variant)

### `BatchMetadata`
- def: [`easydel/inference/esurge/runners/execution_types.py:116`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L116)
- doc: Precomputed tensors describing the current batch layout.
- signature: `class BatchMetadata:`
- members:
  - `frequency_penalties(self)` — [`L273`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L273) — Get frequency penalties for each request.
  - `logits_indices(self)` — [`L220`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L220) — Get indices for extracting logits from hidden states.
  - `min_p(self)` — [`L262`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L262) — Get min-p sampling parameters for each request.
  - `num_requests(self)` — [`L288`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L288) — Get the actual number of active requests (unpadded).
  - `padded_num_reqs(self)` — [`L298`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L298) — Get the padded number of requests for compilation efficiency.
  - `presence_penalties(self)` — [`L278`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L278) — Get presence penalties for each request.
  - `query_start_loc(self)` — [`L182`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L182) — Get cumulative query start locations for each request.
  - `repetition_penalties(self)` — [`L283`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L283) — Get repetition penalties for each request.
  - `request_distribution(self)` — [`L309`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L309) — Get request distribution for v3 attention kernel optimization.
  - `scheduled(self)` — [`L209`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L209) — Get number of tokens scheduled for each request in this step.
  - `seq_lens(self)` — [`L194`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L194) — Get current sequence lengths for each request.
  - `temperature(self)` — [`L241`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L241) — Get temperature sampling parameters for each request.
  - `top_k(self)` — [`L231`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L231) — Get top-k sampling parameters for each request.
  - `top_p(self)` — [`L251`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L251) — Get top-p (nucleus) sampling parameters for each request.
  - `deepstack_visual_embeds` — [`L179`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L179)
  - `image_grid_thw` — [`L165`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L165)
  - `input_ids_buf` — [`L153`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L153)
  - `mrope_position_ids` — [`L173`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L173)
  - `num_kv_update_slices` — [`L161`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L161)
  - `num_tokens` — [`L157`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L157)
  - `packed_f32_padded` — [`L147`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L147)
  - `packed_i32_padded` — [`L146`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L146)
  - `packed_misc_i32` — [`L150`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L150)
  - `packed_qsl_seqlens` — [`L138`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L138)
  - `pages_tables` — [`L152`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L152)
  - `pixel_values` — [`L164`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L164)
  - `pixel_values_videos` — [`L166`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L166)
  - `position_ids_buf` — [`L154`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L154)
  - `prefill_embeds` — [`L174`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L174)
  - `prefill_embeds_mask` — [`L175`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L175)
  - `slot_mapping` — [`L160`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L160)
  - `video_grid_thw` — [`L167`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L167)
  - `visual_pos_masks` — [`L178`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L178)
- used by: [`_backbone_step`](executors/model_executor.md#ModelStepExecutor._backbone_step), [`execute`](execution_manager.md#ExecutionManager.execute), [`print_status`](execution_types.md#StepFunctionInputs.print_status), [`batch_metadata`](execution_types.md#StepFunctionInputs.batch_metadata), [`_build_backbone_fn`](executors/model_executor.md#ModelStepExecutor._build_backbone_fn), [`prepare_batch_metadata`](executors/batch_preparer.md#BatchMetadataPreparer.prepare_batch_metadata), [`get_async_prep_result`](executors/batch_preparer.md#BatchMetadataPreparer.get_async_prep_result), [`prepare_batch_metadata`](execution_manager.md#ExecutionManager.prepare_batch_metadata), [`get_async_prep_result`](execution_manager.md#ExecutionManager.get_async_prep_result)

### `MinimalDeviceState`
- def: [`easydel/inference/esurge/runners/execution_types.py:80`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L80)
- doc: Minimal device state for sampler updates only.
- signature: `class MinimalDeviceState:`
- members:
  - `num_tokens` — [`L112`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L112)
  - `token_ids` — [`L111`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L111)
- used by: [`device_state`](execution_types.md#StepFunctionOutputs.device_state)

### `ModelStepOutputs`
- def: [`easydel/inference/esurge/runners/execution_types.py:339`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L339)
- doc: Outputs returned from the pure model forward pass.
- signature: `class ModelStepOutputs:`
- members:
  - `hidden_states` — [`L376`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L376)
  - `kv_pages` — [`L375`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L375)
  - `logits` — [`L377`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L377)
- uses (calls/refs, reference-scoped): [`HybridCache`](../../../caching/hybrid/cache.md#HybridCache), [`RaggedPagesCache`](../../../caching/ragged_page/cache.md#RaggedPagesCache), [`UnifiedAttentionCache`](../../../caching/unified_attention/cache.md#UnifiedAttentionCache)
- used by: [`execute`](execution_manager.md#ExecutionManager.execute), [`execute_model`](execution_manager.md#ExecutionManager.execute_model), [`compile`](executors/model_executor.md#ModelStepExecutor.compile), [`_combined`](executors/model_executor.md#ModelStepExecutor._combined)

### `StepFunctionInputs`
- def: [`easydel/inference/esurge/runners/execution_types.py:381`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L381)
- doc: Consolidated inputs for fused step execution.
- signature: `class StepFunctionInputs:`
- members:
  - `print_status(self)` — [`L425`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L425) — Print the shapes of all fields in this StepFunctionInputs structure.
  - `active_mask_full` — [`L421`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L421)
  - `batch_metadata` — [`L423`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L423)
  - `kv_pages` — [`L418`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L418)
  - `req_num_tokens_full` — [`L420`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L420)
  - `rng_key` — [`L422`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L422)
  - `scheduled_full` — [`L419`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L419)
- uses (calls/refs, reference-scoped): [`HybridCache`](../../../caching/hybrid/cache.md#HybridCache), [`RaggedPagesCache`](../../../caching/ragged_page/cache.md#RaggedPagesCache), [`views`](../../../caching/hybrid/cache.md#HybridCache.views), [`UnifiedAttentionCache`](../../../caching/unified_attention/cache.md#UnifiedAttentionCache), [`BatchMetadata`](execution_types.md#BatchMetadata), [`key_cache`](../../../caching/unified_attention/cache.md#UnifiedAttentionCacheView.key_cache), [`value_cache`](../../../caching/unified_attention/cache.md#UnifiedAttentionCacheView.value_cache), [`kv_pages`](../../../caching/hybrid/cache.md#ParallelHybridCacheView.kv_pages), [`num_requests`](execution_types.md#BatchMetadata.num_requests), [`query_start_loc`](execution_types.md#BatchMetadata.query_start_loc), [`request_distribution`](execution_types.md#BatchMetadata.request_distribution), [`seq_lens`](execution_types.md#BatchMetadata.seq_lens), [`pixel_values`](execution_types.md#BatchMetadata.pixel_values), [`pixel_values_videos`](execution_types.md#BatchMetadata.pixel_values_videos), [`logits_indices`](execution_types.md#BatchMetadata.logits_indices), [`min_p`](execution_types.md#BatchMetadata.min_p), [`padded_num_reqs`](execution_types.md#BatchMetadata.padded_num_reqs), [`scheduled`](execution_types.md#BatchMetadata.scheduled), [`temperature`](execution_types.md#BatchMetadata.temperature), [`top_k`](execution_types.md#BatchMetadata.top_k), [`top_p`](execution_types.md#BatchMetadata.top_p), [`image_grid_thw`](execution_types.md#BatchMetadata.image_grid_thw), [`input_ids_buf`](execution_types.md#BatchMetadata.input_ids_buf), [`num_kv_update_slices`](execution_types.md#BatchMetadata.num_kv_update_slices), [`slot_mapping`](execution_types.md#BatchMetadata.slot_mapping), [`video_grid_thw`](execution_types.md#BatchMetadata.video_grid_thw), [`pages_tables`](execution_types.md#BatchMetadata.pages_tables), [`position_ids_buf`](execution_types.md#BatchMetadata.position_ids_buf)
- used by: [`execute`](execution_manager.md#ExecutionManager.execute), [`get_compile_configurations`](execution_manager.md#ExecutionManager.get_compile_configurations), [`_compile_sampler_variant`](execution_manager.md#ExecutionManager._compile_sampler_variant), [`compile_backbone`](executors/model_executor.md#ModelStepExecutor.compile_backbone), [`compile_lm_head`](executors/model_executor.md#ModelStepExecutor.compile_lm_head), [`compile`](executors/sampler_executor.md#SamplerExecutor.compile), [`execute_model`](execution_manager.md#ExecutionManager.execute_model), [`compile`](executors/model_executor.md#ModelStepExecutor.compile)

### `StepFunctionOutputs`
- def: [`easydel/inference/esurge/runners/execution_types.py:489`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L489)
- doc: Consolidated outputs from fused step execution.
- signature: `class StepFunctionOutputs:`
- members:
  - `device_state` — [`L533`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L533)
  - `hidden_states` — [`L543`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L543)
  - `input_ids_buf` — [`L535`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L535)
  - `kv_pages` — [`L534`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L534)
  - `logits` — [`L544`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L544)
  - `out_tokens` — [`L541`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L541)
  - `pages_tables` — [`L539`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L539)
  - `position_ids_buf` — [`L536`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L536)
  - `query_start_loc` — [`L537`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L537)
  - `rng_key` — [`L540`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L540)
  - `seq_lens` — [`L538`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L538)
  - `valid_mask` — [`L542`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_types.py#L542)
- uses (calls/refs, reference-scoped): [`HybridCache`](../../../caching/hybrid/cache.md#HybridCache), [`RaggedPagesCache`](../../../caching/ragged_page/cache.md#RaggedPagesCache), [`UnifiedAttentionCache`](../../../caching/unified_attention/cache.md#UnifiedAttentionCache), [`MinimalDeviceState`](execution_types.md#MinimalDeviceState)


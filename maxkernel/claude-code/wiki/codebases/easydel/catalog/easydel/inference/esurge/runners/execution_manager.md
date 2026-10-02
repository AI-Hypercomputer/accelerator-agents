---
title: 'Module: easydel/inference/esurge/runners/execution_manager.py'
type: catalog
provenance: extracted
module: easydel/inference/esurge/runners/execution_manager.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.esurge.runners.execution_manager`/
symbols:
  ExecutionManager.execute: ExecutionManager#execute().
  ExecutionManager.get_compile_configurations: ExecutionManager#get_compile_configurations().
  ExecutionManager._compile_sampler_variant: ExecutionManager#_compile_sampler_variant().
  ExecutionManager._model_executor: ExecutionManager#_model_executor.
  ExecutionManager._prepare_compact_sampler_window: ExecutionManager#_prepare_compact_sampler_window().
  ExecutionManager.update_graphs: ExecutionManager#update_graphs().
  ExecutionManager._empty_sharding: ExecutionManager#_empty_sharding.
  ExecutionManager._compile_backbone_variant: ExecutionManager#_compile_backbone_variant().
  ExecutionManager.compile: ExecutionManager#compile().
  ExecutionManager.get_compiled_key: ExecutionManager#get_compiled_key().
  ExecutionManager.kv_pages: ExecutionManager#kv_pages.
  ExecutionManager.execute_model: ExecutionManager#execute_model().
  ExecutionManager.sample_tokens: ExecutionManager#sample_tokens().
  ExecutionManager._batch_preparer: ExecutionManager#_batch_preparer.
  ExecutionManager._sampler_executor: ExecutionManager#_sampler_executor.
  ExecutionManager.clear_recurrent_slots: ExecutionManager#clear_recurrent_slots().
  ExecutionManager._compile_lm_head_variant: ExecutionManager#_compile_lm_head_variant().
  ExecutionManager.max_num_reqs: ExecutionManager#max_num_reqs.
  ExecutionManager._ensure_sampler_penalty_state: ExecutionManager#_ensure_sampler_penalty_state().
  ExecutionManager.__init__: ExecutionManager#__init__().
  ExecutionManager.clear_cache: ExecutionManager#clear_cache().
  ExecutionManager.model: ExecutionManager#model.
  logger: logger.
  ExecutionManager.invalidate_sampler_penalty_state: ExecutionManager#invalidate_sampler_penalty_state().
  ExecutionManager._sampler_zero_token_counts: ExecutionManager#_sampler_zero_token_counts.
  ExecutionManager.graphother: ExecutionManager#graphother.
  ExecutionManager._use_request_distribution: ExecutionManager#_use_request_distribution.
  ExecutionManager.has_compiled_variants: ExecutionManager#has_compiled_variants().
  ExecutionManager.prepare_batch_metadata: ExecutionManager#prepare_batch_metadata().
  ExecutionManager.rng_key: ExecutionManager#rng_key.
  ExecutionManager.use_aot_forward: ExecutionManager#use_aot_forward.
  ExecutionManager.graphstate: ExecutionManager#graphstate.
  ExecutionManager.get_async_prep_result: ExecutionManager#get_async_prep_result().
  ExecutionManager._sampler_vocab_size: ExecutionManager#_sampler_vocab_size.
  _tree_hash: _tree_hash().
  ExecutionManager._debug_baselines: ExecutionManager#_debug_baselines.
  ExecutionManager._rebuild_penalty_counts: ExecutionManager#_rebuild_penalty_counts().
  ExecutionManager.mesh: ExecutionManager#mesh.
  ExecutionManager._sampler_token_counts: ExecutionManager#_sampler_token_counts.
  ExecutionManager._sampler_gather_positions_cpu: ExecutionManager#_sampler_gather_positions_cpu.
  ExecutionManager._sampler_sampling_seeds_cpu: ExecutionManager#_sampler_sampling_seeds_cpu.
  ExecutionManager._sampler_scatter_positions_cpu: ExecutionManager#_sampler_scatter_positions_cpu.
  ExecutionManager._sampler_window_row_indices_cpu: ExecutionManager#_sampler_window_row_indices_cpu.
  ExecutionManager._sampler_scheduled_cpu: ExecutionManager#_sampler_scheduled_cpu.
  ExecutionManager._sampler_seq_lens_cpu: ExecutionManager#_sampler_seq_lens_cpu.
  ExecutionManager._sampler_active_mask_cpu: ExecutionManager#_sampler_active_mask_cpu.
  ExecutionManager._sampler_temperature_cpu: ExecutionManager#_sampler_temperature_cpu.
  ExecutionManager._sampler_top_p_cpu: ExecutionManager#_sampler_top_p_cpu.
  ExecutionManager._sampler_top_k_cpu: ExecutionManager#_sampler_top_k_cpu.
  ExecutionManager._sampler_min_p_cpu: ExecutionManager#_sampler_min_p_cpu.
  ExecutionManager._sampler_frequency_penalties_cpu: ExecutionManager#_sampler_frequency_penalties_cpu.
  ExecutionManager._sampler_presence_penalties_cpu: ExecutionManager#_sampler_presence_penalties_cpu.
  ExecutionManager._sampler_repetition_penalties_cpu: ExecutionManager#_sampler_repetition_penalties_cpu.
  ExecutionManager._sampler_zero_window_row_indices: ExecutionManager#_sampler_zero_window_row_indices.
  ExecutionManager._compute_slot_mapping_v2: ExecutionManager#_compute_slot_mapping_v2().
  ExecutionManager.start_async_prep: ExecutionManager#start_async_prep().
  DEBUG_MODE: DEBUG_MODE.
  _tree_hash_diff: _tree_hash_diff().
  ExecutionManager.max_model_len: ExecutionManager#max_model_len.
  ExecutionManager.max_num_tokens: ExecutionManager#max_num_tokens.
  ExecutionManager._sampler_penalty_state_ready: ExecutionManager#_sampler_penalty_state_ready.
  _device_put_tree_uniform: _device_put_tree_uniform().
  ExecutionManager._use_slot_mapping: ExecutionManager#_use_slot_mapping.
  ExecutionManager._scatter_sampler_outputs: ExecutionManager#_scatter_sampler_outputs.
  _get_padded_num_reqs_with_upper_limit: _get_padded_num_reqs_with_upper_limit().
  _device_put_tree_with_shardings: _device_put_tree_with_shardings().
  ExecutionManager.min_input_pad: ExecutionManager#min_input_pad.
  ExecutionManager.graphdef: ExecutionManager#graphdef.
  ExecutionManager._sampler_penalty_state_dirty: ExecutionManager#_sampler_penalty_state_dirty.
  ExecutionManager._sampler_penalty_rebuild_token_ids_cpu: ExecutionManager#_sampler_penalty_rebuild_token_ids_cpu.
  ExecutionManager._sampler_penalty_rebuild_seq_lens_cpu: ExecutionManager#_sampler_penalty_rebuild_seq_lens_cpu.
  ExecutionManager._metadata_version: ExecutionManager#_metadata_version.
  ExecutionManager.log_it: ExecutionManager#log_it.
  ExecutionManager: ExecutionManager#
  ExecutionManager.metadata: ExecutionManager#metadata.
  SYNC_INPUTS_FOR_TIMING: SYNC_INPUTS_FOR_TIMING.
  _tree_hash._map: _tree_hash()._map().
  _tree_hash_diff._map: _tree_hash_diff()._map().
  ExecutionManager.bind_graphstate_for_aot: ExecutionManager#bind_graphstate_for_aot.
  ExecutionManager._verbose: ExecutionManager#_verbose.
  ExecutionManager._get_feasible_compile_pairs: ExecutionManager#_get_feasible_compile_pairs().
  _compute_sampling_valid_mask: _compute_sampling_valid_mask().
  ExecutionManager._sampler_min_input_pad: ExecutionManager#_sampler_min_input_pad.
---
# Module: [`easydel/inference/esurge/runners/execution_manager.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py)

## Classes
### `ExecutionManager`
- def: [`easydel/inference/esurge/runners/execution_manager.py:292`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L292)
- doc: Compilation and execution manager for fused inference step functions.
- signature: `class ExecutionManager:`
- members:
  - `__init__(self, model: EasyDeLBaseModule, use_aot_forward: bool = True, bind_graphstate_for_aot: bool = False, min_input_pad: int = 8, max_model_len: int = 2 ** 13, max_num_reqs: int = 16, max_num_tokens: int | None = None, metadata: RaggedPagesCacheConfig | UnifiedAttentionCacheConfig | None = None, verbose: bool = False)` — [`L368`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L368) — Initialize the executor manager.
  - `_compile_backbone_variant(self, *, num_tokens: int, max_num_reqs: int, metadata: RaggedPagesCacheConfig | UnifiedAttentionCacheConfig)` — [`L1407`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L1407) — Compile the backbone (transformer forward) for a token bucket.
  - `_compile_lm_head_variant(self, *, padded_num_reqs: int, max_num_reqs: int, metadata: RaggedPagesCacheConfig | UnifiedAttentionCacheConfig)` — [`L1444`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L1444) — Compile the lm_head (gather + project) for a request bucket.
  - `_compile_sampler_variant(self, *, num_tokens: int, max_num_reqs: int, padded_num_reqs: int, metadata: RaggedPagesCacheConfig | UnifiedAttentionCacheConfig, inputs: StepFunctionInputs | None = None)` — [`L1477`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L1477) — Compile a sampler variant without requiring a matching model variant.
  - `_compute_slot_mapping_v2(self, num_requests: int, scheduled: numpy.ndarray, num_computed_tokens_cpu: numpy.ndarray, page_table_cpu: numpy.ndarray)` — [`L1535`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L1535) — Compute slot mapping tensor for ragged-page attention v2.
  - `_ensure_sampler_penalty_state(self)` — [`L563`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L563) — Rebuild exact device-side token counts from full sequence state when needed.
  - `_get_feasible_compile_pairs(num_tokens_paddings: list[int], reqs_padds: list[int])` — [`L1272`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L1272) — Return only schedulable token/request bucket combinations.
  - `_prepare_compact_sampler_window(self, *, padded_num_reqs: int, scheduled_full_cpu: numpy.ndarray, active_mask_full_cpu: numpy.ndarray, window_row_indices_cpu: numpy.ndarray, num_computed_tokens_cpu: numpy.ndarray, temperature_cpu: numpy.ndarray, top_p_cpu: numpy.ndarray, top_k_cpu: numpy.ndarray, min_p_cpu: numpy.ndarray, frequency_penalties_cpu: numpy.ndarray, presence_penalties_cpu: numpy.ndarray, repetition_penalties_cpu: numpy.ndarray)` — [`L580`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L580) — Compact the sampler workload to rows that can actually emit tokens.
  - `clear_cache(self)` — [`L535`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L535) — Clear all cached compiled functions.
  - `clear_recurrent_slots(self, slot_indices: list[int])` — [`L1054`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L1054) — Zero out recurrent/SSM state for freed request slots.
  - `compile(self, num_tokens_paddings: list[int], num_reqs_max_model_len: int, max_pages_per_req: int, max_num_reqs: int, metadata: RaggedPagesCacheConfig | UnifiedAttentionCacheConfig, num_reqs_paddings: list[int] | None = None, prune_infeasible_pairs: bool = True)` — [`L1291`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L1291) — Compile model execution functions for various input configurations.
  - `execute(self, num_tokens: int, scheduled_full_cpu: numpy.ndarray, req_num_tokens_full: jax.Array, active_mask_full_cpu: numpy.ndarray, window_row_indices_cpu: numpy.ndarray, input_ids_buf: jax.Array, position_ids_buf: jax.Array, padded_num_reqs: int, token_ids_cpu: numpy.ndarray, num_computed_tokens_cpu: numpy.ndarray, temperature_cpu: numpy.ndarray, top_p_cpu: numpy.ndarray, top_k_cpu: numpy.ndarray, min_p_cpu: numpy.ndarray, frequency_penalties_cpu: numpy.ndarray, presence_penalties_cpu: numpy.ndarray, repetition_penalties_cpu: numpy.ndarray, page_table_cpu: numpy.ndarray, page_table_version: int | None = None, mrope_position_ids_cpu: numpy.ndarray | None = None, prefill_embeds_cpu: numpy.ndarray | None = None, prefill_embeds_mask_cpu: numpy.ndarray | None = None, visual_pos_masks_cpu: numpy.ndarray | None = None, deepstack_visual_embeds_cpu: list[numpy.ndarray] | None = None, pixel_values: numpy.ndarray | None = None, image_grid_thw: numpy.ndarray | None = None, pixel_values_videos: numpy.ndarray | None = None, video_grid_thw: numpy.ndarray | None = None, wait_for_outputs: bool = True)` — [`L746`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L746) — Execute a single fused inference step.
  - `execute_model(self, num_tokens: int, padded_num_reqs: int, inputs: StepFunctionInputs)` — [`L1019`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L1019) — Run the compiled model forward step and update self.kv_pages.
  - `get_async_prep_result(self)` — [`L1734`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L1734) — Retrieve results from a previously started async batch preparation.
  - `get_compile_configurations(self, kv_pages: HybridCache | RaggedPagesCache | UnifiedAttentionCache, rng_key: jax.random.PRNGKey, num_tokens: int, max_num_reqs: int, padded_num_reqs: int, metadata: RaggedPagesCacheConfig | UnifiedAttentionCacheConfig)` — [`L1789`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L1789) — Generate compilation arguments for step function.
  - `get_compiled_key(self, num_tokens: int, padded_num_reqs: int)` — [`L1757`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L1757) — Retrieve pre-compiled model and sampler functions for given input dimensions.
  - `has_compiled_variants(self)` — [`L678`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L678) — Check whether both model and sampler executors have compiled variants.
  - `invalidate_sampler_penalty_state(self, token_ids_cpu: numpy.ndarray | None = None, seq_lens_cpu: numpy.ndarray | None = None)` — [`L550`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L550) — Mark incremental sampler penalty state dirty after host-side row changes.
  - `prepare_batch_metadata(self, num_tokens_static: int, scheduled_full_cpu: numpy.ndarray, active_mask_full_cpu: numpy.ndarray, input_ids_buf: jax.Array, position_ids_buf: jax.Array, token_ids_cpu: numpy.ndarray, num_computed_tokens_cpu: numpy.ndarray, temperature_cpu: numpy.ndarray, top_p_cpu: numpy.ndarray, top_k_cpu: numpy.ndarray, min_p_cpu: numpy.ndarray, frequency_penalties_cpu: numpy.ndarray, presence_penalties_cpu: numpy.ndarray, repetition_penalties_cpu: numpy.ndarray, page_table_cpu: numpy.ndarray, padded_num_reqs_in: int, page_table_version: int | None = None, mrope_position_ids_cpu: numpy.ndarray | None = None, prefill_embeds_cpu: numpy.ndarray | None = None, prefill_embeds_mask_cpu: numpy.ndarray | None = None, visual_pos_masks_cpu: numpy.ndarray | None = None, deepstack_visual_embeds_cpu: list[numpy.ndarray] | None = None, pixel_values: numpy.ndarray | None = None, image_grid_thw: numpy.ndarray | None = None, pixel_values_videos: numpy.ndarray | None = None, video_grid_thw: numpy.ndarray | None = None)` — [`L1569`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L1569) — Prepare batch metadata using CPU-first computation.
  - `sample_tokens(self, num_tokens: int, padded_num_reqs: int, *, sampler_padded_num_reqs: int, sampler_num_reqs: int, sampler_total_tokens: int, req_num_tokens_full: jax.Array, logits: jax.Array, rng_key: jax.Array, gather_positions_cpu: numpy.ndarray, sampling_seeds_cpu: numpy.ndarray, scatter_positions_cpu: numpy.ndarray, compact_window_row_indices_cpu: numpy.ndarray, compact_scheduled_cpu: numpy.ndarray, compact_seq_lens_cpu: numpy.ndarray, compact_active_mask_cpu: numpy.ndarray, compact_temperature_cpu: numpy.ndarray, compact_top_p_cpu: numpy.ndarray, compact_top_k_cpu: numpy.ndarray, compact_min_p_cpu: numpy.ndarray, compact_frequency_penalties_cpu: numpy.ndarray, compact_presence_penalties_cpu: numpy.ndarray, compact_repetition_penalties_cpu: numpy.ndarray, need_penalties: bool)` — [`L1106`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L1106) — Run the compiled sampler step over only the rows that need sampling.
  - `start_async_prep(self, num_tokens_static: int, scheduled_full_cpu: numpy.ndarray, active_mask_full_cpu: numpy.ndarray, input_ids_buf: jax.Array, position_ids_buf: jax.Array, token_ids_cpu: numpy.ndarray, num_computed_tokens_cpu: numpy.ndarray, temperature_cpu: numpy.ndarray, top_p_cpu: numpy.ndarray, top_k_cpu: numpy.ndarray, min_p_cpu: numpy.ndarray, frequency_penalties_cpu: numpy.ndarray, presence_penalties_cpu: numpy.ndarray, repetition_penalties_cpu: numpy.ndarray, page_table_cpu: numpy.ndarray, padded_num_reqs_in: int, page_table_version: int | None = None)` — [`L1667`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L1667) — Start async device transfer for double-buffered batch preparation.
  - `update_graphs(self, model: EasyDeLBaseModule | None = None, *, graphdef=None, graphstate=None, graphother=None)` — [`L687`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L687) — Update the graph components (weights) used by the fused executor.
  - `bind_graphstate_for_aot` — [`L409`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L409)
  - `graphdef` — [`L438`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L438)
  - `graphother` — [`L438`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L438)
  - `graphstate` — [`L438`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L438)
  - `kv_pages` — [`L426`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L426)
  - `log_it` — [`L440`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L440)
  - `max_model_len` — [`L411`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L411)
  - `max_num_reqs` — [`L412`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L412)
  - `max_num_tokens` — [`L413`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L413)
  - `mesh` — [`L406`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L406)
  - `metadata` — [`L414`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L414)
  - `min_input_pad` — [`L410`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L410)
  - `model` — [`L405`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L405)
  - `rng_key` — [`L445`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L445)
  - `use_aot_forward` — [`L408`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L408)
- protocol/private: `_batch_preparer`[`L449`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L449), `_debug_baselines`[`L447`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L447), `_empty_sharding`[`L443`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L443), `_metadata_version`[`L415`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L415), `_model_executor`[`L457`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L457), `_rebuild_penalty_counts`[`L509`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L509), `_rebuild_penalty_counts`[`L517`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L517), `_sampler_active_mask_cpu`[`L499`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L499), `_sampler_executor`[`L471`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L471), `_sampler_frequency_penalties_cpu`[`L504`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L504), `_sampler_gather_positions_cpu`[`L493`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L493), `_sampler_min_input_pad`[`L477`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L477), `_sampler_min_p_cpu`[`L503`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L503), `_sampler_penalty_rebuild_seq_lens_cpu`[`L492`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L492), `_sampler_penalty_rebuild_token_ids_cpu`[`L491`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L491), `_sampler_penalty_state_dirty`[`L489`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L489), `_sampler_penalty_state_ready`[`L490`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L490), `_sampler_presence_penalties_cpu`[`L505`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L505), `_sampler_repetition_penalties_cpu`[`L506`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L506), `_sampler_sampling_seeds_cpu`[`L494`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L494), `_sampler_scatter_positions_cpu`[`L495`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L495), `_sampler_scheduled_cpu`[`L497`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L497), `_sampler_seq_lens_cpu`[`L498`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L498), `_sampler_temperature_cpu`[`L500`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L500), `_sampler_token_counts`[`L488`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L488), `_sampler_top_k_cpu`[`L502`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L502), `_sampler_top_p_cpu`[`L501`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L501), `_sampler_vocab_size`[`L470`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L470), `_sampler_window_row_indices_cpu`[`L496`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L496), `_sampler_zero_token_counts`[`L478`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L478), `_sampler_zero_window_row_indices`[`L483`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L483), `_scatter_sampler_outputs`[`L520`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L520), `_scatter_sampler_outputs`[`L533`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L533), `_use_request_distribution`[`L417`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L417), `_use_slot_mapping`[`L416`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L416), `_verbose`[`L441`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L441)
- uses (calls/refs, reference-scoped): [`config`](../../../infra/base_module.md#EasyDeLBaseModule.config), [`EasyDeLBaseModule`](../../../infra/base_module.md#EasyDeLBaseModule), [`HybridCache`](../../../caching/hybrid/cache.md#HybridCache), [`RaggedPagesCache`](../../../caching/ragged_page/cache.md#RaggedPagesCache), [`RaggedPagesCacheConfig`](../../../caching/ragged_page/cache.md#RaggedPagesCacheConfig), [`init_operations_cache`](../../../infra/mixins/generation.md#EasyGenerationMixin.init_operations_cache), [`RecurrentCacheView`](../../../caching/recurrent/cache.md#RecurrentCacheView), [`mesh`](../../../infra/base_module.md#EasyDeLBaseModule.mesh), [`views`](../../../caching/hybrid/cache.md#HybridCache.views), [`UnifiedAttentionCacheConfig`](../../../caching/unified_attention/cache.md#UnifiedAttentionCacheConfig), [`start_async_prep`](executors/batch_preparer.md#BatchMetadataPreparer.start_async_prep), [`compile_backbone`](executors/model_executor.md#ModelStepExecutor.compile_backbone), [`compile_lm_head`](executors/model_executor.md#ModelStepExecutor.compile_lm_head), [`dtype`](../../../infra/base_module.md#EasyDeLBaseModule.dtype), [`batch_metadata`](execution_types.md#StepFunctionInputs.batch_metadata), [`ParallelHybridCacheView`](../../../caching/hybrid/cache.md#ParallelHybridCacheView), [`conv_state`](../../../caching/recurrent/cache.md#RecurrentCacheView.conv_state), [`compile`](executors/sampler_executor.md#SamplerExecutor.compile), [`prepare_batch_metadata`](executors/batch_preparer.md#BatchMetadataPreparer.prepare_batch_metadata), [`recurrent_state`](../../../caching/recurrent/cache.md#RecurrentCacheView.recurrent_state), [`UnifiedAttentionCache`](../../../caching/unified_attention/cache.md#UnifiedAttentionCache), [`_compute_slot_mapping_v2`](executors/batch_preparer.md#BatchMetadataPreparer._compute_slot_mapping_v2), [`get_async_prep_result`](executors/batch_preparer.md#BatchMetadataPreparer.get_async_prep_result), [`num_computed_tokens`](sequence_buffer.md#SequenceBuffer.num_computed_tokens), [`token_ids`](sequence_buffer.md#SequenceBuffer.token_ids), [`TurboQuantConfig`](../../../layers/quantization/_turboquant.md#TurboQuantConfig), [`top_k`](sequence_buffer.md#SequenceBuffer.top_k), [`kv_pages`](execution_types.md#StepFunctionInputs.kv_pages), [`recurrent`](../../../caching/hybrid/cache.md#ParallelHybridCacheView.recurrent), [`min_p`](sequence_buffer.md#SequenceBuffer.min_p), [`temperature`](sequence_buffer.md#SequenceBuffer.temperature), [`top_p`](sequence_buffer.md#SequenceBuffer.top_p), [`frequency_penalties`](sequence_buffer.md#SequenceBuffer.frequency_penalties), [`presence_penalties`](sequence_buffer.md#SequenceBuffer.presence_penalties), [`repetition_penalties`](sequence_buffer.md#SequenceBuffer.repetition_penalties), [`get_compiled`](executors/model_executor.md#ModelStepExecutor.get_compiled), [`page_table`](sequence_buffer.md#SequenceBuffer.page_table), [`logger`](execution_manager.md#logger), [`BatchMetadata`](execution_types.md#BatchMetadata), [`page_size`](../../../caching/ragged_page/cache.md#RaggedPagesCacheConfig.page_size)  (+36 more)
- used by: [`_execute_model_impl`](model_runner.md#eSurgeRunner._execute_model_impl), [`_update_states`](model_runner.md#eSurgeRunner._update_states), [`initialize_kv_cache`](model_runner.md#eSurgeRunner.initialize_kv_cache), [`compile`](model_runner.md#eSurgeRunner.compile), [`executor_manager`](model_runner.md#eSurgeRunner.executor_manager), [`release_model_state`](model_runner.md#eSurgeRunner.release_model_state), [`update_model_weights`](model_runner.md#eSurgeRunner.update_model_weights), [`_window_row_indices_cpu`](model_runner.md#eSurgeRunner._window_row_indices_cpu), [`destroy_kv_cache`](model_runner.md#eSurgeRunner.destroy_kv_cache)

## Functions
- `_compute_sampling_valid_mask(*, i_reqs: jax.Array, num_requests: jax.Array, active_mask_slice: jax.Array, scheduled_slice: jax.Array, seq_lens_now: jax.Array, req_num_tokens_slice: jax.Array)` — [`L146`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L146) — Compute which request slots are valid for sampling.
- `_device_put_tree_uniform(tree, sharding)` — [`L199`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L199) — Place a PyTree on device with uniform sharding for all leaves.
- `_device_put_tree_with_shardings(tree, shardings_tree)` — [`L185`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L185) — Place a PyTree on device with per-leaf shardings.
- `_get_padded_num_reqs_with_upper_limit(x: int, upper_limit: int, min_input_pad: int)` — [`L118`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L118) — Calculate padded request count for compilation efficiency.
- `_map(p, x)` — [`L232`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L232)
- `_map(p, t1, t2)` — [`L282`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L282)
- `_tree_hash(tree)` — [`L214`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L214) — Compute a hash tree for debugging structure/shape/dtype changes.
- `_tree_hash_diff(orgin, new)` — [`L268`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L268) — Compare two hash trees and print differences.

## Module values
- `DEBUG_MODE` — [`L106`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L106)
- `SYNC_INPUTS_FOR_TIMING` — [`L115`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L115)
- `logger` — [`L111`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/execution_manager.py#L111)


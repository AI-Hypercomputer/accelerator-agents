---
title: 'Module: easydel/inference/esurge/runners/executors/batch_preparer.py'
type: catalog
provenance: extracted
module: easydel/inference/esurge/runners/executors/batch_preparer.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.esurge.runners.executors.batch_preparer`/BatchMetadataPreparer#
symbols:
  BatchMetadataPreparer._build_host_payload: _build_host_payload().
  BatchMetadataPreparer.start_async_prep: start_async_prep().
  BatchMetadataPreparer.max_num_reqs: max_num_reqs.
  BatchMetadataPreparer.prepare_batch_metadata: prepare_batch_metadata().
  BatchMetadataPreparer._compute_slot_mapping_v2: _compute_slot_mapping_v2().
  BatchMetadataPreparer.get_async_prep_result: get_async_prep_result().
  BatchMetadataPreparer._enforce_dp_local_page_tables: _enforce_dp_local_page_tables().
  BatchMetadataPreparer._slot_mapping_indices: _slot_mapping_indices.
  BatchMetadataPreparer._use_slot_mapping: _use_slot_mapping.
  BatchMetadataPreparer._empty_sharding: _empty_sharding.
  BatchMetadataPreparer._max_padded_slices: _max_padded_slices.
  BatchMetadataPreparer._enable_dp_local_page_path: _enable_dp_local_page_path.
  BatchMetadataPreparer._pages_tables_cpu: _pages_tables_cpu.
  BatchMetadataPreparer._async_pages_tables_cpu: _async_pages_tables_cpu.
  BatchMetadataPreparer._slot_mapping_cpu: _slot_mapping_cpu.
  BatchMetadataPreparer._async_page_table_cpu: _async_page_table_cpu.
  BatchMetadataPreparer._async_slot_mapping_cpu: _async_slot_mapping_cpu.
  BatchMetadataPreparer._get_zero_dev: _get_zero_dev().
  BatchMetadataPreparer._get_zero_dev_like: _get_zero_dev_like().
  BatchMetadataPreparer.max_num_tokens: max_num_tokens.
  BatchMetadataPreparer._pending_transfer: _pending_transfer.
  BatchMetadataPreparer._cached_rows_to_copy: _cached_rows_to_copy.
  BatchMetadataPreparer.__init__: __init__().
  BatchMetadataPreparer.metadata: metadata.
  BatchMetadataPreparer._use_request_distribution: _use_request_distribution.
  BatchMetadataPreparer._async_num_computed_tokens_cpu: _async_num_computed_tokens_cpu.
  BatchMetadataPreparer._async_scheduled_full_cpu: _async_scheduled_full_cpu.
  BatchMetadataPreparer._async_active_mask_full_cpu: _async_active_mask_full_cpu.
  BatchMetadataPreparer._async_temperature_cpu: _async_temperature_cpu.
  BatchMetadataPreparer._async_top_p_cpu: _async_top_p_cpu.
  BatchMetadataPreparer._async_top_k_cpu: _async_top_k_cpu.
  BatchMetadataPreparer._async_min_p_cpu: _async_min_p_cpu.
  BatchMetadataPreparer._async_frequency_penalties_cpu: _async_frequency_penalties_cpu.
  BatchMetadataPreparer._async_presence_penalties_cpu: _async_presence_penalties_cpu.
  BatchMetadataPreparer._async_repetition_penalties_cpu: _async_repetition_penalties_cpu.
  BatchMetadataPreparer._slices_per_page: _slices_per_page.
  BatchMetadataPreparer._num_reqs_max_model_len: _num_reqs_max_model_len.
  BatchMetadataPreparer._cached_pages_tables_dev: _cached_pages_tables_dev.
  BatchMetadataPreparer._metadata_version: _metadata_version.
  BatchMetadataPreparer._input_ids_cpu: _input_ids_cpu.
  BatchMetadataPreparer._positions_cpu: _positions_cpu.
  BatchMetadataPreparer._query_start_loc_cpu: _query_start_loc_cpu.
  BatchMetadataPreparer._seq_lens_cpu: _seq_lens_cpu.
  BatchMetadataPreparer._logits_indices_cpu: _logits_indices_cpu.
  BatchMetadataPreparer._scheduled_cpu: _scheduled_cpu.
  BatchMetadataPreparer._packed_qsl_seqlens_cpu: _packed_qsl_seqlens_cpu.
  BatchMetadataPreparer._packed_i32_padded_cpu: _packed_i32_padded_cpu.
  BatchMetadataPreparer._packed_f32_padded_cpu: _packed_f32_padded_cpu.
  BatchMetadataPreparer._arange_cpu: _arange_cpu.
  BatchMetadataPreparer._async_input_ids_cpu: _async_input_ids_cpu.
  BatchMetadataPreparer._async_positions_cpu: _async_positions_cpu.
  BatchMetadataPreparer._async_query_start_loc_cpu: _async_query_start_loc_cpu.
  BatchMetadataPreparer._async_seq_lens_cpu: _async_seq_lens_cpu.
  BatchMetadataPreparer._async_logits_indices_cpu: _async_logits_indices_cpu.
  BatchMetadataPreparer._async_scheduled_cpu: _async_scheduled_cpu.
  BatchMetadataPreparer._async_packed_qsl_seqlens_cpu: _async_packed_qsl_seqlens_cpu.
  BatchMetadataPreparer._async_packed_i32_padded_cpu: _async_packed_i32_padded_cpu.
  BatchMetadataPreparer._async_packed_f32_padded_cpu: _async_packed_f32_padded_cpu.
  BatchMetadataPreparer: ''
  BatchMetadataPreparer._max_pages_per_req: _max_pages_per_req.
  BatchMetadataPreparer._cached_page_table_version: _cached_page_table_version.
  BatchMetadataPreparer._pending_transfer_metadata: _pending_transfer_metadata.
  BatchMetadataPreparer.uses_slot_mapping: uses_slot_mapping().
  BatchMetadataPreparer.uses_request_distribution: uses_request_distribution().
  BatchMetadataPreparer.last_prep_stats: last_prep_stats.
  BatchMetadataPreparer._zero_dev_cache: _zero_dev_cache.
  BatchMetadataPreparer.min_input_pad: min_input_pad.
  BatchMetadataPreparer._packed_misc_i32_cpu: _packed_misc_i32_cpu.
  BatchMetadataPreparer._request_distribution_placeholder: _request_distribution_placeholder.
  BatchMetadataPreparer._slot_mapping_placeholder: _slot_mapping_placeholder.
  BatchMetadataPreparer._num_kv_update_placeholder: _num_kv_update_placeholder.
  BatchMetadataPreparer._async_slot_mapping_placeholder: _async_slot_mapping_placeholder.
  BatchMetadataPreparer._async_packed_misc_i32_cpu: _async_packed_misc_i32_cpu.
  BatchMetadataPreparer._async_request_distribution_cpu: _async_request_distribution_cpu.
  BatchMetadataPreparer._async_num_kv_update_cpu: _async_num_kv_update_cpu.
  BatchMetadataPreparer.max_model_len: max_model_len.
---
# Module: [`easydel/inference/esurge/runners/executors/batch_preparer.py`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py)

## Classes
### `BatchMetadataPreparer`
- def: [`easydel/inference/esurge/runners/executors/batch_preparer.py:83`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L83)
- doc: Prepare and transfer per-step metadata for model execution.
- signature: `class BatchMetadataPreparer:`
- members:
  - `__init__(self, *, metadata: RaggedPagesCacheConfig | UnifiedAttentionCacheConfig | None, empty_sharding: jax.sharding.Sharding, max_num_tokens: int, max_num_reqs: int, max_model_len: int, min_input_pad: int)` — [`L118`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L118) — Initialize the BatchMetadataPreparer.
  - `_build_host_payload(self, *, num_tokens_static: int, scheduled_full_cpu: np.ndarray, active_mask_full_cpu: np.ndarray, token_ids_cpu: np.ndarray, num_computed_tokens_cpu: np.ndarray, temperature_cpu: np.ndarray, top_p_cpu: np.ndarray, top_k_cpu: np.ndarray, min_p_cpu: np.ndarray, frequency_penalties_cpu: np.ndarray, presence_penalties_cpu: np.ndarray, repetition_penalties_cpu: np.ndarray, page_table_cpu: np.ndarray, page_table_version: int | None, padded_num_reqs_in: int, copy_slot_mapping: bool, use_async_buffers: bool = False)` — [`L506`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L506) — Build the host-side payload for device transfer.
  - `_compute_slot_mapping_v2(self, *, num_requests: int, scheduled: np.ndarray, num_computed_tokens_cpu: np.ndarray, page_table_cpu: np.ndarray, slot_mapping_out: np.ndarray | None = None, copy_out: bool = True)` — [`L386`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L386) — Compute slot mapping tensor for ragged-page attention v2.
  - `_enforce_dp_local_page_tables(self, *, num_requests: int, scheduled: np.ndarray, num_computed_tokens_cpu: np.ndarray, page_table_cpu: np.ndarray)` — [`L266`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L266) — Validate that active rows only use page IDs local to their DP shard.
  - `_get_zero_dev(self, *, namespace: str, shape: tuple[int, ...], dtype: np.dtype)` — [`L334`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L334) — Get or create a cached zero-filled device array.
  - `_get_zero_dev_like(self, *, namespace: str, arr: np.ndarray)` — [`L356`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L356) — Get a cached zero device array with shape/dtype matching an array.
  - `get_async_prep_result(self)` — [`L1131`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L1131) — Retrieve results from a previously started async batch preparation.
  - `prepare_batch_metadata(self, *, num_tokens_static: int, scheduled_full_cpu: np.ndarray, active_mask_full_cpu: np.ndarray, input_ids_buf: jax.Array, position_ids_buf: jax.Array, token_ids_cpu: np.ndarray, num_computed_tokens_cpu: np.ndarray, temperature_cpu: np.ndarray, top_p_cpu: np.ndarray, top_k_cpu: np.ndarray, min_p_cpu: np.ndarray, frequency_penalties_cpu: np.ndarray, presence_penalties_cpu: np.ndarray, repetition_penalties_cpu: np.ndarray, page_table_cpu: np.ndarray, padded_num_reqs_in: int, page_table_version: int | None = None, mrope_position_ids_cpu: np.ndarray | None = None, prefill_embeds_cpu: np.ndarray | None = None, prefill_embeds_mask_cpu: np.ndarray | None = None, visual_pos_masks_cpu: np.ndarray | None = None, deepstack_visual_embeds_cpu: list[np.ndarray] | None = None, pixel_values: np.ndarray | None = None, image_grid_thw: np.ndarray | None = None, pixel_values_videos: np.ndarray | None = None, video_grid_thw: np.ndarray | None = None)` — [`L782`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L782) — Prepare batch metadata using CPU-first computation strategy.
  - `start_async_prep(self, *, num_tokens_static: int, scheduled_full_cpu: np.ndarray, active_mask_full_cpu: np.ndarray, input_ids_buf: jax.Array, position_ids_buf: jax.Array, token_ids_cpu: np.ndarray, num_computed_tokens_cpu: np.ndarray, temperature_cpu: np.ndarray, top_p_cpu: np.ndarray, top_k_cpu: np.ndarray, min_p_cpu: np.ndarray, frequency_penalties_cpu: np.ndarray, presence_penalties_cpu: np.ndarray, repetition_penalties_cpu: np.ndarray, page_table_cpu: np.ndarray, padded_num_reqs_in: int, page_table_version: int | None = None)` — [`L1018`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L1018) — Start async device transfer for double-buffered batch preparation.
  - `uses_request_distribution(self)` — [`L378`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L378) — Check if this preparer uses v3-style request distribution.
  - `uses_slot_mapping(self)` — [`L369`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L369) — Check if this preparer uses v2-style slot mapping.
  - `last_prep_stats` — [`L259`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L259)
  - `max_model_len` — [`L156`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L156)
  - `max_num_reqs` — [`L155`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L155)
  - `max_num_tokens` — [`L154`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L154)
  - `metadata` — [`L151`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L151)
  - `min_input_pad` — [`L157`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L157)
- protocol/private: `_arange_cpu`[`L184`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L184), `_async_active_mask_full_cpu`[`L216`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L216), `_async_frequency_penalties_cpu`[`L221`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L221), `_async_input_ids_cpu`[`L196`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L196), `_async_logits_indices_cpu`[`L200`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L200), `_async_min_p_cpu`[`L220`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L220), `_async_num_computed_tokens_cpu`[`L214`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L214), `_async_num_kv_update_cpu`[`L212`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L212), `_async_packed_f32_padded_cpu`[`L204`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L204), `_async_packed_i32_padded_cpu`[`L203`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L203), `_async_packed_misc_i32_cpu`[`L205`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L205), `_async_packed_qsl_seqlens_cpu`[`L202`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L202), `_async_page_table_cpu`[`L224`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L224), `_async_pages_tables_cpu`[`L206`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L206), `_async_positions_cpu`[`L197`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L197), `_async_presence_penalties_cpu`[`L222`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L222), `_async_query_start_loc_cpu`[`L198`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L198), `_async_repetition_penalties_cpu`[`L223`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L223), `_async_request_distribution_cpu`[`L211`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L211), `_async_scheduled_cpu`[`L201`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L201), `_async_scheduled_full_cpu`[`L215`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L215), `_async_seq_lens_cpu`[`L199`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L199), `_async_slot_mapping_cpu`[`L241`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L241), `_async_slot_mapping_placeholder`[`L193`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L193), `_async_temperature_cpu`[`L217`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L217), `_async_top_k_cpu`[`L219`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L219), `_async_top_p_cpu`[`L218`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L218), `_cached_page_table_version`[`L230`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L230), `_cached_pages_tables_dev`[`L229`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L229), `_cached_rows_to_copy`[`L231`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L231), `_empty_sharding`[`L152`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L152), `_enable_dp_local_page_path`[`L162`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L162), `_input_ids_cpu`[`L174`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L174), `_logits_indices_cpu`[`L178`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L178), `_max_padded_slices`[`L235`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L235), `_max_pages_per_req`[`L171`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L171), `_metadata_version`[`L159`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L159), `_num_kv_update_placeholder`[`L192`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L192), `_num_reqs_max_model_len`[`L168`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L168), `_packed_f32_padded_cpu`[`L182`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L182), `_packed_i32_padded_cpu`[`L181`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L181), `_packed_misc_i32_cpu`[`L183`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L183), `_packed_qsl_seqlens_cpu`[`L180`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L180), `_pages_tables_cpu`[`L185`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L185), `_pending_transfer`[`L255`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L255), `_pending_transfer_metadata`[`L256`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L256), `_positions_cpu`[`L175`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L175), `_query_start_loc_cpu`[`L176`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L176), `_request_distribution_placeholder`[`L190`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L190), `_scheduled_cpu`[`L179`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L179), `_seq_lens_cpu`[`L177`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L177), `_slices_per_page`[`L234`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L234), `_slot_mapping_cpu`[`L236`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L236), `_slot_mapping_indices`[`L246`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L246), `_slot_mapping_placeholder`[`L191`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L191), `_use_request_distribution`[`L161`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L161), `_use_slot_mapping`[`L160`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L160), `_zero_dev_cache`[`L264`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/batch_preparer.py#L264)
- uses (calls/refs, reference-scoped): [`RaggedPagesCacheConfig`](../../../../caching/ragged_page/cache.md#RaggedPagesCacheConfig), [`UnifiedAttentionCacheConfig`](../../../../caching/unified_attention/cache.md#UnifiedAttentionCacheConfig), [`check_bool_flag`](../../../../utils/helpers.md#check_bool_flag), [`get_padded_num_slices`](../../../../caching/ragged_page/cache.md#RaggedPagesCacheConfig.get_padded_num_slices), [`pages_per_dp_shard`](../../core/dp_sharding.md#pages_per_dp_shard), [`get_max_num_seqs`](../../../../caching/ragged_page/cache.md#RaggedPagesCacheConfig.get_max_num_seqs), [`BatchMetadata`](../execution_types.md#BatchMetadata), [`page_size`](../../../../caching/ragged_page/cache.md#RaggedPagesCacheConfig.page_size), [`version`](../../../../caching/ragged_page/cache.md#RaggedPagesCacheConfig.version), [`dp_shard_page_bounds`](../../core/dp_sharding.md#dp_shard_page_bounds), [`PAGE_TABLE_PADDING_VAL`](../../page_table.md#PAGE_TABLE_PADDING_VAL), [`num_slices_per_kv_cache_update_page`](../../../../caching/ragged_page/cache.md#RaggedPagesCacheConfig.num_slices_per_kv_cache_update_page), [`SLOT_MAPPING_PADDING_VAL`](../../page_table.md#SLOT_MAPPING_PADDING_VAL), [`AttentionMetadataBuilder`](../../../../caching/_metadatabuilder.md#AttentionMetadataBuilder), [`max_num_pages_per_req`](../../../../caching/ragged_page/cache.md#RaggedPagesCacheConfig.max_num_pages_per_req), [`compute_padded_num_reqs`](../../../../caching/_metadatabuilder.md#AttentionMetadataBuilder.compute_padded_num_reqs)
- used by: [`_batch_preparer`](../execution_manager.md#ExecutionManager._batch_preparer), [`prepare_batch_metadata`](../execution_manager.md#ExecutionManager.prepare_batch_metadata), [`get_async_prep_result`](../execution_manager.md#ExecutionManager.get_async_prep_result), [`_compute_slot_mapping_v2`](../execution_manager.md#ExecutionManager._compute_slot_mapping_v2), [`start_async_prep`](../execution_manager.md#ExecutionManager.start_async_prep)


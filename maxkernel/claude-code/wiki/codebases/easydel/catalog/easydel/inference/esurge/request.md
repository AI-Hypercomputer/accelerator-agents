---
title: 'Module: easydel/inference/esurge/request.py'
type: catalog
provenance: extracted
module: easydel/inference/esurge/request.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.esurge.request`/
symbols:
  EngineRequest: EngineRequest#
  EngineRequest.request_id: EngineRequest#request_id.
  EngineRequestStatus: EngineRequestStatus#
  EngineRequest.from_engine_core_request: EngineRequest#from_engine_core_request().
  EngineRequest.status: EngineRequest#status.
  _FINISHED_REASON_MAP: _FINISHED_REASON_MAP.
  EngineRequest.num_computed_tokens: EngineRequest#num_computed_tokens.
  EngineRequest.clear_vision_data: EngineRequest#clear_vision_data().
  EngineRequest.num_tokens: EngineRequest#num_tokens().
  EngineRequest.get_finished_reason: EngineRequest#get_finished_reason().
  EngineRequest.output_token_ids: EngineRequest#output_token_ids.
  EngineRequest.record_event: EngineRequest#record_event().
  EngineRequest.is_finished: EngineRequest#is_finished().
  EngineRequest.max_tokens: EngineRequest#max_tokens.
  EngineRequestStatus.get_finished_reason: EngineRequestStatus#get_finished_reason().
  EngineRequest.num_output_placeholders: EngineRequest#num_output_placeholders.
  EngineRequest.all_token_ids: EngineRequest#all_token_ids.
  EngineRequest._all_token_ids: EngineRequest#_all_token_ids.
  EngineRequest.has_vision: EngineRequest#has_vision().
  EngineRequestStatus.is_finished: EngineRequestStatus#is_finished().
  EngineRequest.events: EngineRequest#events.
  EngineRequest.use_structured_output: EngineRequest#use_structured_output.
  EngineRequest.append_output_token_ids: EngineRequest#append_output_token_ids().
  EngineRequest.num_tokens_with_spec: EngineRequest#num_tokens_with_spec().
  EngineRequest.take_events: EngineRequest#take_events().
  EngineRequest.mm_features: EngineRequest#mm_features.
  EngineRequest.__init__: EngineRequest#__init__().
  EngineRequest.sampling_params: EngineRequest#sampling_params.
  EngineRequest.prompt_token_ids: EngineRequest#prompt_token_ids.
  EngineRequest.spec_token_ids: EngineRequest#spec_token_ids.
  EngineRequestStatus.RUNNING: EngineRequestStatus#RUNNING.
  EngineRequest._output_token_ids: EngineRequest#_output_token_ids.
  EngineRequestStatus.WAITING: EngineRequestStatus#WAITING.
  EngineRequestStatus.FINISHED_STOPPED: EngineRequestStatus#FINISHED_STOPPED.
  EngineRequest.num_output_tokens: EngineRequest#num_output_tokens().
  EngineRequest.eos_token_id: EngineRequest#eos_token_id.
  EngineRequest.num_cached_tokens: EngineRequest#num_cached_tokens.
  EngineRequest.pixel_values: EngineRequest#pixel_values.
  EngineRequest.pixel_values_videos: EngineRequest#pixel_values_videos.
  EngineRequestStatus.PREEMPTED: EngineRequestStatus#PREEMPTED.
  EngineRequestStatus.FINISHED_ABORTED: EngineRequestStatus#FINISHED_ABORTED.
  EngineRequest.num_prompt_tokens: EngineRequest#num_prompt_tokens.
  EngineRequest.vision_processed: EngineRequest#vision_processed().
  EngineRequest.is_output_corrupted: EngineRequest#is_output_corrupted().
  EngineRequest.client_index: EngineRequest#client_index.
  EngineRequest.priority: EngineRequest#priority.
  EngineRequest.arrival_time: EngineRequest#arrival_time.
  EngineRequest.stop_reason: EngineRequest#stop_reason.
  EngineRequest.structured_output_request: EngineRequest#structured_output_request.
  EngineRequest.num_nans_in_logits: EngineRequest#num_nans_in_logits.
  EngineRequest.image_grid_thw: EngineRequest#image_grid_thw.
  EngineRequest.video_grid_thw: EngineRequest#video_grid_thw.
  EngineRequest._vision_processed: EngineRequest#_vision_processed.
  EngineRequestStatus.WAITING_FOR_REMOTE_KVS: EngineRequestStatus#WAITING_FOR_REMOTE_KVS.
  EngineRequestStatus.FINISHED_LENGTH_CAPPED: EngineRequestStatus#FINISHED_LENGTH_CAPPED.
  EngineRequest.kv_transfer_params: EngineRequest#kv_transfer_params.
  EngineRequestStatus.WAITING_FOR_FSM: EngineRequestStatus#WAITING_FOR_FSM.
  EngineRequestStatus.FINISHED_IGNORED: EngineRequestStatus#FINISHED_IGNORED.
  EngineRequest.parent_request_id: EngineRequest#parent_request_id.
  EngineRequest.sample_index: EngineRequest#sample_index.
  EngineRequestStatus.__str__: EngineRequestStatus#__str__().
---
# Module: [`easydel/inference/esurge/request.py`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py)

## Classes
### `EngineRequest`
- def: [`easydel/inference/esurge/request.py:63`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L63)
- doc: Request object for tracking generation through the engine.
- signature: `class EngineRequest:`
- members:
  - `__init__(self, request_id: str, prompt_token_ids: list[int], sampling_params: SamplingParams | None, eos_token_id: int | None, client_index: int = 0, arrival_time: float | None = None, priority: int = 0, parent_request_id: str | None = None, sample_index: int = 0, pixel_values: np.ndarray | None = None, image_grid_thw: np.ndarray | None = None, pixel_values_videos: np.ndarray | None = None, video_grid_thw: np.ndarray | None = None, mm_features: list[MultiModalFeature] | None = None)` — [`L109`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L109) — Initialize EngineRequest.
  - `append_output_token_ids(self, token_ids: int | list[int])` — [`L257`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L257) — Append generated token IDs to the output.
  - `clear_vision_data(self)` — [`L201`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L201) — Clear raw vision data after prefill to free memory.
  - `from_engine_core_request(cls, request: EngineCoreRequest)` — [`L229`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L229) — Create an EngineRequest from an EngineCoreRequest.
  - `get_finished_reason(self)` — [`L319`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L319) — Get the reason why generation finished.
  - `has_vision(self)` — [`L193`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L193) — Check if request has vision data (images or videos).
  - `is_finished(self)` — [`L311`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L311) — Check if request has finished processing.
  - `is_output_corrupted(self)` — [`L276`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L276) — Check if output contains NaN values in logits.
  - `num_output_tokens(self)` — [`L303`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L303) — Get count of generated tokens.
  - `num_tokens(self)` — [`L285`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L285) — Get total token count (prompt + generated).
  - `num_tokens_with_spec(self)` — [`L294`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L294) — Get total tokens including speculative tokens.
  - `record_event(self, event_type: EngineCoreEventType, timestamp: float | None = None)` — [`L327`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L327) — Record a processing event for the request.
  - `take_events(self)` — [`L340`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L340) — Take and clear all recorded events.
  - `vision_processed(self)` — [`L220`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L220) — Check if vision data has been processed (prefill complete).
  - `all_token_ids` — [`L178`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L178)
  - `arrival_time` — [`L152`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L152)
  - `client_index` — [`L146`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L146)
  - `eos_token_id` — [`L151`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L151)
  - `events` — [`L155`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L155)
  - `image_grid_thw` — [`L186`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L186)
  - `kv_transfer_params` — [`L158`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L158)
  - `max_tokens` — [`L165`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L165)
  - `mm_features` — [`L189`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L189)
  - `num_cached_tokens` — [`L180`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L180)
  - `num_computed_tokens` — [`L176`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L176)
  - `num_nans_in_logits` — [`L182`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L182)
  - `num_output_placeholders` — [`L174`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L174)
  - `num_prompt_tokens` — [`L171`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L171)
  - `output_token_ids` — [`L177`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L177)
  - `parent_request_id` — [`L148`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L148)
  - `pixel_values` — [`L185`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L185)
  - `pixel_values_videos` — [`L187`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L187)
  - `priority` — [`L147`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L147)
  - `prompt_token_ids` — [`L170`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L170)
  - `request_id` — [`L145`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L145)
  - `sample_index` — [`L149`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L149)
  - `sampling_params` — [`L150`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L150)
  - `spec_token_ids` — [`L175`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L175)
  - `status` — [`L154`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L154)
  - `stop_reason` — [`L156`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L156)
  - `structured_output_request` — [`L159`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L159)
  - `use_structured_output` — [`L160`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L160)
  - `video_grid_thw` — [`L188`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L188)
- protocol/private: `_all_token_ids`[`L173`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L173), `_output_token_ids`[`L172`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L172), `_vision_processed`[`L190`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L190)
- uses (calls/refs, reference-scoped): [`SamplingParams`](../sampling_params.md#SamplingParams), [`EngineRequestStatus`](request.md#EngineRequestStatus), [`max_tokens`](../sampling_params.md#SamplingParams.max_tokens), [`MultiModalFeature`](multimodal/types.md#MultiModalFeature), [`ConstantList`](utils.md#ConstantList), [`EngineCoreEvent`](engine_types.md#EngineCoreEvent), [`get_finished_reason`](request.md#EngineRequestStatus.get_finished_reason), [`FinishReason`](engine_types.md#FinishReason), [`is_finished`](request.md#EngineRequestStatus.is_finished), [`clear_pixel_values`](multimodal/types.md#MultiModalFeature.clear_pixel_values), [`new_event`](engine_types.md#EngineCoreEvent.new_event), [`EngineCoreEventType`](engine_types.md#EngineCoreEventType), [`WAITING`](request.md#EngineRequestStatus.WAITING), [`sampling_params`](engine_types.md#EngineCoreRequest.sampling_params), [`EngineCoreRequest`](engine_types.md#EngineCoreRequest), [`extra_args`](../sampling_params.md#SamplingParams.extra_args), [`arrival_time`](engine_types.md#EngineCoreRequest.arrival_time), [`client_index`](engine_types.md#EngineCoreRequest.client_index), [`eos_token_id`](engine_types.md#EngineCoreRequest.eos_token_id), [`image_grid_thw`](engine_types.md#EngineCoreRequest.image_grid_thw), [`pixel_values`](engine_types.md#EngineCoreRequest.pixel_values), [`pixel_values_videos`](engine_types.md#EngineCoreRequest.pixel_values_videos), [`priority`](engine_types.md#EngineCoreRequest.priority), [`prompt_token_ids`](engine_types.md#EngineCoreRequest.prompt_token_ids), [`request_id`](engine_types.md#EngineCoreRequest.request_id), [`video_grid_thw`](engine_types.md#EngineCoreRequest.video_grid_thw)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`schedule`](scheduler/scheduler.md#Scheduler.schedule), [`update_from_output`](scheduler/scheduler.md#Scheduler.update_from_output), [`_add_request`](mixins/requests.md#EngineRequestsMixin._add_request), [`allocate_slots`](core/manager.md#CacheManager.allocate_slots), [`add_request`](runners/sequence_buffer.md#SequenceBuffer.add_request), [`_process_optional_params`](runners/sequence_buffer.md#SequenceBuffer._process_optional_params), [`check_stop`](scheduler/utils.md#check_stop), [`get_computed_pages`](core/manager.md#CacheManager.get_computed_pages), [`cache_full_pages`](core/page_pool.md#PagePool.cache_full_pages), [`from_request`](scheduler/output.md#NewRequestData.from_request), [`_update_after_schedule`](scheduler/async_scheduler.md#AsyncScheduler._update_after_schedule), [`running`](scheduler/scheduler.md#Scheduler.running), [`_make_cached_request_data`](scheduler/scheduler.md#Scheduler._make_cached_request_data), [`_update_request_with_output`](scheduler/async_scheduler.md#AsyncScheduler._update_request_with_output), [`cache_pages`](core/single_type_cache_manager.md#SingleTypeCacheManager.cache_pages), [`finish_requests`](scheduler/scheduler.md#Scheduler.finish_requests), [`_free_pages`](scheduler/scheduler.md#Scheduler._free_pages), [`_free_request`](scheduler/scheduler.md#Scheduler._free_request), [`get_num_common_prefix_pages`](core/manager.md#CacheManager.get_num_common_prefix_pages), [`_update_after_schedule`](scheduler/scheduler.md#Scheduler._update_after_schedule), [`_heap`](scheduler/request_queue.md#PriorityRequestQueue._heap), [`_can_prefetch_scheduler_output`](mixins/lifecycle.md#EngineLifecycleMixin._can_prefetch_scheduler_output), [`cache_pages`](core/manager.md#CacheManager.cache_pages), [`_infer_dp_shard_from_pages`](scheduler/scheduler.md#Scheduler._infer_dp_shard_from_pages), [`_update_request_with_output`](scheduler/scheduler.md#Scheduler._update_request_with_output), [`hash_request_tokens`](core/utils.md#hash_request_tokens), [`requests`](scheduler/scheduler.md#Scheduler.requests), [`_pick_new_shard`](scheduler/scheduler.md#Scheduler._pick_new_shard), [`_pick_running_shard`](scheduler/scheduler.md#Scheduler._pick_running_shard), [`add_request`](scheduler/scheduler.md#Scheduler.add_request), [`pop_request`](scheduler/request_queue.md#RequestQueue.pop_request), [`prepend_request`](scheduler/request_queue.md#RequestQueue.prepend_request), [`add_request`](scheduler/request_queue.md#PriorityRequestQueue.add_request), [`cache_pages`](core/coordinator.md#CacheCoordinator.cache_pages), [`free`](core/manager.md#CacheManager.free), [`remove_requests`](scheduler/request_queue.md#RequestQueue.remove_requests), [`add_request`](scheduler/request_queue.md#RequestQueue.add_request), [`free_page_hashes`](core/manager.md#CacheManager.free_page_hashes), [`peek_request`](scheduler/request_queue.md#RequestQueue.peek_request)  (+20 more)

### `EngineRequestStatus`
- def: [`easydel/inference/esurge/request.py:353`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L353)
- members:
  - `__str__(self)` — [`L387`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L387) — Return the status name as a string.
  - `get_finished_reason(status: EngineRequestStatus)` — [`L408`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L408) — Get the finish reason for a finished status.
  - `is_finished(status: EngineRequestStatus)` — [`L396`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L396) — Check if a status indicates the request is finished.
  - `FINISHED_ABORTED` — [`L384`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L384)
  - `FINISHED_IGNORED` — [`L385`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L385)
  - `FINISHED_LENGTH_CAPPED` — [`L383`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L383)
  - `FINISHED_STOPPED` — [`L382`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L382)
  - `PREEMPTED` — [`L380`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L380)
  - `RUNNING` — [`L379`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L379)
  - `WAITING` — [`L376`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L376)
  - `WAITING_FOR_FSM` — [`L377`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L377)
  - `WAITING_FOR_REMOTE_KVS` — [`L378`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L378)
- uses (calls/refs, reference-scoped): [`_FINISHED_REASON_MAP`](request.md#_FINISHED_REASON_MAP), [`FinishReason`](engine_types.md#FinishReason)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`schedule`](scheduler/scheduler.md#Scheduler.schedule), [`update_from_output`](scheduler/scheduler.md#Scheduler.update_from_output), [`_process_engine_outputs`](mixins/parsing.md#EngineParsingMixin._process_engine_outputs), [`check_stop`](scheduler/utils.md#check_stop), [`_update_request_with_output`](scheduler/async_scheduler.md#AsyncScheduler._update_request_with_output), [`status`](request.md#EngineRequest.status), [`_FINISHED_REASON_MAP`](request.md#_FINISHED_REASON_MAP), [`abort_request`](mixins/requests.md#EngineRequestsMixin.abort_request), [`finish_requests`](scheduler/scheduler.md#Scheduler.finish_requests), [`get_num_common_prefix_pages`](core/manager.md#CacheManager.get_num_common_prefix_pages), [`get_finished_reason`](request.md#EngineRequest.get_finished_reason), [`is_finished`](request.md#EngineRequest.is_finished), [`finish_requests`](scheduler/interface.md#SchedulerInterface.finish_requests)

## Module values
- `_FINISHED_REASON_MAP` — [`L420`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/request.py#L420)


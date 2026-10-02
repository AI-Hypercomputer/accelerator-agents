---
title: 'Module: easydel/inference/esurge/scheduler/output.py'
type: catalog
provenance: extracted
module: easydel/inference/esurge/scheduler/output.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.esurge.scheduler.output`/
symbols:
  NewRequestData.from_request: NewRequestData#from_request().
  SchedulerOutput: SchedulerOutput#
  NewRequestData.__repr__: NewRequestData#__repr__().
  NewRequestData.anon_repr: NewRequestData#anon_repr().
  SchedulerOutput.num_scheduled_tokens: SchedulerOutput#num_scheduled_tokens.
  NewRequestData.has_vision: NewRequestData#has_vision().
  CachedRequestData: CachedRequestData#
  NewRequestData.sampling_params: NewRequestData#sampling_params.
  NewRequestData: NewRequestData#
  SchedulerOutput.total_num_scheduled_tokens: SchedulerOutput#total_num_scheduled_tokens.
  SchedulerOutput.async_scheduling: SchedulerOutput#async_scheduling.
  SchedulerOutput.scheduled_new_reqs: SchedulerOutput#scheduled_new_reqs.
  SchedulerOutput.scheduled_cached_reqs: SchedulerOutput#scheduled_cached_reqs.
  NewRequestData.mm_features: NewRequestData#mm_features.
  SchedulerOutput.finished_req_ids: SchedulerOutput#finished_req_ids.
  CachedRequestData.num_reqs: CachedRequestData#num_reqs().
  NewRequestData.req_id: NewRequestData#req_id.
  NewRequestData.prompt_token_ids: NewRequestData#prompt_token_ids.
  NewRequestData.page_ids: NewRequestData#page_ids.
  NewRequestData.num_computed_tokens: NewRequestData#num_computed_tokens.
  CachedRequestData.req_ids: CachedRequestData#req_ids.
  SchedulerOutput.preempted_req_ids: SchedulerOutput#preempted_req_ids.
  CachedRequestData.make_empty: CachedRequestData#make_empty().
  NewRequestData.pixel_values: NewRequestData#pixel_values.
  NewRequestData.pixel_values_videos: NewRequestData#pixel_values_videos.
  CachedRequestData.resumed_from_preemption: CachedRequestData#resumed_from_preemption.
  CachedRequestData.new_page_ids: CachedRequestData#new_page_ids.
  CachedRequestData.num_computed_tokens: CachedRequestData#num_computed_tokens.
  SchedulerOutput.scheduled_spec_decode_tokens: SchedulerOutput#scheduled_spec_decode_tokens.
  NewRequestData.image_grid_thw: NewRequestData#image_grid_thw.
  NewRequestData.video_grid_thw: NewRequestData#video_grid_thw.
  CachedRequestData.new_token_ids: CachedRequestData#new_token_ids.
  SchedulerOutput.num_common_prefix_pages: SchedulerOutput#num_common_prefix_pages.
  SchedulerOutput.suggested_bucket: SchedulerOutput#suggested_bucket.
  SchedulerOutput.num_running_reqs: SchedulerOutput#num_running_reqs.
  SchedulerOutput.num_waiting_reqs: SchedulerOutput#num_waiting_reqs.
  SchedulerOutput.free_pages: SchedulerOutput#free_pages.
  SchedulerOutput.token_budget_initial: SchedulerOutput#token_budget_initial.
  SchedulerOutput.token_budget_remaining: SchedulerOutput#token_budget_remaining.
---
# Module: [`easydel/inference/esurge/scheduler/output.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/output.py)

## Classes
### `CachedRequestData`
- def: [`easydel/inference/esurge/scheduler/output.py:202`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/output.py#L202)
- doc: Data structure for cached/running requests in the batch.
- signature: `class CachedRequestData:`
- members:
  - `make_empty(cls)` — [`L253`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/output.py#L253) — Create an empty CachedRequestData instance.
  - `num_reqs(self)` — [`L244`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/output.py#L244) — Get the number of cached requests.
  - `new_page_ids` — [`L237`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/output.py#L237) — ---
  - `new_token_ids` — [`L234`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/output.py#L234) — ---
  - `num_computed_tokens` — [`L240`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/output.py#L240) — ---
  - `req_ids` — [`L228`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/output.py#L228) — ---
  - `resumed_from_preemption` — [`L231`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/output.py#L231) — ---
- used by: [`_execute_model_impl`](../runners/model_runner.md#eSurgeRunner._execute_model_impl), [`_update_states`](../runners/model_runner.md#eSurgeRunner._update_states), [`_make_cached_request_data`](scheduler.md#Scheduler._make_cached_request_data), [`scheduled_cached_reqs`](output.md#SchedulerOutput.scheduled_cached_reqs)

### `NewRequestData`
- def: [`easydel/inference/esurge/scheduler/output.py:53`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/output.py#L53)
- doc: Data structure for new requests being scheduled for the first time.
- signature: `class NewRequestData:`
- members:
  - `__repr__(self)` — [`L158`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/output.py#L158) — Return a detailed string representation.
  - `anon_repr(self)` — [`L175`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/output.py#L175) — Return an anonymized string representation.
  - `from_request(cls, request: EngineRequest, page_ids: tuple[list[int], ...])` — [`L113`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/output.py#L113) — Create a NewRequestData instance from an EngineRequest.
  - `has_vision(self)` — [`L149`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/output.py#L149) — Check if request has vision data.
  - `image_grid_thw` — [`L100`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/output.py#L100) — ---
  - `mm_features` — [`L109`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/output.py#L109) — ---
  - `num_computed_tokens` — [`L93`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/output.py#L93) — ---
  - `page_ids` — [`L90`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/output.py#L90) — ---
  - `pixel_values` — [`L97`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/output.py#L97) — ---
  - `pixel_values_videos` — [`L103`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/output.py#L103) — ---
  - `prompt_token_ids` — [`L84`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/output.py#L84) — ---
  - `req_id` — [`L81`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/output.py#L81) — ---
  - `sampling_params` — [`L87`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/output.py#L87) — ---
  - `video_grid_thw` — [`L106`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/output.py#L106) — ---
- uses (calls/refs, reference-scoped): [`SamplingParams`](../../sampling_params.md#SamplingParams), [`EngineRequest`](../request.md#EngineRequest), [`request_id`](../request.md#EngineRequest.request_id), [`num_computed_tokens`](../request.md#EngineRequest.num_computed_tokens), [`MultiModalFeature`](../multimodal/types.md#MultiModalFeature), [`mm_features`](../request.md#EngineRequest.mm_features), [`prompt_token_ids`](../request.md#EngineRequest.prompt_token_ids), [`sampling_params`](../request.md#EngineRequest.sampling_params), [`pixel_values`](../request.md#EngineRequest.pixel_values), [`pixel_values_videos`](../request.md#EngineRequest.pixel_values_videos), [`image_grid_thw`](../request.md#EngineRequest.image_grid_thw), [`video_grid_thw`](../request.md#EngineRequest.video_grid_thw)
- used by: [`schedule`](scheduler.md#Scheduler.schedule), [`_update_states`](../runners/model_runner.md#eSurgeRunner._update_states), [`scheduled_new_reqs`](output.md#SchedulerOutput.scheduled_new_reqs)

### `SchedulerOutput`
- def: [`easydel/inference/esurge/scheduler/output.py:277`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/output.py#L277)
- doc: Complete output from a scheduling decision.
- signature: `class SchedulerOutput:`
- members:
  - `async_scheduling` — [`L333`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/output.py#L333) — ---
  - `finished_req_ids` — [`L324`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/output.py#L324) — ---
  - `free_pages` — [`L342`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/output.py#L342) — ---
  - `num_common_prefix_pages` — [`L321`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/output.py#L321) — ---
  - `num_running_reqs` — [`L336`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/output.py#L336) — ---
  - `num_scheduled_tokens` — [`L312`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/output.py#L312) — ---
  - `num_waiting_reqs` — [`L339`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/output.py#L339) — ---
  - `preempted_req_ids` — [`L327`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/output.py#L327) — ---
  - `scheduled_cached_reqs` — [`L309`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/output.py#L309) — ---
  - `scheduled_new_reqs` — [`L306`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/output.py#L306) — ---
  - `scheduled_spec_decode_tokens` — [`L318`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/output.py#L318) — ---
  - `suggested_bucket` — [`L330`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/output.py#L330) — ---
  - `token_budget_initial` — [`L345`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/output.py#L345) — ---
  - `token_budget_remaining` — [`L348`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/output.py#L348) — ---
  - `total_num_scheduled_tokens` — [`L315`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/output.py#L315) — ---
- uses (calls/refs, reference-scoped): [`CachedRequestData`](output.md#CachedRequestData), [`NewRequestData`](output.md#NewRequestData)
- used by: [`_execute_model_impl`](../runners/model_runner.md#eSurgeRunner._execute_model_impl), [`schedule`](scheduler.md#Scheduler.schedule), [`_update_states`](../runners/model_runner.md#eSurgeRunner._update_states), [`update_from_output`](scheduler.md#Scheduler.update_from_output), [`_scheduler_loop`](../mixins/lifecycle.md#EngineLifecycleMixin._scheduler_loop), [`_update_after_schedule`](async_scheduler.md#AsyncScheduler._update_after_schedule), [`_reorder_decode_first_per_shard`](../runners/model_runner.md#eSurgeRunner._reorder_decode_first_per_shard), [`_reorder_decode_first`](../runners/model_runner.md#eSurgeRunner._reorder_decode_first), [`_update_after_schedule`](scheduler.md#Scheduler._update_after_schedule), [`_can_prefetch_scheduler_output`](../mixins/lifecycle.md#EngineLifecycleMixin._can_prefetch_scheduler_output), [`_drain_runner_future`](../mixins/lifecycle.md#EngineLifecycleMixin._drain_runner_future), [`update_from_output`](interface.md#SchedulerInterface.update_from_output), [`_can_prefetch_next`](../mixins/lifecycle.md#EngineLifecycleMixin._can_prefetch_next), [`execute_model`](../runners/model_runner.md#eSurgeRunner.execute_model), [`execute_model_async`](../runners/model_runner.md#eSurgeRunner.execute_model_async), [`schedule`](interface.md#SchedulerInterface.schedule)


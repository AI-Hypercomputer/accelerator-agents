---
title: 'Module: easydel/inference/esurge/scheduler/scheduler.py'
type: catalog
provenance: extracted
module: easydel/inference/esurge/scheduler/scheduler.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.esurge.scheduler.scheduler`/
symbols:
  Scheduler.schedule: Scheduler#schedule().
  Scheduler.update_from_output: Scheduler#update_from_output().
  Scheduler.from_runner: Scheduler#from_runner().
  Scheduler.kv_cache_manager: Scheduler#kv_cache_manager.
  Scheduler.running: Scheduler#running.
  Scheduler._token_budget_manager: Scheduler#_token_budget_manager.
  Scheduler._make_cached_request_data: Scheduler#_make_cached_request_data().
  Scheduler.finish_requests: Scheduler#finish_requests().
  Scheduler.policy: Scheduler#policy.
  Scheduler.waiting: Scheduler#waiting.
  Scheduler._free_pages: Scheduler#_free_pages().
  Scheduler._free_request: Scheduler#_free_request().
  Scheduler.scheduler_config: Scheduler#scheduler_config.
  Scheduler.max_num_scheduled_tokens: Scheduler#max_num_scheduled_tokens.
  Scheduler._update_after_schedule: Scheduler#_update_after_schedule().
  Scheduler: Scheduler#
  Scheduler.num_lookahead_tokens: Scheduler#num_lookahead_tokens.
  Scheduler.requests: Scheduler#requests.
  Scheduler._infer_dp_shard_from_pages: Scheduler#_infer_dp_shard_from_pages().
  Scheduler._update_request_with_output: Scheduler#_update_request_with_output().
  Scheduler.max_model_len: Scheduler#max_model_len.
  Scheduler._pick_running_shard: Scheduler#_pick_running_shard().
  Scheduler._pick_new_shard: Scheduler#_pick_new_shard().
  Scheduler.add_request: Scheduler#add_request().
  Scheduler.max_num_running_reqs: Scheduler#max_num_running_reqs.
  Scheduler.page_size: Scheduler#page_size.
  Scheduler.data_parallel_size: Scheduler#data_parallel_size.
  Scheduler._ensure_capacity: Scheduler#_ensure_capacity().
  Scheduler.req_id_to_row_index: Scheduler#req_id_to_row_index.
  Scheduler.cache_config: Scheduler#cache_config.
  Scheduler._current_seq_bucket: Scheduler#_current_seq_bucket.
  Scheduler.__init__: Scheduler#__init__().
  Scheduler.get_request_counts: Scheduler#get_request_counts().
  Scheduler.get_num_unfinished_requests: Scheduler#get_num_unfinished_requests().
  Scheduler.reset_prefix_cache: Scheduler#reset_prefix_cache().
  Scheduler._select_seq_bucket: Scheduler#_select_seq_bucket().
  logger: logger.
  Scheduler.finished_req_ids: Scheduler#finished_req_ids.
  Scheduler.max_num_seq_buckets: Scheduler#max_num_seq_buckets.
  Scheduler.num_spec_tokens: Scheduler#num_spec_tokens.
  Scheduler.finished_req_ids_dict: Scheduler#finished_req_ids_dict.
  Scheduler.finished_recving_kv_req_ids: Scheduler#finished_recving_kv_req_ids.
  Scheduler._dp_shard_hint_for_row: Scheduler#_dp_shard_hint_for_row().
  Scheduler.has_finished_requests: Scheduler#has_finished_requests().
  Scheduler.use_eagle: Scheduler#use_eagle.
  Scheduler._row_to_dp_shard: Scheduler#_row_to_dp_shard().
  Scheduler.kv_cache_config: Scheduler#kv_cache_config.
  Scheduler._reserve_new_shard: Scheduler#_reserve_new_shard().
  Scheduler.config: Scheduler#config.
  Scheduler.shutdown: Scheduler#shutdown().
---
# Module: [`easydel/inference/esurge/scheduler/scheduler.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py)

## Classes
### `Scheduler`  ·  implements/extends SchedulerInterface
- def: [`easydel/inference/esurge/scheduler/scheduler.py:87`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py#L87)
- doc: Main request scheduler for the eSurge inference engine.
- signature: `class Scheduler(SchedulerInterface):`
- members:
  - `__init__(self, config: Config, kv_cache_config: CacheGroupsConfig, include_finished_set: bool = False, max_num_seq_buckets: list[int] | None = None)` — [`L135`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py#L135) — Initialize the Scheduler with configuration.
  - `_dp_shard_hint_for_row(self, row_index: int)` — [`L246`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py#L246) — Compute a DP shard hint for a logical request row index.
  - `_ensure_capacity(self, desired_running: int)` — [`L357`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py#L357) — Ensure capacity for the desired number of running requests.
  - `_free_pages(self, request: EngineRequest)` — [`L1123`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py#L1123) — Free KV cache pages for a finished request.
  - `_free_request(self, request: EngineRequest)` — [`L1104`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py#L1104) — Free resources associated with a finished request.
  - `_make_cached_request_data(self, running_reqs: list[EngineRequest], resumed_reqs: list[EngineRequest], num_scheduled_tokens: dict[str, int], spec_decode_tokens: dict[str, list[int]], req_to_new_page_ids: dict[str, tuple[list[int], ...]])` — [`L853`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py#L853) — Build CachedRequestData from running and resumed requests.
  - `_pick_new_shard(request: EngineRequest)` — [`L472`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py#L472) — Shard hint for a NEW/WAITING request — balanced distribution.
  - `_pick_running_shard(request: EngineRequest)` — [`L460`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py#L460) — Shard hint for a RUNNING request — always its existing shard.
  - `_reserve_new_shard(shard_hint: int | None)` — [`L487`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py#L487) — Increment shard occupancy when a new request is assigned.
  - `_select_seq_bucket(self, num_reqs: int)` — [`L337`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py#L337) — Select the appropriate sequence bucket for the given request count.
  - `_update_after_schedule(self, scheduler_output: SchedulerOutput)` — [`L831`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py#L831) — Update internal state after scheduling completes.
  - `_update_request_with_output(self, request: EngineRequest, new_token_ids: list[int])` — [`L1009`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py#L1009) — Update a request with newly generated tokens.
  - `add_request(self, request: EngineRequest)` — [`L1045`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py#L1045) — Add a new request to the scheduler.
  - `finish_requests(self, request_ids: str | Iterable[str], finished_status: EngineRequestStatus)` — [`L1057`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py#L1057) — Mark requests as finished from external signal.
  - `from_runner(cls, runner: eSurgeRunner, max_num_batched_tokens: int | None = None, enable_prefix_caching: bool = True, async_scheduling: bool = True, long_prefill_token_threshold: int | None = None)` — [`L256`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py#L256) — Create a Scheduler instance from an eSurgeRunner.
  - `get_num_unfinished_requests(self)` — [`L1139`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py#L1139) — Get the total number of unfinished requests.
  - `get_request_counts(self)` — [`L1037`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py#L1037) — Get the counts of running and waiting requests.
  - `has_finished_requests(self)` — [`L1147`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py#L1147) — Check if there are finished requests pending notification.
  - `reset_prefix_cache(self)` — [`L1155`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py#L1155) — Reset the prefix cache.
  - `schedule(self)` — [`L374`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py#L374) — Schedule requests for the next forward pass.
  - `shutdown(self)` — [`L1165`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py#L1165) — Shutdown the scheduler.
  - `update_from_output(self, scheduler_output: SchedulerOutput, model_runner_output: ModelRunnerOutput)` — [`L902`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py#L902) — Update scheduler state based on model runner output.
  - `cache_config` — [`L171`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py#L171)
  - `config` — [`L169`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py#L169)
  - `data_parallel_size` — [`L183`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py#L183)
  - `finished_recving_kv_req_ids` — [`L216`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py#L216)
  - `finished_req_ids` — [`L214`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py#L214)
  - `finished_req_ids_dict` — [`L174`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py#L174)
  - `kv_cache_config` — [`L172`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py#L172)
  - `kv_cache_manager` — [`L227`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py#L227)
  - `max_model_len` — [`L178`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py#L178)
  - `max_num_running_reqs` — [`L176`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py#L176)
  - `max_num_scheduled_tokens` — [`L177`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py#L177)
  - `max_num_seq_buckets` — [`L243`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py#L243)
  - `num_lookahead_tokens` — [`L220`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py#L220)
  - `num_spec_tokens` — [`L220`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py#L220)
  - `page_size` — [`L188`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py#L188)
  - `policy` — [`L202`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py#L202)
  - `req_id_to_row_index` — [`L212`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py#L212)
  - `requests` — [`L199`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py#L199)
  - `running` — [`L209`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py#L209)
  - `scheduler_config` — [`L170`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py#L170)
  - `use_eagle` — [`L219`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py#L219)
  - `waiting` — [`L208`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py#L208)
- protocol/private: `_current_seq_bucket`[`L244`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py#L244), `_infer_dp_shard_from_pages`[`L442`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py#L442), `_row_to_dp_shard`[`L437`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py#L437), `_token_budget_manager`[`L191`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py#L191)
- uses (calls/refs, reference-scoped): [`config`](../../../infra/base_module.md#EasyDeLBaseModule.config), [`EngineRequest`](../request.md#EngineRequest), [`create_kv_cache_specs_from_config`](../core/interface.md#create_kv_cache_specs_from_config), [`allocate_slots`](../core/manager.md#CacheManager.allocate_slots), [`metadata`](../runners/model_runner.md#eSurgeRunner.metadata), [`request_id`](../request.md#EngineRequest.request_id), [`EngineRequestStatus`](../request.md#EngineRequestStatus), [`check_stop`](utils.md#check_stop), [`model`](../runners/model_runner.md#eSurgeRunner.model), [`get_computed_pages`](../core/manager.md#CacheManager.get_computed_pages), [`from_request`](output.md#NewRequestData.from_request), [`_update_after_schedule`](async_scheduler.md#AsyncScheduler._update_after_schedule), [`record_cache_metrics`](../metrics.md#MetricsCollector.record_cache_metrics), [`record_scheduler_metrics`](../metrics.md#MetricsCollector.record_scheduler_metrics), [`SchedulerOutput`](output.md#SchedulerOutput), [`_update_request_with_output`](async_scheduler.md#AsyncScheduler._update_request_with_output), [`status`](../request.md#EngineRequest.status), [`get_metrics_collector`](../metrics.md#get_metrics_collector), [`begin_cycle`](token_budget.md#TokenBudgetManager.begin_cycle), [`get_num_common_prefix_pages`](../core/manager.md#CacheManager.get_num_common_prefix_pages), [`create_request_queue`](request_queue.md#create_request_queue), [`num_computed_tokens`](../request.md#EngineRequest.num_computed_tokens), [`max_model_len`](../runners/model_runner.md#eSurgeRunner.max_model_len), [`pages_per_dp_shard`](../core/dp_sharding.md#pages_per_dp_shard), [`ModelRunnerOutput`](../outputs.md#ModelRunnerOutput), [`page_pool`](../core/manager.md#CacheManager.page_pool), [`num_scheduled_tokens`](output.md#SchedulerOutput.num_scheduled_tokens), [`pop_request`](request_queue.md#RequestQueue.pop_request), [`prepend_request`](request_queue.md#RequestQueue.prepend_request), [`free`](../core/manager.md#CacheManager.free), [`get_num_free_pages`](../core/page_pool.md#PagePool.get_num_free_pages), [`get_page_ids`](../core/manager.md#CacheManager.get_page_ids), [`long_prefill_token_threshold`](../config.md#SchedulerConfig.long_prefill_token_threshold), [`num_tokens`](../request.md#EngineRequest.num_tokens), [`cached_page_hash_to_page`](../core/page_pool.md#PagePool.cached_page_hash_to_page), [`get_finished_reason`](../request.md#EngineRequest.get_finished_reason), [`max_num_batched_tokens`](../config.md#SchedulerConfig.max_num_batched_tokens), [`is_finished`](../request.md#EngineRequest.is_finished), [`remove_requests`](request_queue.md#RequestQueue.remove_requests), [`page_size`](../../../caching/unified_attention/cache.md#UnifiedAttentionCacheConfig.page_size)  (+106 more)
- used by: [`_scheduler_loop`](../mixins/lifecycle.md#EngineLifecycleMixin._scheduler_loop), [`_update_after_schedule`](async_scheduler.md#AsyncScheduler._update_after_schedule), [`_eos_set`](../esurge_engine.md#eSurge._eos_set), [`_update_request_with_output`](async_scheduler.md#AsyncScheduler._update_request_with_output), [`scheduler`](../mixins/lifecycle.md#EngineLifecycleMixin.scheduler), [`_can_prefetch_scheduler_output`](../mixins/lifecycle.md#EngineLifecycleMixin._can_prefetch_scheduler_output), [`scheduler`](../esurge_engine.md#eSurge.scheduler), [`_drain_runner_future`](../mixins/lifecycle.md#EngineLifecycleMixin._drain_runner_future), [`update_from_output`](interface.md#SchedulerInterface.update_from_output), [`SchedulerInterface`](interface.md#SchedulerInterface), [`AsyncScheduler`](async_scheduler.md#AsyncScheduler), [`add_request`](interface.md#SchedulerInterface.add_request), [`finish_requests`](interface.md#SchedulerInterface.finish_requests), [`schedule`](interface.md#SchedulerInterface.schedule), [`get_num_unfinished_requests`](interface.md#SchedulerInterface.get_num_unfinished_requests), [`has_finished_requests`](interface.md#SchedulerInterface.has_finished_requests), [`get_request_counts`](interface.md#SchedulerInterface.get_request_counts), [`reset_prefix_cache`](interface.md#SchedulerInterface.reset_prefix_cache), [`shutdown`](interface.md#SchedulerInterface.shutdown)

## Module values
- `logger` — [`L84`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/scheduler.py#L84)


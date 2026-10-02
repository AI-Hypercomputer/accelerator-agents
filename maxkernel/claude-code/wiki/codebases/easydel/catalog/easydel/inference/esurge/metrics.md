---
title: 'Module: easydel/inference/esurge/metrics.py'
type: catalog
provenance: extracted
module: easydel/inference/esurge/metrics.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.esurge.metrics`/
symbols:
  MetricsCollector.get_system_metrics: MetricsCollector#get_system_metrics().
  MetricsCollector.complete_request: MetricsCollector#complete_request().
  MetricsCollector.record_scheduler_metrics: MetricsCollector#record_scheduler_metrics().
  MetricsCollector.record_cache_metrics: MetricsCollector#record_cache_metrics().
  MetricsCollector.log_summary: MetricsCollector#log_summary().
  MetricsCollector.record_runner_metrics: MetricsCollector#record_runner_metrics().
  get_metrics_collector: get_metrics_collector().
  MetricsCollector.start_request: MetricsCollector#start_request().
  MetricsCollector.reset_metrics: MetricsCollector#reset_metrics().
  MetricsCollector.export_metrics: MetricsCollector#export_metrics().
  MetricsCollector._lock: MetricsCollector#_lock.
  MetricsCollector.scheduler_metrics: MetricsCollector#scheduler_metrics.
  MetricsCollector.runner_metrics: MetricsCollector#runner_metrics.
  MetricsCollector.record_first_token: MetricsCollector#record_first_token().
  MetricsCollector.add_generated_tokens: MetricsCollector#add_generated_tokens().
  MetricsCollector.request_metrics: MetricsCollector#request_metrics.
  MetricsCollector: MetricsCollector#
  MetricsCollector.cache_metrics: MetricsCollector#cache_metrics.
  MetricsCollector.completed_requests: MetricsCollector#completed_requests.
  RequestMetrics.time_to_first_token: RequestMetrics#time_to_first_token.
  MetricsCollector.counters: MetricsCollector#counters.
  _global_metrics_collector._global_metrics_collector: _global_metrics_collector._global_metrics_collector.
  RequestMetrics.generated_tokens: RequestMetrics#generated_tokens.
  RequestMetrics.total_latency: RequestMetrics#total_latency.
  initialize_metrics: initialize_metrics().
  log_metrics_summary: log_metrics_summary().
  MetricsCollector.logger: MetricsCollector#logger.
  MetricsCollector.record_cache_event: MetricsCollector#record_cache_event().
  RequestMetrics.error: RequestMetrics#error.
  CacheMetrics.total_pages: CacheMetrics#total_pages.
  SystemMetrics.average_throughput: SystemMetrics#average_throughput.
  RequestMetrics: RequestMetrics#
  RequestMetrics.end_time: RequestMetrics#end_time.
  SchedulerMetrics.num_running_requests: SchedulerMetrics#num_running_requests.
  CacheMetrics.used_pages: CacheMetrics#used_pages.
  SystemMetrics: SystemMetrics#
  RequestMetrics.start_time: RequestMetrics#start_time.
  RequestMetrics.tokens_per_second: RequestMetrics#tokens_per_second.
  SchedulerMetrics: SchedulerMetrics#
  SchedulerMetrics.num_waiting_requests: SchedulerMetrics#num_waiting_requests.
  SchedulerMetrics.num_scheduled_tokens: SchedulerMetrics#num_scheduled_tokens.
  ModelRunnerMetrics: ModelRunnerMetrics#
  ModelRunnerMetrics.batch_size: ModelRunnerMetrics#batch_size.
  ModelRunnerMetrics.tokens_per_second: ModelRunnerMetrics#tokens_per_second.
  CacheMetrics: CacheMetrics#
  CacheMetrics.cache_hit_rate: CacheMetrics#cache_hit_rate.
  SystemMetrics.total_requests_completed: SystemMetrics#total_requests_completed.
  SystemMetrics.total_requests_failed: SystemMetrics#total_requests_failed.
  SystemMetrics.total_tokens_generated: SystemMetrics#total_tokens_generated.
  SystemMetrics.average_latency: SystemMetrics#average_latency.
  SystemMetrics.average_ttft: SystemMetrics#average_ttft.
  SystemMetrics.requests_per_second: SystemMetrics#requests_per_second.
  RequestMetrics.request_id: RequestMetrics#request_id.
  RequestMetrics.prompt_tokens: RequestMetrics#prompt_tokens.
  RequestMetrics.total_tokens: RequestMetrics#total_tokens.
  SchedulerMetrics.batch_size: SchedulerMetrics#batch_size.
  SchedulerMetrics.schedule_time: SchedulerMetrics#schedule_time.
  ModelRunnerMetrics.execution_time: ModelRunnerMetrics#execution_time.
  ModelRunnerMetrics.num_tokens: ModelRunnerMetrics#num_tokens.
  CacheMetrics.free_pages: CacheMetrics#free_pages.
  SystemMetrics.timestamp: SystemMetrics#timestamp.
  MetricsCollector.last_log_time: MetricsCollector#last_log_time.
  RequestMetrics.first_token_time: RequestMetrics#first_token_time.
  RequestMetrics.finish_reason: RequestMetrics#finish_reason.
  SchedulerMetrics.timestamp: SchedulerMetrics#timestamp.
  SchedulerMetrics.num_preempted_requests: SchedulerMetrics#num_preempted_requests.
  ModelRunnerMetrics.timestamp: ModelRunnerMetrics#timestamp.
  ModelRunnerMetrics.memory_usage: ModelRunnerMetrics#memory_usage.
  CacheMetrics.timestamp: CacheMetrics#timestamp.
  CacheMetrics.page_allocation_rate: CacheMetrics#page_allocation_rate.
  CacheMetrics.page_free_rate: CacheMetrics#page_free_rate.
  MetricsCollector.log_interval: MetricsCollector#log_interval.
  MetricsCollector.enable_detailed_logging: MetricsCollector#enable_detailed_logging.
  MetricsCollector.timers: MetricsCollector#timers.
  MetricsCollector.__init__: MetricsCollector#__init__().
  MetricsCollector.log_file: MetricsCollector#log_file.
  MetricsCollector.history_size: MetricsCollector#history_size.
---
# Module: [`easydel/inference/esurge/metrics.py`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py)

## Classes
### `CacheMetrics`
- def: [`easydel/inference/esurge/metrics.py:178`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L178)
- doc: Metrics for KV cache operations.
- signature: `class CacheMetrics:`
- members:
  - `cache_hit_rate` — [`L206`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L206)
  - `free_pages` — [`L205`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L205)
  - `page_allocation_rate` — [`L207`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L207)
  - `page_free_rate` — [`L208`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L208)
  - `timestamp` — [`L202`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L202)
  - `total_pages` — [`L203`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L203)
  - `used_pages` — [`L204`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L204)
- used by: [`update_from_metrics_collector`](monitoring.md#PrometheusMetrics.update_from_metrics_collector), [`record_cache_metrics`](metrics.md#MetricsCollector.record_cache_metrics), [`_create_cache_metrics_table`](monitoring.md#RichConsoleMonitor._create_cache_metrics_table), [`cache_metrics`](metrics.md#MetricsCollector.cache_metrics)

### `MetricsCollector`
- def: [`easydel/inference/esurge/metrics.py:247`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L247)
- doc: Centralized metrics collection and logging system for eSurge.
- signature: `class MetricsCollector:`
- members:
  - `__init__(self, log_file: str | None = None, log_interval: float = 10, history_size: int = 1000, enable_detailed_logging: bool = True)` — [`L283`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L283) — Initialize the metrics collector.
  - `add_generated_tokens(self, request_id: str, num_tokens: int)` — [`L368`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L368) — Add generated tokens to a request's metrics.
  - `complete_request(self, request_id: str, finish_reason: str | None = None, error: str | None = None)` — [`L385`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L385) — Complete tracking for a request.
  - `export_metrics(self, file_path: str, format: str = "json")` — [`L631`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L631) — Export all metrics to a file.
  - `get_system_metrics(self, window_seconds: float = 60)` — [`L543`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L543) — Get aggregated system metrics for the specified time window.
  - `log_summary(self, force: bool = False)` — [`L593`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L593) — Log a summary of current metrics.
  - `record_cache_event(self, event: str, details: dict[str, Any] | None = None)` — [`L530`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L530) — Record lifecycle events for the KV cache.
  - `record_cache_metrics(self, total_pages: int, used_pages: int, cache_hit_rate: float = 0, page_allocation_rate: float = 0, page_free_rate: float = 0)` — [`L497`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L497) — Record KV cache metrics.
  - `record_first_token(self, request_id: str)` — [`L351`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L351) — Record when the first token is generated for a request.
  - `record_runner_metrics(self, execution_time: float, batch_size: int, num_tokens: int, memory_usage: dict[str, Any] | None = None)` — [`L466`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L466) — Record model runner performance metrics.
  - `record_scheduler_metrics(self, num_waiting: int, num_running: int, num_scheduled_tokens: int, num_preempted: int = 0, batch_size: int = 0, schedule_time: float = 0)` — [`L432`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L432) — Record scheduler performance metrics.
  - `reset_metrics(self)` — [`L663`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L663) — Reset all metrics and counters.
  - `start_request(self, request_id: str, prompt_tokens: int = 0)` — [`L331`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L331) — Start tracking metrics for a new request.
  - `cache_metrics` — [`L314`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L314)
  - `completed_requests` — [`L311`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L311)
  - `counters` — [`L317`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L317)
  - `enable_detailed_logging` — [`L304`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L304)
  - `history_size` — [`L303`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L303)
  - `last_log_time` — [`L321`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L321)
  - `log_file` — [`L301`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L301)
  - `log_interval` — [`L302`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L302)
  - `logger` — [`L324`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L324)
  - `request_metrics` — [`L310`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L310)
  - `runner_metrics` — [`L313`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L313)
  - `scheduler_metrics` — [`L312`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L312)
  - `timers` — [`L318`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L318)
- protocol/private: `_lock`[`L307`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L307)
- uses (calls/refs, reference-scoped): [`time_to_first_token`](metrics.md#RequestMetrics.time_to_first_token), [`generated_tokens`](metrics.md#RequestMetrics.generated_tokens), [`total_latency`](metrics.md#RequestMetrics.total_latency), [`average_throughput`](metrics.md#SystemMetrics.average_throughput), [`error`](metrics.md#RequestMetrics.error), [`total_pages`](metrics.md#CacheMetrics.total_pages), [`RequestMetrics`](metrics.md#RequestMetrics), [`SystemMetrics`](metrics.md#SystemMetrics), [`end_time`](metrics.md#RequestMetrics.end_time), [`num_running_requests`](metrics.md#SchedulerMetrics.num_running_requests), [`used_pages`](metrics.md#CacheMetrics.used_pages), [`CacheMetrics`](metrics.md#CacheMetrics), [`ModelRunnerMetrics`](metrics.md#ModelRunnerMetrics), [`SchedulerMetrics`](metrics.md#SchedulerMetrics), [`average_latency`](metrics.md#SystemMetrics.average_latency), [`average_ttft`](metrics.md#SystemMetrics.average_ttft), [`batch_size`](metrics.md#ModelRunnerMetrics.batch_size), [`cache_hit_rate`](metrics.md#CacheMetrics.cache_hit_rate), [`num_scheduled_tokens`](metrics.md#SchedulerMetrics.num_scheduled_tokens), [`num_waiting_requests`](metrics.md#SchedulerMetrics.num_waiting_requests), [`requests_per_second`](metrics.md#SystemMetrics.requests_per_second), [`start_time`](metrics.md#RequestMetrics.start_time), [`tokens_per_second`](metrics.md#ModelRunnerMetrics.tokens_per_second), [`tokens_per_second`](metrics.md#RequestMetrics.tokens_per_second), [`total_requests_completed`](metrics.md#SystemMetrics.total_requests_completed), [`total_requests_failed`](metrics.md#SystemMetrics.total_requests_failed), [`total_tokens_generated`](metrics.md#SystemMetrics.total_tokens_generated), [`batch_size`](metrics.md#SchedulerMetrics.batch_size), [`execution_time`](metrics.md#ModelRunnerMetrics.execution_time), [`free_pages`](metrics.md#CacheMetrics.free_pages), [`num_tokens`](metrics.md#ModelRunnerMetrics.num_tokens), [`prompt_tokens`](metrics.md#RequestMetrics.prompt_tokens), [`request_id`](metrics.md#RequestMetrics.request_id), [`schedule_time`](metrics.md#SchedulerMetrics.schedule_time), [`timestamp`](metrics.md#SystemMetrics.timestamp), [`total_tokens`](metrics.md#RequestMetrics.total_tokens), [`finish_reason`](metrics.md#RequestMetrics.finish_reason), [`first_token_time`](metrics.md#RequestMetrics.first_token_time), [`memory_usage`](metrics.md#ModelRunnerMetrics.memory_usage), [`num_preempted_requests`](metrics.md#SchedulerMetrics.num_preempted_requests)  (+5 more)
- used by: [`_execute_model_impl`](runners/model_runner.md#eSurgeRunner._execute_model_impl), [`schedule`](scheduler/scheduler.md#Scheduler.schedule), [`update_from_metrics_collector`](monitoring.md#PrometheusMetrics.update_from_metrics_collector), [`_add_request`](mixins/requests.md#EngineRequestsMixin._add_request), [`_process_engine_outputs`](mixins/parsing.md#EngineParsingMixin._process_engine_outputs), [`_create_system_metrics_table`](monitoring.md#RichConsoleMonitor._create_system_metrics_table), [`get_metrics_collector`](metrics.md#get_metrics_collector), [`abort_request`](mixins/requests.md#EngineRequestsMixin.abort_request), [`get_metrics_summary`](mixins/monitoring.md#EngineMonitoringMixin.get_metrics_summary), [`_create_recent_requests_table`](monitoring.md#RichConsoleMonitor._create_recent_requests_table), [`_create_scheduler_metrics_table`](monitoring.md#RichConsoleMonitor._create_scheduler_metrics_table), [`_create_cache_metrics_table`](monitoring.md#RichConsoleMonitor._create_cache_metrics_table), [`_create_runner_metrics_table`](monitoring.md#RichConsoleMonitor._create_runner_metrics_table), [`_global_metrics_collector`](metrics.md#_global_metrics_collector._global_metrics_collector), [`initialize_metrics`](metrics.md#initialize_metrics), [`log_metrics_summary`](metrics.md#log_metrics_summary), [`_record_cache_event`](mixins/utils.md#EngineUtilsMixin._record_cache_event)

### `ModelRunnerMetrics`
- def: [`easydel/inference/esurge/metrics.py:146`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L146)
- doc: Metrics for model runner operations.
- signature: `class ModelRunnerMetrics:`
- members:
  - `batch_size` — [`L171`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L171)
  - `execution_time` — [`L170`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L170)
  - `memory_usage` — [`L174`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L174)
  - `num_tokens` — [`L172`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L172)
  - `timestamp` — [`L169`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L169)
  - `tokens_per_second` — [`L173`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L173)
- used by: [`update_from_metrics_collector`](monitoring.md#PrometheusMetrics.update_from_metrics_collector), [`record_runner_metrics`](metrics.md#MetricsCollector.record_runner_metrics), [`_create_runner_metrics_table`](monitoring.md#RichConsoleMonitor._create_runner_metrics_table), [`runner_metrics`](metrics.md#MetricsCollector.runner_metrics)

### `RequestMetrics`
- def: [`easydel/inference/esurge/metrics.py:70`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L70)
- doc: Metrics for a single request.
- signature: `class RequestMetrics:`
- members:
  - `end_time` — [`L100`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L100)
  - `error` — [`L109`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L109)
  - `finish_reason` — [`L108`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L108)
  - `first_token_time` — [`L101`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L101)
  - `generated_tokens` — [`L103`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L103)
  - `prompt_tokens` — [`L102`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L102)
  - `request_id` — [`L98`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L98)
  - `start_time` — [`L99`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L99)
  - `time_to_first_token` — [`L106`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L106)
  - `tokens_per_second` — [`L105`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L105)
  - `total_latency` — [`L107`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L107)
  - `total_tokens` — [`L104`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L104)
- used by: [`update_from_metrics_collector`](monitoring.md#PrometheusMetrics.update_from_metrics_collector), [`get_system_metrics`](metrics.md#MetricsCollector.get_system_metrics), [`complete_request`](metrics.md#MetricsCollector.complete_request), [`_create_recent_requests_table`](monitoring.md#RichConsoleMonitor._create_recent_requests_table), [`start_request`](metrics.md#MetricsCollector.start_request), [`add_generated_tokens`](metrics.md#MetricsCollector.add_generated_tokens), [`record_first_token`](metrics.md#MetricsCollector.record_first_token), [`request_metrics`](metrics.md#MetricsCollector.request_metrics), [`completed_requests`](metrics.md#MetricsCollector.completed_requests)

### `SchedulerMetrics`
- def: [`easydel/inference/esurge/metrics.py:113`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L113)
- doc: Metrics for scheduler operations.
- signature: `class SchedulerMetrics:`
- members:
  - `batch_size` — [`L141`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L141)
  - `num_preempted_requests` — [`L140`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L140)
  - `num_running_requests` — [`L138`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L138)
  - `num_scheduled_tokens` — [`L139`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L139)
  - `num_waiting_requests` — [`L137`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L137)
  - `schedule_time` — [`L142`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L142)
  - `timestamp` — [`L136`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L136)
- used by: [`update_from_metrics_collector`](monitoring.md#PrometheusMetrics.update_from_metrics_collector), [`record_scheduler_metrics`](metrics.md#MetricsCollector.record_scheduler_metrics), [`_create_scheduler_metrics_table`](monitoring.md#RichConsoleMonitor._create_scheduler_metrics_table), [`scheduler_metrics`](metrics.md#MetricsCollector.scheduler_metrics)

### `SystemMetrics`
- def: [`easydel/inference/esurge/metrics.py:212`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L212)
- doc: System-wide metrics summary.
- signature: `class SystemMetrics:`
- members:
  - `average_latency` — [`L241`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L241)
  - `average_throughput` — [`L243`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L243)
  - `average_ttft` — [`L242`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L242)
  - `requests_per_second` — [`L244`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L244)
  - `timestamp` — [`L237`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L237)
  - `total_requests_completed` — [`L238`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L238)
  - `total_requests_failed` — [`L239`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L239)
  - `total_tokens_generated` — [`L240`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L240)
- used by: [`update_from_metrics_collector`](monitoring.md#PrometheusMetrics.update_from_metrics_collector), [`get_system_metrics`](metrics.md#MetricsCollector.get_system_metrics), [`_create_system_metrics_table`](monitoring.md#RichConsoleMonitor._create_system_metrics_table), [`get_metrics_summary`](mixins/monitoring.md#EngineMonitoringMixin.get_metrics_summary)

## Functions
- `get_metrics_collector()` — [`L686`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L686) — Get the global metrics collector instance.
- `initialize_metrics(log_file: str | None = None, log_interval: float = 10, history_size: int = 1000, enable_detailed_logging: bool = True)` — [`L700`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L700) — Initialize the global metrics collector.
- `log_metrics_summary()` — [`L738`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L738) — Log a summary of current metrics if collector is initialized.

## Module values
- `_global_metrics_collector` — [`L683`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/metrics.py#L683)


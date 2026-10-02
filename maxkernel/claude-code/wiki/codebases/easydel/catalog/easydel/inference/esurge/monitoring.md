---
title: 'Module: easydel/inference/esurge/monitoring.py'
type: catalog
provenance: extracted
module: easydel/inference/esurge/monitoring.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.esurge.monitoring`/
symbols:
  PrometheusMetrics.update_from_metrics_collector: PrometheusMetrics#update_from_metrics_collector().
  RichConsoleMonitor._create_system_metrics_table: RichConsoleMonitor#_create_system_metrics_table().
  RichConsoleMonitor._create_scheduler_metrics_table: RichConsoleMonitor#_create_scheduler_metrics_table().
  RichConsoleMonitor._create_recent_requests_table: RichConsoleMonitor#_create_recent_requests_table().
  RichConsoleMonitor._create_runner_metrics_table: RichConsoleMonitor#_create_runner_metrics_table().
  RichConsoleMonitor._create_cache_metrics_table: RichConsoleMonitor#_create_cache_metrics_table().
  RichConsoleMonitor._update_layout: RichConsoleMonitor#_update_layout().
  _console_monitor._console_monitor: _console_monitor._console_monitor.
  RichConsoleMonitor._monitor_loop: RichConsoleMonitor#_monitor_loop().
  eSurgeMonitoringServer._update_metrics_loop: eSurgeMonitoringServer#_update_metrics_loop().
  _monitoring_server._monitoring_server: _monitoring_server._monitoring_server.
  RichConsoleMonitor.start: RichConsoleMonitor#start().
  eSurgeMonitoringServer.start: eSurgeMonitoringServer#start().
  stop_monitoring: stop_monitoring().
  RichConsoleMonitor.layout: RichConsoleMonitor#layout.
  eSurgeMonitoringServer.prometheus_metrics: eSurgeMonitoringServer#prometheus_metrics.
  start_monitoring_server: start_monitoring_server().
  start_console_monitor: start_console_monitor().
  RichConsoleMonitor.stop: RichConsoleMonitor#stop().
  eSurgeMonitoringServer.stop: eSurgeMonitoringServer#stop().
  eSurgeMonitoringServer.start_prometheus_server: eSurgeMonitoringServer#start_prometheus_server().
  PROMETHEUS_AVAILABLE.PROMETHEUS_AVAILABLE: PROMETHEUS_AVAILABLE.PROMETHEUS_AVAILABLE.
  RichConsoleMonitor._thread: RichConsoleMonitor#_thread.
  _prometheus_available: _prometheus_available.
  _rich_available: _rich_available.
  RichConsoleMonitor: RichConsoleMonitor#
  RichConsoleMonitor.running: RichConsoleMonitor#running.
  eSurgeMonitoringServer: eSurgeMonitoringServer#
  eSurgeMonitoringServer.running: eSurgeMonitoringServer#running.
  eSurgeMonitoringServer._update_thread: eSurgeMonitoringServer#_update_thread.
  RICH_AVAILABLE.RICH_AVAILABLE: RICH_AVAILABLE.RICH_AVAILABLE.
  PrometheusMetrics.tokens_per_second: PrometheusMetrics#tokens_per_second.
  eSurgeMonitoringServer.prometheus_port: eSurgeMonitoringServer#prometheus_port.
  PrometheusMetrics.__init__: PrometheusMetrics#__init__().
  RichConsoleMonitor.__init__: RichConsoleMonitor#__init__().
  PrometheusMetrics: PrometheusMetrics#
  PrometheusMetrics.requests_total: PrometheusMetrics#requests_total.
  PrometheusMetrics._completions_cursor: PrometheusMetrics#_completions_cursor.
  PrometheusMetrics._last_scheduler_id: PrometheusMetrics#_last_scheduler_id.
  PrometheusMetrics._last_runner_id: PrometheusMetrics#_last_runner_id.
  PrometheusMetrics._observe_new_entries: PrometheusMetrics#_observe_new_entries().
  RichConsoleMonitor.refresh_rate: RichConsoleMonitor#refresh_rate.
  eSurgeMonitoringServer.update_interval: eSurgeMonitoringServer#update_interval.
  PrometheusMetrics.request_duration: PrometheusMetrics#request_duration.
  PrometheusMetrics.time_to_first_token: PrometheusMetrics#time_to_first_token.
  PrometheusMetrics.tokens_generated_total: PrometheusMetrics#tokens_generated_total.
  PrometheusMetrics.waiting_requests: PrometheusMetrics#waiting_requests.
  PrometheusMetrics.running_requests: PrometheusMetrics#running_requests.
  PrometheusMetrics.scheduled_tokens: PrometheusMetrics#scheduled_tokens.
  PrometheusMetrics.schedule_duration: PrometheusMetrics#schedule_duration.
  PrometheusMetrics.model_execution_duration: PrometheusMetrics#model_execution_duration.
  PrometheusMetrics.batch_size: PrometheusMetrics#batch_size.
  PrometheusMetrics.cache_pages_total: PrometheusMetrics#cache_pages_total.
  PrometheusMetrics.cache_pages_used: PrometheusMetrics#cache_pages_used.
  PrometheusMetrics.cache_hit_rate: PrometheusMetrics#cache_hit_rate.
  RichConsoleMonitor.console: RichConsoleMonitor#console.
  PrometheusMetrics.prefix: PrometheusMetrics#prefix.
  PrometheusMetrics.preempted_requests_total: PrometheusMetrics#preempted_requests_total.
  PrometheusMetrics.system_info: PrometheusMetrics#system_info.
  eSurgeMonitoringServer.__init__: eSurgeMonitoringServer#__init__().
  eSurgeMonitoringServer.dashboard_port: eSurgeMonitoringServer#dashboard_port.
---
# Module: [`easydel/inference/esurge/monitoring.py`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py)

## Classes
### `PrometheusMetrics`
- def: [`easydel/inference/esurge/monitoring.py:96`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L96)
- doc: Prometheus metrics exporter for eSurge.
- signature: `class PrometheusMetrics:`
- members:
  - `__init__(self, prefix: str = "esurge_")` — [`L148`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L148) — Initialize Prometheus metrics.
  - `_observe_new_entries(deque, cursor_id: int, observe_fn)` — [`L229`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L229) — Observe deque entries added after the cursor, return new cursor id.
  - `update_from_metrics_collector(self, collector: MetricsCollector)` — [`L257`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L257) — Update Prometheus metrics from the metrics collector.
  - `batch_size` — [`L205`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L205)
  - `cache_hit_rate` — [`L212`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L212)
  - `cache_pages_total` — [`L208`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L208)
  - `cache_pages_used` — [`L210`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L210)
  - `model_execution_duration` — [`L199`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L199)
  - `preempted_requests_total` — [`L188`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L188)
  - `prefix` — [`L160`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L160)
  - `request_duration` — [`L165`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L165)
  - `requests_total` — [`L163`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L163)
  - `running_requests` — [`L184`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L184)
  - `schedule_duration` — [`L192`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L192)
  - `scheduled_tokens` — [`L186`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L186)
  - `system_info` — [`L215`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L215)
  - `time_to_first_token` — [`L171`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L171)
  - `tokens_generated_total` — [`L177`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L177)
  - `tokens_per_second` — [`L179`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L179)
  - `waiting_requests` — [`L182`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L182)
- protocol/private: `_completions_cursor`[`L220`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L220), `_last_runner_id`[`L226`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L226), `_last_scheduler_id`[`L225`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L225)
- uses (calls/refs, reference-scoped): [`get_system_metrics`](metrics.md#MetricsCollector.get_system_metrics), [`_lock`](metrics.md#MetricsCollector._lock), [`scheduler_metrics`](metrics.md#MetricsCollector.scheduler_metrics), [`runner_metrics`](metrics.md#MetricsCollector.runner_metrics), [`MetricsCollector`](metrics.md#MetricsCollector), [`cache_metrics`](metrics.md#MetricsCollector.cache_metrics), [`completed_requests`](metrics.md#MetricsCollector.completed_requests), [`counters`](metrics.md#MetricsCollector.counters), [`time_to_first_token`](metrics.md#RequestMetrics.time_to_first_token), [`generated_tokens`](metrics.md#RequestMetrics.generated_tokens), [`total_latency`](metrics.md#RequestMetrics.total_latency), [`PROMETHEUS_AVAILABLE`](monitoring.md#PROMETHEUS_AVAILABLE.PROMETHEUS_AVAILABLE), [`average_throughput`](metrics.md#SystemMetrics.average_throughput), [`error`](metrics.md#RequestMetrics.error), [`total_pages`](metrics.md#CacheMetrics.total_pages), [`num_running_requests`](metrics.md#SchedulerMetrics.num_running_requests), [`used_pages`](metrics.md#CacheMetrics.used_pages), [`batch_size`](metrics.md#ModelRunnerMetrics.batch_size), [`cache_hit_rate`](metrics.md#CacheMetrics.cache_hit_rate), [`num_scheduled_tokens`](metrics.md#SchedulerMetrics.num_scheduled_tokens), [`num_waiting_requests`](metrics.md#SchedulerMetrics.num_waiting_requests), [`tokens_per_second`](metrics.md#ModelRunnerMetrics.tokens_per_second)
- used by: [`_update_metrics_loop`](monitoring.md#eSurgeMonitoringServer._update_metrics_loop), [`prometheus_metrics`](monitoring.md#eSurgeMonitoringServer.prometheus_metrics)

### `RichConsoleMonitor`
- def: [`easydel/inference/esurge/monitoring.py:339`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L339)
- doc: Rich console-based live monitoring for eSurge.
- signature: `class RichConsoleMonitor:`
- members:
  - `__init__(self, refresh_rate: float = 1)` — [`L366`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L366) — Initialize console monitor.
  - `_create_cache_metrics_table(self, collector: MetricsCollector)` — [`L465`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L465) — Create cache metrics table.
  - `_create_recent_requests_table(self, collector: MetricsCollector)` — [`L492`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L492) — Create recent requests table.
  - `_create_runner_metrics_table(self, collector: MetricsCollector)` — [`L440`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L440) — Create runner metrics table.
  - `_create_scheduler_metrics_table(self, collector: MetricsCollector)` — [`L414`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L414) — Create scheduler metrics table.
  - `_create_system_metrics_table(self, collector: MetricsCollector)` — [`L389`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L389) — Create system metrics table.
  - `_update_layout(self)` — [`L525`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L525) — Update the layout with current metrics.
  - `start(self, blocking: bool = False)` — [`L559`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L559) — Start the live console monitor.
  - `stop(self)` — [`L593`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L593) — Stop the console monitor.
  - `console` — [`L378`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L378)
  - `layout` — [`L384`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L384)
  - `refresh_rate` — [`L379`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L379)
  - `running` — [`L380`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L380)
- protocol/private: `_monitor_loop`[`L575`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L575), `_thread`[`L381`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L381)
- uses (calls/refs, reference-scoped): [`get_system_metrics`](metrics.md#MetricsCollector.get_system_metrics), [`get_metrics_collector`](metrics.md#get_metrics_collector), [`_lock`](metrics.md#MetricsCollector._lock), [`scheduler_metrics`](metrics.md#MetricsCollector.scheduler_metrics), [`runner_metrics`](metrics.md#MetricsCollector.runner_metrics), [`MetricsCollector`](metrics.md#MetricsCollector), [`cache_metrics`](metrics.md#MetricsCollector.cache_metrics), [`completed_requests`](metrics.md#MetricsCollector.completed_requests), [`time_to_first_token`](metrics.md#RequestMetrics.time_to_first_token), [`generated_tokens`](metrics.md#RequestMetrics.generated_tokens), [`total_latency`](metrics.md#RequestMetrics.total_latency), [`average_throughput`](metrics.md#SystemMetrics.average_throughput), [`error`](metrics.md#RequestMetrics.error), [`total_pages`](metrics.md#CacheMetrics.total_pages), [`num_running_requests`](metrics.md#SchedulerMetrics.num_running_requests), [`used_pages`](metrics.md#CacheMetrics.used_pages), [`RICH_AVAILABLE`](monitoring.md#RICH_AVAILABLE.RICH_AVAILABLE), [`average_latency`](metrics.md#SystemMetrics.average_latency), [`average_ttft`](metrics.md#SystemMetrics.average_ttft), [`batch_size`](metrics.md#ModelRunnerMetrics.batch_size), [`cache_hit_rate`](metrics.md#CacheMetrics.cache_hit_rate), [`num_scheduled_tokens`](metrics.md#SchedulerMetrics.num_scheduled_tokens), [`num_waiting_requests`](metrics.md#SchedulerMetrics.num_waiting_requests), [`requests_per_second`](metrics.md#SystemMetrics.requests_per_second), [`tokens_per_second`](metrics.md#ModelRunnerMetrics.tokens_per_second), [`total_requests_completed`](metrics.md#SystemMetrics.total_requests_completed), [`total_requests_failed`](metrics.md#SystemMetrics.total_requests_failed), [`total_tokens_generated`](metrics.md#SystemMetrics.total_tokens_generated), [`batch_size`](metrics.md#SchedulerMetrics.batch_size), [`execution_time`](metrics.md#ModelRunnerMetrics.execution_time), [`free_pages`](metrics.md#CacheMetrics.free_pages), [`num_tokens`](metrics.md#ModelRunnerMetrics.num_tokens), [`request_id`](metrics.md#RequestMetrics.request_id), [`schedule_time`](metrics.md#SchedulerMetrics.schedule_time)
- used by: [`stop_monitoring`](mixins/monitoring.md#EngineMonitoringMixin.stop_monitoring), [`_console_monitor`](monitoring.md#_console_monitor._console_monitor), [`stop_monitoring`](monitoring.md#stop_monitoring), [`start_console_monitor`](monitoring.md#start_console_monitor)

### `eSurgeMonitoringServer`
- def: [`easydel/inference/esurge/monitoring.py:603`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L603)
- doc: Monitoring server for Prometheus export and console monitoring.
- signature: `class eSurgeMonitoringServer:`
- members:
  - `__init__(self, prometheus_port: int = 8000, dashboard_port: int | None = None, metrics_prefix: str = "esurge_", update_interval: float = 1)` — [`L626`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L626) — Initialize monitoring server.
  - `_update_metrics_loop(self)` — [`L653`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L653) — Background thread to update Prometheus metrics.
  - `start(self)` — [`L682`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L682) — Start the monitoring server.
  - `start_prometheus_server(self)` — [`L669`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L669) — Start Prometheus metrics server.
  - `stop(self)` — [`L702`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L702) — Stop the monitoring server.
  - `dashboard_port` — [`L642`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L642)
  - `prometheus_metrics` — [`L646`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L646)
  - `prometheus_port` — [`L641`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L641)
  - `running` — [`L650`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L650)
  - `update_interval` — [`L643`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L643)
- protocol/private: `_update_thread`[`L651`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L651)
- uses (calls/refs, reference-scoped): [`update_from_metrics_collector`](monitoring.md#PrometheusMetrics.update_from_metrics_collector), [`get_metrics_collector`](metrics.md#get_metrics_collector), [`PROMETHEUS_AVAILABLE`](monitoring.md#PROMETHEUS_AVAILABLE.PROMETHEUS_AVAILABLE), [`PrometheusMetrics`](monitoring.md#PrometheusMetrics)
- used by: [`stop_monitoring`](mixins/monitoring.md#EngineMonitoringMixin.stop_monitoring), [`_monitoring_server`](monitoring.md#_monitoring_server._monitoring_server), [`stop_monitoring`](monitoring.md#stop_monitoring), [`start_monitoring_server`](monitoring.md#start_monitoring_server)

## Functions
- `start_console_monitor(refresh_rate: float = 1)` — [`L754`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L754) — Start the global console monitor.
- `start_monitoring_server(prometheus_port: int = 8000, dashboard_port: int | None = None, update_interval: float = 1)` — [`L719`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L719) — Start the global monitoring server.
- `stop_monitoring()` — [`L779`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L779) — Stop all monitoring services.

## Module values
- `PROMETHEUS_AVAILABLE` — [`L73`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L73)
- `RICH_AVAILABLE` — [`L93`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L93)
- `_console_monitor` — [`L716`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L716)
- `_monitoring_server` — [`L715`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L715)
- `_prometheus_available` — [`L64`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L64)
- `_rich_available` — [`L83`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/monitoring.py#L83)


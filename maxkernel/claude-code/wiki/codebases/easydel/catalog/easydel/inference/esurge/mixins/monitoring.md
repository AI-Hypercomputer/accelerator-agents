---
title: 'Module: easydel/inference/esurge/mixins/monitoring.py'
type: catalog
provenance: extracted
module: easydel/inference/esurge/mixins/monitoring.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.esurge.mixins.monitoring`/
symbols:
  EngineMonitoringMixin.get_metrics_summary: EngineMonitoringMixin#get_metrics_summary().
  EngineMonitoringMixin._start_grafana_service: EngineMonitoringMixin#_start_grafana_service().
  EngineMonitoringMixin.stop_monitoring: EngineMonitoringMixin#stop_monitoring().
  EngineMonitoringMixin._stop_grafana_service: EngineMonitoringMixin#_stop_grafana_service().
  EngineMonitoringMixin._grafana_url: EngineMonitoringMixin#_grafana_url.
  EngineMonitoringMixin._monitoring_urls: EngineMonitoringMixin#_monitoring_urls.
  _panel: _panel().
  EngineMonitoringMixin.start_monitoring: EngineMonitoringMixin#start_monitoring().
  EngineMonitoringMixin._monitoring_server: EngineMonitoringMixin#_monitoring_server.
  EngineMonitoringMixin._grafana_container_id: EngineMonitoringMixin#_grafana_container_id.
  EngineMonitoringMixin._monitoring_initialized: EngineMonitoringMixin#_monitoring_initialized.
  EngineMonitoringMixin._start_local_grafana_service: EngineMonitoringMixin#_start_local_grafana_service().
  EngineMonitoringMixin._grafana_process: EngineMonitoringMixin#_grafana_process.
  EngineMonitoringMixin._start_docker_grafana_service: EngineMonitoringMixin#_start_docker_grafana_service().
  EngineMonitoringMixin._prometheus_process: EngineMonitoringMixin#_prometheus_process.
  EngineMonitoringMixin._stop_prometheus_server: EngineMonitoringMixin#_stop_prometheus_server().
  EngineMonitoringMixin._provision_running_grafana: EngineMonitoringMixin#_provision_running_grafana().
  EngineMonitoringMixin: EngineMonitoringMixin#
  EngineMonitoringMixin._grafana_container_name: EngineMonitoringMixin#_grafana_container_name.
  _build_esurge_dashboard_model: _build_esurge_dashboard_model().
  EngineMonitoringMixin._prepare_grafana_provisioning: EngineMonitoringMixin#_prepare_grafana_provisioning().
  EngineMonitoringMixin._grafana_temp_dir: EngineMonitoringMixin#_grafana_temp_dir.
  EngineMonitoringMixin._start_prometheus_server: EngineMonitoringMixin#_start_prometheus_server().
  EngineMonitoringMixin._prometheus_temp_dir: EngineMonitoringMixin#_prometheus_temp_dir.
  EngineMonitoringMixin.monitoring_active: EngineMonitoringMixin#monitoring_active().
---
# Module: [`easydel/inference/esurge/mixins/monitoring.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/monitoring.py)

## Classes
### `EngineMonitoringMixin`
- def: [`easydel/inference/esurge/mixins/monitoring.py:151`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/monitoring.py#L151)
- doc: Mixin providing monitoring and observability for the eSurge engine.
- signature: `class EngineMonitoringMixin:`
- members:
  - `_prepare_grafana_provisioning(self, datasource_name: str, datasource_uid: str, datasource_url: str, for_docker: bool = False)` — [`L167`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/monitoring.py#L167) — Create temporary provisioning config for Grafana.
  - `_provision_running_grafana(self, grafana_host: str | None, grafana_port: int, grafana_admin_user: str, grafana_admin_password: str, datasource_name: str, datasource_uid: str, datasource_url: str)` — [`L468`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/monitoring.py#L468) — Provision datasource and dashboard on an already-running Grafana via HTTP API.
  - `_start_docker_grafana_service(self, provisioning_root: str, grafana_host: str | None, grafana_port: int, grafana_image: str, grafana_admin_user: str, grafana_admin_password: str, allow_anonymous: bool, datasource_url: str)` — [`L312`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/monitoring.py#L312) — Start Grafana using Docker.
  - `_start_grafana_service(self, prometheus_url: str | None, grafana_host: str | None, grafana_port: int, grafana_image: str, grafana_admin_user: str, grafana_admin_password: str, allow_anonymous: bool, datasource_name: str, datasource_uid: str | None, datasource_url: str | None, use_docker: bool)` — [`L573`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/monitoring.py#L573) — Attempt to launch Grafana wired to the Prometheus endpoint.
  - `_start_local_grafana_service(self, provisioning_root: str, grafana_host: str | None, grafana_port: int, grafana_admin_user: str, grafana_admin_password: str, allow_anonymous: bool)` — [`L240`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/monitoring.py#L240) — Start Grafana using a locally installed grafana-server binary.
  - `_start_prometheus_server(self, scrape_target: str, prometheus_server_port: int = 9090)` — [`L385`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/monitoring.py#L385) — Start a Prometheus server that scrapes the prometheus_client endpoint.
  - `_stop_grafana_service(self)` — [`L710`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/monitoring.py#L710) — Stop the Grafana container/process if it was started by the engine.
  - `_stop_prometheus_server(self)` — [`L451`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/monitoring.py#L451) — Stop the Prometheus server if it was started by the engine.
  - `get_metrics_summary(self)` — [`L941`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/monitoring.py#L941) — Get current performance metrics summary.
  - `monitoring_active(self)` — [`L975`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/monitoring.py#L975) — Check if monitoring services are currently active.
  - `start_monitoring(self, dashboard_port: int | None = None, prometheus_port: int = 11184, dashboard_host: str | None = None, enable_prometheus: bool = True, enable_dashboard: bool | None = None, enable_console: bool = False, log_file: str | None = None, log_interval: float = 10, history_size: int = 1000, enable_detailed_logging: bool = True, start_grafana: bool = True, grafana_port: int = 3000, grafana_host: str | None = None, grafana_image: str = "grafana/grafana-oss:latest", grafana_use_docker: bool = False, grafana_admin_user: str = "admin", grafana_admin_password: str = "admin", grafana_allow_anonymous: bool = True, grafana_datasource_name: str = "eSurge Prometheus", grafana_datasource_uid: str | None = None, grafana_datasource_url: str | None = None)` — [`L751`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/monitoring.py#L751) — Start Prometheus-based monitoring for the engine.
  - `stop_monitoring(self)` — [`L906`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/monitoring.py#L906) — Stop all monitoring services.
- protocol/private: `_grafana_container_id`[`L379`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/monitoring.py#L379), `_grafana_container_name`[`L378`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/monitoring.py#L378), `_grafana_process`[`L306`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/monitoring.py#L306), `_grafana_temp_dir`[`L307`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/monitoring.py#L307), `_grafana_url`[`L308`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/monitoring.py#L308), `_monitoring_initialized`[`L897`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/monitoring.py#L897), `_monitoring_server`[`L850`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/monitoring.py#L850), `_monitoring_urls`[`L829`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/monitoring.py#L829), `_prometheus_process`[`L445`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/monitoring.py#L445), `_prometheus_temp_dir`[`L446`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/monitoring.py#L446)
- uses (calls/refs, reference-scoped): [`eSurge`](../esurge_engine.md#eSurge), [`get_system_metrics`](../metrics.md#MetricsCollector.get_system_metrics), [`get_metrics_collector`](../metrics.md#get_metrics_collector), [`_console_monitor`](../monitoring.md#_console_monitor._console_monitor), [`start_console_monitor`](../monitoring.md#start_console_monitor), [`start_monitoring_server`](../monitoring.md#start_monitoring_server), [`initialize_metrics`](../metrics.md#initialize_metrics), [`stop`](../monitoring.md#RichConsoleMonitor.stop), [`stop`](../monitoring.md#eSurgeMonitoringServer.stop), [`average_throughput`](../metrics.md#SystemMetrics.average_throughput), [`_build_esurge_dashboard_model`](monitoring.md#_build_esurge_dashboard_model), [`average_latency`](../metrics.md#SystemMetrics.average_latency), [`average_ttft`](../metrics.md#SystemMetrics.average_ttft), [`requests_per_second`](../metrics.md#SystemMetrics.requests_per_second), [`total_requests_completed`](../metrics.md#SystemMetrics.total_requests_completed), [`total_requests_failed`](../metrics.md#SystemMetrics.total_requests_failed), [`total_tokens_generated`](../metrics.md#SystemMetrics.total_tokens_generated)
- used by: [`eSurge`](../esurge_engine.md#eSurge), [`__del__`](../esurge_engine.md#eSurge.__del__)

## Functions
- `_build_esurge_dashboard_model(datasource_uid: str)` — [`L68`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/monitoring.py#L68) — Return a complete Grafana dashboard dict for eSurge metrics.
- `_panel(title: str, expr: str, ds_uid: str, grid_x: int, grid_y: int, grid_w: int = 12, grid_h: int = 8, panel_id: int = 1, panel_type: str = "timeseries", unit: str = "", legend_mode: str = "list")` — [`L28`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/monitoring.py#L28) — Build a single Grafana dashboard panel dict.


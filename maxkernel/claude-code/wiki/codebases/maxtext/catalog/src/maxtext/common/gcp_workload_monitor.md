---
title: 'Module: src/maxtext/common/gcp_workload_monitor.py'
type: catalog
provenance: extracted
module: src/maxtext/common/gcp_workload_monitor.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.common.gcp_workload_monitor`/
symbols:
  GCPWorkloadMonitor._report_heartbeat: GCPWorkloadMonitor#_report_heartbeat().
  GCPWorkloadMonitor._report_performance: GCPWorkloadMonitor#_report_performance().
  GCPWorkloadMonitor.client: GCPWorkloadMonitor#client.
  monitoring_v3: monitoring_v3.
  _get_gcp_metadata: _get_gcp_metadata().
  GCPWorkloadMonitor._report_heartbeat_thread: GCPWorkloadMonitor#_report_heartbeat_thread().
  GCPWorkloadMonitor._report_performance_thread: GCPWorkloadMonitor#_report_performance_thread().
  _GCLOUD_AVAILABLE: _GCLOUD_AVAILABLE.
  GCPWorkloadMonitor.project_id: GCPWorkloadMonitor#project_id.
  GCPWorkloadMonitor.start_heartbeat_reporting_thread: GCPWorkloadMonitor#start_heartbeat_reporting_thread().
  GCPWorkloadMonitor.start_performance_reporting_thread: GCPWorkloadMonitor#start_performance_reporting_thread().
  GCPWorkloadMonitor.zone: GCPWorkloadMonitor#zone.
  get_gcp_project_id: get_gcp_project_id().
  get_node_zone: get_node_zone().
  GCPWorkloadMonitor.termination_event: GCPWorkloadMonitor#termination_event.
  GCPWorkloadMonitor.__del__: GCPWorkloadMonitor#__del__().
  GCPWorkloadMonitor.workload_id: GCPWorkloadMonitor#workload_id.
  GCPWorkloadMonitor.heartbeat_reporting_started: GCPWorkloadMonitor#heartbeat_reporting_started.
  GCPWorkloadMonitor.performance_reporting_started: GCPWorkloadMonitor#performance_reporting_started.
  metric_pb2: metric_pb2.
  monitored_resource_pb2: monitored_resource_pb2.
  GoogleAPIError: GoogleAPIError.
  _METADATA_SERVER_URL: _METADATA_SERVER_URL.
  _METADATA_HEADERS: _METADATA_HEADERS.
  _MONITORING_STUB: _MONITORING_STUB.
  GCPWorkloadMonitor: GCPWorkloadMonitor#
  GCPWorkloadMonitor.__init__: GCPWorkloadMonitor#__init__().
---
# Module: [`src/maxtext/common/gcp_workload_monitor.py`](../../../../../../../raw/code/maxtext/src/maxtext/common/gcp_workload_monitor.py)

## Classes
### `GCPWorkloadMonitor`
- def: [`src/maxtext/common/gcp_workload_monitor.py:40`](../../../../../../../raw/code/maxtext/src/maxtext/common/gcp_workload_monitor.py#L40)
- doc: Interface for reporting metrics to GCP for monitoring.
- signature: `class GCPWorkloadMonitor:`
- members:
  - `_report_heartbeat(self, local_rank: str, global_rank: str)` — [`L94`](../../../../../../../raw/code/maxtext/src/maxtext/common/gcp_workload_monitor.py#L94) — Reports heartbeat metric for the process specified by the given local rank & global rank.
  - `_report_heartbeat_thread(self, interval: int)` — [`L76`](../../../../../../../raw/code/maxtext/src/maxtext/common/gcp_workload_monitor.py#L76) — Reports heartbeat metric to GCP every {interval} seconds until termination event is set.
  - `_report_performance(self, performance_metric)` — [`L143`](../../../../../../../raw/code/maxtext/src/maxtext/common/gcp_workload_monitor.py#L143) — Reports performance metric to GCP.
  - `_report_performance_thread(self, metrics_queue: queue.Queue)` — [`L84`](../../../../../../../raw/code/maxtext/src/maxtext/common/gcp_workload_monitor.py#L84) — Reports performance metric to GCP whenever new metric arrives at the metrics_queue until termination event is set.
  - `start_heartbeat_reporting_thread(self, interval: int)` — [`L56`](../../../../../../../raw/code/maxtext/src/maxtext/common/gcp_workload_monitor.py#L56) — Starts a thread that reports heartbeat every {interval} seconds until termination event is set.
  - `start_performance_reporting_thread(self, metrics_queue: queue.Queue)` — [`L66`](../../../../../../../raw/code/maxtext/src/maxtext/common/gcp_workload_monitor.py#L66) — Starts a thread that reports performance metric sent to metrics_queue until termination event is set.
  - `client` — [`L48`](../../../../../../../raw/code/maxtext/src/maxtext/common/gcp_workload_monitor.py#L48)
  - `heartbeat_reporting_started` — [`L49`](../../../../../../../raw/code/maxtext/src/maxtext/common/gcp_workload_monitor.py#L49)
  - `performance_reporting_started` — [`L50`](../../../../../../../raw/code/maxtext/src/maxtext/common/gcp_workload_monitor.py#L50)
  - `project_id` — [`L47`](../../../../../../../raw/code/maxtext/src/maxtext/common/gcp_workload_monitor.py#L47)
  - `termination_event` — [`L51`](../../../../../../../raw/code/maxtext/src/maxtext/common/gcp_workload_monitor.py#L51)
  - `workload_id` — [`L45`](../../../../../../../raw/code/maxtext/src/maxtext/common/gcp_workload_monitor.py#L45)
  - `zone` — [`L46`](../../../../../../../raw/code/maxtext/src/maxtext/common/gcp_workload_monitor.py#L46)
- protocol/private: `__del__`[`L53`](../../../../../../../raw/code/maxtext/src/maxtext/common/gcp_workload_monitor.py#L53), `__init__`[`L43`](../../../../../../../raw/code/maxtext/src/maxtext/common/gcp_workload_monitor.py#L43)
- uses (calls/refs, reference-scoped): [`monitoring_v3`](gcp_workload_monitor.md#monitoring_v3), [`_get_gcp_metadata`](gcp_workload_monitor.md#_get_gcp_metadata), [`_GCLOUD_AVAILABLE`](gcp_workload_monitor.md#_GCLOUD_AVAILABLE), [`get_gcp_project_id`](gcp_workload_monitor.md#get_gcp_project_id), [`get_node_zone`](gcp_workload_monitor.md#get_node_zone), [`GoogleAPIError`](gcp_workload_monitor.md#GoogleAPIError), [`metric_pb2`](gcp_workload_monitor.md#metric_pb2), [`monitored_resource_pb2`](gcp_workload_monitor.md#monitored_resource_pb2)

## Functions
- `_get_gcp_metadata(category: str, attribute: str, timeout=5, retries=3)` — [`L187`](../../../../../../../raw/code/maxtext/src/maxtext/common/gcp_workload_monitor.py#L187) — Fetch the specified attribute from GCP metadata server.
- `get_gcp_project_id()` — [`L221`](../../../../../../../raw/code/maxtext/src/maxtext/common/gcp_workload_monitor.py#L221) — Returns the project id of the current GCP project.
- `get_node_zone()` — [`L226`](../../../../../../../raw/code/maxtext/src/maxtext/common/gcp_workload_monitor.py#L226) — Returns the zone of the GCE instance.

## Module values
- `GoogleAPIError` — [`L32`](../../../../../../../raw/code/maxtext/src/maxtext/common/gcp_workload_monitor.py#L32)
- `_GCLOUD_AVAILABLE` — [`L33`](../../../../../../../raw/code/maxtext/src/maxtext/common/gcp_workload_monitor.py#L33)
- `_METADATA_HEADERS` — [`L37`](../../../../../../../raw/code/maxtext/src/maxtext/common/gcp_workload_monitor.py#L37)
- `_METADATA_SERVER_URL` — [`L36`](../../../../../../../raw/code/maxtext/src/maxtext/common/gcp_workload_monitor.py#L36)
- `_MONITORING_STUB` — [`L32`](../../../../../../../raw/code/maxtext/src/maxtext/common/gcp_workload_monitor.py#L32)
- `metric_pb2` — [`L32`](../../../../../../../raw/code/maxtext/src/maxtext/common/gcp_workload_monitor.py#L32)
- `monitored_resource_pb2` — [`L32`](../../../../../../../raw/code/maxtext/src/maxtext/common/gcp_workload_monitor.py#L32)
- `monitoring_v3` — [`L32`](../../../../../../../raw/code/maxtext/src/maxtext/common/gcp_workload_monitor.py#L32)


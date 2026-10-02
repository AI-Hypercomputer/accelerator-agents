---
title: 'Module: src/maxtext/common/metric_logger.py'
type: catalog
provenance: extracted
module: src/maxtext/common/metric_logger.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.common.metric_logger`/
symbols:
  MetricLogger.config: MetricLogger#config.
  MetricLogger.write_metrics: MetricLogger#write_metrics().
  MetricLogger.record_train_metrics: MetricLogger#record_train_metrics().
  MetricLogger.write_setup_info_to_tensorboard: MetricLogger#write_setup_info_to_tensorboard().
  MetricLogger.buffer_and_write_metrics: MetricLogger#buffer_and_write_metrics().
  MetricLogger.cumulative_eval_metrics: MetricLogger#cumulative_eval_metrics.
  MetricLogger._finalize_eval_metrics: MetricLogger#_finalize_eval_metrics().
  MetricLogger._flush_one_buffered_entry: MetricLogger#_flush_one_buffered_entry().
  MetricLogger.log_metrics: MetricLogger#log_metrics().
  MetricLogger.write_metrics_for_gcs: MetricLogger#write_metrics_for_gcs().
  MetricLogger.write_metrics_to_managed_mldiagnostics: MetricLogger#write_metrics_to_managed_mldiagnostics().
  MetricLogger._pending_eval_step_count: MetricLogger#_pending_eval_step_count.
  MetricLogger.flush_metrics_and_cleanup: MetricLogger#flush_metrics_and_cleanup().
  MetricLogger.writer: MetricLogger#writer.
  MetadataKey: MetadataKey#
  MetricLogger.metadata: MetricLogger#metadata.
  MetricLogger.buffered_metrics: MetricLogger#buffered_metrics.
  MetricLogger._log_training_metrics: MetricLogger#_log_training_metrics().
  MetricLogger.write_metrics_locally: MetricLogger#write_metrics_locally().
  MetricLogger.write_metrics_to_tensorboard: MetricLogger#write_metrics_to_tensorboard().
  MetricLogger.get_performance_metric_queue: MetricLogger#get_performance_metric_queue().
  MetricLogger.reset_eval_metrics: MetricLogger#reset_eval_metrics().
  MetricLogger.performance_metric_queue: MetricLogger#performance_metric_queue.
  MetricLogger._log_eval_metrics: MetricLogger#_log_eval_metrics().
  MetricLogger._log_running_eval_metrics: MetricLogger#_log_running_eval_metrics().
  MetricLogger._is_profiler_boundary_step: MetricLogger#_is_profiler_boundary_step().
  MetricLogger._maybe_abort_after_write_metrics: MetricLogger#_maybe_abort_after_write_metrics().
  MetricLogger._accumulate_eval_metrics: MetricLogger#_accumulate_eval_metrics().
  MetadataKey.PER_DEVICE_TFLOPS: MetadataKey#PER_DEVICE_TFLOPS.
  MetadataKey.PER_DEVICE_TOKENS: MetadataKey#PER_DEVICE_TOKENS.
  MetricLogger.running_gcs_metrics: MetricLogger#running_gcs_metrics.
  _prepare_metrics_for_json: _prepare_metrics_for_json().
  _METRICS_TO_MANAGED: _METRICS_TO_MANAGED.
  MetricLogger.learning_rate_schedule: MetricLogger#learning_rate_schedule.
  mldiag: mldiag.
  GCPWorkloadMonitor: GCPWorkloadMonitor.
  _monitor_is_stub: _monitor_is_stub.
  record_activation_metrics: record_activation_metrics().
  MetricLogger: MetricLogger#
  MetricLogger.__init__: MetricLogger#__init__().
---
# Module: [`src/maxtext/common/metric_logger.py`](../../../../../../../raw/code/maxtext/src/maxtext/common/metric_logger.py)

## Classes
### `MetadataKey`  ·  implements/extends Enum
- def: [`src/maxtext/common/metric_logger.py:86`](../../../../../../../raw/code/maxtext/src/maxtext/common/metric_logger.py#L86)
- signature: `class MetadataKey(enum.Enum):`
- members:
  - `PER_DEVICE_TFLOPS` — [`L87`](../../../../../../../raw/code/maxtext/src/maxtext/common/metric_logger.py#L87)
  - `PER_DEVICE_TOKENS` — [`L88`](../../../../../../../raw/code/maxtext/src/maxtext/common/metric_logger.py#L88)
- used by: [`record_train_metrics`](metric_logger.md#MetricLogger.record_train_metrics), [`write_setup_info_to_tensorboard`](metric_logger.md#MetricLogger.write_setup_info_to_tensorboard)

### `MetricLogger`
- def: [`src/maxtext/common/metric_logger.py:91`](../../../../../../../raw/code/maxtext/src/maxtext/common/metric_logger.py#L91)
- doc: Logger for saving metrics to a local file, GCS and TensorBoard.
- signature: `class MetricLogger:`
- members:
  - `_accumulate_eval_metrics(self, metrics)` — [`L419`](../../../../../../../raw/code/maxtext/src/maxtext/common/metric_logger.py#L419) — Accumulates one eval step's raw metrics into cumulative_eval_metrics (eager float()).
  - `_finalize_eval_metrics(self, train_step)` — [`L448`](../../../../../../../raw/code/maxtext/src/maxtext/common/metric_logger.py#L448) — Computes final averaged eval metrics and writes them at train_step.
  - `_flush_one_buffered_entry(self, entry)` — [`L394`](../../../../../../../raw/code/maxtext/src/maxtext/common/metric_logger.py#L394) — Dispatches a single buffered entry to the writer.
  - `_is_profiler_boundary_step(self, step)` — [`L258`](../../../../../../../raw/code/maxtext/src/maxtext/common/metric_logger.py#L258) — Determines if the current step is a profiler start/stop boundary that should be hidden.
  - `_log_eval_metrics(self, metrics, step)` — [`L216`](../../../../../../../raw/code/maxtext/src/maxtext/common/metric_logger.py#L216) — Logs the final accumulated eval summary at the end of an eval run.
  - `_log_running_eval_metrics(self, metrics, step)` — [`L239`](../../../../../../../raw/code/maxtext/src/maxtext/common/metric_logger.py#L239) — Logs a per-eval-step running average (deferred by one eval step).
  - `_log_training_metrics(self, metrics, step)` — [`L148`](../../../../../../../raw/code/maxtext/src/maxtext/common/metric_logger.py#L148) — Handles training-specific metric logging.
  - `_maybe_abort_after_write_metrics(self, metrics)` — [`L273`](../../../../../../../raw/code/maxtext/src/maxtext/common/metric_logger.py#L273) — This function checks whether we have nan or inf values in training
  - `buffer_and_write_metrics(self, metrics, step, step_time_delta=None, is_training=True)` — [`L367`](../../../../../../../raw/code/maxtext/src/maxtext/common/metric_logger.py#L367) — Per-step entry point for both train and eval metrics. Flushes the single deferred entry from
  - `flush_metrics_and_cleanup(self)` — [`L466`](../../../../../../../raw/code/maxtext/src/maxtext/common/metric_logger.py#L466) — This is a terminal operation that uploads any buffered metrics to GCS
  - `get_performance_metric_queue(self, config)` — [`L348`](../../../../../../../raw/code/maxtext/src/maxtext/common/metric_logger.py#L348) — Records heartbeat metrics and performance metrics to GCP.
  - `log_metrics(self, metrics, step, metric_type)` — [`L139`](../../../../../../../raw/code/maxtext/src/maxtext/common/metric_logger.py#L139) — Logs metrics via max_logging.
  - `record_train_metrics(self, metrics, step, step_time)` — [`L432`](../../../../../../../raw/code/maxtext/src/maxtext/common/metric_logger.py#L432) — Records training metrics for the current step.
  - `reset_eval_metrics(self)` — [`L114`](../../../../../../../raw/code/maxtext/src/maxtext/common/metric_logger.py#L114) — Resets the cumulative metrics dictionary for a new evaluation run.
  - `write_metrics(self, metrics, step, metric_type="train")` — [`L119`](../../../../../../../raw/code/maxtext/src/maxtext/common/metric_logger.py#L119) — Entry point for all metrics writing. metric_type is one of 'train', 'eval', 'running_eval'.
  - `write_metrics_for_gcs(self, metrics, step, metric_type)` — [`L292`](../../../../../../../raw/code/maxtext/src/maxtext/common/metric_logger.py#L292) — Writes metrics to GCS.
  - `write_metrics_locally(self, metrics, step)` — [`L283`](../../../../../../../raw/code/maxtext/src/maxtext/common/metric_logger.py#L283) — Writes metrics locally for testing.
  - `write_metrics_to_managed_mldiagnostics(self, metrics, step)` — [`L324`](../../../../../../../raw/code/maxtext/src/maxtext/common/metric_logger.py#L324) — Write metrics to managed profiler.
  - `write_metrics_to_tensorboard(self, metrics, step, metric_type)` — [`L309`](../../../../../../../raw/code/maxtext/src/maxtext/common/metric_logger.py#L309) — Writes metrics to TensorBoard.
  - `write_setup_info_to_tensorboard(self, params)` — [`L336`](../../../../../../../raw/code/maxtext/src/maxtext/common/metric_logger.py#L336) — Writes setup information like train config params, num model params, and XLA flags to TensorBoard.
  - `buffered_metrics` — [`L107`](../../../../../../../raw/code/maxtext/src/maxtext/common/metric_logger.py#L107)
  - `config` — [`L98`](../../../../../../../raw/code/maxtext/src/maxtext/common/metric_logger.py#L98)
  - `cumulative_eval_metrics` — [`L103`](../../../../../../../raw/code/maxtext/src/maxtext/common/metric_logger.py#L103)
  - `learning_rate_schedule` — [`L102`](../../../../../../../raw/code/maxtext/src/maxtext/common/metric_logger.py#L102)
  - `metadata` — [`L99`](../../../../../../../raw/code/maxtext/src/maxtext/common/metric_logger.py#L99)
  - `performance_metric_queue` — [`L101`](../../../../../../../raw/code/maxtext/src/maxtext/common/metric_logger.py#L101)
  - `running_gcs_metrics` — [`L100`](../../../../../../../raw/code/maxtext/src/maxtext/common/metric_logger.py#L100)
  - `writer` — [`L97`](../../../../../../../raw/code/maxtext/src/maxtext/common/metric_logger.py#L97)
- protocol/private: `__init__`[`L96`](../../../../../../../raw/code/maxtext/src/maxtext/common/metric_logger.py#L96), `_pending_eval_step_count`[`L110`](../../../../../../../raw/code/maxtext/src/maxtext/common/metric_logger.py#L110)
- uses (calls/refs, reference-scoped): [`MetadataKey`](metric_logger.md#MetadataKey), [`PER_DEVICE_TFLOPS`](metric_logger.md#MetadataKey.PER_DEVICE_TFLOPS), [`PER_DEVICE_TOKENS`](metric_logger.md#MetadataKey.PER_DEVICE_TOKENS), [`_prepare_metrics_for_json`](metric_logger.md#_prepare_metrics_for_json), [`GCPWorkloadMonitor`](metric_logger.md#GCPWorkloadMonitor), [`_METRICS_TO_MANAGED`](metric_logger.md#_METRICS_TO_MANAGED), [`_monitor_is_stub`](metric_logger.md#_monitor_is_stub), [`mldiag`](metric_logger.md#mldiag)

## Functions
- `_prepare_metrics_for_json(metrics, step, run_name)` — [`L58`](../../../../../../../raw/code/maxtext/src/maxtext/common/metric_logger.py#L58) — Converts metric dictionary into json supported types (e.g. float)
- `record_activation_metrics(output_metrics, intermediate_outputs, config)` — [`L66`](../../../../../../../raw/code/maxtext/src/maxtext/common/metric_logger.py#L66) — Adds the activation metrics to the metrics dict.

## Module values
- `GCPWorkloadMonitor` — [`L42`](../../../../../../../raw/code/maxtext/src/maxtext/common/metric_logger.py#L42)
- `_METRICS_TO_MANAGED` — [`L45`](../../../../../../../raw/code/maxtext/src/maxtext/common/metric_logger.py#L45)
- `_monitor_is_stub` — [`L42`](../../../../../../../raw/code/maxtext/src/maxtext/common/metric_logger.py#L42)
- `mldiag` — [`L41`](../../../../../../../raw/code/maxtext/src/maxtext/common/metric_logger.py#L41)


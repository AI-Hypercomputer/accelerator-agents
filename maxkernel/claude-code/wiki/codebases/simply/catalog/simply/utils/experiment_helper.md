---
title: 'Module: simply/utils/experiment_helper.py'
type: catalog
provenance: extracted
module: simply/utils/experiment_helper.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.utils.experiment_helper`/
symbols:
  ExperimentHelper.ckpt_mngr: ExperimentHelper#ckpt_mngr().
  ExperimentHelper.add_metric: ExperimentHelper#add_metric().
  ExperimentHelper.save_config_info: ExperimentHelper#save_config_info().
  ExperimentHelper.metric_writer: ExperimentHelper#metric_writer().
  ExperimentHelper.save_state_info: ExperimentHelper#save_state_info().
  ExperimentHelper.close: ExperimentHelper#close().
  ExperimentHelper.write_scalars: ExperimentHelper#write_scalars().
  set_notes: set_notes().
  ExperimentHelper.experiment_dir: ExperimentHelper#experiment_dir.
  ExperimentHelper.should_save_data: ExperimentHelper#should_save_data().
  ExperimentHelper.flush: ExperimentHelper#flush().
  ExperimentHelper.metrics_aggregator: ExperimentHelper#metrics_aggregator().
  ExperimentHelper.should_log_additional_info: ExperimentHelper#should_log_additional_info().
  is_primary_task: is_primary_task().
  ExperimentHelper.should_log_metrics: ExperimentHelper#should_log_metrics().
  ExperimentHelper.ckpt_dir: ExperimentHelper#ckpt_dir().
  ExperimentHelper.ckpt_keep_period: ExperimentHelper#ckpt_keep_period.
  ExperimentHelper.get_aggregated_metrics: ExperimentHelper#get_aggregated_metrics().
  ExperimentHelper.write_texts: ExperimentHelper#write_texts().
  ExperimentHelper.save_ckpt: ExperimentHelper#save_ckpt().
  ExperimentHelper.set_notes: ExperimentHelper#set_notes().
  ExperimentHelper.ckpt_interval: ExperimentHelper#ckpt_interval.
  ExperimentHelper.ckpt_save_policy: ExperimentHelper#ckpt_save_policy().
  ExperimentHelper.ckpt_preservation_policy: ExperimentHelper#ckpt_preservation_policy().
  MetricsAggregator.add: MetricsAggregator#add().
  setup_work_unit: setup_work_unit().
  ExperimentHelper.num_train_steps: ExperimentHelper#num_train_steps.
  ExperimentHelper.__post_init__: ExperimentHelper#__post_init__().
  MetricsAggregator.metrics: MetricsAggregator#metrics().
  is_primary_process: is_primary_process().
  ExperimentHelper: ExperimentHelper#
  ExperimentHelper.metric_log_interval: ExperimentHelper#metric_log_interval.
  MetricsAggregator.average_last_n_steps: MetricsAggregator#average_last_n_steps.
  ExperimentHelper.metric_logdir: ExperimentHelper#metric_logdir().
  MetricsAggregator.get_aggregated_metrics: MetricsAggregator#get_aggregated_metrics().
  ExperimentHelper.ckpt_max_to_keep: ExperimentHelper#ckpt_max_to_keep.
  ExperimentHelper.log_additional_info: ExperimentHelper#log_additional_info.
  ExperimentHelper.should_save_ckpt: ExperimentHelper#should_save_ckpt.
  MetricsAggregator.__post_init__: MetricsAggregator#__post_init__().
  MetricsAggregator.reset: MetricsAggregator#reset().
  MetricsAggregator: MetricsAggregator#
  convert_to_scalar: convert_to_scalar().
  ExperimentHelper.write_record: ExperimentHelper#write_record().
---
# Module: [`simply/utils/experiment_helper.py`](../../../../../../raw/code/simply/simply/utils/experiment_helper.py)

## Classes
### `ExperimentHelper`
- def: [`simply/utils/experiment_helper.py:71`](../../../../../../raw/code/simply/simply/utils/experiment_helper.py#L71)
- doc: A utility class that saves all the experiment related data.
- signature: `class ExperimentHelper:`
- members:
  - `add_metric(self, metric_name: str, metric_value: np.typing.ArrayLike)` — [`L210`](../../../../../../raw/code/simply/simply/utils/experiment_helper.py#L210) — documented in [simply-model_lib](../../../concepts/simply-model_lib.md)
  - `ckpt_dir(self)` — [`L88`](../../../../../../raw/code/simply/simply/utils/experiment_helper.py#L88) — documented in [simply-utils-experiment_helper](../../../concepts/simply-utils-experiment_helper.md)
  - `ckpt_mngr(self)` — [`L123`](../../../../../../raw/code/simply/simply/utils/experiment_helper.py#L123) — Creates a checkpoint manager. — documented in [simply-utils-experiment_helper](../../../concepts/simply-utils-experiment_helper.md)
  - `ckpt_preservation_policy(self)` — [`L110`](../../../../../../raw/code/simply/simply/utils/experiment_helper.py#L110) — documented in [simply-utils-experiment_helper](../../../concepts/simply-utils-experiment_helper.md)
  - `ckpt_save_policy(self)` — [`L92`](../../../../../../raw/code/simply/simply/utils/experiment_helper.py#L92) — Creates a checkpoint save policy. — documented in [simply-utils-experiment_helper](../../../concepts/simply-utils-experiment_helper.md)
  - `close(self, final_result=None)` — [`L309`](../../../../../../raw/code/simply/simply/utils/experiment_helper.py#L309) — Closes the experiment helper and saves the final result. — documented in [simply-utils-experiment_helper](../../../concepts/simply-utils-experiment_helper.md)
  - `flush(self)` — [`L254`](../../../../../../raw/code/simply/simply/utils/experiment_helper.py#L254) — Flushes the metric writer.
  - `get_aggregated_metrics(self)` — [`L213`](../../../../../../raw/code/simply/simply/utils/experiment_helper.py#L213)
  - `metric_logdir(self)` — [`L147`](../../../../../../raw/code/simply/simply/utils/experiment_helper.py#L147)
  - `metric_writer(self)` — [`L151`](../../../../../../raw/code/simply/simply/utils/experiment_helper.py#L151) — documented in [simply-utils-experiment_helper](../../../concepts/simply-utils-experiment_helper.md)
  - `metrics_aggregator(self)` — [`L161`](../../../../../../raw/code/simply/simply/utils/experiment_helper.py#L161) — documented in [simply-utils-experiment_helper](../../../concepts/simply-utils-experiment_helper.md)
  - `save_ckpt(self, state, step, data=None)` — [`L305`](../../../../../../raw/code/simply/simply/utils/experiment_helper.py#L305) — documented in [simply-utils-experiment_helper](../../../concepts/simply-utils-experiment_helper.md)
  - `save_config_info(self, config, sharding_config, model=None)` — [`L173`](../../../../../../raw/code/simply/simply/utils/experiment_helper.py#L173) — Save model and config information. — documented in [simply-utils-experiment_helper](../../../concepts/simply-utils-experiment_helper.md)
  - `save_state_info(self, state)` — [`L259`](../../../../../../raw/code/simply/simply/utils/experiment_helper.py#L259) — Save state information.
  - `set_notes(self, notes: str)` — [`L164`](../../../../../../raw/code/simply/simply/utils/experiment_helper.py#L164)
  - `should_log_additional_info(self, step)` — [`L221`](../../../../../../raw/code/simply/simply/utils/experiment_helper.py#L221)
  - `should_log_metrics(self, step)` — [`L216`](../../../../../../raw/code/simply/simply/utils/experiment_helper.py#L216)
  - `should_save_data(self)` — [`L84`](../../../../../../raw/code/simply/simply/utils/experiment_helper.py#L84) — documented in [simply-utils-experiment_helper](../../../concepts/simply-utils-experiment_helper.md)
  - `write_record(self, record: Mapping[str, Any])` — [`L167`](../../../../../../raw/code/simply/simply/utils/experiment_helper.py#L167)
  - `write_scalars(self, step, scalars, filter_nonscalars=True)` — [`L224`](../../../../../../raw/code/simply/simply/utils/experiment_helper.py#L224) — Writes scalar metrics. — documented in [simply-utils-experiment_helper](../../../concepts/simply-utils-experiment_helper.md)
  - `write_texts(self, step, texts)` — [`L244`](../../../../../../raw/code/simply/simply/utils/experiment_helper.py#L244) — Writes text metrics.
  - `ckpt_interval` — [`L75`](../../../../../../raw/code/simply/simply/utils/experiment_helper.py#L75)
  - `ckpt_keep_period` — [`L77`](../../../../../../raw/code/simply/simply/utils/experiment_helper.py#L77)
  - `ckpt_max_to_keep` — [`L76`](../../../../../../raw/code/simply/simply/utils/experiment_helper.py#L76)
  - `experiment_dir` — [`L74`](../../../../../../raw/code/simply/simply/utils/experiment_helper.py#L74)
  - `log_additional_info` — [`L80`](../../../../../../raw/code/simply/simply/utils/experiment_helper.py#L80)
  - `metric_log_interval` — [`L78`](../../../../../../raw/code/simply/simply/utils/experiment_helper.py#L78)
  - `num_train_steps` — [`L79`](../../../../../../raw/code/simply/simply/utils/experiment_helper.py#L79)
  - `should_save_ckpt` — [`L81`](../../../../../../raw/code/simply/simply/utils/experiment_helper.py#L81)
- protocol/private: `__post_init__`[`L142`](../../../../../../raw/code/simply/simply/utils/experiment_helper.py#L142)
- uses (calls/refs, reference-scoped): [`dump`](pytree.md#dump), [`get_raw_arrays`](common.md#get_raw_arrays), [`save_checkpoint`](checkpoint_lib.md#save_checkpoint), [`set_notes`](experiment_helper.md#set_notes), [`create_metric_writer`](metric_writer.md#create_metric_writer), [`BaseMetricWriter`](metric_writer.md#BaseMetricWriter), [`add`](experiment_helper.md#MetricsAggregator.add), [`average_last_n_steps`](experiment_helper.md#MetricsAggregator.average_last_n_steps), [`is_primary_process`](experiment_helper.md#is_primary_process), [`close`](metric_writer.md#BaseMetricWriter.close), [`flush`](metric_writer.md#BaseMetricWriter.flush), [`get_aggregated_metrics`](experiment_helper.md#MetricsAggregator.get_aggregated_metrics), [`write_scalars`](metric_writer.md#BaseMetricWriter.write_scalars), [`write_texts`](metric_writer.md#BaseMetricWriter.write_texts), [`MetricsAggregator`](experiment_helper.md#MetricsAggregator), [`convert_to_scalar`](experiment_helper.md#convert_to_scalar)
- used by: [`run_experiment`](../rl_lib.md#run_experiment), [`main`](../eval/page_decode_eval.md#main), [`main`](../eval/decode_eval.md#main), [`run_experiment`](../model_lib.md#run_experiment), [`ExperimentHelper`](../model_lib.md#ExperimentHelper), [`ExperimentHelper`](../rl_lib.md#ExperimentHelper)

### `MetricsAggregator`
- def: [`simply/utils/experiment_helper.py:323`](../../../../../../raw/code/simply/simply/utils/experiment_helper.py#L323) — documented in [simply-utils-experiment_helper](../../../concepts/simply-utils-experiment_helper.md)
- members:
  - `add(self, name: str, value: np.typing.ArrayLike)` — [`L336`](../../../../../../raw/code/simply/simply/utils/experiment_helper.py#L336) — Adds a metric to the aggregator. — documented in [simply-utils-experiment_helper](../../../concepts/simply-utils-experiment_helper.md)
  - `get_aggregated_metrics(self)` — [`L349`](../../../../../../raw/code/simply/simply/utils/experiment_helper.py#L349)
  - `metrics(self)` — [`L333`](../../../../../../raw/code/simply/simply/utils/experiment_helper.py#L333)
  - `reset(self)` — [`L346`](../../../../../../raw/code/simply/simply/utils/experiment_helper.py#L346)
  - `average_last_n_steps` — [`L326`](../../../../../../raw/code/simply/simply/utils/experiment_helper.py#L326) — documented in [simply-utils-experiment_helper](../../../concepts/simply-utils-experiment_helper.md)
- protocol/private: `__post_init__`[`L328`](../../../../../../raw/code/simply/simply/utils/experiment_helper.py#L328)
- used by: [`add_metric`](experiment_helper.md#ExperimentHelper.add_metric), [`metrics_aggregator`](experiment_helper.md#ExperimentHelper.metrics_aggregator), [`get_aggregated_metrics`](experiment_helper.md#ExperimentHelper.get_aggregated_metrics)

## Functions
- `convert_to_scalar(x: Any)` — [`L50`](../../../../../../raw/code/simply/simply/utils/experiment_helper.py#L50) — Convert x to a single Python scalar.
- `is_primary_process()` — [`L37`](../../../../../../raw/code/simply/simply/utils/experiment_helper.py#L37) — Returns if the current process is the primary one. — documented in [simply-utils-experiment_helper](../../../concepts/simply-utils-experiment_helper.md)
- `is_primary_task()` — [`L43`](../../../../../../raw/code/simply/simply/utils/experiment_helper.py#L43) — Returns if the current process is on task 0.
- `set_notes(notes: str, should_set: bool | None = None)` — [`L62`](../../../../../../raw/code/simply/simply/utils/experiment_helper.py#L62)
- `setup_work_unit()` — [`L66`](../../../../../../raw/code/simply/simply/utils/experiment_helper.py#L66)


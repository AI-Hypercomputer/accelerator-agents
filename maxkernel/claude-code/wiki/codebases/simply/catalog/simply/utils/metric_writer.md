---
title: 'Module: simply/utils/metric_writer.py'
type: catalog
provenance: extracted
module: simply/utils/metric_writer.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.utils.metric_writer`/
symbols:
  create_metric_writer: create_metric_writer().
  BaseMetricWriter: BaseMetricWriter#
  TensorboardXMetricWriter.write_scalars: TensorboardXMetricWriter#write_scalars().
  TensorboardXMetricWriter.write_texts: TensorboardXMetricWriter#write_texts().
  TensorboardXMetricWriter: TensorboardXMetricWriter#
  TensorboardXMetricWriter._writer: TensorboardXMetricWriter#_writer.
  BaseMetricWriter.write_scalars: BaseMetricWriter#write_scalars().
  BaseMetricWriter.write_texts: BaseMetricWriter#write_texts().
  BaseMetricWriter.flush: BaseMetricWriter#flush().
  BaseMetricWriter.close: BaseMetricWriter#close().
  TensorboardXMetricWriter.flush: TensorboardXMetricWriter#flush().
  TensorboardXMetricWriter.close: TensorboardXMetricWriter#close().
  TensorboardXMetricWriter.just_logging: TensorboardXMetricWriter#just_logging.
  _HAS_CLU: _HAS_CLU.
  TensorboardXMetricWriter.__init__: TensorboardXMetricWriter#__init__().
---
# Module: [`simply/utils/metric_writer.py`](../../../../../../raw/code/simply/simply/utils/metric_writer.py)

## Classes
### `BaseMetricWriter`  ·  implements/extends ABC
- def: [`simply/utils/metric_writer.py:24`](../../../../../../raw/code/simply/simply/utils/metric_writer.py#L24) — documented in [simply-utils-experiment_helper](../../../concepts/simply-utils-experiment_helper.md)
- doc: Base class for metric writers.
- signature: `class BaseMetricWriter(abc.ABC):`
- members:
  - `close(self)` — [`L40`](../../../../../../raw/code/simply/simply/utils/metric_writer.py#L40)
  - `flush(self)` — [`L36`](../../../../../../raw/code/simply/simply/utils/metric_writer.py#L36)
  - `write_scalars(self, step: int, scalars: dict[str, Any])` — [`L28`](../../../../../../raw/code/simply/simply/utils/metric_writer.py#L28)
  - `write_texts(self, step: int, texts: dict[str, str])` — [`L32`](../../../../../../raw/code/simply/simply/utils/metric_writer.py#L32)
- uses (calls/refs, reference-scoped): [`write_scalars`](metric_writer.md#TensorboardXMetricWriter.write_scalars), [`write_texts`](metric_writer.md#TensorboardXMetricWriter.write_texts), [`TensorboardXMetricWriter`](metric_writer.md#TensorboardXMetricWriter), [`close`](metric_writer.md#TensorboardXMetricWriter.close), [`flush`](metric_writer.md#TensorboardXMetricWriter.flush)
- used by: [`metric_writer`](experiment_helper.md#ExperimentHelper.metric_writer), [`close`](experiment_helper.md#ExperimentHelper.close), [`write_scalars`](experiment_helper.md#ExperimentHelper.write_scalars), [`flush`](experiment_helper.md#ExperimentHelper.flush), [`create_metric_writer`](metric_writer.md#create_metric_writer), [`test_create_metric_writer`](metric_writer_test.md#MetricWriterTest.test_create_metric_writer), [`write_texts`](experiment_helper.md#ExperimentHelper.write_texts), [`TensorboardXMetricWriter`](metric_writer.md#TensorboardXMetricWriter)

### `TensorboardXMetricWriter`  ·  implements/extends BaseMetricWriter
- def: [`simply/utils/metric_writer.py:44`](../../../../../../raw/code/simply/simply/utils/metric_writer.py#L44)
- doc: Metric writer using tensorboardX.
- signature: `class TensorboardXMetricWriter(BaseMetricWriter):`
- members:
  - `close(self)` — [`L68`](../../../../../../raw/code/simply/simply/utils/metric_writer.py#L68)
  - `flush(self)` — [`L65`](../../../../../../raw/code/simply/simply/utils/metric_writer.py#L65)
  - `write_scalars(self, step: int, scalars: dict[str, Any])` — [`L51`](../../../../../../raw/code/simply/simply/utils/metric_writer.py#L51)
  - `write_texts(self, step: int, texts: dict[str, str])` — [`L58`](../../../../../../raw/code/simply/simply/utils/metric_writer.py#L58)
  - `just_logging` — [`L49`](../../../../../../raw/code/simply/simply/utils/metric_writer.py#L49)
- protocol/private: `__init__`[`L47`](../../../../../../raw/code/simply/simply/utils/metric_writer.py#L47), `_writer`[`L48`](../../../../../../raw/code/simply/simply/utils/metric_writer.py#L48)
- uses (calls/refs, reference-scoped): [`BaseMetricWriter`](metric_writer.md#BaseMetricWriter)
- used by: [`test_tensorboard_x_writer`](metric_writer_test.md#MetricWriterTest.test_tensorboard_x_writer), [`create_metric_writer`](metric_writer.md#create_metric_writer), [`BaseMetricWriter`](metric_writer.md#BaseMetricWriter), [`close`](metric_writer.md#BaseMetricWriter.close), [`flush`](metric_writer.md#BaseMetricWriter.flush), [`write_scalars`](metric_writer.md#BaseMetricWriter.write_scalars), [`write_texts`](metric_writer.md#BaseMetricWriter.write_texts)

## Functions
- `create_metric_writer(logdir: str, just_logging=False)` — [`L72`](../../../../../../raw/code/simply/simply/utils/metric_writer.py#L72) — Creates a metric writer based on the environment.

## Module values
- `_HAS_CLU` — [`L21`](../../../../../../raw/code/simply/simply/utils/metric_writer.py#L21)


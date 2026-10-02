---
title: 'Module: simply/utils/metric_writer_test.py'
type: catalog
provenance: extracted
module: simply/utils/metric_writer_test.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.utils.metric_writer_test`/MetricWriterTest#
symbols:
  MetricWriterTest.test_tensorboard_x_writer: test_tensorboard_x_writer().
  MetricWriterTest.test_create_metric_writer: test_create_metric_writer().
  MetricWriterTest: ''
---
# Module: [`simply/utils/metric_writer_test.py`](../../../../../../raw/code/simply/simply/utils/metric_writer_test.py)

## Classes
### `MetricWriterTest`  ·  implements/extends TestCase
- def: [`simply/utils/metric_writer_test.py:22`](../../../../../../raw/code/simply/simply/utils/metric_writer_test.py#L22)
- signature: `class MetricWriterTest(absltest.TestCase):`
- members:
  - `test_create_metric_writer(self)` — [`L41`](../../../../../../raw/code/simply/simply/utils/metric_writer_test.py#L41)
  - `test_tensorboard_x_writer(self)` — [`L24`](../../../../../../raw/code/simply/simply/utils/metric_writer_test.py#L24)
- uses (calls/refs, reference-scoped): [`create_metric_writer`](metric_writer.md#create_metric_writer), [`BaseMetricWriter`](metric_writer.md#BaseMetricWriter), [`write_scalars`](metric_writer.md#TensorboardXMetricWriter.write_scalars), [`write_texts`](metric_writer.md#TensorboardXMetricWriter.write_texts), [`TensorboardXMetricWriter`](metric_writer.md#TensorboardXMetricWriter), [`close`](metric_writer.md#TensorboardXMetricWriter.close), [`flush`](metric_writer.md#TensorboardXMetricWriter.flush), [`_HAS_CLU`](metric_writer.md#_HAS_CLU)


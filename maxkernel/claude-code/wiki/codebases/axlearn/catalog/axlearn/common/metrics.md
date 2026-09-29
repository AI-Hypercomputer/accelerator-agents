---
title: 'Module: axlearn/common/metrics.py'
type: catalog
provenance: extracted
module: axlearn/common/metrics.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.metrics`/
symbols:
  WeightedSummary: WeightedSummary#
  MetricSummary: MetricSummary#
  _ReducerSummary: _ReducerSummary#
  MetricAccumulator: MetricAccumulator#
  MetricAccumulator.update: MetricAccumulator#update().
  _ReducerSummary.value: _ReducerSummary#value().
  _ReducerSummary._accumulate_op: _ReducerSummary#_accumulate_op().
  _metric_accumulator_unflatten: _metric_accumulator_unflatten().
  WeightedValue.mean: WeightedValue#mean.
  WeightedValue.weight: WeightedValue#weight.
  WeightedValue.__init__: WeightedValue#__init__().
  WeightedSummary.__add__: WeightedSummary#__add__().
  _ReducerSummary.accumulate: _ReducerSummary#accumulate().
  MetricAccumulator.__init__: MetricAccumulator#__init__().
  MetricAccumulator._summaries: MetricAccumulator#_summaries.
  WeightedValue: WeightedValue#
  _metric_accumulator_flatten: _metric_accumulator_flatten().
  WeightedValue.value: WeightedValue#value().
  WeightedSummary.accumulate: WeightedSummary#accumulate().
  _ReducerSummary.validate: _ReducerSummary#validate().
  _ReducerSummary._value: _ReducerSummary#_value.
  MetricAccumulator.summaries: MetricAccumulator#summaries().
  MetricAccumulator._tree_map: MetricAccumulator#_tree_map().
  MetricAccumulator.is_leaf: MetricAccumulator#is_leaf().
  MinSummary: MinSummary#
  MinSummary._accumulate_op: MinSummary#_accumulate_op().
  MaxSummary: MaxSummary#
  MaxSummary._accumulate_op: MaxSummary#_accumulate_op().
  SumSummary: SumSummary#
  SumSummary._accumulate_op: SumSummary#_accumulate_op().
---
# Module: [`axlearn/common/metrics.py`](../../../../../../raw/code/axlearn/axlearn/common/metrics.py)

## Classes
### `MaxSummary`  ·  implements/extends _ReducerSummary
- def: [`axlearn/common/metrics.py:108`](../../../../../../raw/code/axlearn/axlearn/common/metrics.py#L108)
- doc: A summary that computes the maximum value across tensor elements.
- signature: `class MaxSummary(_ReducerSummary):`
- protocol/private: `_accumulate_op`[`L111`](../../../../../../raw/code/axlearn/axlearn/common/metrics.py#L111)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`_ReducerSummary`](metrics.md#_ReducerSummary)
- used by: [`_ReducerSummary`](metrics.md#_ReducerSummary), [`_accumulate_op`](metrics.md#_ReducerSummary._accumulate_op)

### `MetricAccumulator`  ·  implements/extends Configurable
- def: [`axlearn/common/metrics.py:122`](../../../../../../raw/code/axlearn/axlearn/common/metrics.py#L122)
- doc: A MetricAccumulator is used during evaluation to accumulate metrics across batches.
- signature: `class MetricAccumulator(Configurable):`
- members:
  - `is_leaf(x)` — [`L154`](../../../../../../raw/code/axlearn/axlearn/common/metrics.py#L154)
  - `summaries(self)` — [`L149`](../../../../../../raw/code/axlearn/axlearn/common/metrics.py#L149)
  - `update(self, model_outputs: dict[str, Any])` — [`L129`](../../../../../../raw/code/axlearn/axlearn/common/metrics.py#L129)
- protocol/private: `__init__`[`L125`](../../../../../../raw/code/axlearn/axlearn/common/metrics.py#L125), `_summaries`[`L127`](../../../../../../raw/code/axlearn/axlearn/common/metrics.py#L127), `_tree_map`[`L153`](../../../../../../raw/code/axlearn/axlearn/common/metrics.py#L153)
- uses (calls/refs, reference-scoped): [`Configurable`](config.md#Configurable), [`Summary`](summary.md#Summary), [`Config`](config.md#Configurable.Config), [`__init__`](config.md#Configurable.__init__)
- used by: [`Configurable`](config.md#Configurable), [`fwd_helper`](gradient_accumulation.md#with_minibatch_steps.decorator.fwd_helper), [`scan_body`](gradient_accumulation.md#with_minibatch_steps.decorator.fwd_helper.scan_body), [`init_state`](evaler.md#GlobalMetricCalculator.init_state), [`forward`](evaler.md#GlobalMetricCalculator.forward), [`_metric_accumulator_unflatten`](metrics.md#_metric_accumulator_unflatten), [`with_minibatch_steps`](gradient_accumulation.md#with_minibatch_steps), [`metric_accumulator`](evaler.md#ModelSummaryAccumulator.Config.metric_accumulator), [`_metric_accumulator_flatten`](metrics.md#_metric_accumulator_flatten), [`_metric_accumulator`](evaler.md#GlobalMetricCalculator._metric_accumulator)

### `MetricSummary`  ·  implements/extends Summary
- def: [`axlearn/common/metrics.py:16`](../../../../../../raw/code/axlearn/axlearn/common/metrics.py#L16)
- doc: Base class for metric summaries computed during training/evaluation.
- signature: `class MetricSummary(Summary):`
- uses (calls/refs, reference-scoped): [`Summary`](summary.md#Summary), [`_ReducerSummary`](metrics.md#_ReducerSummary), [`WeightedValue`](metrics.md#WeightedValue)
- used by: [`Summary`](summary.md#Summary), [`_ReducerSummary`](metrics.md#_ReducerSummary), [`get_summaries`](evaler.md#BaseMetricCalculator.get_summaries), [`precision_recall_f_score`](metrics_classification.md#precision_recall_f_score), [`forward`](loss_metrics.md#BaseLossMetrics.forward), [`get_summaries`](evaler.md#CompositeMetricCalculator.get_summaries), [`get_summaries`](evaler.md#ModelSummaryAccumulator.get_summaries), [`WeightedValue`](metrics.md#WeightedValue)

### `MinSummary`  ·  implements/extends _ReducerSummary
- def: [`axlearn/common/metrics.py:101`](../../../../../../raw/code/axlearn/axlearn/common/metrics.py#L101)
- doc: A summary that computes the minimum value across tensor elements.
- signature: `class MinSummary(_ReducerSummary):`
- protocol/private: `_accumulate_op`[`L104`](../../../../../../raw/code/axlearn/axlearn/common/metrics.py#L104)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`_ReducerSummary`](metrics.md#_ReducerSummary)
- used by: [`_ReducerSummary`](metrics.md#_ReducerSummary), [`_accumulate_op`](metrics.md#_ReducerSummary._accumulate_op)

### `SumSummary`  ·  implements/extends _ReducerSummary
- def: [`axlearn/common/metrics.py:115`](../../../../../../raw/code/axlearn/axlearn/common/metrics.py#L115)
- doc: A summary that computes the sum of values across tensor elements.
- signature: `class SumSummary(_ReducerSummary):`
- protocol/private: `_accumulate_op`[`L118`](../../../../../../raw/code/axlearn/axlearn/common/metrics.py#L118)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`_ReducerSummary`](metrics.md#_ReducerSummary)
- used by: [`_ReducerSummary`](metrics.md#_ReducerSummary), [`_accumulate_op`](metrics.md#_ReducerSummary._accumulate_op)

### `WeightedSummary`  ·  implements/extends WeightedValue
- def: [`axlearn/common/metrics.py:35`](../../../../../../raw/code/axlearn/axlearn/common/metrics.py#L35)
- doc: A weighted tensor represents a weighted Summable value.
- signature: `class WeightedSummary(WeightedValue):`
- members:
  - `accumulate(self, other: Summary)` — [`L56`](../../../../../../raw/code/axlearn/axlearn/common/metrics.py#L56)
- protocol/private: `__add__`[`L42`](../../../../../../raw/code/axlearn/axlearn/common/metrics.py#L42)
- uses (calls/refs, reference-scoped): [`Summary`](summary.md#Summary), [`mean`](metrics.md#WeightedValue.mean), [`weight`](metrics.md#WeightedValue.weight), [`WeightedValue`](metrics.md#WeightedValue)
- used by: [`forward`](loss_metrics.md#ModalityLossMetrics.forward), [`get_summaries`](evaler.md#GlobalMetricCalculator.get_summaries), [`accumulate`](summary.md#Summary.accumulate), [`forward`](layers.md#BinaryClassificationMetric.forward), [`_maybe_add_clipping_summary`](quantized_dot_general/activation_clipping.md#BaseActivationClippingLayer._maybe_add_clipping_summary), [`precision_recall_f_score`](metrics_classification.md#precision_recall_f_score), [`forward`](layers.md#CategoricalHingeLossMetric.forward), [`forward`](layers.md#ClassificationMetric.forward), [`forward`](loss_metrics.md#BaseLossMetrics.forward), [`__call__`](checkpointer.md#BestMetricPolicy.__call__), [`_weighted_mean`](loss.md#_weighted_mean), [`_add_activation_summary`](base_layer.md#BaseLayer._add_activation_summary), [`bilinear_mean_squared_error`](loss.md#bilinear_mean_squared_error), [`l1_loss`](loss.md#l1_loss), [`mean_squared_error`](loss.md#mean_squared_error), [`WeightedValue`](metrics.md#WeightedValue)

### `WeightedValue`  ·  implements/extends MetricSummary
- def: [`axlearn/common/metrics.py:20`](../../../../../../raw/code/axlearn/axlearn/common/metrics.py#L20)
- doc: A weighted mean tensor represents a mean value and a weight.
- signature: `class WeightedValue(MetricSummary):`
- members:
  - `value(self)` — [`L31`](../../../../../../raw/code/axlearn/axlearn/common/metrics.py#L31)
  - `mean` — [`L23`](../../../../../../raw/code/axlearn/axlearn/common/metrics.py#L23)
  - `weight` — [`L24`](../../../../../../raw/code/axlearn/axlearn/common/metrics.py#L24)
- protocol/private: `__init__`[`L26`](../../../../../../raw/code/axlearn/axlearn/common/metrics.py#L26)
- uses (calls/refs, reference-scoped): [`NestedTensor`](utils.md#NestedTensor), [`WeightedSummary`](metrics.md#WeightedSummary), [`MetricSummary`](metrics.md#MetricSummary)
- used by: [`WeightedSummary`](metrics.md#WeightedSummary), [`MetricSummary`](metrics.md#MetricSummary), [`value`](summary.md#Summary.value), [`__call__`](checkpointer.md#BestMetricPolicy.__call__), [`__add__`](metrics.md#WeightedSummary.__add__)

### `_ReducerSummary`  ·  implements/extends MetricSummary
- def: [`axlearn/common/metrics.py:62`](../../../../../../raw/code/axlearn/axlearn/common/metrics.py#L62)
- doc: Base class for summaries that reduce tensor elements across accumulation.
- signature: `class _ReducerSummary(MetricSummary):`
- members:
  - `_accumulate_op(self, this_value: Tensor, other_value: Tensor)` — [`L80`](../../../../../../raw/code/axlearn/axlearn/common/metrics.py#L80) — Defines how two values are combined during accumulation.
  - `accumulate(self, other: _ReducerSummary)` — [`L92`](../../../../../../raw/code/axlearn/axlearn/common/metrics.py#L92)
  - `validate(self)` — [`L74`](../../../../../../raw/code/axlearn/axlearn/common/metrics.py#L74)
  - `value(self)` — [`L71`](../../../../../../raw/code/axlearn/axlearn/common/metrics.py#L71)
- protocol/private: `_value`[`L69`](../../../../../../raw/code/axlearn/axlearn/common/metrics.py#L69)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`MetricSummary`](metrics.md#MetricSummary), [`MaxSummary`](metrics.md#MaxSummary), [`MinSummary`](metrics.md#MinSummary), [`SumSummary`](metrics.md#SumSummary), [`_accumulate_op`](metrics.md#MaxSummary._accumulate_op), [`_accumulate_op`](metrics.md#MinSummary._accumulate_op), [`_accumulate_op`](metrics.md#SumSummary._accumulate_op)
- used by: [`MetricSummary`](metrics.md#MetricSummary), [`value`](summary.md#Summary.value), [`accumulate`](summary.md#Summary.accumulate), [`validate`](summary.md#Summary.validate), [`MaxSummary`](metrics.md#MaxSummary), [`MinSummary`](metrics.md#MinSummary), [`SumSummary`](metrics.md#SumSummary)

## Functions
- `_metric_accumulator_flatten(v: MetricAccumulator)` — [`L160`](../../../../../../raw/code/axlearn/axlearn/common/metrics.py#L160) — Specifies a flattening recipe for `MetricAccumulator`.
- `_metric_accumulator_unflatten(summaries_keys: tuple, summaries_values: tuple)` — [`L170`](../../../../../../raw/code/axlearn/axlearn/common/metrics.py#L170) — Specifies an unflattening recipe for `MetricAccumulator`.


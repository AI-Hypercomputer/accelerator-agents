---
title: 'Module: axlearn/common/loss_metrics.py'
type: catalog
provenance: extracted
module: axlearn/common/loss_metrics.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.loss_metrics`/
symbols:
  ModalityLossMetrics.forward: ModalityLossMetrics#forward().
  filter_module_outputs: filter_module_outputs().
  BaseLossMetrics.forward: BaseLossMetrics#forward().
  ModalityLossMetrics.Config.inner: ModalityLossMetrics#Config#inner.
  ModalityLossMetrics.Config: ModalityLossMetrics#Config#
  BaseLossMetrics: BaseLossMetrics#
  ModalityLossMetrics.Config.modality_vocab_info: ModalityLossMetrics#Config#modality_vocab_info.
  ModalityLossMetrics.__init__: ModalityLossMetrics#__init__().
  ModalityLossMetrics: ModalityLossMetrics#
  ModalityLossMetrics.Config.target_labels_pattern: ModalityLossMetrics#Config#target_labels_pattern.
---
# Module: [`axlearn/common/loss_metrics.py`](../../../../../../raw/code/axlearn/axlearn/common/loss_metrics.py)

## Classes
### `BaseLossMetrics`  ·  implements/extends BaseLayer
- def: [`axlearn/common/loss_metrics.py:25`](../../../../../../raw/code/axlearn/axlearn/common/loss_metrics.py#L25)
- doc: A module for computing training time metrics.
- signature: `class BaseLossMetrics(BaseLayer):`
- members:
  - `forward(self, input_batch: Nested[Tensor], *, predict_outputs: Nested[Tensor], module_outputs: Nested[Tensor])` — [`L31`](../../../../../../raw/code/axlearn/axlearn/common/loss_metrics.py#L31) — Computes metrics from inputs and predictions.
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested), [`BaseLayer`](base_layer.md#BaseLayer), [`WeightedSummary`](metrics.md#WeightedSummary), [`forward`](loss_metrics.md#ModalityLossMetrics.forward), [`MetricSummary`](metrics.md#MetricSummary), [`ModalityLossMetrics`](loss_metrics.md#ModalityLossMetrics)
- used by: [`BaseLayer`](base_layer.md#BaseLayer), [`inner`](loss_metrics.md#ModalityLossMetrics.Config.inner), [`Config`](loss_metrics.md#ModalityLossMetrics.Config), [`ModalityLossMetrics`](loss_metrics.md#ModalityLossMetrics)

### `Config`  ·  implements/extends Config
- def: [`axlearn/common/loss_metrics.py:96`](../../../../../../raw/code/axlearn/axlearn/common/loss_metrics.py#L96)
- doc: Configures ModalityLossMetrics.
- signature: `class Config(BaseLossMetrics.Config):`
- members:
  - `inner` — [`L106`](../../../../../../raw/code/axlearn/axlearn/common/loss_metrics.py#L106)
  - `modality_vocab_info` — [`L107`](../../../../../../raw/code/axlearn/axlearn/common/loss_metrics.py#L107)
  - `target_labels_pattern` — [`L108`](../../../../../../raw/code/axlearn/axlearn/common/loss_metrics.py#L108)
- uses (calls/refs, reference-scoped): [`config_class`](config.md#config_class), [`REQUIRED`](config.md#REQUIRED), [`Required`](config.md#Required), [`Config`](base_layer.md#BaseLayer.Config), [`BaseLossMetrics`](loss_metrics.md#BaseLossMetrics), [`ModalityVocabInfo`](embedding.md#ModalityVocabInfo)
- used by: [`forward`](loss_metrics.md#ModalityLossMetrics.forward)

### `ModalityLossMetrics`  ·  implements/extends BaseLossMetrics
- def: [`axlearn/common/loss_metrics.py:87`](../../../../../../raw/code/axlearn/axlearn/common/loss_metrics.py#L87)
- doc: Wraps a metrics implementation with per-modality masking.
- signature: `class ModalityLossMetrics(BaseLossMetrics):`
- members:
  - `forward(self, input_batch: Nested[Tensor], *, predict_outputs: Nested[Tensor], module_outputs: Nested[Tensor])` — [`L114`](../../../../../../raw/code/axlearn/axlearn/common/loss_metrics.py#L114) — Computes loss and metrics.
- protocol/private: `__init__`[`L110`](../../../../../../raw/code/axlearn/axlearn/common/loss_metrics.py#L110)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested), [`config`](config.md#Configurable.config), [`_add_child`](module.md#Module._add_child), [`__init__`](base_layer.md#BaseLayer.__init__), [`WeightedSummary`](metrics.md#WeightedSummary), [`validate_contains_paths`](utils.md#validate_contains_paths), [`filter_module_outputs`](loss_metrics.md#filter_module_outputs), [`Config`](loss_metrics.md#ModalityLossMetrics.Config), [`BaseLossMetrics`](loss_metrics.md#BaseLossMetrics), [`modality_vocab_info`](loss_metrics.md#ModalityLossMetrics.Config.modality_vocab_info), [`placeholder_end`](embedding.md#ModalityVocabInfo.placeholder_end), [`placeholder_start`](embedding.md#ModalityVocabInfo.placeholder_start), [`target_labels_pattern`](loss_metrics.md#ModalityLossMetrics.Config.target_labels_pattern)
- used by: [`forward`](loss_metrics.md#BaseLossMetrics.forward), [`BaseLossMetrics`](loss_metrics.md#BaseLossMetrics)

## Functions
- `filter_module_outputs(module_outputs: Nested[Tensor], *, path_regex: str, default: Any = REQUIRED)` — [`L53`](../../../../../../raw/code/axlearn/axlearn/common/loss_metrics.py#L53) — Retrieves the leaf value(s) corresponding to `path_regex` from `module_outputs`.


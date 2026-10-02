---
title: 'Module: axlearn/common/base_model.py'
type: catalog
provenance: extracted
module: axlearn/common/base_model.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.base_model`/BaseModel#
symbols:
  BaseModel: ''
  BaseModel.forward: forward().
---
# Module: [`axlearn/common/base_model.py`](../../../../../../raw/code/axlearn/axlearn/common/base_model.py)

## Classes
### `BaseModel`  ·  implements/extends BaseLayer
- def: [`axlearn/common/base_model.py:46`](../../../../../../raw/code/axlearn/axlearn/common/base_model.py#L46)
- doc: Base class for trainable models that compute losses for optimization.
- signature: `class BaseModel(BaseLayer):`
- members:
  - `forward(self, input_batch: NestedTensor)` — [`L117`](../../../../../../raw/code/axlearn/axlearn/common/base_model.py#L117) — Computes loss and auxiliary outputs with the given inputs.
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`NestedTensor`](utils.md#NestedTensor), [`BaseLayer`](base_layer.md#BaseLayer)  (2 test-only)
- used by: [`BaseLayer`](base_layer.md#BaseLayer), [`__init__`](evaler.md#SpmdEvaler.__init__), [`model`](trainer.md#SpmdTrainer.Config.model), [`__init__`](evaler.md#BaseMetricCalculator.__init__), [`__init__`](evaler.md#CompositeMetricCalculator.__init__), [`__init__`](evaler.md#GlobalMetricCalculator.__init__), [`__init__`](evaler.md#ModelSummaryAccumulator.__init__)  (3 test-only)


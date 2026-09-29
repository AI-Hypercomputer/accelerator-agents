---
title: 'Module: axlearn/common/learner_base.py'
type: catalog
provenance: extracted
module: axlearn/common/learner_base.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.learner_base`/LearnerModule#
symbols:
  LearnerModule.create_state_partition_specs: create_state_partition_specs().
  LearnerModule.init: init().
  LearnerModule: ''
---
# Module: [`axlearn/common/learner_base.py`](../../../../../../raw/code/axlearn/axlearn/common/learner_base.py)

## Classes
### `LearnerModule`  ·  implements/extends Module
- def: [`axlearn/common/learner_base.py:14`](../../../../../../raw/code/axlearn/axlearn/common/learner_base.py#L14)
- doc: Any stateful module used inside a `BaseLearner`, including the learner itself.
- signature: `class LearnerModule(Module):`
- members:
  - `create_state_partition_specs(self, model_param_specs: Nested[ParameterSpec])` — [`L21`](../../../../../../raw/code/axlearn/axlearn/common/learner_base.py#L21) — Creates learner state partition_specs.
  - `init(self, model_params: Nested[OptParam])` — [`L29`](../../../../../../raw/code/axlearn/axlearn/common/learner_base.py#L29) — Initializes learner state.
- uses (calls/refs, reference-scoped): [`Nested`](utils.md#Nested), [`Module`](module.md#Module), [`ParameterSpec`](base_layer.md#ParameterSpec), [`OptParam`](optimizer_base.md#OptParam), [`init`](learner.md#Learner.init), [`create_state_partition_specs`](update_transformation.md#ConditionalUpdateTransformation.create_state_partition_specs), [`init`](learner.md#CompositeLearner.init), [`create_state_partition_specs`](learner.md#Learner.create_state_partition_specs), [`UpdateTransformation`](update_transformation.md#UpdateTransformation), [`create_state_partition_specs`](learner.md#CompositeLearner.create_state_partition_specs), [`init`](update_transformation.md#ConditionalUpdateTransformation.init), [`init`](update_transformation.md#WrappedPartitionedGradientTransformation.init), [`create_state_partition_specs`](update_transformation.md#WrappedPartitionedGradientTransformation.create_state_partition_specs), [`BaseLearner`](learner.md#BaseLearner)
- used by: [`Module`](module.md#Module), [`init`](learner.md#Learner.init), [`create_state_partition_specs`](update_transformation.md#ConditionalUpdateTransformation.create_state_partition_specs), [`__init__`](learner.md#Learner.__init__), [`__init__`](update_transformation.md#ConditionalUpdateTransformation.__init__), [`create_state_partition_specs`](learner.md#Learner.create_state_partition_specs), [`UpdateTransformation`](update_transformation.md#UpdateTransformation), [`init`](update_transformation.md#ConditionalUpdateTransformation.init), [`BaseLearner`](learner.md#BaseLearner)


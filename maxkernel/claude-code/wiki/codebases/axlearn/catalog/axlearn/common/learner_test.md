---
title: 'Module: axlearn/common/learner_test.py'
type: catalog
provenance: extracted
module: axlearn/common/learner_test.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.learner_test`/
symbols:
  _Model: _Model#
  _Model.Config: _Model#Config#
  _Model.Config.dim: _Model#Config#dim.
  _Model.__init__: _Model#__init__().
  _Model.forward: _Model#forward().
  LearnerTest: LearnerTest#
  LearnerTest.test_forward_and_backward: LearnerTest#test_forward_and_backward().
  LearnerTest.test_learner: LearnerTest#test_learner().
  LearnerTest.test_update_rules: LearnerTest#test_update_rules().
  LearnerTest.test_update_rules_on_vdict: LearnerTest#test_update_rules_on_vdict().
  LearnerTest.test_per_variable_summaries: LearnerTest#test_per_variable_summaries().
  LearnerTest.test_inplace_updates_supersede_delta_updates: LearnerTest#test_inplace_updates_supersede_delta_updates().
  LearnerTest.test_gradient_accumulation_init: LearnerTest#test_gradient_accumulation_init().
  LearnerTest.test_grad_accumulation_numeric: LearnerTest#test_grad_accumulation_numeric().
  HelperTest: HelperTest#
  HelperTest.test__apply_updates: HelperTest#test__apply_updates().
  HelperTest._forward: HelperTest#_forward().
  HelperTest.test__split_gradients: HelperTest#test__split_gradients().
  HelperTest.test__value_and_grad: HelperTest#test__value_and_grad().
  CompositeLearnerTest: CompositeLearnerTest#
  CompositeLearnerTest.test_forward_and_backward: CompositeLearnerTest#test_forward_and_backward().
  CompositeLearnerTest.test_learner: CompositeLearnerTest#test_learner().
  CompositeLearnerTest.test_learner_config: CompositeLearnerTest#test_learner_config().
  CompositeLearnerTest.test_sublearner_ema: CompositeLearnerTest#test_sublearner_ema().
  CompositeLearnerTest.test_learner_masking: CompositeLearnerTest#test_learner_masking().
  CompositeLearnerTest.test_fp8_override_update: CompositeLearnerTest#test_fp8_override_update().
---
# Module: [`axlearn/common/learner_test.py`](../../../../../../raw/code/axlearn/axlearn/common/learner_test.py)

## Classes
### `CompositeLearnerTest`
- def: [`axlearn/common/learner_test.py:949`](../../../../../../raw/code/axlearn/axlearn/common/learner_test.py#L949)
- signature: `class CompositeLearnerTest(TestCase):`
- members:
  - `test_forward_and_backward(self, ema_decay)` — [`L951`](../../../../../../raw/code/axlearn/axlearn/common/learner_test.py#L951) — Demonstrates how API users should use the API while ensuring that it works correctly.
  - `test_fp8_override_update(self, use_override_inplace_update)` — [`L1570`](../../../../../../raw/code/axlearn/axlearn/common/learner_test.py#L1570) — Tests FP8 with `OverrideInplaceUpdateTransformation`
  - `test_learner(self, ema_decay: Optional[float], method: str)` — [`L1054`](../../../../../../raw/code/axlearn/axlearn/common/learner_test.py#L1054) — Sets up two sub learners for encoder/decoder respectively.
  - `test_learner_config(self)` — [`L1335`](../../../../../../raw/code/axlearn/axlearn/common/learner_test.py#L1335)
  - `test_learner_masking(test_self)` — [`L1436`](../../../../../../raw/code/axlearn/axlearn/common/learner_test.py#L1436) — In-depth test of the masking of `Learner` and `CompositeLearner`.
  - `test_sublearner_ema(self)` — [`L1417`](../../../../../../raw/code/axlearn/axlearn/common/learner_test.py#L1417)

### `Config`
- def: [`axlearn/common/learner_test.py:76`](../../../../../../raw/code/axlearn/axlearn/common/learner_test.py#L76)
- signature: `class Config(BaseModel.Config):`
- members:
  - `dim` — [`L77`](../../../../../../raw/code/axlearn/axlearn/common/learner_test.py#L77)

### `HelperTest`
- def: [`axlearn/common/learner_test.py:841`](../../../../../../raw/code/axlearn/axlearn/common/learner_test.py#L841)
- doc: Test helper functions.
- signature: `class HelperTest(TestCase):`
- members:
  - `test__apply_updates(self)` — [`L844`](../../../../../../raw/code/axlearn/axlearn/common/learner_test.py#L844)
  - `test__split_gradients(self)` — [`L883`](../../../../../../raw/code/axlearn/axlearn/common/learner_test.py#L883)
  - `test__value_and_grad(self)` — [`L898`](../../../../../../raw/code/axlearn/axlearn/common/learner_test.py#L898)
- protocol/private: `_forward`[`L873`](../../../../../../raw/code/axlearn/axlearn/common/learner_test.py#L873)

### `LearnerTest`
- def: [`axlearn/common/learner_test.py:101`](../../../../../../raw/code/axlearn/axlearn/common/learner_test.py#L101)
- signature: `class LearnerTest(TestCase):`
- members:
  - `test_forward_and_backward(self, ema_decay)` — [`L103`](../../../../../../raw/code/axlearn/axlearn/common/learner_test.py#L103) — Demonstrates how API users should use the API while ensuring that it works correctly.
  - `test_grad_accumulation_numeric(self)` — [`L754`](../../../../../../raw/code/axlearn/axlearn/common/learner_test.py#L754) — Test that the gradient accumulation works as expected.
  - `test_gradient_accumulation_init(self, accumulation_steps)` — [`L733`](../../../../../../raw/code/axlearn/axlearn/common/learner_test.py#L733)
  - `test_inplace_updates_supersede_delta_updates(self)` — [`L706`](../../../../../../raw/code/axlearn/axlearn/common/learner_test.py#L706) — Tests that inplace updates take precedence over delta updates.
  - `test_learner(self, ema_decay: Optional[float], method: str)` — [`L177`](../../../../../../raw/code/axlearn/axlearn/common/learner_test.py#L177)
  - `test_per_variable_summaries(self, enable_per_variable_summaries)` — [`L592`](../../../../../../raw/code/axlearn/axlearn/common/learner_test.py#L592)
  - `test_update_rules(self, update_rules)` — [`L351`](../../../../../../raw/code/axlearn/axlearn/common/learner_test.py#L351)
  - `test_update_rules_on_vdict(self, update_rules)` — [`L507`](../../../../../../raw/code/axlearn/axlearn/common/learner_test.py#L507)

### `_Model`
- def: [`axlearn/common/learner_test.py:72`](../../../../../../raw/code/axlearn/axlearn/common/learner_test.py#L72)
- doc: A simple model for test.
- signature: `class _Model(BaseModel):`
- members:
  - `forward(self, input_batch: NestedTensor)` — [`L93`](../../../../../../raw/code/axlearn/axlearn/common/learner_test.py#L93)
- protocol/private: `__init__`[`L79`](../../../../../../raw/code/axlearn/axlearn/common/learner_test.py#L79)


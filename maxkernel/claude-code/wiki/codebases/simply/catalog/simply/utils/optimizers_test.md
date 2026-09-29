---
title: 'Module: simply/utils/optimizers_test.py'
type: catalog
provenance: extracted
module: simply/utils/optimizers_test.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.utils.optimizers_test`/OptimizerTest#
symbols:
  OptimizerTest.assert_almost_equal: assert_almost_equal().
  OptimizerTest.test_schedule_backward_compatibility_constant.MockConfigV1.lr: test_schedule_backward_compatibility_constant().MockConfigV1#lr.
  OptimizerTest.test_schedule_backward_compatibility_cosine_decay.MockConfigV1.lr: test_schedule_backward_compatibility_cosine_decay().MockConfigV1#lr.
  OptimizerTest.test_schedule_backward_compatibility_cosine_decay_fraction: test_schedule_backward_compatibility_cosine_decay_fraction().
  OptimizerTest.test_schedule_backward_compatibility_constant.MockConfigV1: test_schedule_backward_compatibility_constant().MockConfigV1#
  OptimizerTest.test_schedule_backward_compatibility_cosine_decay.MockConfigV1: test_schedule_backward_compatibility_cosine_decay().MockConfigV1#
  OptimizerTest.test_schedule_backward_compatibility_cosine_decay_fraction.MockConfigV1: test_schedule_backward_compatibility_cosine_decay_fraction().MockConfigV1#
  OptimizerTest.test_dump: test_dump().
  OptimizerTest.test_sgd: test_sgd().
  OptimizerTest.test_adam: test_adam().
  OptimizerTest.test_lion: test_lion().
  OptimizerTest.test_init_step: test_init_step().
  OptimizerTest._assert_tensor_almost_equal: _assert_tensor_almost_equal().
  OptimizerTest.test_schedule_backward_compatibility_constant.MockConfigV0: test_schedule_backward_compatibility_constant().MockConfigV0#
  OptimizerTest.test_schedule_backward_compatibility_cosine_decay.MockConfigV0: test_schedule_backward_compatibility_cosine_decay().MockConfigV0#
  OptimizerTest.test_schedule_backward_compatibility_cosine_decay_fraction.MockConfigV0: test_schedule_backward_compatibility_cosine_decay_fraction().MockConfigV0#
  OptimizerTest: ''
  OptimizerTest.test_schedule_backward_compatibility_constant: test_schedule_backward_compatibility_constant().
  OptimizerTest.test_schedule_backward_compatibility_constant.MockConfigV0.num_train_steps: test_schedule_backward_compatibility_constant().MockConfigV0#num_train_steps.
  OptimizerTest.test_schedule_backward_compatibility_constant.MockConfigV0.lr_schedule_name: test_schedule_backward_compatibility_constant().MockConfigV0#lr_schedule_name.
  OptimizerTest.test_schedule_backward_compatibility_constant.MockConfigV0.lr_schedule_config: test_schedule_backward_compatibility_constant().MockConfigV0#lr_schedule_config.
  OptimizerTest.test_schedule_backward_compatibility_constant.MockConfigV1.num_train_steps: test_schedule_backward_compatibility_constant().MockConfigV1#num_train_steps.
  OptimizerTest.test_schedule_backward_compatibility_cosine_decay: test_schedule_backward_compatibility_cosine_decay().
  OptimizerTest.test_schedule_backward_compatibility_cosine_decay.MockConfigV0.num_train_steps: test_schedule_backward_compatibility_cosine_decay().MockConfigV0#num_train_steps.
  OptimizerTest.test_schedule_backward_compatibility_cosine_decay.MockConfigV0.lr_schedule_name: test_schedule_backward_compatibility_cosine_decay().MockConfigV0#lr_schedule_name.
  OptimizerTest.test_schedule_backward_compatibility_cosine_decay.MockConfigV0.lr_schedule_config: test_schedule_backward_compatibility_cosine_decay().MockConfigV0#lr_schedule_config.
  OptimizerTest.test_schedule_backward_compatibility_cosine_decay.MockConfigV1.num_train_steps: test_schedule_backward_compatibility_cosine_decay().MockConfigV1#num_train_steps.
  OptimizerTest.test_schedule_backward_compatibility_cosine_decay_fraction.MockConfigV0.num_train_steps: test_schedule_backward_compatibility_cosine_decay_fraction().MockConfigV0#num_train_steps.
  OptimizerTest.test_schedule_backward_compatibility_cosine_decay_fraction.MockConfigV0.lr_schedule_name: test_schedule_backward_compatibility_cosine_decay_fraction().MockConfigV0#lr_schedule_name.
  OptimizerTest.test_schedule_backward_compatibility_cosine_decay_fraction.MockConfigV0.lr_schedule_config: test_schedule_backward_compatibility_cosine_decay_fraction().MockConfigV0#lr_schedule_config.
  OptimizerTest.test_schedule_backward_compatibility_cosine_decay_fraction.MockConfigV1.num_train_steps: test_schedule_backward_compatibility_cosine_decay_fraction().MockConfigV1#num_train_steps.
  OptimizerTest.test_schedule_backward_compatibility_cosine_decay_fraction.MockConfigV1.lr: test_schedule_backward_compatibility_cosine_decay_fraction().MockConfigV1#lr.
---
# Module: [`simply/utils/optimizers_test.py`](../../../../../../raw/code/simply/simply/utils/optimizers_test.py)

## Classes
### `MockConfigV0`
- def: [`simply/utils/optimizers_test.py:148`](../../../../../../raw/code/simply/simply/utils/optimizers_test.py#L148)
- signature: `class MockConfigV0:`
- members:
  - `lr_schedule_config` — [`L97`](../../../../../../raw/code/simply/simply/utils/optimizers_test.py#L97)
  - `lr_schedule_config` — [`L122`](../../../../../../raw/code/simply/simply/utils/optimizers_test.py#L122)
  - `lr_schedule_config` — [`L151`](../../../../../../raw/code/simply/simply/utils/optimizers_test.py#L151)
  - `lr_schedule_name` — [`L96`](../../../../../../raw/code/simply/simply/utils/optimizers_test.py#L96)
  - `lr_schedule_name` — [`L121`](../../../../../../raw/code/simply/simply/utils/optimizers_test.py#L121)
  - `lr_schedule_name` — [`L150`](../../../../../../raw/code/simply/simply/utils/optimizers_test.py#L150)
  - `num_train_steps` — [`L95`](../../../../../../raw/code/simply/simply/utils/optimizers_test.py#L95)
  - `num_train_steps` — [`L120`](../../../../../../raw/code/simply/simply/utils/optimizers_test.py#L120)
  - `num_train_steps` — [`L149`](../../../../../../raw/code/simply/simply/utils/optimizers_test.py#L149)
- used by: [`test_schedule_backward_compatibility_cosine_decay_fraction`](optimizers_test.md#OptimizerTest.test_schedule_backward_compatibility_cosine_decay_fraction)

### `MockConfigV1`
- def: [`simply/utils/optimizers_test.py:160`](../../../../../../raw/code/simply/simply/utils/optimizers_test.py#L160)
- signature: `class MockConfigV1:`
- members:
  - `lr` — [`L105`](../../../../../../raw/code/simply/simply/utils/optimizers_test.py#L105)
  - `lr` — [`L132`](../../../../../../raw/code/simply/simply/utils/optimizers_test.py#L132)
  - `lr` — [`L162`](../../../../../../raw/code/simply/simply/utils/optimizers_test.py#L162)
  - `num_train_steps` — [`L104`](../../../../../../raw/code/simply/simply/utils/optimizers_test.py#L104)
  - `num_train_steps` — [`L131`](../../../../../../raw/code/simply/simply/utils/optimizers_test.py#L131)
  - `num_train_steps` — [`L161`](../../../../../../raw/code/simply/simply/utils/optimizers_test.py#L161)
- uses (calls/refs, reference-scoped): [`LinearWarmupCosineDecay`](optimizers.md#LinearWarmupCosineDecay), [`create_lr_schedule`](optimizers.md#create_lr_schedule), [`Schedule`](optimizers.md#Schedule), [`MockConfigV1`](optimizers_test.md#OptimizerTest.test_schedule_backward_compatibility_constant.MockConfigV1), [`MockConfigV1`](optimizers_test.md#OptimizerTest.test_schedule_backward_compatibility_cosine_decay.MockConfigV1), [`MockConfigV0`](optimizers_test.md#OptimizerTest.test_schedule_backward_compatibility_constant.MockConfigV0), [`MockConfigV0`](optimizers_test.md#OptimizerTest.test_schedule_backward_compatibility_cosine_decay.MockConfigV0)
- used by: [`test_schedule_backward_compatibility_cosine_decay_fraction`](optimizers_test.md#OptimizerTest.test_schedule_backward_compatibility_cosine_decay_fraction)

### `OptimizerTest`  ·  implements/extends TestCase
- def: [`simply/utils/optimizers_test.py:26`](../../../../../../raw/code/simply/simply/utils/optimizers_test.py#L26)
- signature: `class OptimizerTest(absltest.TestCase):`
- members:
  - `assert_almost_equal(self, actual: common.PyTree, expected: common.PyTree, delta: float = 0.00001)` — [`L28`](../../../../../../raw/code/simply/simply/utils/optimizers_test.py#L28)
  - `test_adam(self)` — [`L63`](../../../../../../raw/code/simply/simply/utils/optimizers_test.py#L63)
  - `test_dump(self)` — [`L47`](../../../../../../raw/code/simply/simply/utils/optimizers_test.py#L47)
  - `test_init_step(self)` — [`L51`](../../../../../../raw/code/simply/simply/utils/optimizers_test.py#L51)
  - `test_lion(self)` — [`L78`](../../../../../../raw/code/simply/simply/utils/optimizers_test.py#L78)
  - `test_schedule_backward_compatibility_constant(self)` — [`L91`](../../../../../../raw/code/simply/simply/utils/optimizers_test.py#L91)
  - `test_schedule_backward_compatibility_cosine_decay(self)` — [`L116`](../../../../../../raw/code/simply/simply/utils/optimizers_test.py#L116)
  - `test_schedule_backward_compatibility_cosine_decay_fraction(self)` — [`L145`](../../../../../../raw/code/simply/simply/utils/optimizers_test.py#L145)
  - `test_sgd(self)` — [`L55`](../../../../../../raw/code/simply/simply/utils/optimizers_test.py#L55)
- protocol/private: `_assert_tensor_almost_equal`[`L32`](../../../../../../raw/code/simply/simply/utils/optimizers_test.py#L32)
- uses (calls/refs, reference-scoped): [`PyTree`](common.md#PyTree.PyTree), [`dump`](pytree.md#dump), [`traverse_tree_with_path`](pytree.md#traverse_tree_with_path), [`Adam`](optimizers.md#Adam), [`create_lr_schedule`](optimizers.md#create_lr_schedule), [`SGD`](optimizers.md#SGD), [`Lion`](optimizers.md#Lion), [`get_init_steps`](optimizers.md#get_init_steps), [`MockConfigV1`](optimizers_test.md#OptimizerTest.test_schedule_backward_compatibility_cosine_decay_fraction.MockConfigV1), [`MockConfigV0`](optimizers_test.md#OptimizerTest.test_schedule_backward_compatibility_cosine_decay_fraction.MockConfigV0)


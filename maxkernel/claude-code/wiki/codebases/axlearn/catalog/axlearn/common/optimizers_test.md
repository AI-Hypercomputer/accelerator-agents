---
title: 'Module: axlearn/common/optimizers_test.py'
type: catalog
provenance: extracted
module: axlearn/common/optimizers_test.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.optimizers_test`/
symbols:
  rms_norm: rms_norm().
  optax_ema_partition: optax_ema_partition().
  _counter: _counter().
  _mesh: _mesh().
  _checkpointer_config: _checkpointer_config().
  OldSkipClipState: OldSkipClipState#
  OldSkipClipState.nonvalid_count: OldSkipClipState#nonvalid_count.
  OldSkipClipState.inner_state: OldSkipClipState#inner_state.
  OptimizerTest: OptimizerTest#
  OptimizerTest.test_partition_fn: OptimizerTest#test_partition_fn().
  OptimizerTest.test_sgd_optimizer: OptimizerTest#test_sgd_optimizer().
  OptimizerTest.test_adamw_optimizer: OptimizerTest#test_adamw_optimizer().
  OptimizerTest.test_adamw_decoupled_optimizer: OptimizerTest#test_adamw_decoupled_optimizer().
  OptimizerTest.test_adam_optimizer: OptimizerTest#test_adam_optimizer().
  OptimizerTest.test_adafactor_optimizer: OptimizerTest#test_adafactor_optimizer().
  OptimizerTest.test_lion_optimizer: OptimizerTest#test_lion_optimizer().
  OptimizerTest.test_lion_optimizer_dtype: OptimizerTest#test_lion_optimizer_dtype().
  OptimizerTest._test_optimizer: OptimizerTest#_test_optimizer().
  OptimizerTest._test_optimizer_helper: OptimizerTest#_test_optimizer_helper().
  OptimizerTest.test_adafactor_weight_decay: OptimizerTest#test_adafactor_weight_decay().
  OptimizerTest.test_adafactor_beta_schedules: OptimizerTest#test_adafactor_beta_schedules().
  OptimizerTest.test_adamw_multiply_by_parameter_scale: OptimizerTest#test_adamw_multiply_by_parameter_scale().
  OptimizerTest.test_adamw_decoupled_update_schedule: OptimizerTest#test_adamw_decoupled_update_schedule().
  OptimizerTest.test_weight_scaling: OptimizerTest#test_weight_scaling().
  OptimizerTest.test_weight_decay_with_learning_rate_exponent: OptimizerTest#test_weight_decay_with_learning_rate_exponent().
  OptimizerTest.test_weight_decay_per_param_scales: OptimizerTest#test_weight_decay_per_param_scales().
  OptimizerTest.test_gradient_clipping: OptimizerTest#test_gradient_clipping().
  OptimizerTest.test_gradient_skipping_and_clipping: OptimizerTest#test_gradient_skipping_and_clipping().
  OptimizerTest.test_gradient_skipping_backward_compatibility: OptimizerTest#test_gradient_skipping_backward_compatibility().
  OptimizerTest.test_l2_regularizer: OptimizerTest#test_l2_regularizer().
  OptimizerTest.test_scale_by_trust_ratio: OptimizerTest#test_scale_by_trust_ratio().
  OptimizerTest.test_clip_by_block_rms: OptimizerTest#test_clip_by_block_rms().
  OptimizerTest.test_clip_by_block_rms_both_none: OptimizerTest#test_clip_by_block_rms_both_none().
  OptimizerTest.test_scale_by_param_block_rms: OptimizerTest#test_scale_by_param_block_rms().
  OptimizerTest.test_ema_parity: OptimizerTest#test_ema_parity().
  OptimizerTest.test_param_ema: OptimizerTest#test_param_ema().
  OptimizerTest.test_scale_by_schedule: OptimizerTest#test_scale_by_schedule().
  OptimizerTest.test_scale_by_schedule_cosine_with_linear_warmup: OptimizerTest#test_scale_by_schedule_cosine_with_linear_warmup().
  OptimizerTest.test_adastar_vs_adamw_decoupled: OptimizerTest#test_adastar_vs_adamw_decoupled().
  OptimizerTest.test_adastar_vs_adafactor: OptimizerTest#test_adastar_vs_adafactor().
  OptimizerTest._compare_optimizers: OptimizerTest#_compare_optimizers().
  OptimizerTest.test_adastar_summaries: OptimizerTest#test_adastar_summaries().
  OptimizerTest.test_covariance_and_rms: OptimizerTest#test_covariance_and_rms().
---
# Module: [`axlearn/common/optimizers_test.py`](../../../../../../raw/code/axlearn/axlearn/common/optimizers_test.py)

## Classes
### `OldSkipClipState`
- def: [`axlearn/common/optimizers_test.py:109`](../../../../../../raw/code/axlearn/axlearn/common/optimizers_test.py#L109)
- doc: State of an older version of skip_and_clip_by_global_norm() for testing.
- signature: `class OldSkipClipState(NamedTuple):`
- members:
  - `inner_state` — [`L113`](../../../../../../raw/code/axlearn/axlearn/common/optimizers_test.py#L113)
  - `nonvalid_count` — [`L112`](../../../../../../raw/code/axlearn/axlearn/common/optimizers_test.py#L112)

### `OptimizerTest`
- def: [`axlearn/common/optimizers_test.py:116`](../../../../../../raw/code/axlearn/axlearn/common/optimizers_test.py#L116)
- doc: Tests optimization modules.
- signature: `class OptimizerTest(TestCase):`
- members:
  - `test_adafactor_beta_schedules(self, b1: Schedule, b2: Schedule, expected_value)` — [`L525`](../../../../../../raw/code/axlearn/axlearn/common/optimizers_test.py#L525)
  - `test_adafactor_optimizer(self, learning_rate, multiply_by_parameter_scale, clipping_threshold, apply_scale_by_trust_ratio)` — [`L321`](../../../../../../raw/code/axlearn/axlearn/common/optimizers_test.py#L321)
  - `test_adafactor_weight_decay(self, learning_rate: float, weight_decay: Optional[float], weight_decay_scale_by_learning_rate_exponent: Optional[float])` — [`L443`](../../../../../../raw/code/axlearn/axlearn/common/optimizers_test.py#L443)
  - `test_adam_optimizer(self, learning_rate, l2_regularizer_weight)` — [`L303`](../../../../../../raw/code/axlearn/axlearn/common/optimizers_test.py#L303)
  - `test_adamw_decoupled_optimizer(self, learning_rate, weight_decay, update_schedule, multiply_by_parameter_scale)` — [`L284`](../../../../../../raw/code/axlearn/axlearn/common/optimizers_test.py#L284)
  - `test_adamw_decoupled_update_schedule(self, weight_decay: float, update_schedule: float, scale_adam_by: float)` — [`L603`](../../../../../../raw/code/axlearn/axlearn/common/optimizers_test.py#L603)
  - `test_adamw_multiply_by_parameter_scale(self, params)` — [`L553`](../../../../../../raw/code/axlearn/axlearn/common/optimizers_test.py#L553)
  - `test_adamw_optimizer(self, learning_rate, weight_decay, multiply_by_parameter_scale)` — [`L268`](../../../../../../raw/code/axlearn/axlearn/common/optimizers_test.py#L268)
  - `test_adastar_summaries(self, learning_rate, b1, b2, eps_square, update_schedule, clipping_threshold, weight_decay, weight_decay_per_param_scale)` — [`L1551`](../../../../../../raw/code/axlearn/axlearn/common/optimizers_test.py#L1551)
  - `test_adastar_vs_adafactor(self, learning_rate, b1, b2, eps, update_schedule, clipping_threshold, weight_decay)` — [`L1440`](../../../../../../raw/code/axlearn/axlearn/common/optimizers_test.py#L1440)
  - `test_adastar_vs_adamw_decoupled(self, learning_rate, b1, b2, eps, update_schedule, weight_decay)` — [`L1383`](../../../../../../raw/code/axlearn/axlearn/common/optimizers_test.py#L1383)
  - `test_clip_by_block_rms(self, max_norm)` — [`L1035`](../../../../../../raw/code/axlearn/axlearn/common/optimizers_test.py#L1035)
  - `test_clip_by_block_rms_both_none(self)` — [`L1077`](../../../../../../raw/code/axlearn/axlearn/common/optimizers_test.py#L1077) — Tests clip_clip_by_block_rms(threshold=None, summary_suffix=None).
  - `test_covariance_and_rms(self)` — [`L1644`](../../../../../../raw/code/axlearn/axlearn/common/optimizers_test.py#L1644)
  - `test_ema_parity(self, accumulator_dtype, debias, momentum)` — [`L1138`](../../../../../../raw/code/axlearn/axlearn/common/optimizers_test.py#L1138)
  - `test_gradient_clipping(self, max_norm, drop_norm)` — [`L835`](../../../../../../raw/code/axlearn/axlearn/common/optimizers_test.py#L835)
  - `test_gradient_skipping_and_clipping(self, max_norm, drop_norm, offload)` — [`L872`](../../../../../../raw/code/axlearn/axlearn/common/optimizers_test.py#L872)
  - `test_gradient_skipping_backward_compatibility(self)` — [`L929`](../../../../../../raw/code/axlearn/axlearn/common/optimizers_test.py#L929)
  - `test_l2_regularizer(self, regularizer_weight, per_param_scale)` — [`L969`](../../../../../../raw/code/axlearn/axlearn/common/optimizers_test.py#L969)
  - `test_lion_optimizer(self, learning_rate, weight_decay, multiply_by_parameter_scale)` — [`L340`](../../../../../../raw/code/axlearn/axlearn/common/optimizers_test.py#L340)
  - `test_lion_optimizer_dtype(self, mu_dtype, params_dtype)` — [`L354`](../../../../../../raw/code/axlearn/axlearn/common/optimizers_test.py#L354) — Tests that dtypes are consistent between init, update and partition.
  - `test_param_ema(self, decay, dtype)` — [`L1263`](../../../../../../raw/code/axlearn/axlearn/common/optimizers_test.py#L1263)
  - `test_partition_fn(self, optimizer_cfg)` — [`L194`](../../../../../../raw/code/axlearn/axlearn/common/optimizers_test.py#L194) — Tests that opt.{init,mesh_axes} are consistent with each other.
  - `test_scale_by_param_block_rms(self, threshold)` — [`L1109`](../../../../../../raw/code/axlearn/axlearn/common/optimizers_test.py#L1109)
  - `test_scale_by_schedule(self)` — [`L1328`](../../../../../../raw/code/axlearn/axlearn/common/optimizers_test.py#L1328)
  - `test_scale_by_schedule_cosine_with_linear_warmup(self)` — [`L1341`](../../../../../../raw/code/axlearn/axlearn/common/optimizers_test.py#L1341)
  - `test_scale_by_trust_ratio(self)` — [`L1015`](../../../../../../raw/code/axlearn/axlearn/common/optimizers_test.py#L1015)
  - `test_sgd_optimizer(self, learning_rate, weight_decay, decouple_weight_decay)` — [`L239`](../../../../../../raw/code/axlearn/axlearn/common/optimizers_test.py#L239)
  - `test_weight_decay_per_param_scales(self, bias_scale=1.0, optimizer_cfg=None)` — [`L778`](../../../../../../raw/code/axlearn/axlearn/common/optimizers_test.py#L778)
  - `test_weight_decay_with_learning_rate_exponent(self)` — [`L705`](../../../../../../raw/code/axlearn/axlearn/common/optimizers_test.py#L705) — Tests add_decayed_weights with learning_rate_exponent=1.
  - `test_weight_scaling(self, optimizer_cfg, param_scale)` — [`L651`](../../../../../../raw/code/axlearn/axlearn/common/optimizers_test.py#L651)
- protocol/private: `_compare_optimizers`[`L1498`](../../../../../../raw/code/axlearn/axlearn/common/optimizers_test.py#L1498), `_test_optimizer`[`L383`](../../../../../../raw/code/axlearn/axlearn/common/optimizers_test.py#L383), `_test_optimizer_helper`[`L387`](../../../../../../raw/code/axlearn/axlearn/common/optimizers_test.py#L387)

## Functions
- `_checkpointer_config()` — [`L105`](../../../../../../raw/code/axlearn/axlearn/common/optimizers_test.py#L105)
- `_counter()` — [`L86`](../../../../../../raw/code/axlearn/axlearn/common/optimizers_test.py#L86)
- `_mesh(mesh_shape: Sequence[int])` — [`L100`](../../../../../../raw/code/axlearn/axlearn/common/optimizers_test.py#L100)
- `optax_ema_partition(base: optax.GradientTransformation)` — [`L75`](../../../../../../raw/code/axlearn/axlearn/common/optimizers_test.py#L75)
- `rms_norm(x)` — [`L71`](../../../../../../raw/code/axlearn/axlearn/common/optimizers_test.py#L71)


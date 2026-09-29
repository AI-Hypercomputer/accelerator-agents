---
title: 'Module: tests/_src/providers/qt_test.py'
type: catalog
provenance: extracted
module: tests/_src/providers/qt_test.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `tests._src.providers.qt_test`/QtTest#
symbols:
  QtTest.test_nnx_multi_head_attention_qt_bwd: test_nnx_multi_head_attention_qt_bwd().
  QtTest.test_srq_jit_grad: test_srq_jit_grad().
  QtTest.test_dot_general_with_sparsity: test_dot_general_with_sparsity().
  QtTest.create_srq_nnx_model: create_srq_nnx_model().
  QtTest.test_bwd_reuse_noise: test_bwd_reuse_noise().
  QtTest.test_bwd_reuse_noise.TestModule.test_config: test_bwd_reuse_noise().TestModule#test_config().
  QtTest.test_srq_jit_grad_nnx: test_srq_jit_grad_nnx().
  QtTest.jit_apply: jit_apply().
  QtTest.test_dot_general_with_sparsity.TestModule.__call__: test_dot_general_with_sparsity().TestModule#__call__().
  QtTest.test_bwd_reuse_noise.TestModule.provider: test_bwd_reuse_noise().TestModule#provider.
  QtTest.test_dot_general_with_sparsity.TestModule.provider: test_dot_general_with_sparsity().TestModule#provider.
  QtTest.train_step: train_step().
  QtTest.loss_fn: loss_fn().
  QtTest.test_bwd_reuse_noise.TestModule: test_bwd_reuse_noise().TestModule#
  QtTest.test_dot_general_with_sparsity.TestModule: test_dot_general_with_sparsity().TestModule#
  QtTest: ''
---
# Module: [`tests/_src/providers/qt_test.py`](../../../../../../../raw/code/qwix/tests/_src/providers/qt_test.py)

## Classes
### `QtTest`  ·  implements/extends TestCase
- def: [`tests/_src/providers/qt_test.py:27`](../../../../../../../raw/code/qwix/tests/_src/providers/qt_test.py#L27)
- signature: `class QtTest(absltest.TestCase):`
- members:
  - `create_srq_nnx_model(model_input)` — [`L109`](../../../../../../../raw/code/qwix/tests/_src/providers/qt_test.py#L109)
  - `jit_apply(variables)` — [`L88`](../../../../../../../raw/code/qwix/tests/_src/providers/qt_test.py#L88)
  - `loss_fn(params)` — [`L89`](../../../../../../../raw/code/qwix/tests/_src/providers/qt_test.py#L89)
  - `test_bwd_reuse_noise(self)` — [`L29`](../../../../../../../raw/code/qwix/tests/_src/providers/qt_test.py#L29) — Tests that noise is reused in bwd pass for lhs/rhs.
  - `test_dot_general_with_sparsity(self)` — [`L172`](../../../../../../../raw/code/qwix/tests/_src/providers/qt_test.py#L172) — Tests that dot_general applies sparsity to rhs.
  - `test_nnx_multi_head_attention_qt_bwd(self)` — [`L203`](../../../../../../../raw/code/qwix/tests/_src/providers/qt_test.py#L203)
  - `test_srq_jit_grad(self)` — [`L67`](../../../../../../../raw/code/qwix/tests/_src/providers/qt_test.py#L67) — Test that the grad of SRQ can be taken inside a jitted function.
  - `test_srq_jit_grad_nnx(self)` — [`L106`](../../../../../../../raw/code/qwix/tests/_src/providers/qt_test.py#L106) — Test creating and train an SRQ NNX model inside jit.
  - `train_step(model, x)` — [`L237`](../../../../../../../raw/code/qwix/tests/_src/providers/qt_test.py#L237)
- uses (calls/refs, reference-scoped): [`weight_qtype`](../../../qwix/_src/qconfig.md#QuantizationRule.weight_qtype), [`quantize_model`](../../../qwix/_src/model.md#quantize_model), [`QuantizationRule`](../../../qwix/_src/qconfig.md#QuantizationRule), [`act_qtype`](../../../qwix/_src/qconfig.md#QuantizationRule.act_qtype), [`module_path`](../../../qwix/_src/qconfig.md#QuantizationRule.module_path), [`tile_size`](../../../qwix/_src/qconfig.md#QuantizationRule.tile_size), [`act_static_scale`](../../../qwix/_src/qconfig.md#QuantizationRule.act_static_scale), [`QtRule`](../../../qwix/_src/providers/qt.md#QtRule), [`QtProvider`](../../../qwix/_src/providers/qt.md#QtProvider), [`bwd_qtype`](../../../qwix/_src/providers/qt.md#QtRule.bwd_qtype), [`QuantStat`](../../../qwix/_src/utils/flax_util.md#QuantStat), [`SparsityRule`](../../../qwix/_src/core/sparsity.md#SparsityRule), [`disable_channelwise_axes`](../../../qwix/_src/providers/qt.md#QtRule.disable_channelwise_axes), [`weight_sparsity_m`](../../../qwix/_src/core/sparsity.md#SparsityRule.weight_sparsity_m), [`weight_sparsity_n`](../../../qwix/_src/core/sparsity.md#SparsityRule.weight_sparsity_n), [`additional_qt_config`](../../../qwix/_src/providers/qt.md#QtRule.additional_qt_config), [`bwd_stochastic_rounding`](../../../qwix/_src/providers/qt.md#QtRule.bwd_stochastic_rounding), [`bwd_weight_grad_tile_size`](../../../qwix/_src/providers/qt.md#QtRule.bwd_weight_grad_tile_size)  (3 test-only)

### `TestModule`  ·  implements/extends Module
- def: [`tests/_src/providers/qt_test.py:188`](../../../../../../../raw/code/qwix/tests/_src/providers/qt_test.py#L188)
- signature: `class TestModule(nn.Module):`
- members:
  - `test_config(self, lhs, rhs)` — [`L35`](../../../../../../../raw/code/qwix/tests/_src/providers/qt_test.py#L35)
  - `provider` — [`L33`](../../../../../../../raw/code/qwix/tests/_src/providers/qt_test.py#L33)
  - `provider` — [`L189`](../../../../../../../raw/code/qwix/tests/_src/providers/qt_test.py#L189)
- protocol/private: `__call__`[`L191`](../../../../../../../raw/code/qwix/tests/_src/providers/qt_test.py#L191)
- uses (calls/refs, reference-scoped): [`_create_dot_general_qt_config`](../../../qwix/_src/providers/qt.md#QtProvider._create_dot_general_qt_config), [`QtProvider`](../../../qwix/_src/providers/qt.md#QtProvider), [`dot_general`](../../../qwix/_src/providers/qt.md#QtProvider.dot_general), [`_rules`](../../../qwix/_src/qconfig.md#QuantizationProvider._rules)
- used by: (2 test-only callers)


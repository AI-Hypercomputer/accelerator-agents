---
title: 'Module: tests/_src/core/dot_general_qt_test.py'
type: catalog
provenance: extracted
module: tests/_src/core/dot_general_qt_test.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `tests._src.core.dot_general_qt_test`/
symbols:
  DotGeneralQtTest.test_mxfp_tiled_residual_fallback_single_tile: DotGeneralQtTest#test_mxfp_tiled_residual_fallback_single_tile().
  DotGeneralQtTest.test_grad_against_fq: DotGeneralQtTest#test_grad_against_fq().
  DotGeneralQtTest.test_tiled_residual_fallback: DotGeneralQtTest#test_tiled_residual_fallback().
  DotGeneralQtTest.test_sparsity_rule: DotGeneralQtTest#test_sparsity_rule().
  _fake_quant: _fake_quant().
  dot_general_fq: dot_general_fq().
  DotGeneralQtTest.run_step: DotGeneralQtTest#run_step().
  DotGeneralQtTest.mlp: DotGeneralQtTest#mlp().
  DotGeneralQtTest.f: DotGeneralQtTest#f().
  DotGeneralQtTest.loss_fn_fq: DotGeneralQtTest#loss_fn_fq().
  DotGeneralQtTest.loss_fn_qt: DotGeneralQtTest#loss_fn_qt().
  DotGeneralQtTest.train_step: DotGeneralQtTest#train_step().
  DotGeneralQtTest.loss_fn: DotGeneralQtTest#loss_fn().
  DotGeneralQtTest.test_verify_jaxpr: DotGeneralQtTest#test_verify_jaxpr().
  DotGeneralQtTest.test_gradient_clipping_integration: DotGeneralQtTest#test_gradient_clipping_integration().
  DotGeneralQtTest.loss_fn_fp: DotGeneralQtTest#loss_fn_fp().
  DotGeneralQtTest: DotGeneralQtTest#
---
# Module: [`tests/_src/core/dot_general_qt_test.py`](../../../../../../../raw/code/qwix/tests/_src/core/dot_general_qt_test.py)

## Classes
### `DotGeneralQtTest`  ·  implements/extends TestCase
- def: [`tests/_src/core/dot_general_qt_test.py:71`](../../../../../../../raw/code/qwix/tests/_src/core/dot_general_qt_test.py#L71)
- doc: Test class for dot_general_qt.
- signature: `class DotGeneralQtTest(parameterized.TestCase):`
- members:
  - `f(l, r)` — [`L395`](../../../../../../../raw/code/qwix/tests/_src/core/dot_general_qt_test.py#L395)
  - `loss_fn(l, r)` — [`L318`](../../../../../../../raw/code/qwix/tests/_src/core/dot_general_qt_test.py#L318)
  - `loss_fn_fp(lhs_arr, rhs_arr)` — [`L226`](../../../../../../../raw/code/qwix/tests/_src/core/dot_general_qt_test.py#L226)
  - `loss_fn_fq(lhs_arr, rhs_arr)` — [`L214`](../../../../../../../raw/code/qwix/tests/_src/core/dot_general_qt_test.py#L214)
  - `loss_fn_qt(lhs_arr, rhs_arr)` — [`L219`](../../../../../../../raw/code/qwix/tests/_src/core/dot_general_qt_test.py#L219)
  - `mlp(x, w1, w2, w3)` — [`L266`](../../../../../../../raw/code/qwix/tests/_src/core/dot_general_qt_test.py#L266)
  - `run_step(calibration_method)` — [`L310`](../../../../../../../raw/code/qwix/tests/_src/core/dot_general_qt_test.py#L310)
  - `test_grad_against_fq(self, *, lhs_shape=(2, 4), rhs_shape=(4, 2), lhs_qtype, rhs_qtype, bwd_qtype=None, tile_size=None, bwd_drhs_tile_size=None, use_original_residuals=False, residual_qtype=None, expected_mae_fq_out, expected_mae_fq_grads, expected_mae_fp_out, expected_mae_fp_grads)` — [`L179`](../../../../../../../raw/code/qwix/tests/_src/core/dot_general_qt_test.py#L179)
  - `test_gradient_clipping_integration(self)` — [`L303`](../../../../../../../raw/code/qwix/tests/_src/core/dot_general_qt_test.py#L303) — Verifies that calibration scaling triggers gradient masking (STE).
  - `test_mxfp_tiled_residual_fallback_single_tile(self)` — [`L405`](../../../../../../../raw/code/qwix/tests/_src/core/dot_general_qt_test.py#L405) — Verifies that MXFP single-tile residuals (size 32) correctly fallback to float inputs in bwd pass.
  - `test_sparsity_rule(self)` — [`L339`](../../../../../../../raw/code/qwix/tests/_src/core/dot_general_qt_test.py#L339) — Verifies that DotGeneralQtConfig accepts sparsity_rule and applies it to rhs.
  - `test_tiled_residual_fallback(self)` — [`L375`](../../../../../../../raw/code/qwix/tests/_src/core/dot_general_qt_test.py#L375) — Verifies that tiled residuals correctly fallback to original inputs in bwd pass.
  - `test_verify_jaxpr(self)` — [`L263`](../../../../../../../raw/code/qwix/tests/_src/core/dot_general_qt_test.py#L263) — Verify quantized training with full int8 via checking the jaxpr.
  - `train_step(x, weights)` — [`L277`](../../../../../../../raw/code/qwix/tests/_src/core/dot_general_qt_test.py#L277)
- uses (calls/refs, reference-scoped): [`QArray`](../../../qwix/_src/core/qarray.md#QArray), [`dot_general_qt`](../../../qwix/_src/core/dot_general_qt.md#dot_general_qt), [`calibrate`](../../../qwix/_src/core/qarray.md#calibrate), [`get_how_to_quantize`](../../../qwix/_src/core/dot_general.md#get_how_to_quantize), [`sparsify`](../../../qwix/_src/core/qarray.md#sparsify), [`DotGeneralQtConfig`](../../../qwix/_src/core/dot_general_qt.md#DotGeneralQtConfig), [`lhs_qtype`](../../../qwix/_src/core/dot_general_qt.md#DotGeneralQtConfig.lhs_qtype), [`rhs_qtype`](../../../qwix/_src/core/dot_general_qt.md#DotGeneralQtConfig.rhs_qtype), [`tile_size`](../../../qwix/_src/core/dot_general_qt.md#DotGeneralQtConfig.tile_size), [`SparsityRule`](../../../qwix/_src/core/sparsity.md#SparsityRule), [`dlhs_grad_qtype`](../../../qwix/_src/core/dot_general_qt.md#DotGeneralQtConfig.dlhs_grad_qtype), [`dot_general_qt_fwd_bwd`](../../../qwix/_src/core/dot_general_qt.md#dot_general_qt_fwd_bwd), [`drhs_grad_qtype`](../../../qwix/_src/core/dot_general_qt.md#DotGeneralQtConfig.drhs_grad_qtype), [`weight_sparsity_m`](../../../qwix/_src/core/sparsity.md#SparsityRule.weight_sparsity_m), [`sparsity_rule`](../../../qwix/_src/core/dot_general_qt.md#DotGeneralQtConfig.sparsity_rule), [`weight_sparsity_n`](../../../qwix/_src/core/sparsity.md#SparsityRule.weight_sparsity_n), [`drhs_tile_size`](../../../qwix/_src/core/dot_general_qt.md#DotGeneralQtConfig.drhs_tile_size), [`dlhs_tile_size`](../../../qwix/_src/core/dot_general_qt.md#DotGeneralQtConfig.dlhs_tile_size), [`lhs_calibration_method`](../../../qwix/_src/core/dot_general_qt.md#DotGeneralQtConfig.lhs_calibration_method), [`rhs_calibration_method`](../../../qwix/_src/core/dot_general_qt.md#DotGeneralQtConfig.rhs_calibration_method), [`use_original_residuals`](../../../qwix/_src/core/dot_general_qt.md#DotGeneralQtConfig.use_original_residuals), [`dlhs_residual_qtype`](../../../qwix/_src/core/dot_general_qt.md#DotGeneralQtConfig.dlhs_residual_qtype), [`drhs_residual_qtype`](../../../qwix/_src/core/dot_general_qt.md#DotGeneralQtConfig.drhs_residual_qtype), `fwd`  (1 test-only)

## Functions
- `_fake_quant(array: jax.Array, how: qarray.HowToQuantize)` — [`L25`](../../../../../../../raw/code/qwix/tests/_src/core/dot_general_qt_test.py#L25)
- `dot_general_fq(lhs: jax.Array, rhs: jax.Array, dimension_numbers: jax.lax.DotDimensionNumbers, config: dot_general_qt.DotGeneralQtConfig)` — [`L39`](../../../../../../../raw/code/qwix/tests/_src/core/dot_general_qt_test.py#L39) — dot_general implemented with fake quantization.


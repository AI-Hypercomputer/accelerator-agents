---
title: 'Module: tests/_src/core/two_dim_scales_test.py'
type: catalog
provenance: extracted
module: tests/_src/core/two_dim_scales_test.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `tests._src.core.two_dim_scales_test`/TwoDimScalesTest#
symbols:
  TwoDimScalesTest.test_dot_general_qt_2d_scales_backward: test_dot_general_qt_2d_scales_backward().
  TwoDimScalesTest.test_dot_general_2d_scales: test_dot_general_2d_scales().
  TwoDimScalesTest.test_quantize_2d_scales: test_quantize_2d_scales().
  TwoDimScalesTest.test_dot_general_qt_2d_scales: test_dot_general_qt_2d_scales().
  TwoDimScalesTest.loss_fn_qt: loss_fn_qt().
  TwoDimScalesTest.final_loss_fn: final_loss_fn().
  TwoDimScalesTest: ''
---
# Module: [`tests/_src/core/two_dim_scales_test.py`](../../../../../../../raw/code/qwix/tests/_src/core/two_dim_scales_test.py)

## Classes
### `TwoDimScalesTest`  ·  implements/extends TestCase
- def: [`tests/_src/core/two_dim_scales_test.py:26`](../../../../../../../raw/code/qwix/tests/_src/core/two_dim_scales_test.py#L26)
- doc: Tests for two dimensional scales.
- signature: `class TwoDimScalesTest(parameterized.TestCase):`
- members:
  - `final_loss_fn(x)` — [`L201`](../../../../../../../raw/code/qwix/tests/_src/core/two_dim_scales_test.py#L201)
  - `loss_fn_qt(x, y)` — [`L191`](../../../../../../../raw/code/qwix/tests/_src/core/two_dim_scales_test.py#L191)
  - `test_dot_general_2d_scales(self)` — [`L56`](../../../../../../../raw/code/qwix/tests/_src/core/two_dim_scales_test.py#L56) — Tests dot_general implementations with 2d scales.
  - `test_dot_general_qt_2d_scales(self)` — [`L94`](../../../../../../../raw/code/qwix/tests/_src/core/two_dim_scales_test.py#L94) — Tests dot_general_qt with 2d scales.
  - `test_dot_general_qt_2d_scales_backward(self, dlhs_grad_qtype, drhs_grad_qtype)` — [`L160`](../../../../../../../raw/code/qwix/tests/_src/core/two_dim_scales_test.py#L160) — Tests backwards for dot_general_qt with 2d scales.
  - `test_quantize_2d_scales(self)` — [`L29`](../../../../../../../raw/code/qwix/tests/_src/core/two_dim_scales_test.py#L29) — Tests quantization with 2d scales.
- uses (calls/refs, reference-scoped): [`HowToQuantize`](../../../qwix/_src/core/qarray.md#HowToQuantize), [`qtype`](../../../qwix/_src/core/qarray.md#HowToQuantize.qtype), [`quantize`](../../../qwix/_src/core/qarray.md#quantize), [`qvalue`](../../../qwix/_src/core/qarray.md#QArray.qvalue), [`scale`](../../../qwix/_src/core/qarray.md#QArray.scale), [`dequantize`](../../../qwix/_src/core/qarray.md#dequantize), [`tiled_axes`](../../../qwix/_src/core/qarray.md#HowToQuantize.tiled_axes), [`dot_general_qt`](../../../qwix/_src/core/dot_general_qt.md#dot_general_qt), [`dot_general`](../../../qwix/_src/core/dot_general.md#dot_general), [`calibration_method`](../../../qwix/_src/core/qarray.md#HowToQuantize.calibration_method), [`_fast_dot_general`](../../../qwix/_src/core/dot_general.md#_fast_dot_general), [`loop_dot_general`](../../../qwix/_src/core/dot_general.md#loop_dot_general), [`get_how_to_quantize`](../../../qwix/_src/core/dot_general.md#get_how_to_quantize), [`DotGeneralQtConfig`](../../../qwix/_src/core/dot_general_qt.md#DotGeneralQtConfig), [`lhs_qtype`](../../../qwix/_src/core/dot_general_qt.md#DotGeneralQtConfig.lhs_qtype), [`rhs_qtype`](../../../qwix/_src/core/dot_general_qt.md#DotGeneralQtConfig.rhs_qtype), [`_slow_dot_general`](../../../qwix/_src/core/dot_general.md#_slow_dot_general), [`tile_size`](../../../qwix/_src/core/dot_general_qt.md#DotGeneralQtConfig.tile_size), [`dlhs_grad_qtype`](../../../qwix/_src/core/dot_general_qt.md#DotGeneralQtConfig.dlhs_grad_qtype), [`drhs_grad_qtype`](../../../qwix/_src/core/dot_general_qt.md#DotGeneralQtConfig.drhs_grad_qtype), [`drhs_tile_size`](../../../qwix/_src/core/dot_general_qt.md#DotGeneralQtConfig.drhs_tile_size), [`dlhs_tile_size`](../../../qwix/_src/core/dot_general_qt.md#DotGeneralQtConfig.dlhs_tile_size), [`T`](../../../qwix/_src/core/qarray.md#QArray.T)


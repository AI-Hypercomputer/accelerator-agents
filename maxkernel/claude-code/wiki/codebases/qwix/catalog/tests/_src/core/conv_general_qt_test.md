---
title: 'Module: tests/_src/core/conv_general_qt_test.py'
type: catalog
provenance: extracted
module: tests/_src/core/conv_general_qt_test.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `tests._src.core.conv_general_qt_test`/
symbols:
  _fake_quant: _fake_quant().
  conv_general_fq: conv_general_fq().
  ConvGeneralQtTest.test_grad_against_fq: ConvGeneralQtTest#test_grad_against_fq().
  ConvGeneralQtTest.f: ConvGeneralQtTest#f().
  mae: mae().
  ConvGeneralQtTest.loss_fn_fq: ConvGeneralQtTest#loss_fn_fq().
  ConvGeneralQtTest.loss_fn_qt: ConvGeneralQtTest#loss_fn_qt().
  ConvGeneralQtTest.loss_fn_fp: ConvGeneralQtTest#loss_fn_fp().
  ConvGeneralQtTest: ConvGeneralQtTest#
---
# Module: [`tests/_src/core/conv_general_qt_test.py`](../../../../../../../raw/code/qwix/tests/_src/core/conv_general_qt_test.py)

## Classes
### `ConvGeneralQtTest`  ·  implements/extends TestCase
- def: [`tests/_src/core/conv_general_qt_test.py:86`](../../../../../../../raw/code/qwix/tests/_src/core/conv_general_qt_test.py#L86)
- doc: Test class for conv_general_qt.
- signature: `class ConvGeneralQtTest(parameterized.TestCase):`
- members:
  - `f(lhs, rhs)` — [`L217`](../../../../../../../raw/code/qwix/tests/_src/core/conv_general_qt_test.py#L217)
  - `loss_fn_fp(lhs_arr, rhs_arr)` — [`L203`](../../../../../../../raw/code/qwix/tests/_src/core/conv_general_qt_test.py#L203)
  - `loss_fn_fq(lhs_arr, rhs_arr)` — [`L175`](../../../../../../../raw/code/qwix/tests/_src/core/conv_general_qt_test.py#L175)
  - `loss_fn_qt(lhs_arr, rhs_arr)` — [`L189`](../../../../../../../raw/code/qwix/tests/_src/core/conv_general_qt_test.py#L189)
  - `test_grad_against_fq(self, *, data_format, fwd_qtype, bwd_qtype=None, padding='SAME', lhs_dilation=None, rhs_dilation=None, expected_maes)` — [`L141`](../../../../../../../raw/code/qwix/tests/_src/core/conv_general_qt_test.py#L141)
- uses (calls/refs, reference-scoped): [`conv_general_qt`](../../../qwix/_src/core/conv_general_qt.md#conv_general_qt), [`ConvGeneralQtConfig`](../../../qwix/_src/core/conv_general_qt.md#ConvGeneralQtConfig), [`dlhs_grad_qtype`](../../../qwix/_src/core/conv_general_qt.md#ConvGeneralQtConfig.dlhs_grad_qtype), [`drhs_grad_qtype`](../../../qwix/_src/core/conv_general_qt.md#ConvGeneralQtConfig.drhs_grad_qtype), [`lhs_qtype`](../../../qwix/_src/core/conv_general_qt.md#ConvGeneralQtConfig.lhs_qtype), [`rhs_qtype`](../../../qwix/_src/core/conv_general_qt.md#ConvGeneralQtConfig.rhs_qtype)  (2 test-only)

## Functions
- `_fake_quant(array: jax.Array, how: qarray.HowToQuantize)` — [`L30`](../../../../../../../raw/code/qwix/tests/_src/core/conv_general_qt_test.py#L30) — Generic fake quantization function using a Straight-Through Estimator.
- `conv_general_fq(lhs: jax.Array, rhs: jax.Array, config: conv_general_qt.ConvGeneralQtConfig, window_strides: Sequence[int], padding: str | Sequence[tuple[int, int]], dimension_numbers: jax.lax.ConvGeneralDilatedDimensionNumbers | None, lhs_dilation: Sequence[int] | None = None, rhs_dilation: Sequence[int] | None = None)` — [`L46`](../../../../../../../raw/code/qwix/tests/_src/core/conv_general_qt_test.py#L46) — conv_general_dilated implemented with fake quantization.
- `mae(lhs: jax.Array, rhs: jax.Array)` — [`L25`](../../../../../../../raw/code/qwix/tests/_src/core/conv_general_qt_test.py#L25)


---
title: 'Module: tests/_src/core/ragged_dot_qt_test.py'
type: catalog
provenance: extracted
module: tests/_src/core/ragged_dot_qt_test.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `tests._src.core.ragged_dot_qt_test`/
symbols:
  RaggedDotQtTest.test_grad_against_fq_and_fp: RaggedDotQtTest#test_grad_against_fq_and_fp().
  ragged_dot_fq: ragged_dot_fq().
  _fake_quant: _fake_quant().
  RaggedDotQtTest.test_traced_group_sizes: RaggedDotQtTest#test_traced_group_sizes().
  RaggedDotQtTest.f: RaggedDotQtTest#f().
  _mae: _mae().
  RaggedDotQtTest: RaggedDotQtTest#
---
# Module: [`tests/_src/core/ragged_dot_qt_test.py`](../../../../../../../raw/code/qwix/tests/_src/core/ragged_dot_qt_test.py)

## Classes
### `RaggedDotQtTest`  ·  implements/extends TestCase
- def: [`tests/_src/core/ragged_dot_qt_test.py:54`](../../../../../../../raw/code/qwix/tests/_src/core/ragged_dot_qt_test.py#L54)
- doc: Test class for ragged_dot_qt.
- signature: `class RaggedDotQtTest(parameterized.TestCase):`
- members:
  - `f(lhs, rhs)` — [`L122`](../../../../../../../raw/code/qwix/tests/_src/core/ragged_dot_qt_test.py#L122)
  - `test_grad_against_fq_and_fp(self, lhs_qtype, rhs_qtype, expected_mae_fq_out, expected_mae_fq_dlhs, expected_mae_fq_drhs, expected_mae_fp_out, expected_mae_fp_dlhs, expected_mae_fp_drhs, bwd_qtype=None)` — [`L93`](../../../../../../../raw/code/qwix/tests/_src/core/ragged_dot_qt_test.py#L93)
  - `test_traced_group_sizes(self)` — [`L153`](../../../../../../../raw/code/qwix/tests/_src/core/ragged_dot_qt_test.py#L153)
- uses (calls/refs, reference-scoped): [`ragged_dot_qt`](../../../qwix/_src/core/ragged_dot_qt.md#ragged_dot_qt), [`RaggedDotQtConfig`](../../../qwix/_src/core/ragged_dot_qt.md#RaggedDotQtConfig), [`drhs_grad_qtype`](../../../qwix/_src/core/ragged_dot_qt.md#RaggedDotQtConfig.drhs_grad_qtype), [`lhs_qtype`](../../../qwix/_src/core/ragged_dot_qt.md#RaggedDotQtConfig.lhs_qtype), [`rhs_qtype`](../../../qwix/_src/core/ragged_dot_qt.md#RaggedDotQtConfig.rhs_qtype), [`dlhs_grad_qtype`](../../../qwix/_src/core/ragged_dot_qt.md#RaggedDotQtConfig.dlhs_grad_qtype)  (2 test-only)

## Functions
- `_fake_quant(array: jax.Array, how: qarray.HowToQuantize)` — [`L28`](../../../../../../../raw/code/qwix/tests/_src/core/ragged_dot_qt_test.py#L28) — Simulates quantization in full precision using STE.
- `_mae(a, b)` — [`L23`](../../../../../../../raw/code/qwix/tests/_src/core/ragged_dot_qt_test.py#L23) — Computes mean absolute error normalized by the mean absolute value of a.
- `ragged_dot_fq(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, config: ragged_dot_qt.RaggedDotQtConfig)` — [`L40`](../../../../../../../raw/code/qwix/tests/_src/core/ragged_dot_qt_test.py#L40) — Ragged dot implemented with fake quantization for baseline comparison.


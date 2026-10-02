---
title: 'Module: tests/contrib/hadamard_rot_test.py'
type: catalog
provenance: extracted
module: tests/contrib/hadamard_rot_test.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `tests.contrib.hadamard_rot_test`/HadamardRotTest#
symbols:
  HadamardRotTest.test_apply_hadamard_lhs: test_apply_hadamard_lhs().
  HadamardRotTest.test_apply_hadamard_rhs: test_apply_hadamard_rhs().
  HadamardRotTest.test_hadamard_rotate_multiply_different_axes: test_hadamard_rotate_multiply_different_axes().
  HadamardRotTest.test_create_hadamard_matrix: test_create_hadamard_matrix().
  HadamardRotTest.test_hadamard_rotate_multiply_identity: test_hadamard_rotate_multiply_identity().
  HadamardRotTest.test_hadamard_rotate_matmul: test_hadamard_rotate_matmul().
  HadamardRotTest.test_hadmard_matrix_construction_errors: test_hadmard_matrix_construction_errors().
  HadamardRotTest.test_error_on_multiple_reduction_axes: test_error_on_multiple_reduction_axes().
  HadamardRotTest.op: op().
  HadamardRotTest: ''
---
# Module: [`tests/contrib/hadamard_rot_test.py`](../../../../../../raw/code/qwix/tests/contrib/hadamard_rot_test.py)

## Classes
### `HadamardRotTest`  ·  implements/extends TestCase
- def: [`tests/contrib/hadamard_rot_test.py:22`](../../../../../../raw/code/qwix/tests/contrib/hadamard_rot_test.py#L22)
- signature: `class HadamardRotTest(parameterized.TestCase):`
- members:
  - `op(x, y)` — [`L128`](../../../../../../raw/code/qwix/tests/contrib/hadamard_rot_test.py#L128)
  - `test_apply_hadamard_lhs(self, idx)` — [`L44`](../../../../../../raw/code/qwix/tests/contrib/hadamard_rot_test.py#L44) — Tests that apply_hadamard_lhs works as expected.
  - `test_apply_hadamard_rhs(self, idx)` — [`L60`](../../../../../../raw/code/qwix/tests/contrib/hadamard_rot_test.py#L60) — Tests that apply_hadamard_rhs works as expected.
  - `test_create_hadamard_matrix(self, power, row_sign_flip, col_sign_flip)` — [`L31`](../../../../../../raw/code/qwix/tests/contrib/hadamard_rot_test.py#L31) — Tests that HH^T = dI.
  - `test_error_on_multiple_reduction_axes(self)` — [`L187`](../../../../../../raw/code/qwix/tests/contrib/hadamard_rot_test.py#L187) — Tests that multiple reduction axes raise errors.
  - `test_hadamard_rotate_matmul(self)` — [`L92`](../../../../../../raw/code/qwix/tests/contrib/hadamard_rot_test.py#L92) — Tests that (xH)(H^T w) = x w.
  - `test_hadamard_rotate_multiply_different_axes(self, idx, row_sign_flip, col_sign_flip)` — [`L119`](../../../../../../raw/code/qwix/tests/contrib/hadamard_rot_test.py#L119) — Tests that (xH)(H^T w / d) = x w along different reduction axes.
  - `test_hadamard_rotate_multiply_identity(self)` — [`L75`](../../../../../../raw/code/qwix/tests/contrib/hadamard_rot_test.py#L75) — Tests that (IH)(H^T I / d) = I.
  - `test_hadmard_matrix_construction_errors(self)` — [`L149`](../../../../../../raw/code/qwix/tests/contrib/hadamard_rot_test.py#L149) — Tests that the Hadamard matrix construction raises errors as expected.
- uses (calls/refs, reference-scoped): [`hadamard_rotate_inputs`](../../qwix/contrib/hadamard_rot.md#hadamard_rotate_inputs), [`_create_hadamard_matrix`](../../qwix/contrib/hadamard_rot.md#_create_hadamard_matrix), [`_apply_hadamard_lhs`](../../qwix/contrib/hadamard_rot.md#_apply_hadamard_lhs), [`_apply_hadamard_rhs`](../../qwix/contrib/hadamard_rot.md#_apply_hadamard_rhs)


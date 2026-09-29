---
title: 'Module: aqt/jax/v2/transpose_test.py'
type: catalog
provenance: extracted
module: aqt/jax/v2/transpose_test.py
status: fresh
symbol_base: scip-python python aqt 0.0.0 `aqt.jax.v2.transpose_test`/AqtTransposeTest#
symbols:
  AqtTransposeTest: ''
  AqtTransposeTest.test_transpose: test_transpose().
  AqtTransposeTest.test_lhs_scale_transpose_to_output: test_lhs_scale_transpose_to_output().
  AqtTransposeTest.test_rhs_scale_transpose_to_output: test_rhs_scale_transpose_to_output().
  AqtTransposeTest.test_lhs_scale_transpose_for_rhs_input: test_lhs_scale_transpose_for_rhs_input().
  AqtTransposeTest.test_rhs_scale_transpose_for_lhs_input: test_rhs_scale_transpose_for_lhs_input().
---
# Module: [`aqt/jax/v2/transpose_test.py`](../../../../../../../raw/code/aqt/aqt/jax/v2/transpose_test.py)

## Classes
### `AqtTransposeTest`
- def: [`aqt/jax/v2/transpose_test.py:24`](../../../../../../../raw/code/aqt/aqt/jax/v2/transpose_test.py#L24)
- signature: `class AqtTransposeTest(parameterized.TestCase):`
- members:
  - `test_lhs_scale_transpose_for_rhs_input(self, lhs_scale_shape, rhs_shape, dimension_numbers, expected_shape)` — [`L142`](../../../../../../../raw/code/aqt/aqt/jax/v2/transpose_test.py#L142)
  - `test_lhs_scale_transpose_to_output(self, lhs_scale_shape, lhs_shape, rhs_shape, dimension_numbers, expected_qlhs_scale_t_shape)` — [`L58`](../../../../../../../raw/code/aqt/aqt/jax/v2/transpose_test.py#L58)
  - `test_rhs_scale_transpose_for_lhs_input(self, lhs_shape, rhs_scale_shape, dimension_numbers, expected_shape)` — [`L169`](../../../../../../../raw/code/aqt/aqt/jax/v2/transpose_test.py#L169)
  - `test_rhs_scale_transpose_to_output(self, lhs_shape, rhs_scale_shape, rhs_shape, dimension_numbers, expected_qrhs_scale_t_shape)` — [`L101`](../../../../../../../raw/code/aqt/aqt/jax/v2/transpose_test.py#L101)
  - `test_transpose(self, tensor_shape, transpose_axes)` — [`L35`](../../../../../../../raw/code/aqt/aqt/jax/v2/transpose_test.py#L35)


---
title: 'Module: jaxite/jaxite_ckks/bat_utils_test.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_ckks/bat_utils_test.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_ckks.bat_utils_test`/BatUtilsTest#
symbols:
  BatUtilsTest.test_bat_key_vector_matmul: test_bat_key_vector_matmul().
  BatUtilsTest.test_large_values_no_overflow: test_large_values_no_overflow().
  BatUtilsTest.test_bat_merge_byte_dimension_rank1_lhs: test_bat_merge_byte_dimension_rank1_lhs().
  BatUtilsTest.test_basis_aligned_transformation_shape: test_basis_aligned_transformation_shape().
  BatUtilsTest: ''
---
# Module: [`jaxite/jaxite_ckks/bat_utils_test.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/bat_utils_test.py)

## Classes
### `BatUtilsTest`  ·  implements/extends TestCase
- def: [`jaxite/jaxite_ckks/bat_utils_test.py:14`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/bat_utils_test.py#L14)
- signature: `class BatUtilsTest(parameterized.TestCase):`
- members:
  - `test_basis_aligned_transformation_shape(self)` — [`L85`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/bat_utils_test.py#L85)
  - `test_bat_key_vector_matmul(self)` — [`L16`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/bat_utils_test.py#L16)
  - `test_bat_merge_byte_dimension_rank1_lhs(self)` — [`L146`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/bat_utils_test.py#L146)
  - `test_large_values_no_overflow(self, merge_byte_dimension)` — [`L94`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/bat_utils_test.py#L94)
- uses (calls/refs, reference-scoped): [`matmul_bat_einsum`](bat_utils.md#matmul_bat_einsum), [`basis_aligned_transformation`](bat_utils.md#basis_aligned_transformation)


---
title: 'Module: jaxite/jaxite_cggi/matrix_utils_test.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_cggi/matrix_utils_test.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_cggi.matrix_utils_test`/
symbols:
  MatrixUtilsTest.test_toeplitz_kernelized: MatrixUtilsTest#test_toeplitz_kernelized().
  MatrixUtilsTest.test_scale_by_x_power_n_minus_1: MatrixUtilsTest#test_scale_by_x_power_n_minus_1().
  MatrixUtilsTest.test_poly_mul: MatrixUtilsTest#test_poly_mul().
  MatrixUtilsTest.test_i32_as_u8_matmul: MatrixUtilsTest#test_i32_as_u8_matmul().
  test_hpmatmul_bat: test_hpmatmul_bat().
  test_hpmatmul_bat_full_precision: test_hpmatmul_bat_full_precision().
  vectors: vectors().
  test_hpmatmul_outerproduct: test_hpmatmul_outerproduct().
  POLY_MUL_IMPLS: POLY_MUL_IMPLS.
  MatrixUtilsTest._np_polymul: MatrixUtilsTest#_np_polymul().
  MatrixUtilsTest.test_generate_sign_matrix: MatrixUtilsTest#test_generate_sign_matrix().
  MatrixUtilsTest.test_get_cyclic_matrix: MatrixUtilsTest#test_get_cyclic_matrix().
  MatrixUtilsTest.test_monomial_mul: MatrixUtilsTest#test_monomial_mul().
  MatrixUtilsTest.test_monomial_mul_neg: MatrixUtilsTest#test_monomial_mul_neg().
  MatrixUtilsTest.test_monomial_div: MatrixUtilsTest#test_monomial_div().
  MatrixUtilsTest.test_monomial_div_neg: MatrixUtilsTest#test_monomial_div_neg().
  MatrixUtilsTest.test_integer_div: MatrixUtilsTest#test_integer_div().
  MatrixUtilsTest.test_integer_div_hypothesis: MatrixUtilsTest#test_integer_div_hypothesis().
  MatrixUtilsTest.test_monomial_mul_32_bit_modulus: MatrixUtilsTest#test_monomial_mul_32_bit_modulus().
  MatrixUtilsTest.test_x_power_n_minus_1: MatrixUtilsTest#test_x_power_n_minus_1().
  MatrixUtilsTest.test_x_power_n_minus_1_zero: MatrixUtilsTest#test_x_power_n_minus_1_zero().
  MatrixUtilsTest.test_x_power_n_minus_1_reduced_degree_with_sign_flip: MatrixUtilsTest#test_x_power_n_minus_1_reduced_degree_with_sign_flip().
  MatrixUtilsTest.test_x_power_n_minus_1_reduced_degree_without_sign_flip: MatrixUtilsTest#test_x_power_n_minus_1_reduced_degree_without_sign_flip().
  matrices: matrices().
  MatrixUtilsTest.cast_float64_to_int32: MatrixUtilsTest#cast_float64_to_int32().
  MatrixUtilsTest: MatrixUtilsTest#
---
# Module: [`jaxite/jaxite_cggi/matrix_utils_test.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils_test.py)

## Classes
### `MatrixUtilsTest`  ·  implements/extends TestCase
- def: [`jaxite/jaxite_cggi/matrix_utils_test.py:56`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils_test.py#L56)
- signature: `class MatrixUtilsTest(parameterized.TestCase):`
- members:
  - `cast_float64_to_int32(self, x)` — [`L67`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils_test.py#L67)
  - `test_generate_sign_matrix(self)` — [`L58`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils_test.py#L58)
  - `test_get_cyclic_matrix(self)` — [`L62`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils_test.py#L62)
  - `test_i32_as_u8_matmul(self, lhs, rhs)` — [`L246`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils_test.py#L246)
  - `test_integer_div(self, values, divisor, expected)` — [`L185`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils_test.py#L185)
  - `test_integer_div_hypothesis(self, value, divisor)` — [`L197`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils_test.py#L197)
  - `test_monomial_div(self, degree, expected)` — [`L146`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils_test.py#L146)
  - `test_monomial_div_neg(self, degree, expected)` — [`L159`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils_test.py#L159)
  - `test_monomial_mul(self, degree, expected)` — [`L120`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils_test.py#L120)
  - `test_monomial_mul_32_bit_modulus(self, degree, expected)` — [`L217`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils_test.py#L217)
  - `test_monomial_mul_neg(self, degree, expected)` — [`L133`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils_test.py#L133)
  - `test_poly_mul(self, poly1, poly2, impl)` — [`L105`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils_test.py#L105)
  - `test_scale_by_x_power_n_minus_1(self, power, poly)` — [`L265`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils_test.py#L265)
  - `test_toeplitz_kernelized(self, poly)` — [`L256`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils_test.py#L256)
  - `test_x_power_n_minus_1(self)` — [`L224`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils_test.py#L224)
  - `test_x_power_n_minus_1_reduced_degree_with_sign_flip(self)` — [`L234`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils_test.py#L234)
  - `test_x_power_n_minus_1_reduced_degree_without_sign_flip(self)` — [`L239`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils_test.py#L239)
  - `test_x_power_n_minus_1_zero(self)` — [`L229`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils_test.py#L229)
- protocol/private: `_np_polymul`[`L77`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils_test.py#L77)
- uses (calls/refs, reference-scoped): [`monomial_mul`](matrix_utils.md#monomial_mul), [`toeplitz`](matrix_utils.md#toeplitz), [`i32_as_u8_matmul`](matrix_utils.md#i32_as_u8_matmul), [`vectors`](matrix_utils_test.md#vectors), [`x_power_n_minus_1`](matrix_utils.md#x_power_n_minus_1), [`toeplitz_kernelized`](matrix_utils.md#toeplitz_kernelized), [`get_tpu_version`](jax_helpers.md#get_tpu_version), [`scale_by_x_power_n_minus_1`](matrix_utils.md#scale_by_x_power_n_minus_1), [`POLY_MUL_IMPLS`](matrix_utils_test.md#POLY_MUL_IMPLS), [`poly_mul_const_matrix`](matrix_utils.md#poly_mul_const_matrix), [`_generate_sign_matrix`](matrix_utils.md#_generate_sign_matrix), [`integer_div`](matrix_utils.md#integer_div), [`matrices`](matrix_utils_test.md#matrices)

## Functions
- `matrices(draw, shape, min_value=-(2 ** 31), max_value=2 ** 31 - 1)` — [`L22`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils_test.py#L22)
- `test_hpmatmul_bat()` — [`L296`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils_test.py#L296) — Test the correctness of the Basis Align Transformation (BAT) algorithm.
- `test_hpmatmul_bat_full_precision()` — [`L326`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils_test.py#L326) — Test the correctness of the Conv-Adapt-Conv algorithm.
- `test_hpmatmul_outerproduct()` — [`L275`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils_test.py#L275) — Test the correctness of the Conv-Adapt-Conv algorithm.
- `vectors(draw, size, min_value=-(2 ** 31), max_value=2 ** 31 - 1)` — [`L42`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils_test.py#L42)

## Module values
- `POLY_MUL_IMPLS` — [`L16`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils_test.py#L16)


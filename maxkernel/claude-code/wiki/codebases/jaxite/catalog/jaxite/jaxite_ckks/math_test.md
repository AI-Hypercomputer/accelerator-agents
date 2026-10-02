---
title: 'Module: jaxite/jaxite_ckks/math_test.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_ckks/math_test.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_ckks.math_test`/MathTest#
symbols:
  MathTest.test_root_of_unity: test_root_of_unity().
  MathTest.test_gen_twiddle_matrix_inv: test_gen_twiddle_matrix_inv().
  MathTest.test_prime_factors: test_prime_factors().
  MathTest.test_find_generator: test_find_generator().
  MathTest.test_gen_twiddle_matrix: test_gen_twiddle_matrix().
  MathTest.test_get_bit_reverse_perm: test_get_bit_reverse_perm().
  MathTest: ''
---
# Module: [`jaxite/jaxite_ckks/math_test.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/math_test.py)

## Classes
### `MathTest`  ·  implements/extends TestCase
- def: [`jaxite/jaxite_ckks/math_test.py:9`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/math_test.py#L9)
- signature: `class MathTest(parameterized.TestCase):`
- members:
  - `test_find_generator(self, q, expected_generator)` — [`L23`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/math_test.py#L23)
  - `test_gen_twiddle_matrix(self)` — [`L47`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/math_test.py#L47)
  - `test_gen_twiddle_matrix_inv(self)` — [`L61`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/math_test.py#L61)
  - `test_get_bit_reverse_perm(self)` — [`L78`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/math_test.py#L78)
  - `test_prime_factors(self, n, expected_factors)` — [`L16`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/math_test.py#L16)
  - `test_root_of_unity(self, m, q)` — [`L33`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/math_test.py#L33)
- uses (calls/refs, reference-scoped): [`gen_twiddle_matrix`](math.md#gen_twiddle_matrix), [`root_of_unity`](math.md#root_of_unity), [`find_generator`](math.md#find_generator), [`gen_twiddle_matrix_inv`](math.md#gen_twiddle_matrix_inv), [`get_bit_reverse_perm`](math.md#get_bit_reverse_perm), [`prime_factors`](math.md#prime_factors)


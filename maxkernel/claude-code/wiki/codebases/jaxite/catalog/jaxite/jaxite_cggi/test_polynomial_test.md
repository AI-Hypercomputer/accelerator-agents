---
title: 'Module: jaxite/jaxite_cggi/test_polynomial_test.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_cggi/test_polynomial_test.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_cggi.test_polynomial_test`/
symbols:
  TestPolynomialTest.test_gen_test_polynomial_parameterized: TestPolynomialTest#test_gen_test_polynomial_parameterized().
  TestPolynomialTest.test_gen_polynomial_modded: TestPolynomialTest#test_gen_polynomial_modded().
  DEGREE_32_POLY: DEGREE_32_POLY.
  TestPolynomialTest.test_manually_gen_identity_test_polynomial: TestPolynomialTest#test_manually_gen_identity_test_polynomial().
  THREE_BIT_ENCODING: THREE_BIT_ENCODING.
  TestPolynomialTest.test_manually_gen_nonidentity_test_polynomial: TestPolynomialTest#test_manually_gen_nonidentity_test_polynomial().
  TestPolynomialTest.test_gen_test_polynomial_wrong_dims: TestPolynomialTest#test_gen_test_polynomial_wrong_dims().
  TestPolynomialTest: TestPolynomialTest#
---
# Module: [`jaxite/jaxite_cggi/test_polynomial_test.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/test_polynomial_test.py)

## Classes
### `TestPolynomialTest`  ·  implements/extends TestCase
- def: [`jaxite/jaxite_cggi/test_polynomial_test.py:29`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/test_polynomial_test.py#L29)
- signature: `class TestPolynomialTest(parameterized.TestCase):`
- members:
  - `test_gen_polynomial_modded(self)` — [`L31`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/test_polynomial_test.py#L31)
  - `test_gen_test_polynomial_parameterized(self, p, q, N)` — [`L136`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/test_polynomial_test.py#L136)
  - `test_gen_test_polynomial_wrong_dims(self, coeffs)` — [`L120`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/test_polynomial_test.py#L120)
  - `test_manually_gen_identity_test_polynomial(self)` — [`L56`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/test_polynomial_test.py#L56)
  - `test_manually_gen_nonidentity_test_polynomial(self)` — [`L87`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/test_polynomial_test.py#L87)
- uses (calls/refs, reference-scoped): [`message_bit_length`](encoding.md#EncodingParameters.message_bit_length), [`EncodingParameters`](encoding.md#EncodingParameters), [`message`](rlwe.md#RlwePlaintext.message), [`padding_bit_length`](encoding.md#EncodingParameters.padding_bit_length), [`SchemeParameters`](parameters.md#SchemeParameters), [`total_bit_length`](encoding.md#EncodingParameters.total_bit_length), [`polynomial_modulus_degree`](parameters.md#SchemeParameters.polynomial_modulus_degree), [`lwe_dimension`](parameters.md#SchemeParameters.lwe_dimension), [`rlwe_dimension`](parameters.md#SchemeParameters.rlwe_dimension), [`plaintext_modulus`](parameters.md#SchemeParameters.plaintext_modulus), [`error_bit_length`](encoding.md#EncodingParameters.error_bit_length)  (3 test-only)

## Module values
- `DEGREE_32_POLY` — [`L21`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/test_polynomial_test.py#L21)
- `THREE_BIT_ENCODING` — [`L15`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/test_polynomial_test.py#L15)


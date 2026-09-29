---
title: 'Module: jaxite/jaxite_ckks/rns_test.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_ckks/rns_test.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_ckks.rns_test`/Rns
symbols:
  RnsNegativeTest.degree: NegativeTest#degree.
  RnsPolynomialTest.test_iterative_gentleman_sande: PolynomialTest#test_iterative_gentleman_sande().
  RnsNegativeTest.moduli: NegativeTest#moduli.
  RnsPolynomialTest.test_iterative_cooley_tukey: PolynomialTest#test_iterative_cooley_tukey().
  RnsPolynomialTest._random_rns_polynomial: PolynomialTest#_random_rns_polynomial().
  RnsPolynomialTest.test_rns_polynomial_addition: PolynomialTest#test_rns_polynomial_addition().
  RnsPolynomialTest.test_rns_polynomial_negation: PolynomialTest#test_rns_polynomial_negation().
  RnsPolynomialTest.test_ntt: PolynomialTest#test_ntt().
  RnsNegativeTest.test_add_polynomials_with_different_degrees: NegativeTest#test_add_polynomials_with_different_degrees().
  RnsNegativeTest.test_add_polynomials_with_incompatible_coeffs: NegativeTest#test_add_polynomials_with_incompatible_coeffs().
  RnsNegativeTest.test_add_polynomials_with_different_moduli: NegativeTest#test_add_polynomials_with_different_moduli().
  RnsNegativeTest.test_add_polynomials_with_different_forms: NegativeTest#test_add_polynomials_with_different_forms().
  RnsNegativeTest.test_multiply_polynomials_in_coefficient_form: NegativeTest#test_multiply_polynomials_in_coefficient_form().
  RnsPolynomialTest.ntt_params: PolynomialTest#ntt_params.
  RnsPolynomialTest.test_rns_polynomial_multiplication: PolynomialTest#test_rns_polynomial_multiplication().
  RnsNegativeTest.test_create_ntt_with_invalid_n: NegativeTest#test_create_ntt_with_invalid_n().
  RnsNegativeTest.test_create_ntt_with_invalid_q: NegativeTest#test_create_ntt_with_invalid_q().
  RnsNegativeTest.test_ntt_loop: NegativeTest#test_ntt_loop().
  RnsPolynomialTest._random_coeffs: PolynomialTest#_random_coeffs().
  RnsPolynomialTest: PolynomialTest#
  RnsPolynomialTest.setUp: PolynomialTest#setUp().
  RnsNegativeTest: NegativeTest#
  RnsNegativeTest.setUp: NegativeTest#setUp().
---
# Module: [`jaxite/jaxite_ckks/rns_test.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns_test.py)

## Classes
### `RnsNegativeTest`  ·  implements/extends TestCase
- def: [`jaxite/jaxite_ckks/rns_test.py:153`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns_test.py#L153)
- doc: Testing negative cases for RNS implementation.
- signature: `class RnsNegativeTest(parameterized_test.TestCase):`
- members:
  - `setUp(self)` — [`L156`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns_test.py#L156)
  - `test_add_polynomials_with_different_degrees(self)` — [`L180`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns_test.py#L180)
  - `test_add_polynomials_with_different_forms(self)` — [`L208`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns_test.py#L208)
  - `test_add_polynomials_with_different_moduli(self)` — [`L198`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns_test.py#L198)
  - `test_add_polynomials_with_incompatible_coeffs(self)` — [`L190`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns_test.py#L190)
  - `test_create_ntt_with_invalid_n(self, invalid_n)` — [`L171`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns_test.py#L171)
  - `test_create_ntt_with_invalid_q(self)` — [`L175`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns_test.py#L175)
  - `test_multiply_polynomials_in_coefficient_form(self)` — [`L215`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns_test.py#L215)
  - `test_ntt_loop(self, coeffs)` — [`L228`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns_test.py#L228)
  - `degree` — [`L158`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns_test.py#L158)
  - `moduli` — [`L159`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns_test.py#L159)
- uses (calls/refs, reference-scoped): [`is_ntt`](rns.md#RnsPolynomial.is_ntt), [`RnsPolynomial`](rns.md#RnsPolynomial), [`ntt_negacyclic_poly`](ntt_cpu.md#ntt_negacyclic_poly), [`intt_negacyclic_poly`](ntt_cpu.md#intt_negacyclic_poly), [`Ntt`](rns.md#Ntt)

### `RnsPolynomialTest`  ·  implements/extends TestCase
- def: [`jaxite/jaxite_ckks/rns_test.py:23`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns_test.py#L23)
- signature: `class RnsPolynomialTest(absltest.TestCase):`
- members:
  - `setUp(self)` — [`L25`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns_test.py#L25)
  - `test_iterative_cooley_tukey(self)` — [`L48`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns_test.py#L48)
  - `test_iterative_gentleman_sande(self)` — [`L70`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns_test.py#L70)
  - `test_ntt(self)` — [`L38`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns_test.py#L38)
  - `test_rns_polynomial_addition(self)` — [`L97`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns_test.py#L97)
  - `test_rns_polynomial_multiplication(self)` — [`L135`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns_test.py#L135)
  - `test_rns_polynomial_negation(self)` — [`L116`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns_test.py#L116)
  - `ntt_params` — [`L27`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns_test.py#L27)
- protocol/private: `_random_coeffs`[`L29`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns_test.py#L29), `_random_rns_polynomial`[`L32`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns_test.py#L32)
- uses (calls/refs, reference-scoped): [`is_ntt`](rns.md#RnsPolynomial.is_ntt), [`RnsPolynomial`](rns.md#RnsPolynomial), [`to_coeffs_form`](rns.md#RnsPolynomial.to_coeffs_form), [`to_ntt_form`](rns.md#RnsPolynomial.to_ntt_form), [`backward`](rns.md#Ntt.backward), [`bit_reversal_array`](rns_utils.md#bit_reversal_array), [`Ntt`](rns.md#Ntt), [`_primitive_root`](rns.md#_primitive_root), [`_iterative_cooley_tukey`](rns.md#Ntt._iterative_cooley_tukey), [`_iterative_gentleman_sande`](rns.md#Ntt._iterative_gentleman_sande), [`forward`](rns.md#Ntt.forward), [`num_bits`](rns_utils.md#num_bits), [`inverse_mod`](rns_utils.md#inverse_mod)


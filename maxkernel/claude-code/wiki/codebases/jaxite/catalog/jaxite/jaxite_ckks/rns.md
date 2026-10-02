---
title: 'Module: jaxite/jaxite_ckks/rns.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_ckks/rns.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_ckks.rns`/
symbols:
  RnsPolynomial.is_ntt: RnsPolynomial#is_ntt.
  RnsPolynomial: RnsPolynomial#
  Ntt.__post_init__: Ntt#__post_init__().
  RnsPolynomial.moduli: RnsPolynomial#moduli.
  RnsPolynomial.degree: RnsPolynomial#degree.
  RnsPolynomial._check_compatible: RnsPolynomial#_check_compatible().
  RnsPolynomial.__add__: RnsPolynomial#__add__().
  RnsPolynomial.__sub__: RnsPolynomial#__sub__().
  RnsPolynomial.__mul__: RnsPolynomial#__mul__().
  RnsPolynomial.to_ntt_form: RnsPolynomial#to_ntt_form().
  RnsPolynomial.to_coeffs_form: RnsPolynomial#to_coeffs_form().
  RnsPolynomial.coeffs: RnsPolynomial#coeffs.
  RnsPolynomial.__neg__: RnsPolynomial#__neg__().
  Ntt.backward: Ntt#backward().
  Ntt.psis_inv_bitrev: Ntt#psis_inv_bitrev.
  Ntt: Ntt#
  RnsParams.__post_init__: RnsParams#__post_init__().
  _primitive_root: _primitive_root().
  Ntt.q: Ntt#q.
  Ntt.forward: Ntt#forward().
  Ntt._iterative_cooley_tukey: Ntt#_iterative_cooley_tukey().
  Ntt._iterative_gentleman_sande: Ntt#_iterative_gentleman_sande().
  gen_rns_polynomial: gen_rns_polynomial().
  gen_rns_polynomial_from_jnp_array: gen_rns_polynomial_from_jnp_array().
  Ntt.psis_bitrev: Ntt#psis_bitrev.
  RnsParams.ntt_params: RnsParams#ntt_params.
  _mod_exp: _mod_exp().
  Ntt.n: Ntt#n.
  RnsPolynomial.to_jnp_array: RnsPolynomial#to_jnp_array().
  RnsPolynomial.__len__: RnsPolynomial#__len__().
  Ntt.n_inv_mod_q: Ntt#n_inv_mod_q.
  RnsParams.degree: RnsParams#degree.
  RnsParams.moduli: RnsParams#moduli.
  RnsParams: RnsParams#
---
# Module: [`jaxite/jaxite_ckks/rns.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns.py)

## Classes
### `Ntt`
- def: [`jaxite/jaxite_ckks/rns.py:49`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns.py#L49)
- doc: Number Theoretic Transformations in Z\[X\]/(q, X^n+1).
- signature: `class Ntt:`
- members:
  - `_iterative_cooley_tukey(self, coeffs: list[int], log_len: int)` — [`L104`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns.py#L104) — Cooley-Tukey NTT on coeffs in log_len iterations.
  - `_iterative_gentleman_sande(self, coeffs: list[int], log_len: int)` — [`L123`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns.py#L123) — Gentleman-Sande NTT on coeffs in log_len iterations.
  - `backward(self, coeffs: list[int])` — [`L98`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns.py#L98) — Backward NTT (normalized).
  - `forward(self, coeffs: list[int])` — [`L94`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns.py#L94) — Forward NTT.
  - `n` — [`L55`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns.py#L55) — documented in [jaxite-jaxite_ckks-rns](../../../concepts/jaxite-jaxite_ckks-rns.md)
  - `n_inv_mod_q` — [`L58`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns.py#L58) — documented in [jaxite-jaxite_ckks-rns](../../../concepts/jaxite-jaxite_ckks-rns.md)
  - `psis_bitrev` — [`L62`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns.py#L62) — documented in [jaxite-jaxite_ckks-rns](../../../concepts/jaxite-jaxite_ckks-rns.md)
  - `psis_inv_bitrev` — [`L65`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns.py#L65) — documented in [jaxite-jaxite_ckks-rns](../../../concepts/jaxite-jaxite_ckks-rns.md)
  - `q` — [`L56`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns.py#L56) — documented in [jaxite-jaxite_ckks-rns](../../../concepts/jaxite-jaxite_ckks-rns.md)
- protocol/private: `__post_init__`[`L67`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns.py#L67)
- uses (calls/refs, reference-scoped): [`bit_reversal_array`](rns_utils.md#bit_reversal_array), [`_primitive_root`](rns.md#_primitive_root), [`num_bits`](rns_utils.md#num_bits), [`_mod_exp`](rns.md#_mod_exp), [`inverse_mod`](rns_utils.md#inverse_mod), [`is_power_of_two`](rns_utils.md#is_power_of_two)
- used by: [`test_iterative_gentleman_sande`](rns_test.md#RnsPolynomialTest.test_iterative_gentleman_sande), [`test_iterative_cooley_tukey`](rns_test.md#RnsPolynomialTest.test_iterative_cooley_tukey), [`to_coeffs_form`](rns.md#RnsPolynomial.to_coeffs_form), [`to_ntt_form`](rns.md#RnsPolynomial.to_ntt_form), [`__post_init__`](rns.md#RnsParams.__post_init__), [`test_ntt`](rns_test.md#RnsPolynomialTest.test_ntt), [`ntt_params`](rns_test.md#RnsPolynomialTest.ntt_params), [`test_create_ntt_with_invalid_n`](rns_test.md#RnsNegativeTest.test_create_ntt_with_invalid_n), [`test_create_ntt_with_invalid_q`](rns_test.md#RnsNegativeTest.test_create_ntt_with_invalid_q), [`ntt_params`](rns.md#RnsParams.ntt_params)

### `RnsParams`
- def: [`jaxite/jaxite_ckks/rns.py:145`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns.py#L145)
- doc: Parameters for an RNS instance over Z\[X\]/(Q, X^N+1).
- signature: `class RnsParams:`
- members:
  - `degree` — [`L149`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns.py#L149)
  - `moduli` — [`L152`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns.py#L152)
  - `ntt_params` — [`L155`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns.py#L155)
- protocol/private: `__post_init__`[`L157`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns.py#L157)
- uses (calls/refs, reference-scoped): [`Ntt`](rns.md#Ntt)

### `RnsPolynomial`
- def: [`jaxite/jaxite_ckks/rns.py:162`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns.py#L162) — documented in [jaxite-jaxite_ckks-rns](../../../concepts/jaxite-jaxite_ckks-rns.md)
- doc: A polynomial in the quotient ring R_Q = Z\[X\] / (Q, X^N + 1).
- signature: `class RnsPolynomial:`
- members:
  - `__add__(self, other: RnsPolynomial)` — [`L238`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns.py#L238) — Add two polynomials in R_Q. — documented in [jaxite-jaxite_ckks-rns](../../../concepts/jaxite-jaxite_ckks-rns.md)
  - `__len__(self)` — [`L197`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns.py#L197) — Returns the length of the polynomial.
  - `__mul__(self, other: RnsPolynomial)` — [`L262`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns.py#L262) — Multiply two polynomials in R_Q. — documented in [jaxite-jaxite_ckks-rns](../../../concepts/jaxite-jaxite_ckks-rns.md)
  - `__neg__(self)` — [`L201`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns.py#L201) — Compute the negative of a polynomial.
  - `__sub__(self, other: RnsPolynomial)` — [`L250`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns.py#L250) — Subtract two polynomials in R_Q. — documented in [jaxite-jaxite_ckks-rns](../../../concepts/jaxite-jaxite_ckks-rns.md)
  - `_check_compatible(self, other: RnsPolynomial)` — [`L209`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns.py#L209) — Check if two polynomials are compatible. — documented in [jaxite-jaxite_ckks-rns](../../../concepts/jaxite-jaxite_ckks-rns.md)
  - `to_coeffs_form(self, ntt_params: list[Ntt])` — [`L185`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns.py#L185) — Convert the polynomial to the Coefficient form. — documented in [jaxite-jaxite_ckks-rns](../../../concepts/jaxite-jaxite_ckks-rns.md)
  - `to_jnp_array(self)` — [`L193`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns.py#L193) — Convert the polynomial to a jax.Array.
  - `to_ntt_form(self, ntt_params: list[Ntt])` — [`L177`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns.py#L177) — Convert the polynomial to the NTT form. — documented in [jaxite-jaxite_ckks-rns](../../../concepts/jaxite-jaxite_ckks-rns.md)
  - `coeffs` — [`L172`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns.py#L172)
  - `degree` — [`L166`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns.py#L166)
  - `is_ntt` — [`L175`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns.py#L175) — documented in [jaxite-jaxite_ckks-rns](../../../concepts/jaxite-jaxite_ckks-rns.md)
  - `moduli` — [`L169`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns.py#L169)
- uses (calls/refs, reference-scoped): [`backward`](rns.md#Ntt.backward), [`Ntt`](rns.md#Ntt), [`forward`](rns.md#Ntt.forward)
- used by: [`_random_rns_polynomial`](rns_test.md#RnsPolynomialTest._random_rns_polynomial), [`test_rns_polynomial_addition`](rns_test.md#RnsPolynomialTest.test_rns_polynomial_addition), [`test_rns_polynomial_negation`](rns_test.md#RnsPolynomialTest.test_rns_polynomial_negation), [`test_add_polynomials_with_different_degrees`](rns_test.md#RnsNegativeTest.test_add_polynomials_with_different_degrees), [`test_add_polynomials_with_different_forms`](rns_test.md#RnsNegativeTest.test_add_polynomials_with_different_forms), [`test_add_polynomials_with_different_moduli`](rns_test.md#RnsNegativeTest.test_add_polynomials_with_different_moduli), [`test_add_polynomials_with_incompatible_coeffs`](rns_test.md#RnsNegativeTest.test_add_polynomials_with_incompatible_coeffs), [`test_multiply_polynomials_in_coefficient_form`](rns_test.md#RnsNegativeTest.test_multiply_polynomials_in_coefficient_form), [`gen_rns_polynomial`](rns.md#gen_rns_polynomial), [`gen_rns_polynomial_from_jnp_array`](rns.md#gen_rns_polynomial_from_jnp_array), [`test_rns_polynomial_multiplication`](rns_test.md#RnsPolynomialTest.test_rns_polynomial_multiplication)

## Functions
- `_mod_exp(x: int, n: int, q: int)` — [`L14`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns.py#L14) — Returns x^n mod q. — documented in [jaxite-jaxite_ckks-rns](../../../concepts/jaxite-jaxite_ckks-rns.md)
- `_primitive_root(m: int, q: int)` — [`L25`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns.py#L25) — Returns a m'th primitive root of unity mod q. — documented in [jaxite-jaxite_ckks-rns](../../../concepts/jaxite-jaxite_ckks-rns.md)
- `gen_rns_polynomial(degree: int, coeffs: list[int], moduli: list[int])` — [`L278`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns.py#L278) — Generate a RNS polynomial from the given mod-Q coefficients. — documented in [jaxite-jaxite_ckks-rns](../../../concepts/jaxite-jaxite_ckks-rns.md)
- `gen_rns_polynomial_from_jnp_array(degree: int, rns_coeffs: jnp.ndarray, moduli: list[int], is_ntt: bool = False)` — [`L288`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns.py#L288) — Generate a RNS polynomial from the given RNS coefficients. — documented in [jaxite-jaxite_ckks-rns](../../../concepts/jaxite-jaxite_ckks-rns.md)


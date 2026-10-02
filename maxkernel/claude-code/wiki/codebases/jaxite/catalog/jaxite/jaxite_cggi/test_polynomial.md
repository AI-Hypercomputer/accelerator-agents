---
title: 'Module: jaxite/jaxite_cggi/test_polynomial.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_cggi/test_polynomial.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_cggi.test_polynomial`/
symbols:
  gen_test_polynomial: gen_test_polynomial().
  trivial_encryption: trivial_encryption().
  identity_test_polynomial: identity_test_polynomial().
  gen_and_encrypt: gen_and_encrypt().
  _pad_for_error: _pad_for_error().
---
# Module: [`jaxite/jaxite_cggi/test_polynomial.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/test_polynomial.py)

## Functions
- `_pad_for_error(polynomial: jnp.ndarray, mod_degree: int, coeff_mod: int)` — [`L48`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/test_polynomial.py#L48) — Pads the coefficients of the input polynomial into blocks.
- `gen_and_encrypt(cleartext_coefficients: jnp.ndarray, encoding_params: encoding.EncodingParameters, scheme_params: parameters.SchemeParameters)` — [`L214`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/test_polynomial.py#L214) — Generates and encrypts a test polynomial for a lookup table.
- `gen_test_polynomial(cleartext_coefficients: jnp.ndarray, encoding_params: encoding.EncodingParameters, scheme_params: parameters.SchemeParameters)` — [`L93`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/test_polynomial.py#L93) — Generates a test polynomial with padding and encoding.
- `identity_test_polynomial(encoding_params: encoding.EncodingParameters, scheme_params: parameters.SchemeParameters)` — [`L153`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/test_polynomial.py#L153) — Generates a test polynomial for the identity function.
- `trivial_encryption(test_polynomial: rlwe.RlwePlaintext, scheme_params: parameters.SchemeParameters)` — [`L182`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/test_polynomial.py#L182) — Trivially encrypts the test polynomial.


---
title: 'Module: jaxite/jaxite_cggi/rlwe_test.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_cggi/rlwe_test.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_cggi.rlwe_test`/RlweTest#
symbols:
  RlweTest.test_encrypt_add_decrypt_prod_security_params: test_encrypt_add_decrypt_prod_security_params().
  RlweTest.test_encrypt_add_decrypt: test_encrypt_add_decrypt().
  RlweTest.setUp: setUp().
  RlweTest.test_gen_key: test_gen_key().
  RlweTest.test_encrypt_dimension: test_encrypt_dimension().
  RlweTest.test_error_free_encrypt_decrypt: test_error_free_encrypt_decrypt().
  RlweTest.default_key: default_key.
  RlweTest.test_rlwe_plaintext_str: test_rlwe_plaintext_str().
  RlweTest.test_rlwe_ciphertext_str: test_rlwe_ciphertext_str().
  RlweTest.noise_free_rng: noise_free_rng.
  RlweTest.dim: dim.
  RlweTest.plaintext_modulus: plaintext_modulus.
  RlweTest.polynomial_modulus_degree: polynomial_modulus_degree.
  RlweTest.rlwe_dimension: rlwe_dimension.
  RlweTest: ''
---
# Module: [`jaxite/jaxite_cggi/rlwe_test.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rlwe_test.py)

## Classes
### `RlweTest`  ·  implements/extends TestCase
- def: [`jaxite/jaxite_cggi/rlwe_test.py:18`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rlwe_test.py#L18)
- signature: `class RlweTest(parameterized.TestCase):`
- members:
  - `setUp(self)` — [`L20`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rlwe_test.py#L20)
  - `test_encrypt_add_decrypt(self, message1, message2, rng)` — [`L147`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rlwe_test.py#L147)
  - `test_encrypt_add_decrypt_prod_security_params(self)` — [`L179`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rlwe_test.py#L179)
  - `test_encrypt_dimension(self, plaintext)` — [`L96`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rlwe_test.py#L96)
  - `test_error_free_encrypt_decrypt(self, message)` — [`L115`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rlwe_test.py#L115)
  - `test_gen_key(self, dim: int, deg: int)` — [`L71`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rlwe_test.py#L71)
  - `test_rlwe_ciphertext_str(self)` — [`L52`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rlwe_test.py#L52)
  - `test_rlwe_plaintext_str(self, coeffs: list[int], expected: str)` — [`L45`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rlwe_test.py#L45)
  - `default_key` — [`L27`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rlwe_test.py#L27)
  - `dim` — [`L22`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rlwe_test.py#L22)
  - `noise_free_rng` — [`L26`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rlwe_test.py#L26)
  - `plaintext_modulus` — [`L23`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rlwe_test.py#L23)
  - `polynomial_modulus_degree` — [`L24`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rlwe_test.py#L24)
  - `rlwe_dimension` — [`L25`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rlwe_test.py#L25)
- uses (calls/refs, reference-scoped): [`encrypt`](rlwe.md#encrypt), [`decrypt`](rlwe.md#decrypt), [`message_bit_length`](encoding.md#EncodingParameters.message_bit_length), [`message`](rlwe.md#RlwePlaintext.message), [`padding_bit_length`](encoding.md#EncodingParameters.padding_bit_length), [`gen_key`](rlwe.md#gen_key), [`SchemeParameters`](parameters.md#SchemeParameters), [`total_bit_length`](encoding.md#EncodingParameters.total_bit_length), [`polynomial_modulus_degree`](parameters.md#SchemeParameters.polynomial_modulus_degree), [`RlweCiphertext`](rlwe.md#RlweCiphertext), [`lwe_dimension`](parameters.md#SchemeParameters.lwe_dimension), [`message`](rlwe.md#RlweCiphertext.message), [`data`](rlwe.md#RlweSecretKey.data), [`RlwePlaintext`](rlwe.md#RlwePlaintext), [`rlwe_dimension`](parameters.md#SchemeParameters.rlwe_dimension), [`plaintext_modulus`](parameters.md#SchemeParameters.plaintext_modulus), [`log_coefficient_modulus`](rlwe.md#RlwePlaintext.log_coefficient_modulus), [`log_coefficient_modulus`](rlwe.md#RlweSecretKey.log_coefficient_modulus), [`modulus_degree`](rlwe.md#RlwePlaintext.modulus_degree), [`modulus_degree`](rlwe.md#RlweCiphertext.modulus_degree), [`log_coefficient_modulus`](rlwe.md#RlweCiphertext.log_coefficient_modulus), [`modulus_degree`](rlwe.md#RlweSecretKey.modulus_degree), [`CycleRng`](random_source.md#CycleRng), [`log_plaintext_modulus`](parameters.md#SchemeParameters.log_plaintext_modulus), [`rlwe_dimension`](rlwe.md#RlweSecretKey.rlwe_dimension), [`VARYING_MAGNITUDE_TEST_RNGS`](random_source.md#VARYING_MAGNITUDE_TEST_RNGS)  (4 test-only)


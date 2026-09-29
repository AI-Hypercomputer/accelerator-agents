---
title: 'Module: jaxite/jaxite_cggi/lwe_test.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_cggi/lwe_test.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_cggi.lwe_test`/Lwe
symbols:
  LweModulusSwitchingTest.test_modulus_switch_preserves_message: ModulusSwitchingTest#test_modulus_switch_preserves_message().
  LweEncryptDecryptTest.default_key: EncryptDecryptTest#default_key.
  LweEncryptDecryptTest.test_gen_key: EncryptDecryptTest#test_gen_key().
  LweEncryptDecryptTest.test_deterministic_noise_encrypt: EncryptDecryptTest#test_deterministic_noise_encrypt().
  LweEncryptDecryptTest.test_error_free_encrypt_decrypt_uint32: EncryptDecryptTest#test_error_free_encrypt_decrypt_uint32().
  LweEncryptDecryptTest.test_encrypt_add_decrypt_prod_security_params: EncryptDecryptTest#test_encrypt_add_decrypt_prod_security_params().
  LweEncryptDecryptTest.test_error_free_encrypt_decrypt: EncryptDecryptTest#test_error_free_encrypt_decrypt().
  LweEncryptDecryptTest.test_encrypt_add_decrypt: EncryptDecryptTest#test_encrypt_add_decrypt().
  LweEncryptDecryptTest.test_max_error_encrypt_decrypt: EncryptDecryptTest#test_max_error_encrypt_decrypt().
  LweEncryptDecryptTest.test_too_much_error_cannot_decrypt: EncryptDecryptTest#test_too_much_error_cannot_decrypt().
  LweEncryptDecryptTest.test_encrypt_dimension: EncryptDecryptTest#test_encrypt_dimension().
  LweEncryptDecryptTest.noise_free_rng: EncryptDecryptTest#noise_free_rng.
  LweEncryptDecryptTest.polynomial_modulus_degree: EncryptDecryptTest#polynomial_modulus_degree.
  LweEncryptDecryptTest.test_noiseless_embedding_succeeds: EncryptDecryptTest#test_noiseless_embedding_succeeds().
  LweEncryptDecryptTest.dim: EncryptDecryptTest#dim.
  LweEncryptDecryptTest.plaintext_modulus: EncryptDecryptTest#plaintext_modulus.
  LweEncryptDecryptTest.rlwe_dimension: EncryptDecryptTest#rlwe_dimension.
  LweEncryptDecryptTest: EncryptDecryptTest#
  LweEncryptDecryptTest.setUp: EncryptDecryptTest#setUp().
  LweModulusSwitchingTest: ModulusSwitchingTest#
---
# Module: [`jaxite/jaxite_cggi/lwe_test.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/lwe_test.py)

## Classes
### `LweEncryptDecryptTest`  ·  implements/extends TestCase
- def: [`jaxite/jaxite_cggi/lwe_test.py:17`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/lwe_test.py#L17)
- signature: `class LweEncryptDecryptTest(parameterized.TestCase):`
- members:
  - `setUp(self)` — [`L19`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/lwe_test.py#L19)
  - `test_deterministic_noise_encrypt(self)` — [`L62`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/lwe_test.py#L62)
  - `test_encrypt_add_decrypt(self, message1, message2, rng)` — [`L136`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/lwe_test.py#L136)
  - `test_encrypt_add_decrypt_prod_security_params(self)` — [`L205`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/lwe_test.py#L205)
  - `test_encrypt_dimension(self, plaintext)` — [`L56`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/lwe_test.py#L56)
  - `test_error_free_encrypt_decrypt(self, message)` — [`L97`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/lwe_test.py#L97)
  - `test_error_free_encrypt_decrypt_uint32(self, message)` — [`L114`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/lwe_test.py#L114)
  - `test_gen_key(self, dim: int)` — [`L39`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/lwe_test.py#L39)
  - `test_max_error_encrypt_decrypt(self, sign: int)` — [`L158`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/lwe_test.py#L158)
  - `test_noiseless_embedding_succeeds(self)` — [`L198`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/lwe_test.py#L198)
  - `test_too_much_error_cannot_decrypt(self, sign: int)` — [`L182`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/lwe_test.py#L182)
  - `default_key` — [`L26`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/lwe_test.py#L26)
  - `dim` — [`L22`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/lwe_test.py#L22)
  - `noise_free_rng` — [`L21`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/lwe_test.py#L21)
  - `plaintext_modulus` — [`L23`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/lwe_test.py#L23)
  - `polynomial_modulus_degree` — [`L25`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/lwe_test.py#L25)
  - `rlwe_dimension` — [`L24`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/lwe_test.py#L24)
- uses (calls/refs, reference-scoped): [`message_bit_length`](encoding.md#EncodingParameters.message_bit_length), [`encode`](encoding.md#encode), [`encrypt`](lwe.md#encrypt), [`EncodingParameters`](encoding.md#EncodingParameters), [`gen_key`](lwe.md#gen_key), [`padding_bit_length`](encoding.md#EncodingParameters.padding_bit_length), [`SchemeParameters`](parameters.md#SchemeParameters), [`total_bit_length`](encoding.md#EncodingParameters.total_bit_length), [`decode`](encoding.md#decode), [`polynomial_modulus_degree`](parameters.md#SchemeParameters.polynomial_modulus_degree), [`decrypt`](lwe.md#decrypt), [`lwe_dimension`](parameters.md#SchemeParameters.lwe_dimension), [`rlwe_dimension`](parameters.md#SchemeParameters.rlwe_dimension), [`plaintext_modulus`](parameters.md#SchemeParameters.plaintext_modulus), [`lwe_dimension`](lwe.md#LweSecretKey.lwe_dimension), [`LwePlaintext`](types.md#LwePlaintext), [`key_data`](lwe.md#LweSecretKey.key_data), [`CycleRng`](random_source.md#CycleRng), [`VARYING_MAGNITUDE_TEST_RNGS`](random_source.md#VARYING_MAGNITUDE_TEST_RNGS), [`noiseless_embedding`](lwe.md#noiseless_embedding), [`modulus`](lwe.md#LweSecretKey.modulus)  (4 test-only)

### `LweModulusSwitchingTest`
- def: [`jaxite/jaxite_cggi/lwe_test.py:234`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/lwe_test.py#L234)
- signature: `class LweModulusSwitchingTest(parameterized.TestCase):`
- members:
  - `test_modulus_switch_preserves_message(self, log_output_modulus: int, lwe_dim: int, cleartext: int, injected_noise: int, seed: int)` — [`L236`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/lwe_test.py#L236) — Ensure that modulus switch preserves the encrypted message.
- uses (calls/refs, reference-scoped): [`message_bit_length`](encoding.md#EncodingParameters.message_bit_length), [`encode`](encoding.md#encode), [`encrypt`](lwe.md#encrypt), [`EncodingParameters`](encoding.md#EncodingParameters), [`gen_key`](lwe.md#gen_key), [`padding_bit_length`](encoding.md#EncodingParameters.padding_bit_length), [`SchemeParameters`](parameters.md#SchemeParameters), [`total_bit_length`](encoding.md#EncodingParameters.total_bit_length), [`decode`](encoding.md#decode), [`polynomial_modulus_degree`](parameters.md#SchemeParameters.polynomial_modulus_degree), [`PseudorandomSource`](random_source.md#PseudorandomSource), [`lwe_dimension`](parameters.md#SchemeParameters.lwe_dimension), [`decrypt_without_denoising`](lwe.md#decrypt_without_denoising), [`rlwe_dimension`](parameters.md#SchemeParameters.rlwe_dimension), [`plaintext_modulus`](parameters.md#SchemeParameters.plaintext_modulus), [`extract_noise`](encoding.md#extract_noise), [`lwe_dimension`](lwe.md#LweSecretKey.lwe_dimension), [`LweSecretKey`](lwe.md#LweSecretKey), [`key_data`](lwe.md#LweSecretKey.key_data), [`log_modulus`](lwe.md#LweSecretKey.log_modulus), [`switch_modulus`](lwe.md#switch_modulus)


---
title: 'Module: jaxite/jaxite_cggi/blind_rotate_test.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_cggi/blind_rotate_test.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_cggi.blind_rotate_test`/BlindRotateTest#
symbols:
  BlindRotateTest.test_cmux_noisy: test_cmux_noisy().
  BlindRotateTest.test_blind_rotate_with_noise: test_blind_rotate_with_noise().
  BlindRotateTest.test_sample_extract_noisy: test_sample_extract_noisy().
  BlindRotateTest.test_blind_rotate_encoded: test_blind_rotate_encoded().
  BlindRotateTest.test_blind_rotate_doesnt_set_padding_bit: test_blind_rotate_doesnt_set_padding_bit().
  BlindRotateTest.test_external_product_noisy_multiply: test_external_product_noisy_multiply().
  BlindRotateTest.test_cmux_noise_free_encoded: test_cmux_noise_free_encoded().
  BlindRotateTest.test_external_product_noise_free_multiply: test_external_product_noise_free_multiply().
  BlindRotateTest.test_cmux_noise_free: test_cmux_noise_free().
  BlindRotateTest.test_sample_extract_noisefree: test_sample_extract_noisefree().
  BlindRotateTest.rlwe_key: rlwe_key.
  BlindRotateTest.decomposition_params: decomposition_params.
  BlindRotateTest.scheme_params: scheme_params.
  BlindRotateTest.polynomial_modulus_degree: polynomial_modulus_degree.
  BlindRotateTest.noisy_encoding: noisy_encoding.
  BlindRotateTest.noise_free_rng: noise_free_rng.
  BlindRotateTest.log_plaintext_modulus: log_plaintext_modulus.
  BlindRotateTest.rgsw_key: rgsw_key.
  BlindRotateTest.encoding: encoding.
  BlindRotateTest.rlwe_dimension: rlwe_dimension.
  BlindRotateTest: ''
  BlindRotateTest.setUp: setUp().
---
# Module: [`jaxite/jaxite_cggi/blind_rotate_test.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/blind_rotate_test.py)

## Classes
### `BlindRotateTest`  ·  implements/extends TestCase
- def: [`jaxite/jaxite_cggi/blind_rotate_test.py:20`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/blind_rotate_test.py#L20)
- signature: `class BlindRotateTest(parameterized.TestCase):`
- members:
  - `setUp(self)` — [`L22`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/blind_rotate_test.py#L22)
  - `test_blind_rotate_doesnt_set_padding_bit(self)` — [`L552`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/blind_rotate_test.py#L552)
  - `test_blind_rotate_encoded(self, j)` — [`L416`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/blind_rotate_test.py#L416)
  - `test_blind_rotate_with_noise(self, j, seed, lwe_dim)` — [`L486`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/blind_rotate_test.py#L486)
  - `test_cmux_noise_free(self, control)` — [`L173`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/blind_rotate_test.py#L173)
  - `test_cmux_noise_free_encoded(self, control)` — [`L220`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/blind_rotate_test.py#L220)
  - `test_cmux_noisy(self, control)` — [`L278`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/blind_rotate_test.py#L278)
  - `test_external_product_noise_free_multiply(self, bit)` — [`L73`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/blind_rotate_test.py#L73)
  - `test_external_product_noisy_multiply(self, bit)` — [`L114`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/blind_rotate_test.py#L114)
  - `test_sample_extract_noisefree(self, message)` — [`L351`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/blind_rotate_test.py#L351)
  - `test_sample_extract_noisy(self, message)` — [`L376`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/blind_rotate_test.py#L376)
  - `decomposition_params` — [`L32`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/blind_rotate_test.py#L32)
  - `encoding` — [`L41`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/blind_rotate_test.py#L41)
  - `log_plaintext_modulus` — [`L24`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/blind_rotate_test.py#L24)
  - `noise_free_rng` — [`L37`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/blind_rotate_test.py#L37)
  - `noisy_encoding` — [`L50`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/blind_rotate_test.py#L50)
  - `polynomial_modulus_degree` — [`L35`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/blind_rotate_test.py#L35)
  - `rgsw_key` — [`L67`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/blind_rotate_test.py#L67)
  - `rlwe_dimension` — [`L36`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/blind_rotate_test.py#L36)
  - `rlwe_key` — [`L64`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/blind_rotate_test.py#L64)
  - `scheme_params` — [`L57`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/blind_rotate_test.py#L57)
- uses (calls/refs, reference-scoped): [`encrypt`](rgsw.md#encrypt), [`gen_bootstrapping_key`](bootstrap.md#gen_bootstrapping_key), [`encrypt`](rlwe.md#encrypt), [`decrypt`](rlwe.md#decrypt), [`message_bit_length`](encoding.md#EncodingParameters.message_bit_length), [`encode`](encoding.md#encode), [`encrypt`](lwe.md#encrypt), [`EncodingParameters`](encoding.md#EncodingParameters), [`gen_key`](lwe.md#gen_key), [`message`](rlwe.md#RlwePlaintext.message), [`padding_bit_length`](encoding.md#EncodingParameters.padding_bit_length), [`gen_key`](rlwe.md#gen_key), [`SchemeParameters`](parameters.md#SchemeParameters), [`total_bit_length`](encoding.md#EncodingParameters.total_bit_length), [`DecompositionParameters`](decomposition.md#DecompositionParameters), [`decode`](encoding.md#decode), [`polynomial_modulus_degree`](parameters.md#SchemeParameters.polynomial_modulus_degree), [`blind_rotate`](bootstrap.md#blind_rotate), [`RlweCiphertext`](rlwe.md#RlweCiphertext), [`PseudorandomSource`](random_source.md#PseudorandomSource), [`decrypt`](lwe.md#decrypt), [`lwe_dimension`](parameters.md#SchemeParameters.lwe_dimension), [`flatten_key`](rlwe.md#flatten_key), [`message`](rlwe.md#RlweCiphertext.message), [`data`](rlwe.md#RlweSecretKey.data), [`level_count`](decomposition.md#DecompositionParameters.level_count), [`cmux`](bootstrap.md#cmux), [`RlwePlaintext`](rlwe.md#RlwePlaintext), [`log_base`](decomposition.md#DecompositionParameters.log_base), [`rlwe_dimension`](parameters.md#SchemeParameters.rlwe_dimension), [`plaintext_modulus`](parameters.md#SchemeParameters.plaintext_modulus), [`external_product`](bootstrap.md#external_product), [`log_coefficient_modulus`](rlwe.md#RlwePlaintext.log_coefficient_modulus), [`key`](rgsw.md#RgswSecretKey.key), [`log_coefficient_modulus`](rlwe.md#RlweSecretKey.log_coefficient_modulus), [`modulus_degree`](rlwe.md#RlwePlaintext.modulus_degree), [`decode_without_removing_padding`](encoding.md#decode_without_removing_padding), [`message`](rgsw.md#RgswPlaintext.message), [`modulus_degree`](rlwe.md#RlweCiphertext.modulus_degree), [`lwe_dimension`](lwe.md#LweSecretKey.lwe_dimension)  (+11 more; 3 test-only)


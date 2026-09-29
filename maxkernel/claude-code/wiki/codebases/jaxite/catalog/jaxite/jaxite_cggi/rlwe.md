---
title: 'Module: jaxite/jaxite_cggi/rlwe.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_cggi/rlwe.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_cggi.rlwe`/
symbols:
  encrypt: encrypt().
  decrypt: decrypt().
  RlwePlaintext.message: RlwePlaintext#message.
  gen_key: gen_key().
  RlweCiphertext: RlweCiphertext#
  flatten_key: flatten_key().
  RlweCiphertext.message: RlweCiphertext#message.
  RlweSecretKey.data: RlweSecretKey#data.
  RlwePlaintext: RlwePlaintext#
  RlwePlaintext.log_coefficient_modulus: RlwePlaintext#log_coefficient_modulus.
  RlwePlaintext.modulus_degree: RlwePlaintext#modulus_degree.
  RlweSecretKey.log_coefficient_modulus: RlweSecretKey#log_coefficient_modulus.
  RlweCiphertext.modulus_degree: RlweCiphertext#modulus_degree.
  RlweCiphertext._single_row_str: RlweCiphertext#_single_row_str().
  RlweCiphertext.log_coefficient_modulus: RlweCiphertext#log_coefficient_modulus.
  RlweSecretKey.modulus_degree: RlweSecretKey#modulus_degree.
  RlweSecretKey: RlweSecretKey#
  RlweSecretKey.rlwe_dimension: RlweSecretKey#rlwe_dimension.
  RlweCiphertext.coefficient_modulus: RlweCiphertext#coefficient_modulus().
  RlweCiphertext.__str__: RlweCiphertext#__str__().
  jit_encrypt: jit_encrypt().
  RlweSecretKey.coefficient_modulus: RlweSecretKey#coefficient_modulus().
  RlwePlaintext.coefficient_modulus: RlwePlaintext#coefficient_modulus().
  RlwePlaintext.__str__: RlwePlaintext#__str__().
  RlweCiphertext.__repr__: RlweCiphertext#__repr__().
---
# Module: [`jaxite/jaxite_cggi/rlwe.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rlwe.py)

## Classes
### `RlweCiphertext`
- def: [`jaxite/jaxite_cggi/rlwe.py:44`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rlwe.py#L44) — documented in [jaxite-jaxite_cggi-rlwe](../../../concepts/jaxite-jaxite_cggi-rlwe.md)
- doc: An RLWE ciphertext.
- signature: `class RlweCiphertext:`
- members:
  - `coefficient_modulus(self)` — [`L63`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rlwe.py#L63)
  - `log_coefficient_modulus` — [`L52`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rlwe.py#L52)
  - `message` — [`L60`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rlwe.py#L60) — documented in [jaxite-jaxite_cggi-rlwe](../../../concepts/jaxite-jaxite_cggi-rlwe.md)
  - `modulus_degree` — [`L55`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rlwe.py#L55)
- protocol/private: `__repr__`[`L80`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rlwe.py#L80), `__str__`[`L75`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rlwe.py#L75), `_single_row_str`[`L66`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rlwe.py#L66)
- uses (calls/refs, reference-scoped): [`message`](rlwe.md#RlwePlaintext.message), [`RlwePlaintext`](rlwe.md#RlwePlaintext), [`log_coefficient_modulus`](rlwe.md#RlwePlaintext.log_coefficient_modulus), [`modulus_degree`](rlwe.md#RlwePlaintext.modulus_degree)
- used by: [`run_bootstrap_test`](bootstrap_test.md#BootstrapBaseTest.run_bootstrap_test), [`bootstrap`](bootstrap.md#bootstrap), [`test_blind_rotate_encoded`](blind_rotate_test.md#BlindRotateTest.test_blind_rotate_encoded), [`encrypt`](rlwe.md#encrypt), [`test_encrypt_add_decrypt_prod_security_params`](rlwe_test.md#RlweTest.test_encrypt_add_decrypt_prod_security_params), [`decrypt`](rlwe.md#decrypt), [`test_encrypt_add_decrypt`](rlwe_test.md#RlweTest.test_encrypt_add_decrypt), [`blind_rotate`](bootstrap.md#blind_rotate), [`cmux`](bootstrap.md#cmux), [`test_encrypt_dimension`](rlwe_test.md#RlweTest.test_encrypt_dimension), [`as_rlwe_test_polynomial`](../jaxite_bool/lut.md#LookUpTable.as_rlwe_test_polynomial), [`external_product`](bootstrap.md#external_product), [`jit_external_product`](bootstrap.md#jit_external_product), [`lut_poly_by_name`](../jaxite_bool/bool_params.md#Parameters.lut_poly_by_name), [`pmap_lut2`](../jaxite_bool/jaxite_bool.md#pmap_lut2), [`pmap_lut3`](../jaxite_bool/jaxite_bool.md#pmap_lut3), [`jit_blind_rotate`](bootstrap.md#jit_blind_rotate), [`sample_extract`](bootstrap.md#sample_extract), [`lut_poly`](../jaxite_bool/bool_params.md#Parameters.lut_poly), [`lut_poly`](../jaxite_bool/lut.md#LutCache.lut_poly), [`lut_poly_by_name`](../jaxite_bool/lut.md#LutCache.lut_poly_by_name), [`test_rlwe_ciphertext_str`](rlwe_test.md#RlweTest.test_rlwe_ciphertext_str), [`jit_cmux`](bootstrap.md#jit_cmux)  (3 test-only)

### `RlwePlaintext`
- def: [`jaxite/jaxite_cggi/rlwe.py:16`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rlwe.py#L16) — documented in [jaxite-jaxite_cggi-rlwe](../../../concepts/jaxite-jaxite_cggi-rlwe.md)
- doc: An RLWE plaintext is a polynomial in (Z/qZ)\[X\] / (X^N + 1).
- signature: `class RlwePlaintext:`
- members:
  - `coefficient_modulus(self)` — [`L30`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rlwe.py#L30)
  - `log_coefficient_modulus` — [`L20`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rlwe.py#L20) — documented in [jaxite-jaxite_cggi-rlwe](../../../concepts/jaxite-jaxite_cggi-rlwe.md)
  - `message` — [`L27`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rlwe.py#L27) — documented in [jaxite-jaxite_cggi-rlwe](../../../concepts/jaxite-jaxite_cggi-rlwe.md)
  - `modulus_degree` — [`L23`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rlwe.py#L23) — documented in [jaxite-jaxite_cggi-rlwe](../../../concepts/jaxite-jaxite_cggi-rlwe.md)
- protocol/private: `__str__`[`L33`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rlwe.py#L33)
- used by: [`test_blind_rotate_with_noise`](blind_rotate_test.md#BlindRotateTest.test_blind_rotate_with_noise), [`test_cmux_noisy`](blind_rotate_test.md#BlindRotateTest.test_cmux_noisy), [`test_blind_rotate_doesnt_set_padding_bit`](blind_rotate_test.md#BlindRotateTest.test_blind_rotate_doesnt_set_padding_bit), [`test_blind_rotate_encoded`](blind_rotate_test.md#BlindRotateTest.test_blind_rotate_encoded), [`test_sample_extract_noisy`](blind_rotate_test.md#BlindRotateTest.test_sample_extract_noisy), [`test_external_product_noisy_multiply`](blind_rotate_test.md#BlindRotateTest.test_external_product_noisy_multiply), [`test_cmux_noise_free_encoded`](blind_rotate_test.md#BlindRotateTest.test_cmux_noise_free_encoded), [`encrypt`](rlwe.md#encrypt), [`test_cmux_noise_free`](blind_rotate_test.md#BlindRotateTest.test_cmux_noise_free), [`test_encrypt_add_decrypt_prod_security_params`](rlwe_test.md#RlweTest.test_encrypt_add_decrypt_prod_security_params), [`test_external_product_noise_free_multiply`](blind_rotate_test.md#BlindRotateTest.test_external_product_noise_free_multiply), [`decrypt`](rlwe.md#decrypt), [`test_encrypt_add_decrypt`](rlwe_test.md#RlweTest.test_encrypt_add_decrypt), [`test_sample_extract_noisefree`](blind_rotate_test.md#BlindRotateTest.test_sample_extract_noisefree), [`test_encrypt_dimension`](rlwe_test.md#RlweTest.test_encrypt_dimension), [`test_error_free_encrypt_decrypt`](rlwe_test.md#RlweTest.test_error_free_encrypt_decrypt), [`_single_row_str`](rlwe.md#RlweCiphertext._single_row_str), [`test_rlwe_plaintext_str`](rlwe_test.md#RlweTest.test_rlwe_plaintext_str)  (8 test-only)

### `RlweSecretKey`
- def: [`jaxite/jaxite_cggi/rlwe.py:85`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rlwe.py#L85) — documented in [jaxite-jaxite_cggi-rgsw](../../../concepts/jaxite-jaxite_cggi-rgsw.md)
- doc: A secret key for the RLWE encryption scheme.
- signature: `class RlweSecretKey:`
- members:
  - `coefficient_modulus(self)` — [`L103`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rlwe.py#L103)
  - `data` — [`L100`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rlwe.py#L100) — documented in [jaxite-jaxite_cggi-rlwe](../../../concepts/jaxite-jaxite_cggi-rlwe.md)
  - `log_coefficient_modulus` — [`L89`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rlwe.py#L89)
  - `modulus_degree` — [`L92`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rlwe.py#L92)
  - `rlwe_dimension` — [`L95`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rlwe.py#L95) — documented in [jaxite-jaxite_cggi-rlwe](../../../concepts/jaxite-jaxite_cggi-rlwe.md)
- used by: [`test_cmux_noisy`](blind_rotate_test.md#BlindRotateTest.test_cmux_noisy), [`encrypt`](rgsw.md#encrypt), [`test_sample_extract_noisy`](blind_rotate_test.md#BlindRotateTest.test_sample_extract_noisy), [`gen_bootstrapping_key`](bootstrap.md#gen_bootstrapping_key), [`test_cmux_noise_free_encoded`](blind_rotate_test.md#BlindRotateTest.test_cmux_noise_free_encoded), [`encrypt`](rlwe.md#encrypt), [`test_cmux_noise_free`](blind_rotate_test.md#BlindRotateTest.test_cmux_noise_free), [`test_encrypt_add_decrypt_prod_security_params`](rlwe_test.md#RlweTest.test_encrypt_add_decrypt_prod_security_params), [`test_external_product_noise_free_multiply`](blind_rotate_test.md#BlindRotateTest.test_external_product_noise_free_multiply), [`test_encrypt_decrypt_with_varying_rngs`](rgsw_test.md#RgswTest.test_encrypt_decrypt_with_varying_rngs), [`decrypt`](rlwe.md#decrypt), [`decrypt`](rgsw.md#decrypt), [`test_encrypt_decrypt_with_varying_nonzero_sk_entry`](rgsw_test.md#RgswTest.test_encrypt_decrypt_with_varying_nonzero_sk_entry), [`gen_key`](rlwe.md#gen_key), [`test_encrypt_add_decrypt`](rlwe_test.md#RlweTest.test_encrypt_add_decrypt), [`test_encrypt_diagonal`](rgsw_test.md#RgswTest.test_encrypt_diagonal), [`test_encrypt_decrypt_prod_security_params`](rgsw_test.md#RgswTest.test_encrypt_decrypt_prod_security_params), [`flatten_key`](rlwe.md#flatten_key), [`test_gen_key`](rlwe_test.md#RlweTest.test_gen_key), [`test_encrypt_dimension`](rlwe_test.md#RlweTest.test_encrypt_dimension), [`test_error_free_encrypt_decrypt`](rlwe_test.md#RlweTest.test_error_free_encrypt_decrypt), [`key`](rgsw.md#RgswSecretKey.key), [`key_from_rlwe`](rgsw.md#key_from_rlwe), [`rlwe_sk`](../jaxite_bool/jaxite_bool.md#ClientKeySet.rlwe_sk), [`process_one_batch`](bootstrap.md#gen_bootstrapping_key.process_one_batch), [`to_rlwe_secret_key`](rgsw.md#RgswSecretKey.to_rlwe_secret_key), [`data_at_index`](rgsw.md#RgswSecretKey.data_at_index)  (2 test-only)

## Functions
- `decrypt(ciphertext: RlweCiphertext, sk: RlweSecretKey, encoding_params: encoding.EncodingParameters)` — [`L173`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rlwe.py#L173) — Decrypt an RLWE ciphertext. — documented in [jaxite-jaxite_cggi-encoding](../../../concepts/jaxite-jaxite_cggi-encoding.md)
- `encrypt(plaintext: RlwePlaintext, sk: RlweSecretKey, prg: random_source.RandomSource)` — [`L122`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rlwe.py#L122) — Encrypt an RLWE plaintext. — documented in [jaxite-jaxite_cggi-random_source](../../../concepts/jaxite-jaxite_cggi-random_source.md)
- `flatten_key(sk: RlweSecretKey)` — [`L192`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rlwe.py#L192) — Flattens rlwe secret key to use for decrypting after sample extraction. — documented in [jaxite-jaxite_bool](../../../concepts/jaxite-jaxite_bool.md)
- `gen_key(params: parameters.SchemeParameters, prg: random_source.RandomSource)` — [`L107`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rlwe.py#L107) — Generate an RLWE secret key. — documented in [jaxite-jaxite_bool](../../../concepts/jaxite-jaxite_bool.md)
- `jit_encrypt(plaintext: jnp.ndarray, key_data: jnp.ndarray, ai_samples: jnp.ndarray, error_sample: jnp.ndarray, log_coefficient_modulus: int)` — [`L149`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rlwe.py#L149) — Encrypt an RLWE plaintext with pre-computed randomness.


---
title: 'Module: jaxite/jaxite_cggi/lwe.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_cggi/lwe.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_cggi.lwe`/
symbols:
  encrypt: encrypt().
  gen_key: gen_key().
  decrypt: decrypt().
  decrypt_without_denoising: decrypt_without_denoising().
  LweSecretKey.lwe_dimension: LweSecretKey#lwe_dimension.
  LweSecretKey: LweSecretKey#
  LweSecretKey.key_data: LweSecretKey#key_data.
  LweSecretKey.log_modulus: LweSecretKey#log_modulus.
  switch_modulus: switch_modulus().
  jit_encrypt: jit_encrypt().
  noiseless_embedding: noiseless_embedding().
  LweSecretKey.modulus: LweSecretKey#modulus().
---
# Module: [`jaxite/jaxite_cggi/lwe.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/lwe.py)

## Classes
### `LweSecretKey`
- def: [`jaxite/jaxite_cggi/lwe.py:16`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/lwe.py#L16) — documented in [jaxite-jaxite_bool](../../../concepts/jaxite-jaxite_bool.md)
- doc: A secret key for the LWE encryption scheme.
- signature: `class LweSecretKey:`
- members:
  - `modulus(self)` — [`L33`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/lwe.py#L33)
  - `key_data` — [`L30`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/lwe.py#L30) — documented in [jaxite-jaxite_cggi-lwe](../../../concepts/jaxite-jaxite_cggi-lwe.md)
  - `log_modulus` — [`L20`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/lwe.py#L20) — documented in [jaxite-jaxite_cggi-lwe](../../../concepts/jaxite-jaxite_cggi-lwe.md)
  - `lwe_dimension` — [`L26`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/lwe.py#L26) — documented in [jaxite-jaxite_cggi-lwe](../../../concepts/jaxite-jaxite_cggi-lwe.md)
- used by: [`run_bootstrap_test`](bootstrap_test.md#BootstrapBaseTest.run_bootstrap_test), [`test_key_switch_128_bit_security`](key_switch_test.md#KeySwitchTest.test_key_switch_128_bit_security), [`test_sample_extract_noisy`](blind_rotate_test.md#BlindRotateTest.test_sample_extract_noisy), [`gen_bootstrapping_key`](bootstrap.md#gen_bootstrapping_key), [`gen_key`](key_switch.md#gen_key), [`test_modulus_switch_preserves_message`](lwe_test.md#LweModulusSwitchingTest.test_modulus_switch_preserves_message), [`encrypt`](lwe.md#encrypt), [`gen_key`](lwe.md#gen_key), [`decrypt`](lwe.md#decrypt), [`flatten_key`](rlwe.md#flatten_key), [`test_deterministic_noise_encrypt`](lwe_test.md#LweEncryptDecryptTest.test_deterministic_noise_encrypt), [`test_gen_key`](lwe_test.md#LweEncryptDecryptTest.test_gen_key), [`decrypt_without_denoising`](lwe.md#decrypt_without_denoising), [`test_key_switch_gen`](key_switch_test.md#KeySwitchTest.test_key_switch_gen), [`decompose_and_encrypt`](key_switch.md#gen_key.decompose_and_encrypt), [`lwe_sk`](../jaxite_bool/jaxite_bool.md#ClientKeySet.lwe_sk), [`test_encrypt_dimension`](lwe_test.md#LweEncryptDecryptTest.test_encrypt_dimension)  (2 test-only)

## Functions
- `decrypt(ciphertext: types.LweCiphertext, sk: LweSecretKey, encoding_params: encoding.EncodingParameters)` — [`L93`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/lwe.py#L93) — Decrypt and remove the noise from an LWE ciphertext. — documented in [jaxite-jaxite_cggi-encoding](../../../concepts/jaxite-jaxite_cggi-encoding.md)
- `decrypt_without_denoising(ciphertext: types.LweCiphertext, sk: LweSecretKey)` — [`L81`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/lwe.py#L81) — Decrypt an LWE ciphertext without removing noise. — documented in [jaxite-jaxite_cggi-lwe](../../../concepts/jaxite-jaxite_cggi-lwe.md)
- `encrypt(plaintext: types.LwePlaintext, sk: LweSecretKey, prg: random_source.RandomSource)` — [`L48`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/lwe.py#L48) — Encrypt an LWE plaintext. — documented in [jaxite-jaxite_bool](../../../concepts/jaxite-jaxite_bool.md)
- `gen_key(params: parameters.SchemeParameters, prg: random_source.RandomSource)` — [`L37`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/lwe.py#L37) — Generate an LWE secret key. — documented in [jaxite-jaxite_bool](../../../concepts/jaxite-jaxite_bool.md)
- `jit_encrypt(plaintext: types.LwePlaintext, key_data: jnp.ndarray, ai_samples: jnp.ndarray, error_sample: jnp.uint32, log_modulus: int)` — [`L66`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/lwe.py#L66) — Encrypt an LWE plaintext with pre-computed randomness. — documented in [jaxite-jaxite_cggi-lwe](../../../concepts/jaxite-jaxite_cggi-lwe.md)
- `noiseless_embedding(plaintext: types.LwePlaintext, lwe_dimension: int)` — [`L104`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/lwe.py#L104) — Returns a noiseless LweCiphertext embedding of `plaintext`. — documented in [jaxite-jaxite_bool-bool_params](../../../concepts/jaxite-jaxite_bool-bool_params.md)
- `switch_modulus(ciphertext: types.LweCiphertext, log_input_modulus: jnp.uint32, log_output_modulus: jnp.uint32)` — [`L113`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/lwe.py#L113) — Perform a modulus switch on the input ciphertext. — documented in [jaxite-jaxite_cggi-bootstrap](../../../concepts/jaxite-jaxite_cggi-bootstrap.md)


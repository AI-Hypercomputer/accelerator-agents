---
title: 'Module: jaxite/jaxite_cggi/rgsw.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_cggi/rgsw.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_cggi.rgsw`/
symbols:
  encrypt: encrypt().
  decrypt: decrypt().
  RgswSecretKey.key: RgswSecretKey#key.
  RgswPlaintext.message: RgswPlaintext#message.
  gen_key: gen_key().
  RgswPlaintext: RgswPlaintext#
  jit_encrypt: jit_encrypt().
  RgswPlaintext.modulus_degree: RgswPlaintext#modulus_degree.
  key_from_rlwe: key_from_rlwe().
  RgswCiphertext.message: RgswCiphertext#message.
  RgswSecretKey.to_rlwe_secret_key: RgswSecretKey#to_rlwe_secret_key().
  RgswSecretKey: RgswSecretKey#
  RgswCiphertext: RgswCiphertext#
  RgswCiphertext.__str__: RgswCiphertext#__str__().
  RgswSecretKey.data_at_index: RgswSecretKey#data_at_index().
  jit_encrypt.encrypt_and_modify_one_row: jit_encrypt().encrypt_and_modify_one_row().
  jit_encrypt.encrypt_block: jit_encrypt().encrypt_block().
  RgswCiphertext.log_coefficient_modulus: RgswCiphertext#log_coefficient_modulus.
  RgswPlaintext.__str__: RgswPlaintext#__str__().
  RgswCiphertext.coefficient_modulus: RgswCiphertext#coefficient_modulus().
  RgswCiphertext.__len__: RgswCiphertext#__len__().
  RgswCiphertext.modulus_degree: RgswCiphertext#modulus_degree.
  RgswCiphertext.pretty_print_polynomial: RgswCiphertext#pretty_print_polynomial().
  RgswCiphertext.__repr__: RgswCiphertext#__repr__().
---
# Module: [`jaxite/jaxite_cggi/rgsw.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rgsw.py)

## Classes
### `RgswCiphertext`
- def: [`jaxite/jaxite_cggi/rgsw.py:39`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rgsw.py#L39) — documented in [jaxite-jaxite_cggi-rgsw](../../../concepts/jaxite-jaxite_cggi-rgsw.md)
- doc: An RGSW ciphertext.
- signature: `class RgswCiphertext:`
- members:
  - `coefficient_modulus(self)` — [`L56`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rgsw.py#L56)
  - `pretty_print_polynomial(self, row: jnp.ndarray)` — [`L62`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rgsw.py#L62)
  - `log_coefficient_modulus` — [`L50`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rgsw.py#L50)
  - `message` — [`L47`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rgsw.py#L47) — documented in [jaxite-jaxite_cggi-rgsw](../../../concepts/jaxite-jaxite_cggi-rgsw.md)
  - `modulus_degree` — [`L53`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rgsw.py#L53)
- protocol/private: `__len__`[`L59`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rgsw.py#L59), `__repr__`[`L79`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rgsw.py#L79), `__str__`[`L69`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rgsw.py#L69)
- used by: [`encrypt`](rgsw.md#encrypt), [`decrypt`](rgsw.md#decrypt), [`test_encrypt_diagonal`](rgsw_test.md#RgswTest.test_encrypt_diagonal), [`test_encrypt_dimension`](rgsw_test.md#RgswTest.test_encrypt_dimension), [`cmux`](bootstrap.md#cmux), [`external_product`](bootstrap.md#external_product), [`test_rgsw_ciphertext_str`](rgsw_test.md#RgswTest.test_rgsw_ciphertext_str)

### `RgswPlaintext`
- def: [`jaxite/jaxite_cggi/rgsw.py:18`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rgsw.py#L18) — documented in [jaxite-jaxite_cggi-rgsw](../../../concepts/jaxite-jaxite_cggi-rgsw.md)
- doc: An RGSW plaintext is an unsigned integer and a polynomial modulus degree.
- signature: `class RgswPlaintext:`
- members:
  - `message` — [`L32`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rgsw.py#L32) — documented in [jaxite-jaxite_cggi-rgsw](../../../concepts/jaxite-jaxite_cggi-rgsw.md)
  - `modulus_degree` — [`L29`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rgsw.py#L29) — documented in [jaxite-jaxite_cggi-rgsw](../../../concepts/jaxite-jaxite_cggi-rgsw.md)
- protocol/private: `__str__`[`L34`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rgsw.py#L34)
- used by: [`test_cmux_noisy`](blind_rotate_test.md#BlindRotateTest.test_cmux_noisy), [`encrypt`](rgsw.md#encrypt), [`test_external_product_noisy_multiply`](blind_rotate_test.md#BlindRotateTest.test_external_product_noisy_multiply), [`test_cmux_noise_free_encoded`](blind_rotate_test.md#BlindRotateTest.test_cmux_noise_free_encoded), [`test_cmux_noise_free`](blind_rotate_test.md#BlindRotateTest.test_cmux_noise_free), [`test_external_product_noise_free_multiply`](blind_rotate_test.md#BlindRotateTest.test_external_product_noise_free_multiply), [`test_encrypt_decrypt_with_varying_rngs`](rgsw_test.md#RgswTest.test_encrypt_decrypt_with_varying_rngs), [`decrypt`](rgsw.md#decrypt), [`test_encrypt_decrypt_with_varying_nonzero_sk_entry`](rgsw_test.md#RgswTest.test_encrypt_decrypt_with_varying_nonzero_sk_entry), [`test_encrypt_diagonal`](rgsw_test.md#RgswTest.test_encrypt_diagonal), [`test_encrypt_decrypt_prod_security_params`](rgsw_test.md#RgswTest.test_encrypt_decrypt_prod_security_params), [`test_encrypt_dimension`](rgsw_test.md#RgswTest.test_encrypt_dimension), [`test_encrypt_decrypt_no_noise`](rgsw_test.md#RgswTest.test_encrypt_decrypt_no_noise)

### `RgswSecretKey`
- def: [`jaxite/jaxite_cggi/rgsw.py:84`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rgsw.py#L84) — documented in [jaxite-jaxite_cggi-rgsw](../../../concepts/jaxite-jaxite_cggi-rgsw.md)
- doc: This is a wrapper around RlweSecretKey.
- signature: `class RgswSecretKey:`
- members:
  - `data_at_index(self, i: int)` — [`L93`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rgsw.py#L93) — documented in [jaxite-jaxite_cggi-rgsw](../../../concepts/jaxite-jaxite_cggi-rgsw.md)
  - `to_rlwe_secret_key(self)` — [`L90`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rgsw.py#L90) — documented in [jaxite-jaxite_cggi-rgsw](../../../concepts/jaxite-jaxite_cggi-rgsw.md)
  - `key` — [`L88`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rgsw.py#L88) — documented in [jaxite-jaxite_cggi-rgsw](../../../concepts/jaxite-jaxite_cggi-rgsw.md)
- uses (calls/refs, reference-scoped): [`data`](rlwe.md#RlweSecretKey.data), [`RlweSecretKey`](rlwe.md#RlweSecretKey)
- used by: [`test_cmux_noisy`](blind_rotate_test.md#BlindRotateTest.test_cmux_noisy), [`encrypt`](rgsw.md#encrypt), [`gen_bootstrapping_key`](bootstrap.md#gen_bootstrapping_key), [`test_cmux_noise_free_encoded`](blind_rotate_test.md#BlindRotateTest.test_cmux_noise_free_encoded), [`test_cmux_noise_free`](blind_rotate_test.md#BlindRotateTest.test_cmux_noise_free), [`test_external_product_noise_free_multiply`](blind_rotate_test.md#BlindRotateTest.test_external_product_noise_free_multiply), [`test_encrypt_decrypt_with_varying_rngs`](rgsw_test.md#RgswTest.test_encrypt_decrypt_with_varying_rngs), [`decrypt`](rgsw.md#decrypt), [`test_encrypt_decrypt_with_varying_nonzero_sk_entry`](rgsw_test.md#RgswTest.test_encrypt_decrypt_with_varying_nonzero_sk_entry), [`test_encrypt_diagonal`](rgsw_test.md#RgswTest.test_encrypt_diagonal), [`test_encrypt_decrypt_prod_security_params`](rgsw_test.md#RgswTest.test_encrypt_decrypt_prod_security_params), [`gen_key`](rgsw.md#gen_key), [`key_from_rlwe`](rgsw.md#key_from_rlwe)

## Functions
- `decrypt(ciphertext: RgswCiphertext, decomposition_params: decomposition.DecompositionParameters, sk: RgswSecretKey)` — [`L267`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rgsw.py#L267) — Decrypts an RGSW ciphertext. — documented in [jaxite-jaxite_cggi-rgsw](../../../concepts/jaxite-jaxite_cggi-rgsw.md)
- `encrypt(plaintext: RgswPlaintext, sk: RgswSecretKey, decomposition_params: decomposition.DecompositionParameters, prg: random_source.RandomSource)` — [`L109`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rgsw.py#L109) — Create an RGSW ciphertext. — documented in [jaxite-jaxite_cggi-rgsw](../../../concepts/jaxite-jaxite_cggi-rgsw.md)
- `encrypt_and_modify_one_row(ai_samples, error_sample, level, block, plaintext_message)` — [`L216`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rgsw.py#L216)
- `encrypt_block(ai_samples, error_samples, block, plaintext_message)` — [`L244`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rgsw.py#L244)
- `gen_key(params: parameters.SchemeParameters, prg: random_source.RandomSource)` — [`L97`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rgsw.py#L97) — Generate an RGSW secret key. — documented in [jaxite-jaxite_cggi-random_source](../../../concepts/jaxite-jaxite_cggi-random_source.md)
- `jit_encrypt(plaintext: jnp.ndarray, rlwe_sk: jnp.ndarray, ai_samples: jnp.ndarray, error_samples: jnp.ndarray, decomposition_params: decomposition.DecompositionParameters, num_blocks: int, log_coefficient_modulus: int, modulus_degree: int)` — [`L162`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rgsw.py#L162) — Create an RGSW ciphertext. — documented in [jaxite-jaxite_cggi-bootstrap](../../../concepts/jaxite-jaxite_cggi-bootstrap.md)
- `key_from_rlwe(rlwe_key: rlwe.RlweSecretKey)` — [`L104`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rgsw.py#L104) — Convert an RLWE secret key to the corresponding RGSW secret key. — documented in [jaxite-jaxite_cggi-rgsw](../../../concepts/jaxite-jaxite_cggi-rgsw.md)


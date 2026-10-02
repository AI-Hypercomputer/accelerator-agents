---
title: 'Module: jaxite/jaxite_cggi/test_utils.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_cggi/test_utils.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_cggi.test_utils`/
symbols:
  MidBootstrapDecrypter.decrypt: MidBootstrapDecrypter#decrypt().
  DEFAULT_ENCODING_PARAMS: DEFAULT_ENCODING_PARAMS.
  SCHEME_PARAMS_128_BIT_SECURITY: SCHEME_PARAMS_128_BIT_SECURITY.
  ENCODING_PARAMS_128_BIT_SECURITY: ENCODING_PARAMS_128_BIT_SECURITY.
  MidBootstrapDecrypter.encoding_params: MidBootstrapDecrypter#encoding_params.
  MidBootstrapDecrypter.__init__: MidBootstrapDecrypter#__init__().
  MidBootstrapDecrypter.starting_cleartext: MidBootstrapDecrypter#starting_cleartext.
  RLWE_RNG_128_BIT_SECURITY: RLWE_RNG_128_BIT_SECURITY.
  LWE_RNG_128_BIT_SECURITY: LWE_RNG_128_BIT_SECURITY.
  ConsistencyChecker.store: ConsistencyChecker#store().
  ConsistencyChecker.check: ConsistencyChecker#check().
  BSK_DECOMP_PARAMS_128_BIT_SECURITY: BSK_DECOMP_PARAMS_128_BIT_SECURITY.
  KSK_DECOMP_PARAMS_128_BIT_SECURITY: KSK_DECOMP_PARAMS_128_BIT_SECURITY.
  ConsistencyChecker.records: ConsistencyChecker#records.
  MidBootstrapDecrypter.lwe_key: MidBootstrapDecrypter#lwe_key.
  MidBootstrapDecrypter.rlwe_key: MidBootstrapDecrypter#rlwe_key.
  MidBootstrapDecrypter._noise_and_bits: MidBootstrapDecrypter#_noise_and_bits().
  assert_safe_modulus_switch: assert_safe_modulus_switch().
  MidBootstrapDecrypter: MidBootstrapDecrypter#
  ConsistencyChecker.insertion_order: ConsistencyChecker#insertion_order.
  MidBootstrapDecrypter.scheme_params: MidBootstrapDecrypter#scheme_params.
  ConsistencyChecker: ConsistencyChecker#
  ConsistencyChecker.__init__: ConsistencyChecker#__init__().
---
# Module: [`jaxite/jaxite_cggi/test_utils.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/test_utils.py)

## Classes
### `ConsistencyChecker`
- def: [`jaxite/jaxite_cggi/test_utils.py:33`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/test_utils.py#L33)
- doc: A helper for checking consistency of intermediate values across runs.
- signature: `class ConsistencyChecker:`
- members:
  - `check(self, asserter=None)` — [`L48`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/test_utils.py#L48)
  - `store(self, name: str, value: jnp.ndarray)` — [`L40`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/test_utils.py#L40)
  - `insertion_order` — [`L37`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/test_utils.py#L37)
  - `records` — [`L38`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/test_utils.py#L38)
- protocol/private: `__init__`[`L36`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/test_utils.py#L36)
- used by: [`test_3_bit_consistency`](bootstrap_test.md#ConsistencyTest.test_3_bit_consistency)

### `MidBootstrapDecrypter`
- def: [`jaxite/jaxite_cggi/test_utils.py:93`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/test_utils.py#L93)
- doc: A helper for decrypting intermediate encrypted values during bootstrap.
- signature: `class MidBootstrapDecrypter:`
- members:
  - `decrypt(self, name: str, value: jnp.ndarray, callback_lut: Optional[list[int]] = None)` — [`L119`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/test_utils.py#L119) — Decrypt an intermediate value of bootstrap and report noise growth.
  - `encoding_params` — [`L108`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/test_utils.py#L108)
  - `lwe_key` — [`L109`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/test_utils.py#L109)
  - `rlwe_key` — [`L110`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/test_utils.py#L110)
  - `scheme_params` — [`L107`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/test_utils.py#L107)
  - `starting_cleartext` — [`L111`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/test_utils.py#L111)
- protocol/private: `__init__`[`L100`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/test_utils.py#L100), `_noise_and_bits`[`L113`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/test_utils.py#L113)
- uses (calls/refs, reference-scoped): [`decrypt`](rlwe.md#decrypt), [`message_bit_length`](encoding.md#EncodingParameters.message_bit_length), [`EncodingParameters`](encoding.md#EncodingParameters), [`message`](rlwe.md#RlwePlaintext.message), [`padding_bit_length`](encoding.md#EncodingParameters.padding_bit_length), [`SchemeParameters`](parameters.md#SchemeParameters), [`total_bit_length`](encoding.md#EncodingParameters.total_bit_length), [`decode`](encoding.md#decode), [`RlweCiphertext`](rlwe.md#RlweCiphertext), [`decrypt`](lwe.md#decrypt), [`lwe_dimension`](parameters.md#SchemeParameters.lwe_dimension), [`flatten_key`](rlwe.md#flatten_key), [`message`](rlwe.md#RlweCiphertext.message), [`decrypt_without_denoising`](lwe.md#decrypt_without_denoising), [`log_coefficient_modulus`](rlwe.md#RlweSecretKey.log_coefficient_modulus), [`modulus_degree`](rlwe.md#RlweCiphertext.modulus_degree), [`extract_noise`](encoding.md#extract_noise), [`lwe_dimension`](lwe.md#LweSecretKey.lwe_dimension), [`LweSecretKey`](lwe.md#LweSecretKey), [`key_data`](lwe.md#LweSecretKey.key_data), [`log_coefficient_modulus`](rlwe.md#RlweCiphertext.log_coefficient_modulus), [`modulus_degree`](rlwe.md#RlweSecretKey.modulus_degree), [`RlweSecretKey`](rlwe.md#RlweSecretKey), [`log_modulus`](lwe.md#LweSecretKey.log_modulus), [`log_mod_degree`](parameters.md#SchemeParameters.log_mod_degree)
- used by: [`run_bootstrap_test`](bootstrap_test.md#BootstrapBaseTest.run_bootstrap_test), [`callback`](../jaxite_bool/jaxite_bool_test.md#BoolBasicOperationsTest.callback), [`callback`](../jaxite_bool/jaxite_bool_multigate_test.md#BoolMultigateTest.callback)

## Functions
- `assert_safe_modulus_switch(mod_degree: int, message_bits: int, lwe_dimension: int)` — [`L57`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/test_utils.py#L57) — Guard against corruption of the message due to modulus switch error.

## Module values
- `BSK_DECOMP_PARAMS_128_BIT_SECURITY` — [`L25`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/test_utils.py#L25)
- `DEFAULT_ENCODING_PARAMS` — [`L14`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/test_utils.py#L14)
- `ENCODING_PARAMS_128_BIT_SECURITY` — [`L17`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/test_utils.py#L17)
- `KSK_DECOMP_PARAMS_128_BIT_SECURITY` — [`L28`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/test_utils.py#L28)
- `LWE_RNG_128_BIT_SECURITY` — [`L21`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/test_utils.py#L21)
- `RLWE_RNG_128_BIT_SECURITY` — [`L22`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/test_utils.py#L22)
- `SCHEME_PARAMS_128_BIT_SECURITY` — [`L23`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/test_utils.py#L23)


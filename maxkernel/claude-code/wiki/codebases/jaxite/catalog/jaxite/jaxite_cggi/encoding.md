---
title: 'Module: jaxite/jaxite_cggi/encoding.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_cggi/encoding.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_cggi.encoding`/
symbols:
  EncodingParameters.message_bit_length: EncodingParameters#message_bit_length.
  encode: encode().
  EncodingParameters: EncodingParameters#
  EncodingParameters.padding_bit_length: EncodingParameters#padding_bit_length.
  EncodingParameters.total_bit_length: EncodingParameters#total_bit_length.
  decode: decode().
  decode_without_removing_padding: decode_without_removing_padding().
  EncodingParameters.__post_init__: EncodingParameters#__post_init__().
  extract_noise: extract_noise().
  remove_noise: remove_noise().
  EncodingParameters.error_bit_length: EncodingParameters#error_bit_length.
  EncodingParameters.message_min: EncodingParameters#message_min.
  EncodingParameters.message_max: EncodingParameters#message_max.
  EncodingParameters.message_range: EncodingParameters#message_range.
  round_to_power_of_2: round_to_power_of_2().
---
# Module: [`jaxite/jaxite_cggi/encoding.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/encoding.py)

## Classes
### `EncodingParameters`
- def: [`jaxite/jaxite_cggi/encoding.py:11`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/encoding.py#L11) — documented in [jaxite-jaxite_cggi-encoding](../../../concepts/jaxite-jaxite_cggi-encoding.md)
- doc: The parameters for encoding a cleartext for use in TFHE.
- signature: `class EncodingParameters:`
- members:
  - `error_bit_length` — [`L33`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/encoding.py#L33) — documented in [jaxite-jaxite_cggi-encoding](../../../concepts/jaxite-jaxite_cggi-encoding.md)
  - `message_bit_length` — [`L22`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/encoding.py#L22) — documented in [jaxite-jaxite_cggi-encoding](../../../concepts/jaxite-jaxite_cggi-encoding.md)
  - `message_max` — [`L39`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/encoding.py#L39) — documented in [jaxite-jaxite_cggi-encoding](../../../concepts/jaxite-jaxite_cggi-encoding.md)
  - `message_min` — [`L36`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/encoding.py#L36)
  - `message_range` — [`L42`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/encoding.py#L42)
  - `padding_bit_length` — [`L30`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/encoding.py#L30) — documented in [jaxite-jaxite_cggi-encoding](../../../concepts/jaxite-jaxite_cggi-encoding.md)
  - `total_bit_length` — [`L19`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/encoding.py#L19)
- protocol/private: `__post_init__`[`L44`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/encoding.py#L44)
- uses (calls/refs, reference-scoped): [`LweCleartext`](types.md#LweCleartext)
- used by: [`run_bootstrap_test`](bootstrap_test.md#BootstrapBaseTest.run_bootstrap_test), [`test_key_switch_128_bit_security`](key_switch_test.md#KeySwitchTest.test_key_switch_128_bit_security), [`test_blind_rotate_with_noise`](blind_rotate_test.md#BlindRotateTest.test_blind_rotate_with_noise), [`test_cmux_noisy`](blind_rotate_test.md#BlindRotateTest.test_cmux_noisy), [`test_blind_rotate_doesnt_set_padding_bit`](blind_rotate_test.md#BlindRotateTest.test_blind_rotate_doesnt_set_padding_bit), [`test_blind_rotate_encoded`](blind_rotate_test.md#BlindRotateTest.test_blind_rotate_encoded), [`test_sample_extract_noisy`](blind_rotate_test.md#BlindRotateTest.test_sample_extract_noisy), [`test_external_product_noisy_multiply`](blind_rotate_test.md#BlindRotateTest.test_external_product_noisy_multiply), [`test_modulus_switch_preserves_message`](lwe_test.md#LweModulusSwitchingTest.test_modulus_switch_preserves_message), [`test_encrypt_add_decrypt_prod_security_params`](rlwe_test.md#RlweTest.test_encrypt_add_decrypt_prod_security_params), [`decrypt`](rlwe.md#decrypt), [`encode`](encoding.md#encode), [`test_switch_key_error_free`](key_switch_test.md#KeySwitchTest.test_switch_key_error_free), [`test_switch_key_with_error`](key_switch_test.md#KeySwitchTest.test_switch_key_with_error), [`decode`](encoding.md#decode), [`decrypt`](lwe.md#decrypt), [`as_rlwe_test_polynomial`](../jaxite_bool/lut.md#LookUpTable.as_rlwe_test_polynomial), [`test_encode_decode_message_bounds_succeeds`](encoding_test.md#EncodingDecodingTest.test_encode_decode_message_bounds_succeeds), [`test_encode_decode_succeeds`](encoding_test.md#EncodingDecodingTest.test_encode_decode_succeeds), [`test_error_free_encrypt_decrypt_uint32`](lwe_test.md#LweEncryptDecryptTest.test_error_free_encrypt_decrypt_uint32), [`decode_without_removing_padding`](encoding.md#decode_without_removing_padding), [`ENCODING_PARAMS`](../jaxite_bool/bool_encoding.md#ENCODING_PARAMS), [`noisy_encoding`](blind_rotate_test.md#BlindRotateTest.noisy_encoding), [`extract_noise`](encoding.md#extract_noise), [`test_encode_add_noise_decode_succeeds`](encoding_test.md#EncodingDecodingTest.test_encode_add_noise_decode_succeeds), [`test_encode_add_noise_extract_noise`](encoding_test.md#EncodingDecodingTest.test_encode_add_noise_extract_noise), [`test_encode_invalid_message_raises`](encoding_test.md#EncodingDecodingTest.test_encode_invalid_message_raises), [`encoding`](blind_rotate_test.md#BlindRotateTest.encoding), [`remove_noise`](encoding.md#remove_noise), [`test_encode_test_polynomial_coefficients`](encoding_test.md#EncodingDecodingTest.test_encode_test_polynomial_coefficients), [`test_encode_test_polynomial_coefficients_too_large`](encoding_test.md#EncodingDecodingTest.test_encode_test_polynomial_coefficients_too_large), [`test_invalid_message_bit_length_raises`](encoding_test.md#EncodingParametersTest.test_invalid_message_bit_length_raises), [`test_invalid_padding_bit_length_raises`](encoding_test.md#EncodingParametersTest.test_invalid_padding_bit_length_raises), [`test_invalid_total_bit_length_raises`](encoding_test.md#EncodingParametersTest.test_invalid_total_bit_length_raises), [`test_unavailable_space_raises`](encoding_test.md#EncodingParametersTest.test_unavailable_space_raises)  (12 test-only)

## Functions
- `decode(plaintext: types.LwePlaintext, params: EncodingParameters)` — [`L130`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/encoding.py#L130) — Decode a plaintext. — documented in [jaxite-jaxite_cggi-encoding](../../../concepts/jaxite-jaxite_cggi-encoding.md)
- `decode_without_removing_padding(plaintext: types.LwePlaintext, params: EncodingParameters)` — [`L111`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/encoding.py#L111) — Decode a plaintext without removing padding. — documented in [jaxite-jaxite_cggi-encoding](../../../concepts/jaxite-jaxite_cggi-encoding.md)
- `encode(message: Union[types.LweCleartext, jnp.ndarray], params: EncodingParameters, test_polynomial_encoding: bool = False)` — [`L66`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/encoding.py#L66) — Encode a plaintext or array of plaintexts for use in a TFHE ciphertext. — documented in [jaxite-jaxite_bool-bool_params](../../../concepts/jaxite-jaxite_bool-bool_params.md)
- `extract_noise(plaintext: types.LwePlaintext, encoding_params: EncodingParameters)` — [`L178`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/encoding.py#L178) — Extracts the noise bits of a plaintext as a (signed) int. — documented in [jaxite-jaxite_cggi-encoding](../../../concepts/jaxite-jaxite_cggi-encoding.md)
- `remove_noise(inp: jnp.ndarray, encoding_params: EncodingParameters)` — [`L156`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/encoding.py#L156) — Uses encoding_params to round the inp to remove noise. — documented in [jaxite-jaxite_cggi-encoding](../../../concepts/jaxite-jaxite_cggi-encoding.md)
- `round_to_power_of_2(arr: jnp.ndarray, log_pow_of_2: int)` — [`L163`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/encoding.py#L163) — Rounds to the nearest multiple of a given power of 2.


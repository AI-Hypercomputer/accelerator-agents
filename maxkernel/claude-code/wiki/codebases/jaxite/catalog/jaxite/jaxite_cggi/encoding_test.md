---
title: 'Module: jaxite/jaxite_cggi/encoding_test.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_cggi/encoding_test.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_cggi.encoding_test`/Encoding
symbols:
  EncodingDecodingTest.test_encode_decode_message_bounds_succeeds: DecodingTest#test_encode_decode_message_bounds_succeeds().
  EncodingDecodingTest.test_encode_decode_succeeds: DecodingTest#test_encode_decode_succeeds().
  EncodingDecodingTest.test_encode_add_noise_decode_succeeds: DecodingTest#test_encode_add_noise_decode_succeeds().
  EncodingDecodingTest.test_encode_add_noise_extract_noise: DecodingTest#test_encode_add_noise_extract_noise().
  EncodingDecodingTest.test_encode_invalid_message_raises: DecodingTest#test_encode_invalid_message_raises().
  EncodingDecodingTest.test_encode_test_polynomial_coefficients: DecodingTest#test_encode_test_polynomial_coefficients().
  EncodingDecodingTest.test_encode_test_polynomial_coefficients_too_large: DecodingTest#test_encode_test_polynomial_coefficients_too_large().
  EncodingParametersTest.test_invalid_total_bit_length_raises: ParametersTest#test_invalid_total_bit_length_raises().
  EncodingParametersTest.test_invalid_message_bit_length_raises: ParametersTest#test_invalid_message_bit_length_raises().
  EncodingParametersTest.test_invalid_padding_bit_length_raises: ParametersTest#test_invalid_padding_bit_length_raises().
  EncodingParametersTest.test_unavailable_space_raises: ParametersTest#test_unavailable_space_raises().
  EncodingParametersTest: ParametersTest#
  EncodingDecodingTest: DecodingTest#
---
# Module: [`jaxite/jaxite_cggi/encoding_test.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/encoding_test.py)

## Classes
### `EncodingDecodingTest`  ·  implements/extends TestCase
- def: [`jaxite/jaxite_cggi/encoding_test.py:40`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/encoding_test.py#L40)
- doc: Exercises encoding and decoding logic.
- signature: `class EncodingDecodingTest(parameterized.TestCase):`
- members:
  - `test_encode_add_noise_decode_succeeds(self, cleartext: types.LweCleartext, noise: int)` — [`L76`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/encoding_test.py#L76)
  - `test_encode_add_noise_extract_noise(self, cleartext: types.LweCleartext, noise: int)` — [`L105`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/encoding_test.py#L105)
  - `test_encode_decode_message_bounds_succeeds(self, cleartext: types.LweCleartext)` — [`L47`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/encoding_test.py#L47) — Tests encoding+decoding produces exactly the input cleartext.
  - `test_encode_decode_succeeds(self, cleartext: types.LweCleartext)` — [`L61`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/encoding_test.py#L61)
  - `test_encode_invalid_message_raises(self, cleartext: types.LweCleartext)` — [`L91`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/encoding_test.py#L91)
  - `test_encode_test_polynomial_coefficients(self)` — [`L116`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/encoding_test.py#L116)
  - `test_encode_test_polynomial_coefficients_too_large(self)` — [`L126`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/encoding_test.py#L126)
- uses (calls/refs, reference-scoped): [`message_bit_length`](encoding.md#EncodingParameters.message_bit_length), [`encode`](encoding.md#encode), [`EncodingParameters`](encoding.md#EncodingParameters), [`padding_bit_length`](encoding.md#EncodingParameters.padding_bit_length), [`total_bit_length`](encoding.md#EncodingParameters.total_bit_length), [`decode`](encoding.md#decode), [`LweCleartext`](types.md#LweCleartext), [`extract_noise`](encoding.md#extract_noise), [`LwePlaintext`](types.md#LwePlaintext)

### `EncodingParametersTest`  ·  implements/extends TestCase
- def: [`jaxite/jaxite_cggi/encoding_test.py:12`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/encoding_test.py#L12)
- doc: Tests encoding.EncodingParameters initialization.
- signature: `class EncodingParametersTest(absltest.TestCase):`
- members:
  - `test_invalid_message_bit_length_raises(self)` — [`L21`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/encoding_test.py#L21)
  - `test_invalid_padding_bit_length_raises(self)` — [`L27`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/encoding_test.py#L27)
  - `test_invalid_total_bit_length_raises(self)` — [`L15`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/encoding_test.py#L15)
  - `test_unavailable_space_raises(self)` — [`L33`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/encoding_test.py#L33)
- uses (calls/refs, reference-scoped): [`message_bit_length`](encoding.md#EncodingParameters.message_bit_length), [`EncodingParameters`](encoding.md#EncodingParameters), [`padding_bit_length`](encoding.md#EncodingParameters.padding_bit_length), [`total_bit_length`](encoding.md#EncodingParameters.total_bit_length)


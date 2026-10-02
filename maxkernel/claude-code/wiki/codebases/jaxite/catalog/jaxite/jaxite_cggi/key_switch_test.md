---
title: 'Module: jaxite/jaxite_cggi/key_switch_test.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_cggi/key_switch_test.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_cggi.key_switch_test`/
symbols:
  KeySwitchTest.test_key_switch_128_bit_security: KeySwitchTest#test_key_switch_128_bit_security().
  KeySwitchTest.test_switch_key_error_free: KeySwitchTest#test_switch_key_error_free().
  KeySwitchTest.test_switch_key_with_error: KeySwitchTest#test_switch_key_with_error().
  KeySwitchTest.test_key_switch_gen: KeySwitchTest#test_key_switch_gen().
  KeySwitchTest.in_params: KeySwitchTest#in_params.
  KeySwitchTest.out_params: KeySwitchTest#out_params.
  KeySwitchTest.decomposition_params: KeySwitchTest#decomposition_params.
  ZERO_RNG: ZERO_RNG.
  A_I_BOUNDS: A_I_BOUNDS.
  KeySwitchTest.polynomial_modulus_degree: KeySwitchTest#polynomial_modulus_degree.
  KeySwitchTest: KeySwitchTest#
  KeySwitchTest.setUp: KeySwitchTest#setUp().
  KeySwitchTest.dim: KeySwitchTest#dim.
  KeySwitchTest.plaintext_modulus: KeySwitchTest#plaintext_modulus.
  KeySwitchTest.rlwe_dimension: KeySwitchTest#rlwe_dimension.
---
# Module: [`jaxite/jaxite_cggi/key_switch_test.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/key_switch_test.py)

## Classes
### `KeySwitchTest`  ·  implements/extends TestCase
- def: [`jaxite/jaxite_cggi/key_switch_test.py:19`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/key_switch_test.py#L19)
- signature: `class KeySwitchTest(parameterized.TestCase):`
- members:
  - `setUp(self)` — [`L21`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/key_switch_test.py#L21)
  - `test_key_switch_128_bit_security(self, message: int)` — [`L105`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/key_switch_test.py#L105)
  - `test_key_switch_gen(self)` — [`L43`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/key_switch_test.py#L43)
  - `test_switch_key_error_free(self, ai_bound: int, message: int)` — [`L63`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/key_switch_test.py#L63)
  - `test_switch_key_with_error(self, ai_bound: int, message: int)` — [`L86`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/key_switch_test.py#L86)
  - `decomposition_params` — [`L25`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/key_switch_test.py#L25)
  - `dim` — [`L23`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/key_switch_test.py#L23)
  - `in_params` — [`L30`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/key_switch_test.py#L30)
  - `out_params` — [`L36`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/key_switch_test.py#L36)
  - `plaintext_modulus` — [`L24`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/key_switch_test.py#L24)
  - `polynomial_modulus_degree` — [`L28`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/key_switch_test.py#L28)
  - `rlwe_dimension` — [`L29`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/key_switch_test.py#L29)
- uses (calls/refs, reference-scoped): [`gen_key`](key_switch.md#gen_key), [`message_bit_length`](encoding.md#EncodingParameters.message_bit_length), [`encode`](encoding.md#encode), [`encrypt`](lwe.md#encrypt), [`EncodingParameters`](encoding.md#EncodingParameters), [`gen_key`](lwe.md#gen_key), [`padding_bit_length`](encoding.md#EncodingParameters.padding_bit_length), [`gen_key`](rlwe.md#gen_key), [`SchemeParameters`](parameters.md#SchemeParameters), [`total_bit_length`](encoding.md#EncodingParameters.total_bit_length), [`DecompositionParameters`](decomposition.md#DecompositionParameters), [`decode`](encoding.md#decode), [`polynomial_modulus_degree`](parameters.md#SchemeParameters.polynomial_modulus_degree), [`PseudorandomSource`](random_source.md#PseudorandomSource), [`decrypt`](lwe.md#decrypt), [`lwe_dimension`](parameters.md#SchemeParameters.lwe_dimension), [`flatten_key`](rlwe.md#flatten_key), [`level_count`](decomposition.md#DecompositionParameters.level_count), [`decrypt_without_denoising`](lwe.md#decrypt_without_denoising), [`log_base`](decomposition.md#DecompositionParameters.log_base), [`rlwe_dimension`](parameters.md#SchemeParameters.rlwe_dimension), [`plaintext_modulus`](parameters.md#SchemeParameters.plaintext_modulus), [`switch_key`](key_switch.md#switch_key), [`extract_noise`](encoding.md#extract_noise), [`lwe_dimension`](lwe.md#LweSecretKey.lwe_dimension), [`key_data`](key_switch.md#LweKeySwitchingKey.key_data), [`A_I_BOUNDS`](key_switch_test.md#A_I_BOUNDS)  (2 test-only)

## Module values
- `A_I_BOUNDS` — [`L16`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/key_switch_test.py#L16)
- `ZERO_RNG` — [`L15`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/key_switch_test.py#L15)


---
title: 'Module: jaxite/jaxite_ckks/key_switching_key_test.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_ckks/key_switching_key_test.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_ckks.key_switching_key_test`/
symbols:
  KeySwitchingTest.test_key_switch: KeySwitchingTest#test_key_switch().
  KeySwitchingTest.test_gen_key_switching_key: KeySwitchingTest#test_gen_key_switching_key().
  TEST_PRIMES: TEST_PRIMES.
  KeySwitchingTest: KeySwitchingTest#
---
# Module: [`jaxite/jaxite_ckks/key_switching_key_test.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/key_switching_key_test.py)

## Classes
### `KeySwitchingTest`  ·  implements/extends TestCase
- def: [`jaxite/jaxite_ckks/key_switching_key_test.py:34`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/key_switching_key_test.py#L34)
- signature: `class KeySwitchingTest(parameterized.TestCase):`
- members:
  - `test_gen_key_switching_key(self)` — [`L36`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/key_switching_key_test.py#L36)
  - `test_key_switch(self, q_limbs, p_limbs, dnum)` — [`L118`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/key_switching_key_test.py#L118)
- uses (calls/refs, reference-scoped): [`data`](types.md#Ciphertext.data), [`encrypt`](encrypt.md#Encrypt.encrypt), [`moduli`](types.md#Ciphertext.moduli), [`key_switch`](key_switching.md#KeySwitcher.key_switch), [`encode`](encode.md#Encode.encode), [`keygen`](key_gen.md#keygen), [`Ciphertext`](types.md#Ciphertext), [`decrypt`](encrypt.md#Decrypt.decrypt), [`precompute_constants`](basis_conversion.md#BasisConversionBarrett.precompute_constants), [`Encode`](encode.md#Encode), [`Encrypt`](encrypt.md#Encrypt), [`gen_key_switching_key`](key_gen.md#gen_key_switching_key), [`data`](types.md#Plaintext.data), [`Decrypt`](encrypt.md#Decrypt), [`precompute_barrett_constants`](barrett.md#precompute_barrett_constants), [`BasisConversionBarrett`](basis_conversion.md#BasisConversionBarrett), [`extend_secret_key`](key_gen.md#extend_secret_key), [`TEST_PRIMES`](key_switching_key_test.md#TEST_PRIMES), [`intt_negacyclic_poly`](ntt_cpu.md#intt_negacyclic_poly), [`precompute_constants`](key_switching.md#KeySwitcher.precompute_constants), [`MulPlaintextCiphertextBarrett`](mul.md#MulPlaintextCiphertextBarrett), [`ZeroNoiseRandomSource`](random.md#ZeroNoiseRandomSource), [`compute_scaled_source_key_partition`](key_gen.md#compute_scaled_source_key_partition), [`a`](types.md#EvaluationKeys.a), [`b`](types.md#EvaluationKeys.b), [`KeySwitcher`](key_switching.md#KeySwitcher)

## Module values
- `TEST_PRIMES` — [`L24`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/key_switching_key_test.py#L24)


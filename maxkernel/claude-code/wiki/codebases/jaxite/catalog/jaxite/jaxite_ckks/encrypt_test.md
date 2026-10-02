---
title: 'Module: jaxite/jaxite_ckks/encrypt_test.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_ckks/encrypt_test.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_ckks.encrypt_test`/EncryptTest#
symbols:
  EncryptTest.test_encrypt_decrypt_loop: test_encrypt_decrypt_loop().
  EncryptTest.test_encrypt_decrypt_loop_with_hamming_weight: test_encrypt_decrypt_loop_with_hamming_weight().
  EncryptTest.test_exact_encrypt_decrypt: test_exact_encrypt_decrypt().
  EncryptTest.test_decrypt_moduli_mismatch_raises_error: test_decrypt_moduli_mismatch_raises_error().
  EncryptTest.test_jax_compatibility: test_jax_compatibility().
  EncryptTest.test_encrypt_moduli_mismatch_raises_error: test_encrypt_moduli_mismatch_raises_error().
  EncryptTest.get_data: get_data().
  EncryptTest: ''
---
# Module: [`jaxite/jaxite_ckks/encrypt_test.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encrypt_test.py)

## Classes
### `EncryptTest`  ·  implements/extends TestCase
- def: [`jaxite/jaxite_ckks/encrypt_test.py:22`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encrypt_test.py#L22)
- signature: `class EncryptTest(absltest.TestCase):`
- members:
  - `get_data(c)` — [`L180`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encrypt_test.py#L180)
  - `test_decrypt_moduli_mismatch_raises_error(self)` — [`L143`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encrypt_test.py#L143)
  - `test_encrypt_decrypt_loop(self, slots, scale, seed)` — [`L34`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encrypt_test.py#L34)
  - `test_encrypt_decrypt_loop_with_hamming_weight(self, slots, scale, seed, hw)` — [`L68`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encrypt_test.py#L68)
  - `test_encrypt_moduli_mismatch_raises_error(self)` — [`L125`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encrypt_test.py#L125)
  - `test_exact_encrypt_decrypt(self, seed)` — [`L97`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encrypt_test.py#L97)
  - `test_jax_compatibility(self)` — [`L164`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encrypt_test.py#L164) — Ensure Ciphertext and keys are valid JAX types.
- uses (calls/refs, reference-scoped): [`data`](types.md#Ciphertext.data), [`encrypt`](encrypt.md#Encrypt.encrypt), [`encode`](encode.md#Encode.encode), [`keygen`](key_gen.md#keygen), [`Ciphertext`](types.md#Ciphertext), [`decrypt`](encrypt.md#Decrypt.decrypt), [`decode`](encode.md#Decode.decode), [`Encode`](encode.md#Encode), [`Encrypt`](encrypt.md#Encrypt), [`data`](types.md#Plaintext.data), [`Decrypt`](encrypt.md#Decrypt), [`ntt_negacyclic_poly`](ntt_cpu.md#ntt_negacyclic_poly), [`Decode`](encode.md#Decode), [`Plaintext`](types.md#Plaintext), [`TestRandomSource`](random.md#TestRandomSource), [`ZeroNoiseRandomSource`](random.md#ZeroNoiseRandomSource)


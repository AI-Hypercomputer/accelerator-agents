---
title: 'Module: jaxite/jaxite_ckks/mul_test.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_ckks/mul_test.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_ckks.mul_test`/
symbols:
  CiphertextCiphertextMulTest.test_encrypt_multiply_decrypt: CiphertextCiphertextMulTest#test_encrypt_multiply_decrypt().
  CiphertextCiphertextMulTest.test_relinearize_batched: CiphertextCiphertextMulTest#test_relinearize_batched().
  PlaintextCiphertextMulTest.test_full_pipeline_mul: PlaintextCiphertextMulTest#test_full_pipeline_mul().
  PlaintextCiphertextMulTest.test_mul_simple: PlaintextCiphertextMulTest#test_mul_simple().
  PlaintextCiphertextMulTest.test_mul_modular_barrett_hypothesis: PlaintextCiphertextMulTest#test_mul_modular_barrett_hypothesis().
  CiphertextCiphertextMulTest.test_mul_pytree: CiphertextCiphertextMulTest#test_mul_pytree().
  _get_kernel: _get_kernel().
  SLOTS_STRATEGY: SLOTS_STRATEGY.
  NUM_SLOTS: NUM_SLOTS.
  PlaintextCiphertextMulTest: PlaintextCiphertextMulTest#
  CiphertextCiphertextMulTest: CiphertextCiphertextMulTest#
---
# Module: [`jaxite/jaxite_ckks/mul_test.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul_test.py)

## Classes
### `CiphertextCiphertextMulTest`  ·  implements/extends TestCase
- def: [`jaxite/jaxite_ckks/mul_test.py:163`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul_test.py#L163)
- signature: `class CiphertextCiphertextMulTest(absltest.TestCase):`
- members:
  - `test_encrypt_multiply_decrypt(self)` — [`L165`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul_test.py#L165)
  - `test_mul_pytree(self)` — [`L314`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul_test.py#L314)
  - `test_relinearize_batched(self)` — [`L243`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul_test.py#L243)
- uses (calls/refs, reference-scoped): [`data`](types.md#Ciphertext.data), [`relinearize`](mul.md#Mul.relinearize), [`encrypt`](encrypt.md#Encrypt.encrypt), [`precompute_constants`](ntt.md#NTTBarrett.precompute_constants), [`moduli`](types.md#Ciphertext.moduli), [`encode`](encode.md#Encode.encode), [`keygen`](key_gen.md#keygen), [`Ciphertext`](types.md#Ciphertext), [`rescale`](rescale.md#Rescale.rescale), [`precompute_constants`](rescale.md#Rescale.precompute_constants), [`decrypt`](encrypt.md#Decrypt.decrypt), [`decode`](encode.md#Decode.decode), [`NTTBarrett`](ntt.md#NTTBarrett), [`precompute_constants`](basis_conversion.md#BasisConversionBarrett.precompute_constants), [`Encode`](encode.md#Encode), [`Encrypt`](encrypt.md#Encrypt), [`data`](types.md#Plaintext.data), [`Decrypt`](encrypt.md#Decrypt), [`Decode`](encode.md#Decode), [`data`](types.md#SecretKey.data), [`BasisConversionBarrett`](basis_conversion.md#BasisConversionBarrett), [`gen_evaluation_key`](key_gen.md#gen_evaluation_key), [`Rescale`](rescale.md#Rescale), [`SecretKey`](types.md#SecretKey), [`tensor_multiply`](mul.md#Mul.tensor_multiply), [`moduli`](types.md#SecretKey.moduli), [`extend_moduli`](mul.md#Mul.extend_moduli), [`Mul`](mul.md#Mul), [`original_moduli`](mul.md#Mul.original_moduli), [`precompute_constants`](mul.md#Mul.precompute_constants), [`compute_control_indices`](mul.md#Mul.compute_control_indices)

### `PlaintextCiphertextMulTest`  ·  implements/extends TestCase
- def: [`jaxite/jaxite_ckks/mul_test.py:49`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul_test.py#L49)
- signature: `class PlaintextCiphertextMulTest(parameterized.TestCase):`
- members:
  - `test_full_pipeline_mul(self, slots1, slots2)` — [`L125`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul_test.py#L125)
  - `test_mul_modular_barrett_hypothesis(self, ct_list, pt_list)` — [`L92`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul_test.py#L92)
  - `test_mul_simple(self, kernel_name)` — [`L55`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul_test.py#L55)
- uses (calls/refs, reference-scoped): [`data`](types.md#Ciphertext.data), [`encrypt`](encrypt.md#Encrypt.encrypt), [`moduli`](types.md#Ciphertext.moduli), [`encode`](encode.md#Encode.encode), [`keygen`](key_gen.md#keygen), [`Ciphertext`](types.md#Ciphertext), [`decrypt`](encrypt.md#Decrypt.decrypt), [`decode`](encode.md#Decode.decode), [`Encode`](encode.md#Encode), [`Encrypt`](encrypt.md#Encrypt), [`data`](types.md#Plaintext.data), [`Decrypt`](encrypt.md#Decrypt), [`precompute_barrett_constants`](barrett.md#precompute_barrett_constants), [`Decode`](encode.md#Decode), [`mul`](mul.md#MulPlaintextCiphertextBarrett.mul), [`moduli`](types.md#Plaintext.moduli), [`mul`](mul.md#MulPlaintextCiphertextSimple.mul), [`Plaintext`](types.md#Plaintext), [`TestRandomSource`](random.md#TestRandomSource), [`MulPlaintextCiphertextBarrett`](mul.md#MulPlaintextCiphertextBarrett), [`_get_kernel`](mul_test.md#_get_kernel), [`SLOTS_STRATEGY`](mul_test.md#SLOTS_STRATEGY)

## Functions
- `_get_kernel(kernel_name, moduli=None)` — [`L37`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul_test.py#L37)

## Module values
- `NUM_SLOTS` — [`L24`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul_test.py#L24)
- `SLOTS_STRATEGY` — [`L25`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul_test.py#L25)


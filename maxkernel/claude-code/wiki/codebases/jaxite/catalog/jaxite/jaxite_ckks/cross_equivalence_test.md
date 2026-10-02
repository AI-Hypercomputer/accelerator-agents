---
title: 'Module: jaxite/jaxite_ckks/cross_equivalence_test.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_ckks/cross_equivalence_test.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_ckks.cross_equivalence_test`/CrossEquivalenceTest#
symbols:
  CrossEquivalenceTest.test_encrypt_equivalence: test_encrypt_equivalence().
  CrossEquivalenceTest.test_mul_equivalence: test_mul_equivalence().
  CrossEquivalenceTest.test_decrypt_equivalence: test_decrypt_equivalence().
  CrossEquivalenceTest.test_composition_equivalence: test_composition_equivalence().
  CrossEquivalenceTest.test_ntt_equivalence: test_ntt_equivalence().
  CrossEquivalenceTest.test_encode_equivalence: test_encode_equivalence().
  CrossEquivalenceTest._test_data: _test_data.
  CrossEquivalenceTest.test_encrypt_equivalence.MockRandomSource: test_encrypt_equivalence().MockRandomSource#
  CrossEquivalenceTest.test_composition_equivalence.MockRandomSource: test_composition_equivalence().MockRandomSource#
  CrossEquivalenceTest: ''
  CrossEquivalenceTest.setUpClass: setUpClass().
  CrossEquivalenceTest.test_encrypt_equivalence.MockRandomSource.gen_ternary_poly: test_encrypt_equivalence().MockRandomSource#gen_ternary_poly().
  CrossEquivalenceTest.test_encrypt_equivalence.MockRandomSource.gen_gaussian_poly: test_encrypt_equivalence().MockRandomSource#gen_gaussian_poly().
  CrossEquivalenceTest.test_encrypt_equivalence.MockRandomSource.gen_uniform_poly: test_encrypt_equivalence().MockRandomSource#gen_uniform_poly().
  CrossEquivalenceTest.test_encrypt_equivalence.MockRandomSource.gen_sparse_binary: test_encrypt_equivalence().MockRandomSource#gen_sparse_binary().
  CrossEquivalenceTest.test_composition_equivalence.MockRandomSource.gen_ternary_poly: test_composition_equivalence().MockRandomSource#gen_ternary_poly().
  CrossEquivalenceTest.test_composition_equivalence.MockRandomSource.gen_gaussian_poly: test_composition_equivalence().MockRandomSource#gen_gaussian_poly().
  CrossEquivalenceTest.test_composition_equivalence.MockRandomSource.gen_uniform_poly: test_composition_equivalence().MockRandomSource#gen_uniform_poly().
  CrossEquivalenceTest.test_composition_equivalence.MockRandomSource.gen_sparse_binary: test_composition_equivalence().MockRandomSource#gen_sparse_binary().
---
# Module: [`jaxite/jaxite_ckks/cross_equivalence_test.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/cross_equivalence_test.py)

## Classes
### `CrossEquivalenceTest`  ·  implements/extends TestCase
- def: [`jaxite/jaxite_ckks/cross_equivalence_test.py:24`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/cross_equivalence_test.py#L24)
- signature: `class CrossEquivalenceTest(absltest.TestCase):`
- members:
  - `setUpClass(cls)` — [`L27`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/cross_equivalence_test.py#L27)
  - `test_composition_equivalence(self)` — [`L115`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/cross_equivalence_test.py#L115)
  - `test_decrypt_equivalence(self)` — [`L93`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/cross_equivalence_test.py#L93)
  - `test_encode_equivalence(self)` — [`L35`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/cross_equivalence_test.py#L35)
  - `test_encrypt_equivalence(self)` — [`L50`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/cross_equivalence_test.py#L50)
  - `test_mul_equivalence(self)` — [`L196`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/cross_equivalence_test.py#L196)
  - `test_ntt_equivalence(self)` — [`L155`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/cross_equivalence_test.py#L155)
- protocol/private: `_test_data`[`L33`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/cross_equivalence_test.py#L33)
- uses (calls/refs, reference-scoped): [`data`](types.md#Ciphertext.data), [`relinearize`](mul.md#Mul.relinearize), [`encrypt`](encrypt.md#Encrypt.encrypt), [`precompute_constants`](ntt.md#NTTBarrett.precompute_constants), [`moduli`](types.md#Ciphertext.moduli), [`encode`](encode.md#Encode.encode), [`Ciphertext`](types.md#Ciphertext), [`rescale`](rescale.md#Rescale.rescale), [`precompute_constants`](rescale.md#Rescale.precompute_constants), [`decrypt`](encrypt.md#Decrypt.decrypt), [`NTTBarrett`](ntt.md#NTTBarrett), [`Encode`](encode.md#Encode), [`Encrypt`](encrypt.md#Encrypt), [`data`](types.md#Plaintext.data), [`ntt`](ntt.md#NTTBarrett.ntt), [`Decrypt`](encrypt.md#Decrypt), [`intt`](ntt.md#NTTBarrett.intt), [`data`](types.md#SecretKey.data), [`moduli`](types.md#Plaintext.moduli), [`Plaintext`](types.md#Plaintext), [`intt_negacyclic_poly`](ntt_cpu.md#intt_negacyclic_poly), [`Rescale`](rescale.md#Rescale), [`SecretKey`](types.md#SecretKey), [`tensor_multiply`](mul.md#Mul.tensor_multiply), [`moduli`](types.md#SecretKey.moduli), [`Mul`](mul.md#Mul), [`data`](types.md#PublicKey.data), [`PublicKey`](types.md#PublicKey), [`moduli`](types.md#PublicKey.moduli), [`precompute_constants`](mul.md#Mul.precompute_constants), [`MockRandomSource`](cross_equivalence_test.md#CrossEquivalenceTest.test_composition_equivalence.MockRandomSource), [`MockRandomSource`](cross_equivalence_test.md#CrossEquivalenceTest.test_encrypt_equivalence.MockRandomSource), [`EvaluationKeys`](mul.md#EvaluationKeys)

### `MockRandomSource`  ·  implements/extends RandomSource
- def: [`jaxite/jaxite_ckks/cross_equivalence_test.py:127`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/cross_equivalence_test.py#L127)
- signature: `class MockRandomSource(random.RandomSource):`
- members:
  - `gen_gaussian_poly(self, d, m, sigma=3.19)` — [`L71`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/cross_equivalence_test.py#L71)
  - `gen_gaussian_poly(self, d, m, sigma=3.19)` — [`L132`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/cross_equivalence_test.py#L132)
  - `gen_sparse_binary(self, d, w, m)` — [`L77`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/cross_equivalence_test.py#L77)
  - `gen_sparse_binary(self, d, w, m)` — [`L138`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/cross_equivalence_test.py#L138)
  - `gen_ternary_poly(self, d, m)` — [`L68`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/cross_equivalence_test.py#L68)
  - `gen_ternary_poly(self, d, m)` — [`L129`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/cross_equivalence_test.py#L129)
  - `gen_uniform_poly(self, d, m)` — [`L74`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/cross_equivalence_test.py#L74)
  - `gen_uniform_poly(self, d, m)` — [`L135`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/cross_equivalence_test.py#L135)
- uses (calls/refs, reference-scoped): [`RandomSource`](random.md#RandomSource)
- used by: [`test_composition_equivalence`](cross_equivalence_test.md#CrossEquivalenceTest.test_composition_equivalence), [`RandomSource`](random.md#RandomSource), [`gen_gaussian_poly`](random.md#RandomSource.gen_gaussian_poly), [`gen_uniform_poly`](random.md#RandomSource.gen_uniform_poly), [`gen_ternary_poly`](random.md#RandomSource.gen_ternary_poly), [`gen_sparse_binary`](random.md#RandomSource.gen_sparse_binary)


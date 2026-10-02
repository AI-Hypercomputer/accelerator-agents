---
title: 'Module: jaxite/jaxite_ckks/random.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_ckks/random.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_ckks.random`/
symbols:
  RandomSource: RandomSource#
  SecureRandomSource: SecureRandomSource#
  TestRandomSource: TestRandomSource#
  RandomSource.gen_gaussian_poly: RandomSource#gen_gaussian_poly().
  RandomSource.gen_uniform_poly: RandomSource#gen_uniform_poly().
  ZeroNoiseRandomSource: ZeroNoiseRandomSource#
  RandomSource.gen_ternary_poly: RandomSource#gen_ternary_poly().
  RandomSource.gen_sparse_binary: RandomSource#gen_sparse_binary().
  SecureRandomSource.gen_gaussian_poly: SecureRandomSource#gen_gaussian_poly().
  SecureRandomSource.rng: SecureRandomSource#rng.
  SecureRandomSource.gen_ternary_poly: SecureRandomSource#gen_ternary_poly().
  SecureRandomSource.gen_uniform_poly: SecureRandomSource#gen_uniform_poly().
  SecureRandomSource.gen_sparse_binary: SecureRandomSource#gen_sparse_binary().
  SecureRandomSource.__init__: SecureRandomSource#__init__().
  ZeroNoiseRandomSource.gen_gaussian_poly: ZeroNoiseRandomSource#gen_gaussian_poly().
  TestRandomSource.__init__: TestRandomSource#__init__().
  TestRandomSource.rng: TestRandomSource#rng.
---
# Module: [`jaxite/jaxite_ckks/random.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/random.py)

## Classes
### `RandomSource`  ·  implements/extends ABC
- def: [`jaxite/jaxite_ckks/random.py:8`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/random.py#L8) — documented in [jaxite-jaxite_ckks-encrypt](../../../concepts/jaxite-jaxite_ckks-encrypt.md)
- doc: An interface for random number generation in CKKS.
- signature: `class RandomSource(abc.ABC):`
- members:
  - `gen_gaussian_poly(self, degree: int, moduli: list[int], sigma: float = 3.19)` — [`L12`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/random.py#L12) — documented in [jaxite-jaxite_ckks-encrypt](../../../concepts/jaxite-jaxite_ckks-encrypt.md)
  - `gen_sparse_binary(self, degree: int, weight: int, moduli: list[int])` — [`L26`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/random.py#L26)
  - `gen_ternary_poly(self, degree: int, moduli: list[int])` — [`L18`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/random.py#L18) — documented in [jaxite-jaxite_ckks-encrypt](../../../concepts/jaxite-jaxite_ckks-encrypt.md)
  - `gen_uniform_poly(self, degree: int, moduli: list[int])` — [`L22`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/random.py#L22)
- uses (calls/refs, reference-scoped): [`SecureRandomSource`](random.md#SecureRandomSource), [`gen_gaussian_poly`](random.md#SecureRandomSource.gen_gaussian_poly), [`MockRandomSource`](cross_equivalence_test.md#CrossEquivalenceTest.test_composition_equivalence.MockRandomSource), [`MockRandomSource`](cross_equivalence_test.md#CrossEquivalenceTest.test_encrypt_equivalence.MockRandomSource), [`gen_sparse_binary`](random.md#SecureRandomSource.gen_sparse_binary), [`gen_ternary_poly`](random.md#SecureRandomSource.gen_ternary_poly), [`gen_uniform_poly`](random.md#SecureRandomSource.gen_uniform_poly), [`gen_gaussian_poly`](cross_equivalence_test.md#CrossEquivalenceTest.test_composition_equivalence.MockRandomSource.gen_gaussian_poly), [`gen_gaussian_poly`](cross_equivalence_test.md#CrossEquivalenceTest.test_encrypt_equivalence.MockRandomSource.gen_gaussian_poly), [`gen_sparse_binary`](cross_equivalence_test.md#CrossEquivalenceTest.test_composition_equivalence.MockRandomSource.gen_sparse_binary), [`gen_sparse_binary`](cross_equivalence_test.md#CrossEquivalenceTest.test_encrypt_equivalence.MockRandomSource.gen_sparse_binary), [`gen_ternary_poly`](cross_equivalence_test.md#CrossEquivalenceTest.test_composition_equivalence.MockRandomSource.gen_ternary_poly), [`gen_ternary_poly`](cross_equivalence_test.md#CrossEquivalenceTest.test_encrypt_equivalence.MockRandomSource.gen_ternary_poly), [`gen_uniform_poly`](cross_equivalence_test.md#CrossEquivalenceTest.test_composition_equivalence.MockRandomSource.gen_uniform_poly), [`gen_uniform_poly`](cross_equivalence_test.md#CrossEquivalenceTest.test_encrypt_equivalence.MockRandomSource.gen_uniform_poly)
- used by: [`encrypt`](encrypt.md#Encrypt.encrypt), [`keygen`](key_gen.md#keygen), [`gen_hmuxrot_key`](key_gen.md#gen_hmuxrot_key), [`gen_key_switching_key`](key_gen.md#gen_key_switching_key), [`gen_conjugate_key`](key_gen.md#gen_conjugate_key), [`gen_evaluation_key`](key_gen.md#gen_evaluation_key), [`gen_mux_rotation_key`](key_gen.md#gen_mux_rotation_key), [`SecureRandomSource`](random.md#SecureRandomSource), [`encrypt`](encrypt.md#EncryptBase.encrypt), [`MockRandomSource`](cross_equivalence_test.md#CrossEquivalenceTest.test_composition_equivalence.MockRandomSource), [`MockRandomSource`](cross_equivalence_test.md#CrossEquivalenceTest.test_encrypt_equivalence.MockRandomSource)

### `SecureRandomSource`  ·  implements/extends RandomSource
- def: [`jaxite/jaxite_ckks/random.py:32`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/random.py#L32)
- doc: Random generation utilities for CKKS.
- signature: `class SecureRandomSource(RandomSource):`
- members:
  - `gen_gaussian_poly(self, degree: int, moduli: list[int], sigma: float = 3.19)` — [`L38`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/random.py#L38)
  - `gen_sparse_binary(self, degree: int, weight: int, moduli: list[int])` — [`L60`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/random.py#L60)
  - `gen_ternary_poly(self, degree: int, moduli: list[int])` — [`L47`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/random.py#L47)
  - `gen_uniform_poly(self, degree: int, moduli: list[int])` — [`L54`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/random.py#L54)
  - `rng` — [`L36`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/random.py#L36)
- protocol/private: `__init__`[`L35`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/random.py#L35)
- uses (calls/refs, reference-scoped): [`RandomSource`](random.md#RandomSource), [`TestRandomSource`](random.md#TestRandomSource), [`ZeroNoiseRandomSource`](random.md#ZeroNoiseRandomSource), [`gen_gaussian_poly`](random.md#ZeroNoiseRandomSource.gen_gaussian_poly)
- used by: [`encrypt`](encrypt.md#Encrypt.encrypt), [`keygen`](key_gen.md#keygen), [`gen_hmuxrot_key`](key_gen.md#gen_hmuxrot_key), [`gen_key_switching_key`](key_gen.md#gen_key_switching_key), [`RandomSource`](random.md#RandomSource), [`TestRandomSource`](random.md#TestRandomSource), [`gen_gaussian_poly`](random.md#RandomSource.gen_gaussian_poly), [`ZeroNoiseRandomSource`](random.md#ZeroNoiseRandomSource), [`gen_uniform_poly`](random.md#RandomSource.gen_uniform_poly), [`gen_ternary_poly`](random.md#RandomSource.gen_ternary_poly), [`gen_sparse_binary`](random.md#RandomSource.gen_sparse_binary)

### `TestRandomSource`  ·  implements/extends SecureRandomSource
- def: [`jaxite/jaxite_ckks/random.py:83`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/random.py#L83) — documented in [jaxite-jaxite_ckks-encrypt](../../../concepts/jaxite-jaxite_ckks-encrypt.md)
- doc: A random source that can be seeded, for testing.
- signature: `class TestRandomSource(SecureRandomSource):`
- members:
  - `rng` — [`L88`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/random.py#L88)
- protocol/private: `__init__`[`L86`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/random.py#L86)
- uses (calls/refs, reference-scoped): [`SecureRandomSource`](random.md#SecureRandomSource)
- used by: [`_run_conjugate_test`](conjugate_test.md#ConjugateTest._run_conjugate_test), [`test_rescale_different_scale`](rescale_test.md#RescaleTest.test_rescale_different_scale), [`test_rescale_e2e`](rescale_test.md#RescaleTest.test_rescale_e2e), [`test_double_rescale`](rescale_test.md#RescaleTest.test_double_rescale), [`test_full_pipeline_mul`](mul_test.md#PlaintextCiphertextMulTest.test_full_pipeline_mul), [`test_encrypt_decrypt_3_moduli`](rescale_test.md#RescaleTest.test_encrypt_decrypt_3_moduli), [`test_rescale_different_degree`](rescale_test.md#RescaleTest.test_rescale_different_degree), [`test_rescale_more_moduli`](rescale_test.md#RescaleTest.test_rescale_more_moduli), [`test_encrypt_decrypt_loop`](encrypt_test.md#EncryptTest.test_encrypt_decrypt_loop), [`test_encrypt_decrypt_loop_with_hamming_weight`](encrypt_test.md#EncryptTest.test_encrypt_decrypt_loop_with_hamming_weight), [`SecureRandomSource`](random.md#SecureRandomSource), [`random_source`](rescale_test.md#RescaleHypothesisTest.random_source)

### `ZeroNoiseRandomSource`  ·  implements/extends SecureRandomSource
- def: [`jaxite/jaxite_ckks/random.py:69`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/random.py#L69) — documented in [jaxite-jaxite_ckks-encrypt](../../../concepts/jaxite-jaxite_ckks-encrypt.md)
- doc: A random source that zeros out the gaussian noise.
- signature: `class ZeroNoiseRandomSource(SecureRandomSource):`
- members:
  - `gen_gaussian_poly(self, degree: int, moduli: list[int], sigma: float = 3.19)` — [`L77`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/random.py#L77)
- uses (calls/refs, reference-scoped): [`SecureRandomSource`](random.md#SecureRandomSource)
- used by: [`_run_conjugate_test`](conjugate_test.md#ConjugateTest._run_conjugate_test), [`test_conjugate_of_conjugate`](conjugate_test.md#ConjugateTest.test_conjugate_of_conjugate), [`test_key_switch`](key_switching_key_test.md#KeySwitchingTest.test_key_switch), [`test_blind_rotate_cm`](blind_rotate_test.md#BlindRotateTest.test_blind_rotate_cm), [`test_brot_mux`](blind_rotate_test.md#BlindRotateTest.test_brot_mux), [`test_brot_mux_cyclic_identity`](blind_rotate_test.md#BlindRotateTest.test_brot_mux_cyclic_identity), [`test_exact_encrypt_decrypt`](encrypt_test.md#EncryptTest.test_exact_encrypt_decrypt), [`SecureRandomSource`](random.md#SecureRandomSource), [`test_random_source`](blind_rotate_test.md#BlindRotationHypothesisTest.test_random_source), [`gen_gaussian_poly`](random.md#SecureRandomSource.gen_gaussian_poly)


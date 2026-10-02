---
title: 'Module: jaxite/jaxite_ckks/conjugate_test.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_ckks/conjugate_test.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_ckks.conjugate_test`/ConjugateTest#
symbols:
  ConjugateTest._run_conjugate_test: _run_conjugate_test().
  ConjugateTest.test_conjugate_of_conjugate: test_conjugate_of_conjugate().
  ConjugateTest.test_conjugate_pipeline: test_conjugate_pipeline().
  ConjugateTest.test_conjugate_hypothesis: test_conjugate_hypothesis().
  ConjugateTest.test_conjugation_ntt_preservation: test_conjugation_ntt_preservation().
  ConjugateTest: ''
---
# Module: [`jaxite/jaxite_ckks/conjugate_test.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/conjugate_test.py)

## Classes
### `ConjugateTest`  ·  implements/extends TestCase
- def: [`jaxite/jaxite_ckks/conjugate_test.py:24`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/conjugate_test.py#L24)
- signature: `class ConjugateTest(parameterized.TestCase):`
- members:
  - `test_conjugate_hypothesis(self, slots)` — [`L175`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/conjugate_test.py#L175)
  - `test_conjugate_of_conjugate(self)` — [`L187`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/conjugate_test.py#L187)
  - `test_conjugate_pipeline(self, q_limbs, p_limbs, dnum, use_noise)` — [`L157`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/conjugate_test.py#L157)
  - `test_conjugation_ntt_preservation(self)` — [`L265`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/conjugate_test.py#L265)
- protocol/private: `_run_conjugate_test`[`L26`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/conjugate_test.py#L26)
- uses (calls/refs, reference-scoped): [`encrypt`](encrypt.md#Encrypt.encrypt), [`encode`](encode.md#Encode.encode), [`keygen`](key_gen.md#keygen), [`precompute_constants`](rescale.md#Rescale.precompute_constants), [`decrypt`](encrypt.md#Decrypt.decrypt), [`decode`](encode.md#Decode.decode), [`precompute_constants`](basis_conversion.md#BasisConversionBarrett.precompute_constants), [`Encode`](encode.md#Encode), [`Encrypt`](encrypt.md#Encrypt), [`conjugate`](conjugate.md#Conjugation.conjugate), [`Decrypt`](encrypt.md#Decrypt), [`ntt_negacyclic_poly`](ntt_cpu.md#ntt_negacyclic_poly), [`precompute_barrett_constants`](barrett.md#precompute_barrett_constants), [`Decode`](encode.md#Decode), [`BasisConversionBarrett`](basis_conversion.md#BasisConversionBarrett), [`gen_conjugate_key`](key_gen.md#gen_conjugate_key), [`TestRandomSource`](random.md#TestRandomSource), [`Rescale`](rescale.md#Rescale), [`MulPlaintextCiphertextBarrett`](mul.md#MulPlaintextCiphertextBarrett), [`ZeroNoiseRandomSource`](random.md#ZeroNoiseRandomSource), [`precompute_constants`](conjugate.md#Conjugation.precompute_constants), [`Conjugation`](conjugate.md#Conjugation)


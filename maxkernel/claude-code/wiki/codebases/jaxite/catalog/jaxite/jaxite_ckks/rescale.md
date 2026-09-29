---
title: 'Module: jaxite/jaxite_ckks/rescale.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_ckks/rescale.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_ckks.rescale`/
symbols:
  Rescale.rescale: Rescale#rescale().
  Rescale.precompute_constants: Rescale#precompute_constants().
  Rescale.tree_flatten: Rescale#tree_flatten().
  Rescale.tree_unflatten: Rescale#tree_unflatten().
  Rescale: Rescale#
  Rescale.moduli: Rescale#moduli.
  Rescale.r: Rescale#r.
  Rescale.c: Rescale#c.
  Rescale.thresholds: Rescale#thresholds.
  Rescale.ntt_last_limb_kernels: Rescale#ntt_last_limb_kernels.
  Rescale.ntt_remaining_limbs_kernels: Rescale#ntt_remaining_limbs_kernels.
  Rescale.num_rescales: Rescale#num_rescales.
  Rescale.gammas_stacked: Rescale#gammas_stacked.
  Rescale.betas_stacked: Rescale#betas_stacked.
  Ciphertext: Ciphertext.
  _gamma_beta_calculation: _gamma_beta_calculation().
  Rescale.__init__: Rescale#__init__().
---
# Module: [`jaxite/jaxite_ckks/rescale.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale.py)

## Classes
### `Rescale`
- def: [`jaxite/jaxite_ckks/rescale.py:59`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale.py#L59)
- doc: Kernel for in-place ciphertext rescaling.
- signature: `class Rescale:`
- members:
  - `precompute_constants(self, moduli: Iterable[int], num_rescales: int, r: int, c: int)` — [`L96`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale.py#L96) — Precompute constants for rescale computation. — documented in [jaxite-jaxite_ckks-rescale](../../../concepts/jaxite-jaxite_ckks-rescale.md)
  - `rescale(self, ciphertext: Ciphertext)` — [`L151`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale.py#L151) — Performs in-place rescaling on the ciphertext. — documented in [jaxite-jaxite_ckks-rescale](../../../concepts/jaxite-jaxite_ckks-rescale.md)
  - `tree_flatten(self)` — [`L227`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale.py#L227)
  - `tree_unflatten(cls, aux_data, children)` — [`L239`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale.py#L239)
  - `betas_stacked` — [`L91`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale.py#L91) — documented in [jaxite-jaxite_ckks-rescale](../../../concepts/jaxite-jaxite_ckks-rescale.md)
  - `c` — [`L89`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale.py#L89) — documented in [jaxite-jaxite_ckks-rescale](../../../concepts/jaxite-jaxite_ckks-rescale.md)
  - `gammas_stacked` — [`L90`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale.py#L90) — documented in [jaxite-jaxite_ckks-rescale](../../../concepts/jaxite-jaxite_ckks-rescale.md)
  - `moduli` — [`L86`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale.py#L86) — documented in [jaxite-jaxite_ckks-rescale](../../../concepts/jaxite-jaxite_ckks-rescale.md)
  - `ntt_last_limb_kernels` — [`L93`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale.py#L93) — documented in [jaxite-jaxite_ckks-rescale](../../../concepts/jaxite-jaxite_ckks-rescale.md)
  - `ntt_remaining_limbs_kernels` — [`L94`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale.py#L94) — documented in [jaxite-jaxite_ckks-rescale](../../../concepts/jaxite-jaxite_ckks-rescale.md)
  - `num_rescales` — [`L87`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale.py#L87) — documented in [jaxite-jaxite_ckks-rescale](../../../concepts/jaxite-jaxite_ckks-rescale.md)
  - `r` — [`L88`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale.py#L88) — documented in [jaxite-jaxite_ckks-rescale](../../../concepts/jaxite-jaxite_ckks-rescale.md)
  - `thresholds` — [`L92`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale.py#L92) — documented in [jaxite-jaxite_ckks-rescale](../../../concepts/jaxite-jaxite_ckks-rescale.md)
- protocol/private: `__init__`[`L85`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale.py#L85)
- uses (calls/refs, reference-scoped): [`data`](types.md#Ciphertext.data), [`precompute_constants`](ntt.md#NTTBarrett.precompute_constants), [`moduli`](types.md#Ciphertext.moduli), [`NTTBarrett`](ntt.md#NTTBarrett), [`slice_moduli`](ntt.md#NTTBarrettConstants.slice_moduli), [`constants`](ntt.md#NTTBarrett.constants), [`Ciphertext`](rescale.md#Ciphertext), [`_gamma_beta_calculation`](rescale.md#_gamma_beta_calculation)
- used by: [`test_encrypt_multiply_decrypt`](mul_test.md#CiphertextCiphertextMulTest.test_encrypt_multiply_decrypt), [`test_relinearize_batched`](mul_test.md#CiphertextCiphertextMulTest.test_relinearize_batched), [`_run_conjugate_test`](conjugate_test.md#ConjugateTest._run_conjugate_test), [`test_conjugate_of_conjugate`](conjugate_test.md#ConjugateTest.test_conjugate_of_conjugate), [`hmuxrot`](blind_rotate.md#BlindRotation.hmuxrot), [`rescale_kernel`](rescale_test.md#RescaleTest.rescale_kernel), [`test_rescale_different_scale`](rescale_test.md#RescaleTest.test_rescale_different_scale), [`test_rescale_e2e`](rescale_test.md#RescaleTest.test_rescale_e2e), [`test_double_rescale`](rescale_test.md#RescaleTest.test_double_rescale), [`precompute_constants`](blind_rotate.md#BlindRotation.precompute_constants), [`test_rescale_different_degree`](rescale_test.md#RescaleTest.test_rescale_different_degree), [`test_rescale_hypothesis`](rescale_test.md#RescaleHypothesisTest.test_rescale_hypothesis), [`test_rescale_more_moduli`](rescale_test.md#RescaleTest.test_rescale_more_moduli), [`brot_cm`](blind_rotate.md#BlindRotation.brot_cm), [`rescale_kernel`](rescale_test.md#RescaleHypothesisTest.rescale_kernel), [`conjugate`](conjugate.md#Conjugation.conjugate), [`test_mul_equivalence`](cross_equivalence_test.md#CrossEquivalenceTest.test_mul_equivalence), [`test_rescale_equivalence`](rescale_test.md#RescaleTest.test_rescale_equivalence), [`rescale_kernel`](blind_rotate.md#BlindRotation.rescale_kernel)

## Functions
- `_gamma_beta_calculation(moduli_list: list[int])` — [`L13`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale.py#L13) — Computes gamma and beta parameters for approximate modulus switching.

## Module values
- `Ciphertext` — [`L10`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale.py#L10) — documented in [jaxite-jaxite_ckks-rescale](../../../concepts/jaxite-jaxite_ckks-rescale.md)


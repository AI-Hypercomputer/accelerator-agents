---
title: 'Module: jaxite/jaxite_ckks/conjugate.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_ckks/conjugate.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_ckks.conjugate`/Conjugation#
symbols:
  Conjugation.conjugate: conjugate().
  Conjugation.precompute_constants: precompute_constants().
  Conjugation.key_switcher: key_switcher.
  Conjugation.__init__: __init__().
  Conjugation.tree_flatten: tree_flatten().
  Conjugation.tree_unflatten: tree_unflatten().
  Conjugation: ''
---
# Module: [`jaxite/jaxite_ckks/conjugate.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/conjugate.py)

## Classes
### `Conjugation`
- def: [`jaxite/jaxite_ckks/conjugate.py:27`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/conjugate.py#L27)
- doc: Kernel for homomorphic conjugation on TPU.
- signature: `class Conjugation:`
- members:
  - `conjugate(self, ct: types.Ciphertext, conj_key: types.EvaluationKeys, p_limbs: jax.Array, bc_kernel: basis_conversion.BasisConversionBarrett, mul_kernel: mul.MulPlaintextCiphertextBarrett, rescale_kernel: rescale.Rescale, start_control_index: int)` — [`L55`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/conjugate.py#L55) — Homomorphically conjugates a CKKS ciphertext. — documented in [jaxite-jaxite_ckks-types](../../../concepts/jaxite-jaxite_ckks-types.md)
  - `precompute_constants(self, q_limbs: list[int], p_limbs: list[int], dnum: int, r: int, c: int)` — [`L33`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/conjugate.py#L33)
  - `tree_flatten(self)` — [`L43`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/conjugate.py#L43)
  - `tree_unflatten(cls, aux_data, children)` — [`L49`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/conjugate.py#L49)
  - `key_switcher` — [`L31`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/conjugate.py#L31)
- protocol/private: `__init__`[`L30`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/conjugate.py#L30)
- uses (calls/refs, reference-scoped): [`data`](types.md#Ciphertext.data), [`moduli`](types.md#Ciphertext.moduli), [`key_switch`](key_switching.md#KeySwitcher.key_switch), [`Ciphertext`](types.md#Ciphertext), [`rescale`](rescale.md#Rescale.rescale), [`BasisConversionBarrett`](basis_conversion.md#BasisConversionBarrett), [`Rescale`](rescale.md#Rescale), [`precompute_constants`](key_switching.md#KeySwitcher.precompute_constants), [`MulPlaintextCiphertextBarrett`](mul.md#MulPlaintextCiphertextBarrett), [`EvaluationKeys`](types.md#EvaluationKeys), [`KeySwitcher`](key_switching.md#KeySwitcher)
- used by: [`_run_conjugate_test`](conjugate_test.md#ConjugateTest._run_conjugate_test), [`test_conjugate_of_conjugate`](conjugate_test.md#ConjugateTest.test_conjugate_of_conjugate)


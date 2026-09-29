---
title: 'Module: jaxite/jaxite_ckks/key_switching.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_ckks/key_switching.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_ckks.key_switching`/KeySwitcher#
symbols:
  KeySwitcher.key_switch: key_switch().
  KeySwitcher.precompute_constants: precompute_constants().
  KeySwitcher.ntt_kernels_q: ntt_kernels_q.
  KeySwitcher.ntt_kernels_out: ntt_kernels_out.
  KeySwitcher.tree_flatten: tree_flatten().
  KeySwitcher.tree_unflatten: tree_unflatten().
  KeySwitcher: ''
  KeySwitcher.__init__: __init__().
---
# Module: [`jaxite/jaxite_ckks/key_switching.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/key_switching.py)

## Classes
### `KeySwitcher`
- def: [`jaxite/jaxite_ckks/key_switching.py:27`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/key_switching.py#L27)
- doc: Kernel for homomorphic key switching on TPU.
- signature: `class KeySwitcher:`
- members:
  - `key_switch(self, ct: types.Ciphertext, ksk: types.EvaluationKeys, p_limbs: jax.Array, bc_kernel: basis_conversion.BasisConversionBarrett, mul_kernel: mul.MulPlaintextCiphertextBarrett, start_control_index: int)` — [`L81`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/key_switching.py#L81) — Switch ciphertext from source key to destination key modulo QP.
  - `precompute_constants(self, q_limbs: list[int], p_limbs: list[int], dnum: int, r: int, c: int)` — [`L34`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/key_switching.py#L34) — Precomputes NTT kernels for key switching modular partitions.
  - `tree_flatten(self)` — [`L68`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/key_switching.py#L68)
  - `tree_unflatten(cls, aux_data, children)` — [`L74`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/key_switching.py#L74)
  - `ntt_kernels_out` — [`L32`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/key_switching.py#L32)
  - `ntt_kernels_q` — [`L31`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/key_switching.py#L31)
- protocol/private: `__init__`[`L30`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/key_switching.py#L30)
- uses (calls/refs, reference-scoped): [`data`](types.md#Ciphertext.data), [`precompute_constants`](ntt.md#NTTBarrett.precompute_constants), [`moduli`](types.md#Ciphertext.moduli), [`Ciphertext`](types.md#Ciphertext), [`NTTBarrett`](ntt.md#NTTBarrett), [`constants`](ntt.md#NTTBarrett.constants), [`data`](types.md#Plaintext.data), [`ntt`](ntt.md#NTTBarrett.ntt), [`intt`](ntt.md#NTTBarrett.intt), [`mul`](mul.md#MulPlaintextCiphertextBarrett.mul), [`basis_change`](basis_conversion.md#BasisConversionBarrett.basis_change), [`moduli`](types.md#Plaintext.moduli), [`BasisConversionBarrett`](basis_conversion.md#BasisConversionBarrett), [`Plaintext`](types.md#Plaintext), [`MulPlaintextCiphertextBarrett`](mul.md#MulPlaintextCiphertextBarrett), [`EvaluationKeys`](types.md#EvaluationKeys), [`c`](ntt.md#NTTBarrettConstants.c), [`r`](ntt.md#NTTBarrettConstants.r), [`a`](types.md#EvaluationKeys.a), [`b`](types.md#EvaluationKeys.b)
- used by: [`test_key_switch`](key_switching_key_test.md#KeySwitchingTest.test_key_switch), [`hmuxrot`](blind_rotate.md#BlindRotation.hmuxrot), [`precompute_constants`](blind_rotate.md#BlindRotation.precompute_constants), [`conjugate`](conjugate.md#Conjugation.conjugate), [`key_switcher`](blind_rotate.md#BlindRotation.key_switcher), [`precompute_constants`](conjugate.md#Conjugation.precompute_constants), [`__init__`](conjugate.md#Conjugation.__init__)


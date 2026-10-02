---
title: 'Module: jaxite/jaxite_ckks/blind_rotate.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_ckks/blind_rotate.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_ckks.blind_rotate`/BlindRotation#
symbols:
  BlindRotation.hmuxrot: hmuxrot().
  BlindRotation.precompute_constants: precompute_constants().
  BlindRotation.brot_cm: brot_cm().
  BlindRotation.brot_mux: brot_mux().
  BlindRotation.tree_flatten: tree_flatten().
  BlindRotation.tree_unflatten: tree_unflatten().
  BlindRotation.mul_kernel: mul_kernel.
  BlindRotation.key_switcher: key_switcher.
  BlindRotation.bc_kernel: bc_kernel.
  BlindRotation.rescale_kernel: rescale_kernel.
  BlindRotation.ntt_q: ntt_q.
  BlindRotation.ntt_p: ntt_p.
  BlindRotation: ''
  BlindRotation.__init__: __init__().
---
# Module: [`jaxite/jaxite_ckks/blind_rotate.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate.py)

## Classes
### `BlindRotation`
- def: [`jaxite/jaxite_ckks/blind_rotate.py:31`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate.py#L31)
- doc: Kernel for homomorphic blind rotation on TPU.
- signature: `class BlindRotation:`
- members:
  - `brot_cm(self, cmkey_j: list[types.Ciphertext], pt_rot_mu_all: list[types.Plaintext])` — [`L234`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate.py#L234) — Homomorphic Blind Rotation using the Column Method (BRotCM).
  - `brot_mux(self, ct_in: types.Ciphertext, mux_key: types.MuxRotationKey, p_limbs: jax.Array, control_index: int = 0, stride: int = 1)` — [`L172`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate.py#L172) — Homomorphic Blind Rotation using the Mux Method (BRotMux).
  - `hmuxrot(self, ct: types.Ciphertext, hmrkey: types.HMuxRotKey, j: int, p_limbs: jax.Array, control_index: int = 0)` — [`L105`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate.py#L105) — Evaluates HMuxRot^(j)(hmrkey_beta, ct). — documented in [jaxite-jaxite_ckks-types](../../../concepts/jaxite-jaxite_ckks-types.md)
  - `precompute_constants(self, q_limbs: list[int], p_limbs: list[int], dnum: int, r: int, c: int, num_rescales: int = 1)` — [`L66`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate.py#L66) — Precomputes constants and sub-kernels for blind rotation.
  - `tree_flatten(self)` — [`L42`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate.py#L42)
  - `tree_unflatten(cls, aux_data, children)` — [`L55`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate.py#L55)
  - `bc_kernel` — [`L36`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate.py#L36)
  - `key_switcher` — [`L35`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate.py#L35)
  - `mul_kernel` — [`L37`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate.py#L37)
  - `ntt_p` — [`L40`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate.py#L40)
  - `ntt_q` — [`L39`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate.py#L39)
  - `rescale_kernel` — [`L38`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate.py#L38)
- protocol/private: `__init__`[`L34`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate.py#L34)
- uses (calls/refs, reference-scoped): [`data`](types.md#Ciphertext.data), [`precompute_constants`](ntt.md#NTTBarrett.precompute_constants), [`moduli`](types.md#Ciphertext.moduli), [`key_switch`](key_switching.md#KeySwitcher.key_switch), [`Ciphertext`](types.md#Ciphertext), [`rescale`](rescale.md#Rescale.rescale), [`precompute_constants`](rescale.md#Rescale.precompute_constants), [`NTTBarrett`](ntt.md#NTTBarrett), [`precompute_constants`](basis_conversion.md#BasisConversionBarrett.precompute_constants), [`modular_reduction`](barrett.md#modular_reduction), [`data`](types.md#Plaintext.data), [`precompute_barrett_constants`](barrett.md#precompute_barrett_constants), [`mul`](mul.md#MulPlaintextCiphertextBarrett.mul), [`moduli`](types.md#Plaintext.moduli), [`BasisConversionBarrett`](basis_conversion.md#BasisConversionBarrett), [`Plaintext`](types.md#Plaintext), [`Rescale`](rescale.md#Rescale), [`precompute_constants`](key_switching.md#KeySwitcher.precompute_constants), [`MulPlaintextCiphertextBarrett`](mul.md#MulPlaintextCiphertextBarrett), [`EvaluationKeys`](types.md#EvaluationKeys), [`apply_automorphism_ntt`](blind_rotate_utils.md#apply_automorphism_ntt), [`key0`](types.md#HMuxRotKey.key0), [`key1`](types.md#HMuxRotKey.key1), [`HMuxRotKey`](types.md#HMuxRotKey), [`a`](types.md#EvaluationKeys.a), [`b`](types.md#EvaluationKeys.b), [`keys`](types.md#MuxRotationKey.keys), [`KeySwitcher`](key_switching.md#KeySwitcher), [`barrett_constants`](mul.md#MulPlaintextCiphertextBarrett.barrett_constants), [`MuxRotationKey`](types.md#MuxRotationKey), [`moduli`](types.md#EvaluationKeys.moduli)
- used by: [`test_blind_rotate_cm`](blind_rotate_test.md#BlindRotateTest.test_blind_rotate_cm), [`test_brot_mux_hypothesis`](blind_rotate_test.md#BlindRotationHypothesisTest.test_brot_mux_hypothesis), [`test_brot_mux`](blind_rotate_test.md#BlindRotateTest.test_brot_mux), [`test_brot_mux_cyclic_identity`](blind_rotate_test.md#BlindRotateTest.test_brot_mux_cyclic_identity), [`test_brot_cm_hypothesis`](blind_rotate_test.md#BlindRotationHypothesisTest.test_brot_cm_hypothesis), [`brot_kernel`](blind_rotate_test.md#BlindRotationHypothesisTest.brot_kernel)


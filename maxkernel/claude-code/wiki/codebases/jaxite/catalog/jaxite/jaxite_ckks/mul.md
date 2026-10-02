---
title: 'Module: jaxite/jaxite_ckks/mul.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_ckks/mul.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_ckks.mul`/
symbols:
  Mul.relinearize: Mul#relinearize().
  Mul.tree_unflatten: Mul#tree_unflatten().
  Mul.tree_flatten: Mul#tree_flatten().
  Mul.ntt_current: Mul#ntt_current.
  Mul.ntt_extend: Mul#ntt_extend.
  Mul.ks_ntt_kernels: Mul#ks_ntt_kernels.
  Mul.bconv: Mul#bconv.
  Mul.full_ntt_constants: Mul#full_ntt_constants.
  MulPlaintextCiphertextBarrett.mul: MulPlaintextCiphertextBarrett#mul().
  Mul.drop_last_moduli: Mul#drop_last_moduli.
  Mul.drop_last_extend_moduli: Mul#drop_last_extend_moduli.
  MulPlaintextCiphertextSimple.mul: MulPlaintextCiphertextSimple#mul().
  Mul.full_barrett_constants: Mul#full_barrett_constants.
  Mul.tensor_multiply: Mul#tensor_multiply().
  MulPlaintextCiphertextBase.mul: MulPlaintextCiphertextBase#mul().
  MulPlaintextCiphertextBarrett: MulPlaintextCiphertextBarrett#
  Mul.extend_moduli: Mul#extend_moduli.
  Ciphertext: Ciphertext.
  MulPlaintextCiphertextBase: MulPlaintextCiphertextBase#
  Mul._injected_ntt_factory: Mul#_injected_ntt_factory.
  EvaluationKeys.tree_flatten: EvaluationKeys#tree_flatten().
  Mul.ks_num_parts_ql: Mul#ks_num_parts_ql.
  Mul: Mul#
  Mul.r: Mul#r.
  Mul.c: Mul#c.
  Mul.barrett_constants: Mul#barrett_constants.
  Mul.q_correction_uint32: Mul#q_correction_uint32.
  Mul.__init__: Mul#__init__().
  Mul.is_initialized: Mul#is_initialized.
  Mul.original_moduli: Mul#original_moduli.
  Mul.dnum: Mul#dnum.
  Mul.degree: Mul#degree.
  Mul.drop_last_extend_moduli_jax: Mul#drop_last_extend_moduli_jax.
  Mul.q_correction: Mul#q_correction.
  MulPlaintextCiphertextBarrett.barrett_constants: MulPlaintextCiphertextBarrett#barrett_constants.
  Mul.precompute_constants: Mul#precompute_constants().
  MulPlaintextCiphertextSimple: MulPlaintextCiphertextSimple#
  Mul._injected_full_ntt: Mul#_injected_full_ntt.
  Mul.compute_control_indices: Mul#compute_control_indices().
  Mul.composite_degree: Mul#composite_degree.
  Mul.ks_select_tower_index_overall: Mul#ks_select_tower_index_overall.
  Mul.ks_restore_indices: Mul#ks_restore_indices.
  Mul.PInvModq: Mul#PInvModq.
  MulPlaintextCiphertextBarrett.__init__: MulPlaintextCiphertextBarrett#__init__().
  MulPlaintextCiphertextBarrett.tree_flatten: MulPlaintextCiphertextBarrett#tree_flatten().
  EvaluationKeys: EvaluationKeys#
  EvaluationKeys.a: EvaluationKeys#a.
  EvaluationKeys.b: EvaluationKeys#b.
  Mul._injected_ntt_current: Mul#_injected_ntt_current.
  Mul._injected_ntt_extend: Mul#_injected_ntt_extend.
  Mul.ks_alpha: Mul#ks_alpha.
  Mul.ks_non_select_tower_index_overall: Mul#ks_non_select_tower_index_overall.
  ABC: ABC.
  abstractmethod: abstractmethod.
  EvaluationKeys.moduli: EvaluationKeys#moduli.
  Mul._injected_bconv: Mul#_injected_bconv.
  MulPlaintextCiphertextSimple.__init__: MulPlaintextCiphertextSimple#__init__().
  MulPlaintextCiphertextSimple.tree_flatten: MulPlaintextCiphertextSimple#tree_flatten().
  MulPlaintextCiphertextSimple.tree_unflatten: MulPlaintextCiphertextSimple#tree_unflatten().
  MulPlaintextCiphertextBarrett.tree_unflatten: MulPlaintextCiphertextBarrett#tree_unflatten().
  EvaluationKeys.tree_unflatten: EvaluationKeys#tree_unflatten().
---
# Module: [`jaxite/jaxite_ckks/mul.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py)

## Classes
### `EvaluationKeys`
- def: [`jaxite/jaxite_ckks/mul.py:101`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L101)
- doc: Evaluation keys for relinearization.
- signature: `class EvaluationKeys:`
- members:
  - `tree_flatten(self)` — [`L108`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L108) — Flatten EvaluationKey into its children and auxiliary data.
  - `tree_unflatten(cls, _, children)` — [`L113`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L113) — Reconstruct EvaluationKeys from auxiliary data and children.
  - `a` — [`L104`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L104)
  - `b` — [`L105`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L105)
  - `moduli` — [`L106`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L106)
- used by: [`relinearize`](mul.md#Mul.relinearize), [`test_mul_equivalence`](cross_equivalence_test.md#CrossEquivalenceTest.test_mul_equivalence)

### `Mul`
- def: [`jaxite/jaxite_ckks/mul.py:119`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L119) — documented in [jaxite-jaxite_ckks-mul](../../../concepts/jaxite-jaxite_ckks-mul.md)
- doc: Kernel for ciphertext multiplication and relinearization.
- signature: `class Mul:`
- members:
  - `compute_control_indices(drop_last_moduli: list[int], extend_moduli: list[int], dnum: int)` — [`L150`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L150) — Computes control indices for key switching.
  - `precompute_constants(self, original_moduli: Iterable[int], extend_moduli: Iterable[int], dnum: int, r: int, c: int, composite_degree: int = 1)` — [`L198`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L198) — Precomputes constants needed for multiplication and relinearization.
  - `relinearize(self, ct_3elem: Ciphertext, evk: EvaluationKeys)` — [`L380`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L380) — Performs relinearization on a 3-element ciphertext. — documented in [jaxite-jaxite_ckks-mul](../../../concepts/jaxite-jaxite_ckks-mul.md)
  - `tensor_multiply(self, ct1: Ciphertext, ct2: Ciphertext)` — [`L361`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L361) — Performs tensor multiplication of two ciphertexts. — documented in [jaxite-jaxite_ckks-types](../../../concepts/jaxite-jaxite_ckks-types.md)
  - `tree_flatten(self)` — [`L576`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L576) — Flattens the Mul instance into children and auxiliary data for JAX PyTree.
  - `tree_unflatten(cls, aux_data, children)` — [`L611`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L611) — Reconstructs a Mul instance from auxiliary data and children.
  - `PInvModq` — [`L345`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L345)
  - `barrett_constants` — [`L352`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L352)
  - `bconv` — [`L261`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L261) — documented in [jaxite-jaxite_ckks-mul](../../../concepts/jaxite-jaxite_ckks-mul.md)
  - `c` — [`L212`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L212) — documented in [jaxite-jaxite_ckks-mul](../../../concepts/jaxite-jaxite_ckks-mul.md)
  - `composite_degree` — [`L214`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L214)
  - `degree` — [`L213`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L213) — documented in [jaxite-jaxite_ckks-mul](../../../concepts/jaxite-jaxite_ckks-mul.md)
  - `dnum` — [`L210`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L210) — documented in [jaxite-jaxite_ckks-mul](../../../concepts/jaxite-jaxite_ckks-mul.md)
  - `drop_last_extend_moduli` — [`L225`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L225) — documented in [jaxite-jaxite_ckks-mul](../../../concepts/jaxite-jaxite_ckks-mul.md)
  - `drop_last_extend_moduli_jax` — [`L318`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L318)
  - `drop_last_moduli` — [`L216`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L216) — documented in [jaxite-jaxite_ckks-mul](../../../concepts/jaxite-jaxite_ckks-mul.md)
  - `extend_moduli` — [`L209`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L209) — documented in [jaxite-jaxite_ckks-mul](../../../concepts/jaxite-jaxite_ckks-mul.md)
  - `full_barrett_constants` — [`L355`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L355) — documented in [jaxite-jaxite_ckks-mul](../../../concepts/jaxite-jaxite_ckks-mul.md)
  - `full_ntt_constants` — [`L277`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L277) — documented in [jaxite-jaxite_ckks-mul](../../../concepts/jaxite-jaxite_ckks-mul.md)
  - `is_initialized` — [`L142`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L142) — documented in [jaxite-jaxite_ckks-mul](../../../concepts/jaxite-jaxite_ckks-mul.md)
  - `ks_alpha` — [`L234`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L234)
  - `ks_non_select_tower_index_overall` — [`L258`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L258)
  - `ks_ntt_kernels` — [`L322`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L322) — documented in [jaxite-jaxite_ckks-mul](../../../concepts/jaxite-jaxite_ckks-mul.md)
  - `ks_num_parts_ql` — [`L235`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L235) — documented in [jaxite-jaxite_ckks-mul](../../../concepts/jaxite-jaxite_ckks-mul.md)
  - `ks_restore_indices` — [`L259`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L259)
  - `ks_select_tower_index_overall` — [`L257`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L257)
  - `ntt_current` — [`L288`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L288) — documented in [jaxite-jaxite_ckks-mul](../../../concepts/jaxite-jaxite_ckks-mul.md)
  - `ntt_extend` — [`L302`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L302) — documented in [jaxite-jaxite_ckks-mul](../../../concepts/jaxite-jaxite_ckks-mul.md)
  - `original_moduli` — [`L208`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L208) — documented in [jaxite-jaxite_ckks-mul](../../../concepts/jaxite-jaxite_ckks-mul.md)
  - `q_correction` — [`L347`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L347)
  - `q_correction_uint32` — [`L348`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L348)
  - `r` — [`L211`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L211) — documented in [jaxite-jaxite_ckks-mul](../../../concepts/jaxite-jaxite_ckks-mul.md)
- protocol/private: `__init__`[`L134`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L134), `_injected_bconv`[`L143`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L143), `_injected_full_ntt`[`L147`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L147), `_injected_ntt_current`[`L144`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L144), `_injected_ntt_extend`[`L145`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L145), `_injected_ntt_factory`[`L146`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L146)
- uses (calls/refs, reference-scoped): [`data`](types.md#Ciphertext.data), [`precompute_constants`](ntt.md#NTTBarrett.precompute_constants), [`moduli`](types.md#Ciphertext.moduli), [`NTTBarrett`](ntt.md#NTTBarrett), [`precompute_constants`](basis_conversion.md#BasisConversionBarrett.precompute_constants), [`slice_moduli`](ntt.md#NTTBarrettConstants.slice_moduli), [`modular_reduction`](barrett.md#modular_reduction), [`constants`](ntt.md#NTTBarrett.constants), [`ntt`](ntt.md#NTTBarrett.ntt), [`precompute_barrett_constants`](barrett.md#precompute_barrett_constants), [`intt`](ntt.md#NTTBarrett.intt), [`basis_change`](basis_conversion.md#BasisConversionBarrett.basis_change), [`BasisConversionBarrett`](basis_conversion.md#BasisConversionBarrett), [`Ciphertext`](mul.md#Ciphertext), [`precomputed_constants`](basis_conversion.md#BasisConversionBarrett.precomputed_constants), [`EvaluationKeys`](mul.md#EvaluationKeys), [`a`](mul.md#EvaluationKeys.a), [`b`](mul.md#EvaluationKeys.b)
- used by: [`test_encrypt_multiply_decrypt`](mul_test.md#CiphertextCiphertextMulTest.test_encrypt_multiply_decrypt), [`test_relinearize_batched`](mul_test.md#CiphertextCiphertextMulTest.test_relinearize_batched), [`test_mul_equivalence`](cross_equivalence_test.md#CrossEquivalenceTest.test_mul_equivalence), [`test_mul_pytree`](mul_test.md#CiphertextCiphertextMulTest.test_mul_pytree)

### `MulPlaintextCiphertextBarrett`  ·  implements/extends MulPlaintextCiphertextBase
- def: [`jaxite/jaxite_ckks/mul.py:53`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L53)
- doc: Kernel for plaintext-ciphertext multiplication using Barrett reduction.
- signature: `class MulPlaintextCiphertextBarrett(MulPlaintextCiphertextBase):`
- members:
  - `mul(self, ct: types.Ciphertext, pt: types.Plaintext)` — [`L72`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L72)
  - `tree_flatten(self)` — [`L90`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L90)
  - `tree_unflatten(cls, _, children)` — [`L94`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L94)
  - `barrett_constants` — [`L70`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L70)
- protocol/private: `__init__`[`L69`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L69)
- uses (calls/refs, reference-scoped): [`data`](types.md#Ciphertext.data), [`moduli`](types.md#Ciphertext.moduli), [`Ciphertext`](types.md#Ciphertext), [`modular_reduction`](barrett.md#modular_reduction), [`data`](types.md#Plaintext.data), [`moduli`](types.md#Plaintext.moduli), [`Plaintext`](types.md#Plaintext), [`MulPlaintextCiphertextBase`](mul.md#MulPlaintextCiphertextBase), [`BarrettConstants`](barrett.md#BarrettConstants)
- used by: [`_run_conjugate_test`](conjugate_test.md#ConjugateTest._run_conjugate_test), [`key_switch`](key_switching.md#KeySwitcher.key_switch), [`test_conjugate_of_conjugate`](conjugate_test.md#ConjugateTest.test_conjugate_of_conjugate), [`test_key_switch`](key_switching_key_test.md#KeySwitchingTest.test_key_switch), [`test_full_pipeline_mul`](mul_test.md#PlaintextCiphertextMulTest.test_full_pipeline_mul), [`precompute_constants`](blind_rotate.md#BlindRotation.precompute_constants), [`brot_cm`](blind_rotate.md#BlindRotation.brot_cm), [`conjugate`](conjugate.md#Conjugation.conjugate), [`mul`](mul.md#MulPlaintextCiphertextBase.mul), [`MulPlaintextCiphertextBase`](mul.md#MulPlaintextCiphertextBase), [`_get_kernel`](mul_test.md#_get_kernel), [`mul_kernel`](blind_rotate.md#BlindRotation.mul_kernel)

### `MulPlaintextCiphertextBase`  ·  implements/extends ABC
- def: [`jaxite/jaxite_ckks/mul.py:19`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L19)
- doc: Abstract base class for plaintext-ciphertext multiplication kernels.
- signature: `class MulPlaintextCiphertextBase(ABC):`
- members:
  - `mul(self, ct: types.Ciphertext, pt: types.Plaintext)` — [`L23`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L23) — Multiplies ciphertext with plaintext. — documented in [jaxite-jaxite_ckks-types](../../../concepts/jaxite-jaxite_ckks-types.md)
- uses (calls/refs, reference-scoped): [`Ciphertext`](types.md#Ciphertext), [`mul`](mul.md#MulPlaintextCiphertextBarrett.mul), [`mul`](mul.md#MulPlaintextCiphertextSimple.mul), [`Plaintext`](types.md#Plaintext), [`MulPlaintextCiphertextBarrett`](mul.md#MulPlaintextCiphertextBarrett), [`MulPlaintextCiphertextSimple`](mul.md#MulPlaintextCiphertextSimple), [`ABC`](mul.md#ABC), [`abstractmethod`](mul.md#abstractmethod)
- used by: [`MulPlaintextCiphertextBarrett`](mul.md#MulPlaintextCiphertextBarrett), [`MulPlaintextCiphertextSimple`](mul.md#MulPlaintextCiphertextSimple)

### `MulPlaintextCiphertextSimple`  ·  implements/extends MulPlaintextCiphertextBase
- def: [`jaxite/jaxite_ckks/mul.py:28`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L28)
- doc: Kernel for raw plaintext-ciphertext multiplication without reduction.
- signature: `class MulPlaintextCiphertextSimple(MulPlaintextCiphertextBase):`
- members:
  - `mul(self, ct: types.Ciphertext, pt: types.Plaintext)` — [`L37`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L37) — documented in [jaxite-jaxite_ckks-types](../../../concepts/jaxite-jaxite_ckks-types.md)
  - `tree_flatten(self)` — [`L44`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L44)
  - `tree_unflatten(cls, _, _children)` — [`L48`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L48)
- protocol/private: `__init__`[`L34`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L34)
- uses (calls/refs, reference-scoped): [`data`](types.md#Ciphertext.data), [`moduli`](types.md#Ciphertext.moduli), [`Ciphertext`](types.md#Ciphertext), [`data`](types.md#Plaintext.data), [`moduli`](types.md#Plaintext.moduli), [`Plaintext`](types.md#Plaintext), [`MulPlaintextCiphertextBase`](mul.md#MulPlaintextCiphertextBase)
- used by: [`test_mul_modular_barrett_hypothesis`](mul_test.md#PlaintextCiphertextMulTest.test_mul_modular_barrett_hypothesis), [`test_mul_simple`](mul_test.md#PlaintextCiphertextMulTest.test_mul_simple), [`mul`](mul.md#MulPlaintextCiphertextBase.mul), [`MulPlaintextCiphertextBase`](mul.md#MulPlaintextCiphertextBase), [`_get_kernel`](mul_test.md#_get_kernel)

## Module values
- `ABC` — [`L14`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L14)
- `Ciphertext` — [`L15`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L15) — documented in [jaxite-jaxite_ckks-mul](../../../concepts/jaxite-jaxite_ckks-mul.md)
- `abstractmethod` — [`L16`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/mul.py#L16)


---
title: 'Module: jaxite/jaxite_ckks/basis_conversion.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_ckks/basis_conversion.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_ckks.basis_conversion`/
symbols:
  BasisConversionBarrett.precompute_constants: BasisConversionBarrett#precompute_constants().
  BasisConversionBarrett.basis_change: BasisConversionBarrett#basis_change().
  BasisConversionBarrett: BasisConversionBarrett#
  BarrettBasisConversionConstants.tree_flatten: BarrettBasisConversionConstants#tree_flatten().
  BasisConversion: BasisConversion#
  BasisConversionBarrett.precomputed_constants: BasisConversionBarrett#precomputed_constants.
  BarrettBasisConversionConstants.origin_barrett: BarrettBasisConversionConstants#origin_barrett.
  BarrettBasisConversionConstants.target_barrett: BarrettBasisConversionConstants#target_barrett.
  BasisConversion.precompute_constants: BasisConversion#precompute_constants().
  BasisConversion.basis_change: BasisConversion#basis_change().
  BarrettBasisConversionConstants.q_hat_inv_mod_q: BarrettBasisConversionConstants#q_hat_inv_mod_q.
  BarrettBasisConversionConstants.q_hat_mod_p_bat: BarrettBasisConversionConstants#q_hat_mod_p_bat.
  BasisConversionBarrett.__init__: BasisConversionBarrett#__init__().
  BasisConversionBarrett.tree_flatten: BasisConversionBarrett#tree_flatten().
  BasisConversionBarrett.tree_unflatten: BasisConversionBarrett#tree_unflatten().
  abstractmethod: abstractmethod.
  BarrettBasisConversionConstants: BarrettBasisConversionConstants#
  ABC: ABC.
  BarrettBasisConversionConstants.tree_unflatten: BarrettBasisConversionConstants#tree_unflatten().
  BarrettBasisConversionConstants.__hash__: BarrettBasisConversionConstants#__hash__().
  BasisConversionBarrett.__hash__: BasisConversionBarrett#__hash__().
---
# Module: [`jaxite/jaxite_ckks/basis_conversion.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/basis_conversion.py)

## Classes
### `BarrettBasisConversionConstants`
- def: [`jaxite/jaxite_ckks/basis_conversion.py:63`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/basis_conversion.py#L63)
- doc: Precomputed constants for Barrett-based basis conversion.
- signature: `class BarrettBasisConversionConstants:`
- members:
  - `tree_flatten(self)` — [`L71`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/basis_conversion.py#L71)
  - `tree_unflatten(cls, aux_data, children)` — [`L82`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/basis_conversion.py#L82)
  - `origin_barrett` — [`L68`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/basis_conversion.py#L68)
  - `q_hat_inv_mod_q` — [`L66`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/basis_conversion.py#L66)
  - `q_hat_mod_p_bat` — [`L67`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/basis_conversion.py#L67)
  - `target_barrett` — [`L69`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/basis_conversion.py#L69)
- protocol/private: `__hash__`[`L86`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/basis_conversion.py#L86)
- uses (calls/refs, reference-scoped): [`BarrettConstants`](barrett.md#BarrettConstants)
- used by: [`precompute_constants`](basis_conversion.md#BasisConversionBarrett.precompute_constants), [`basis_change`](basis_conversion.md#BasisConversionBarrett.basis_change), [`__init__`](basis_conversion.md#BasisConversionBarrett.__init__)

### `BasisConversion`  ·  implements/extends ABC
- def: [`jaxite/jaxite_ckks/basis_conversion.py:20`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/basis_conversion.py#L20)
- doc: Abstract base class for basis conversion kernels.
- signature: `class BasisConversion(ABC):`
- members:
  - `basis_change(self, in_tower: jnp.ndarray, control_index: int = 0)` — [`L45`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/basis_conversion.py#L45) — Performs a basis conversion.
  - `precompute_constants(self, modulus_chain: Iterable[int], control_indices: Iterable[Iterable[Iterable[int]]])` — [`L24`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/basis_conversion.py#L24) — Generates precomputed constants needed for basis conversion.
- uses (calls/refs, reference-scoped): [`precompute_constants`](basis_conversion.md#BasisConversionBarrett.precompute_constants), [`basis_change`](basis_conversion.md#BasisConversionBarrett.basis_change), [`BasisConversionBarrett`](basis_conversion.md#BasisConversionBarrett), [`abstractmethod`](basis_conversion.md#abstractmethod), [`ABC`](basis_conversion.md#ABC)
- used by: [`BasisConversionBarrett`](basis_conversion.md#BasisConversionBarrett)

### `BasisConversionBarrett`  ·  implements/extends BasisConversion
- def: [`jaxite/jaxite_ckks/basis_conversion.py:91`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/basis_conversion.py#L91) — documented in [jaxite-jaxite_ckks-mul](../../../concepts/jaxite-jaxite_ckks-mul.md)
- doc: Kernel for Basis Conversion with Barrett reduction.
- signature: `class BasisConversionBarrett(BasisConversion):`
- members:
  - `basis_change(self, in_tower: jnp.ndarray, control_index: int = 0)` — [`L147`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/basis_conversion.py#L147) — Performs the approximate basis change using BAT optimization. — documented in [jaxite-jaxite_ckks-mul](../../../concepts/jaxite-jaxite_ckks-mul.md)
  - `precompute_constants(self, modulus_chain: Iterable[int], control_indices: Iterable[Iterable[Iterable[int]]])` — [`L112`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/basis_conversion.py#L112)
  - `tree_flatten(self)` — [`L97`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/basis_conversion.py#L97)
  - `tree_unflatten(cls, aux_data, children)` — [`L103`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/basis_conversion.py#L103)
  - `precomputed_constants` — [`L95`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/basis_conversion.py#L95)
- protocol/private: `__hash__`[`L109`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/basis_conversion.py#L109), `__init__`[`L94`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/basis_conversion.py#L94)
- uses (calls/refs, reference-scoped): [`modular_reduction`](barrett.md#modular_reduction), [`precompute_barrett_constants`](barrett.md#precompute_barrett_constants), [`matmul_bat_einsum`](bat_utils.md#matmul_bat_einsum), [`basis_aligned_transformation`](bat_utils.md#basis_aligned_transformation), [`BasisConversion`](basis_conversion.md#BasisConversion), [`origin_barrett`](basis_conversion.md#BarrettBasisConversionConstants.origin_barrett), [`target_barrett`](basis_conversion.md#BarrettBasisConversionConstants.target_barrett), [`compute_q_hat_inv_mod_q`](rns_utils.md#compute_q_hat_inv_mod_q), [`q_hat_inv_mod_q`](basis_conversion.md#BarrettBasisConversionConstants.q_hat_inv_mod_q), [`q_hat_mod_p_bat`](basis_conversion.md#BarrettBasisConversionConstants.q_hat_mod_p_bat), [`BarrettBasisConversionConstants`](basis_conversion.md#BarrettBasisConversionConstants), [`compute_q_hat_mod_p`](rns_utils.md#compute_q_hat_mod_p)
- used by: [`relinearize`](mul.md#Mul.relinearize), [`test_encrypt_multiply_decrypt`](mul_test.md#CiphertextCiphertextMulTest.test_encrypt_multiply_decrypt), [`test_relinearize_batched`](mul_test.md#CiphertextCiphertextMulTest.test_relinearize_batched), [`_run_conjugate_test`](conjugate_test.md#ConjugateTest._run_conjugate_test), [`key_switch`](key_switching.md#KeySwitcher.key_switch), [`test_conjugate_of_conjugate`](conjugate_test.md#ConjugateTest.test_conjugate_of_conjugate), [`test_key_switch`](key_switching_key_test.md#KeySwitchingTest.test_key_switch), [`precompute_constants`](blind_rotate.md#BlindRotation.precompute_constants), [`conjugate`](conjugate.md#Conjugation.conjugate), [`bconv`](mul.md#Mul.bconv), [`lift_ciphertext`](blind_rotate_utils.md#lift_ciphertext), [`test_lift_ciphertext`](blind_rotate_utils_test.md#BlindRotateUtilsTest.test_lift_ciphertext), [`test_hypothesis`](basis_conversion_test.md#BasisConversionBarrettTest.test_hypothesis), [`test_mul_pytree`](mul_test.md#CiphertextCiphertextMulTest.test_mul_pytree), [`test_basis_conversion`](basis_conversion_test.md#BasisConversionBarrettTest.test_basis_conversion), [`bc_kernel`](blind_rotate.md#BlindRotation.bc_kernel), [`BasisConversion`](basis_conversion.md#BasisConversion), [`__init__`](mul.md#Mul.__init__), [`basis_change`](basis_conversion.md#BasisConversion.basis_change), [`precompute_constants`](basis_conversion.md#BasisConversion.precompute_constants)

## Module values
- `ABC` — [`L16`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/basis_conversion.py#L16)
- `abstractmethod` — [`L17`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/basis_conversion.py#L17)


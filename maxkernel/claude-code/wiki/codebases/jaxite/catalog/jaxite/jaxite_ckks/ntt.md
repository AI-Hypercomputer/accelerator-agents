---
title: 'Module: jaxite/jaxite_ckks/ntt.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_ckks/ntt.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_ckks.ntt`/
symbols:
  NTTBarrett.precompute_constants: NTTBarrett#precompute_constants().
  NTTBarrett: NTTBarrett#
  NTTBarrettConstants.slice_moduli: NTTBarrettConstants#slice_moduli().
  NTTBarrett.constants: NTTBarrett#constants.
  NTTBarrett.ntt: NTTBarrett#ntt().
  NTTBarrettConstants.tree_flatten: NTTBarrettConstants#tree_flatten().
  NTTBarrett.intt: NTTBarrett#intt().
  NTTBarrettConstants.barrett_constants: NTTBarrettConstants#barrett_constants.
  NTTBarrettConstants.ntt_bat_tf_step1: NTTBarrettConstants#ntt_bat_tf_step1.
  NTTBarrettConstants.ntt_tf_step2: NTTBarrettConstants#ntt_tf_step2.
  NTTBarrettConstants.ntt_bat_tf_step3: NTTBarrettConstants#ntt_bat_tf_step3.
  NTTBarrettConstants.intt_bat_tf_step1: NTTBarrettConstants#intt_bat_tf_step1.
  NTTBarrettConstants.intt_tf_step2: NTTBarrettConstants#intt_tf_step2.
  NTTBarrettConstants.intt_bat_tf_step3: NTTBarrettConstants#intt_bat_tf_step3.
  NTTBarrett.to_bat: NTTBarrett#to_bat().
  NTTBase: NTTBase#
  NTTBarrettConstants.r: NTTBarrettConstants#r.
  NTTBarrettConstants.c: NTTBarrettConstants#c.
  NTTBarrettConstants.moduli: NTTBarrettConstants#moduli.
  NTTBase.precompute_constants: NTTBase#precompute_constants().
  NTTBase.ntt: NTTBase#ntt().
  NTTBase.intt: NTTBase#intt().
  abstractmethod: abstractmethod.
  NTTBarrettConstants: NTTBarrettConstants#
  NTTBarrett.__init__: NTTBarrett#__init__().
  NTTBarrett.tree_flatten: NTTBarrett#tree_flatten().
  NTTBarrett.tree_unflatten: NTTBarrett#tree_unflatten().
  ABC: ABC.
  NTTBarrettConstants.tree_unflatten: NTTBarrettConstants#tree_unflatten().
  NTTBarrettConstants.__hash__: NTTBarrettConstants#__hash__().
  NTTBarrett.__hash__: NTTBarrett#__hash__().
---
# Module: [`jaxite/jaxite_ckks/ntt.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt.py)

## Classes
### `NTTBarrett`  ·  implements/extends NTTBase
- def: [`jaxite/jaxite_ckks/ntt.py:107`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt.py#L107) — documented in [jaxite-jaxite_ckks-mul](../../../concepts/jaxite-jaxite_ckks-mul.md)
- doc: Kernel for NTT with Barrett reduction.
- signature: `class NTTBarrett(NTTBase):`
- members:
  - `intt(self, v: jnp.ndarray)` — [`L269`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt.py#L269) — Performs the inverse NTT using BAT optimization. — documented in [jaxite-jaxite_ckks-ntt](../../../concepts/jaxite-jaxite_ckks-ntt.md)
  - `ntt(self, v: jnp.ndarray)` — [`L244`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt.py#L244) — Performs the forward NTT using BAT optimization. — documented in [jaxite-jaxite_ckks-ntt](../../../concepts/jaxite-jaxite_ckks-ntt.md)
  - `precompute_constants(self, moduli: Iterable[int], r: int, c: int)` — [`L128`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt.py#L128) — documented in [jaxite-jaxite_ckks-ntt](../../../concepts/jaxite-jaxite_ckks-ntt.md)
  - `to_bat(tf, moduli)` — [`L223`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt.py#L223) — documented in [jaxite-jaxite_ckks-ntt](../../../concepts/jaxite-jaxite_ckks-ntt.md)
  - `tree_flatten(self)` — [`L113`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt.py#L113)
  - `tree_unflatten(cls, aux_data, children)` — [`L119`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt.py#L119)
  - `constants` — [`L111`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt.py#L111) — documented in [jaxite-jaxite_ckks-ntt](../../../concepts/jaxite-jaxite_ckks-ntt.md)
- protocol/private: `__hash__`[`L125`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt.py#L125), `__init__`[`L110`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt.py#L110)
- uses (calls/refs, reference-scoped): [`modular_reduction`](barrett.md#modular_reduction), [`precompute_barrett_constants`](barrett.md#precompute_barrett_constants), [`barrett_constants`](ntt.md#NTTBarrettConstants.barrett_constants), [`gen_twiddle_matrix`](math.md#gen_twiddle_matrix), [`matmul_bat_einsum`](bat_utils.md#matmul_bat_einsum), [`basis_aligned_transformation`](bat_utils.md#basis_aligned_transformation), [`intt_bat_tf_step1`](ntt.md#NTTBarrettConstants.intt_bat_tf_step1), [`intt_bat_tf_step3`](ntt.md#NTTBarrettConstants.intt_bat_tf_step3), [`intt_tf_step2`](ntt.md#NTTBarrettConstants.intt_tf_step2), [`ntt_bat_tf_step1`](ntt.md#NTTBarrettConstants.ntt_bat_tf_step1), [`ntt_bat_tf_step3`](ntt.md#NTTBarrettConstants.ntt_bat_tf_step3), [`ntt_tf_step2`](ntt.md#NTTBarrettConstants.ntt_tf_step2), [`root_of_unity`](math.md#root_of_unity), [`NTTBase`](ntt.md#NTTBase), [`c`](ntt.md#NTTBarrettConstants.c), [`moduli`](ntt.md#NTTBarrettConstants.moduli), [`r`](ntt.md#NTTBarrettConstants.r), [`gen_twiddle_matrix_inv`](math.md#gen_twiddle_matrix_inv), [`get_bit_reverse_perm`](math.md#get_bit_reverse_perm), [`NTTBarrettConstants`](ntt.md#NTTBarrettConstants)
- used by: [`relinearize`](mul.md#Mul.relinearize), [`test_encrypt_multiply_decrypt`](mul_test.md#CiphertextCiphertextMulTest.test_encrypt_multiply_decrypt), [`test_relinearize_batched`](mul_test.md#CiphertextCiphertextMulTest.test_relinearize_batched), [`key_switch`](key_switching.md#KeySwitcher.key_switch), [`precompute_constants`](rescale.md#Rescale.precompute_constants), [`test_ntt_constants_slicing`](ntt_test.md#NTTTest.test_ntt_constants_slicing), [`precompute_constants`](blind_rotate.md#BlindRotation.precompute_constants), [`ntt_current`](mul.md#Mul.ntt_current), [`ntt_extend`](mul.md#Mul.ntt_extend), [`ks_ntt_kernels`](mul.md#Mul.ks_ntt_kernels), [`bconv`](mul.md#Mul.bconv), [`full_ntt_constants`](mul.md#Mul.full_ntt_constants), [`lift_ciphertext`](blind_rotate_utils.md#lift_ciphertext), [`test_lift_ciphertext`](blind_rotate_utils_test.md#BlindRotateUtilsTest.test_lift_ciphertext), [`test_mul_pytree`](mul_test.md#CiphertextCiphertextMulTest.test_mul_pytree), [`precompute_constants`](key_switching.md#KeySwitcher.precompute_constants), [`test_ntt_cpu_equivalence`](ntt_test.md#NTTTest.test_ntt_cpu_equivalence), [`test_ntt_cpu_equivalence_property`](ntt_test.md#NTTTest.test_ntt_cpu_equivalence_property), [`test_ntt_equivalence`](cross_equivalence_test.md#CrossEquivalenceTest.test_ntt_equivalence), [`test_ntt_intt_identity`](ntt_test.md#NTTTest.test_ntt_intt_identity), [`test_ntt_intt_identity_property`](ntt_test.md#NTTTest.test_ntt_intt_identity_property), [`ntt_pq`](blind_rotate_test.md#BlindRotationHypothesisTest.ntt_pq), [`ntt_kernels_q`](key_switching.md#KeySwitcher.ntt_kernels_q), [`test_ntt_linearity`](ntt_test.md#NTTTest.test_ntt_linearity), [`ntt_kernels_out`](key_switching.md#KeySwitcher.ntt_kernels_out), [`NTTBase`](ntt.md#NTTBase), [`ntt_p`](blind_rotate.md#BlindRotation.ntt_p), [`ntt_q`](blind_rotate.md#BlindRotation.ntt_q), [`__init__`](mul.md#Mul.__init__), [`intt`](ntt.md#NTTBase.intt), [`ntt`](ntt.md#NTTBase.ntt), [`precompute_constants`](ntt.md#NTTBase.precompute_constants)

### `NTTBarrettConstants`
- def: [`jaxite/jaxite_ckks/ntt.py:55`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt.py#L55) — documented in [jaxite-jaxite_ckks-ntt](../../../concepts/jaxite-jaxite_ckks-ntt.md)
- doc: Precomputed constants for Barrett-based NTT.
- signature: `class NTTBarrettConstants:`
- members:
  - `slice_moduli(self, slice_obj)` — [`L91`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt.py#L91) — documented in [jaxite-jaxite_ckks-ntt](../../../concepts/jaxite-jaxite_ckks-ntt.md)
  - `tree_flatten(self)` — [`L69`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt.py#L69) — Flattens this object into a JAX pytree.
  - `tree_unflatten(cls, aux_data, children)` — [`L85`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt.py#L85)
  - `barrett_constants` — [`L64`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt.py#L64) — documented in [jaxite-jaxite_ckks-ntt](../../../concepts/jaxite-jaxite_ckks-ntt.md)
  - `c` — [`L66`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt.py#L66) — documented in [jaxite-jaxite_ckks-ntt](../../../concepts/jaxite-jaxite_ckks-ntt.md)
  - `intt_bat_tf_step1` — [`L61`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt.py#L61) — documented in [jaxite-jaxite_ckks-ntt](../../../concepts/jaxite-jaxite_ckks-ntt.md)
  - `intt_bat_tf_step3` — [`L63`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt.py#L63) — documented in [jaxite-jaxite_ckks-ntt](../../../concepts/jaxite-jaxite_ckks-ntt.md)
  - `intt_tf_step2` — [`L62`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt.py#L62) — documented in [jaxite-jaxite_ckks-ntt](../../../concepts/jaxite-jaxite_ckks-ntt.md)
  - `moduli` — [`L67`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt.py#L67)
  - `ntt_bat_tf_step1` — [`L58`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt.py#L58)
  - `ntt_bat_tf_step3` — [`L60`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt.py#L60)
  - `ntt_tf_step2` — [`L59`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt.py#L59)
  - `r` — [`L65`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt.py#L65) — documented in [jaxite-jaxite_ckks-ntt](../../../concepts/jaxite-jaxite_ckks-ntt.md)
- protocol/private: `__hash__`[`L88`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt.py#L88)
- uses (calls/refs, reference-scoped): [`BarrettConstants`](barrett.md#BarrettConstants)
- used by: [`precompute_constants`](ntt.md#NTTBarrett.precompute_constants), [`key_switch`](key_switching.md#KeySwitcher.key_switch), [`precompute_constants`](rescale.md#Rescale.precompute_constants), [`test_ntt_constants_slicing`](ntt_test.md#NTTTest.test_ntt_constants_slicing), [`ntt`](ntt.md#NTTBarrett.ntt), [`ntt_current`](mul.md#Mul.ntt_current), [`ntt_extend`](mul.md#Mul.ntt_extend), [`ks_ntt_kernels`](mul.md#Mul.ks_ntt_kernels), [`intt`](ntt.md#NTTBarrett.intt), [`__init__`](ntt.md#NTTBarrett.__init__)

### `NTTBase`  ·  implements/extends ABC
- def: [`jaxite/jaxite_ckks/ntt.py:20`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt.py#L20)
- doc: Abstract base class for NTT/INTT kernels.
- signature: `class NTTBase(ABC):`
- members:
  - `intt(self, v: jnp.ndarray)` — [`L43`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt.py#L43) — Performs an INTT.
  - `ntt(self, v: jnp.ndarray)` — [`L33`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt.py#L33) — Performs an NTT.
  - `precompute_constants(self, moduli: Iterable[int], r: int, c: int)` — [`L24`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt.py#L24) — Generates precomputed constants needed for NTT.
- uses (calls/refs, reference-scoped): [`precompute_constants`](ntt.md#NTTBarrett.precompute_constants), [`NTTBarrett`](ntt.md#NTTBarrett), [`ntt`](ntt.md#NTTBarrett.ntt), [`intt`](ntt.md#NTTBarrett.intt), [`abstractmethod`](ntt.md#abstractmethod), [`ABC`](ntt.md#ABC)
- used by: [`NTTBarrett`](ntt.md#NTTBarrett)

## Module values
- `ABC` — [`L16`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt.py#L16)
- `abstractmethod` — [`L17`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt.py#L17)


---
title: 'Module: jaxite/jaxite_ckks/barrett.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_ckks/barrett.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_ckks.barrett`/
symbols:
  modular_reduction: modular_reduction().
  precompute_barrett_constants: precompute_barrett_constants().
  BarrettConstants.slice: BarrettConstants#slice().
  BarrettConstants.tree_flatten: BarrettConstants#tree_flatten().
  BarrettConstants: BarrettConstants#
  BarrettConstants.m: BarrettConstants#m.
  BarrettConstants.moduli: BarrettConstants#moduli.
  BarrettConstants.w: BarrettConstants#w.
  BarrettConstants.s_w: BarrettConstants#s_w.
  BarrettConstants.tree_unflatten: BarrettConstants#tree_unflatten().
---
# Module: [`jaxite/jaxite_ckks/barrett.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/barrett.py)

## Classes
### `BarrettConstants`
- def: [`jaxite/jaxite_ckks/barrett.py:16`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/barrett.py#L16)
- doc: Precomputed constants for Barrett reduction.
- signature: `class BarrettConstants:`
- members:
  - `slice(self, slice_obj: slice)` — [`L33`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/barrett.py#L33)
  - `tree_flatten(self)` — [`L24`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/barrett.py#L24)
  - `tree_unflatten(cls, _, children)` — [`L30`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/barrett.py#L30)
  - `m` — [`L19`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/barrett.py#L19)
  - `moduli` — [`L20`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/barrett.py#L20)
  - `s_w` — [`L22`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/barrett.py#L22)
  - `w` — [`L21`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/barrett.py#L21)
- used by: [`test_ntt_constants_slicing`](ntt_test.md#NTTTest.test_ntt_constants_slicing), [`modular_reduction`](barrett.md#modular_reduction), [`precompute_barrett_constants`](barrett.md#precompute_barrett_constants), [`barrett_constants`](ntt.md#NTTBarrettConstants.barrett_constants), [`origin_barrett`](basis_conversion.md#BarrettBasisConversionConstants.origin_barrett), [`target_barrett`](basis_conversion.md#BarrettBasisConversionConstants.target_barrett), [`__init__`](add.md#AddModularBarrett.__init__), [`__init__`](mul.md#MulPlaintextCiphertextBarrett.__init__)

## Functions
- `modular_reduction(z: jax.Array, constants: BarrettConstants)` — [`L71`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/barrett.py#L71) — Vectorized implementation of the Barrett reduction.
- `precompute_barrett_constants(moduli: list[int] | int)` — [`L42`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/barrett.py#L42) — Precomputes Barrett constants for a list of moduli. — documented in [jaxite-jaxite_ckks-ntt](../../../concepts/jaxite-jaxite_ckks-ntt.md)


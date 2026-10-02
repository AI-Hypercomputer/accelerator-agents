---
title: 'Module: jaxite/jaxite_cggi/decomposition_test.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_cggi/decomposition_test.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_cggi.decomposition_test`/
symbols:
  DecomposeTest.test_gadget_inverse_dot_gadget: DecomposeTest#test_gadget_inverse_dot_gadget().
  NUM_BITS: NUM_BITS.
  DecomposeTest.test_gadget_matrix_32_bit: DecomposeTest#test_gadget_matrix_32_bit().
  DecomposeTest.test_gadget_matrix: DecomposeTest#test_gadget_matrix().
  DecomposeTest.test_signed_decomposition: DecomposeTest#test_signed_decomposition().
  DecomposeTest.test_decompose_rlwe_ciphertext_vmap_compatibility: DecomposeTest#test_decompose_rlwe_ciphertext_vmap_compatibility().
  DecomposeTest.test_decompose_recompose: DecomposeTest#test_decompose_recompose().
  MAX_SIGNED_REPRESENTABLE: MAX_SIGNED_REPRESENTABLE.
  DecomposeTest.test_exact_reconstruction: DecomposeTest#test_exact_reconstruction().
  BASE: BASE.
  BASE_LOG: BASE_LOG.
  DecomposeTest.test_decompose_specific_example: DecomposeTest#test_decompose_specific_example().
  DecomposeTest.test_recompose_specific_example: DecomposeTest#test_recompose_specific_example().
  DecomposeTest.test_decomposition_summand_specific_example: DecomposeTest#test_decomposition_summand_specific_example().
  DecomposeTest: DecomposeTest#
---
# Module: [`jaxite/jaxite_cggi/decomposition_test.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/decomposition_test.py)

## Classes
### `DecomposeTest`  ·  implements/extends TestCase
- def: [`jaxite/jaxite_cggi/decomposition_test.py:19`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/decomposition_test.py#L19)
- signature: `class DecomposeTest(absltest.TestCase):`
- members:
  - `test_decompose_recompose(self, x: int, base_log: int, num_levels: int)` — [`L68`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/decomposition_test.py#L68)
  - `test_decompose_rlwe_ciphertext_vmap_compatibility(self)` — [`L168`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/decomposition_test.py#L168)
  - `test_decompose_specific_example(self)` — [`L21`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/decomposition_test.py#L21)
  - `test_decomposition_summand_specific_example(self)` — [`L28`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/decomposition_test.py#L28)
  - `test_exact_reconstruction(self, x: int)` — [`L49`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/decomposition_test.py#L49)
  - `test_gadget_inverse_dot_gadget(self, x: float, y: float)` — [`L137`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/decomposition_test.py#L137)
  - `test_gadget_matrix(self)` — [`L93`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/decomposition_test.py#L93)
  - `test_gadget_matrix_32_bit(self)` — [`L107`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/decomposition_test.py#L107)
  - `test_recompose_specific_example(self)` — [`L38`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/decomposition_test.py#L38)
  - `test_signed_decomposition(self, number)` — [`L157`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/decomposition_test.py#L157)
- uses (calls/refs, reference-scoped): [`DecompositionParameters`](decomposition.md#DecompositionParameters), [`level_count`](decomposition.md#DecompositionParameters.level_count), [`log_base`](decomposition.md#DecompositionParameters.log_base), [`NUM_BITS`](decomposition_test.md#NUM_BITS), [`decompose_rlwe_ciphertext`](decomposition.md#decompose_rlwe_ciphertext), [`inverse_gadget`](decomposition.md#inverse_gadget), [`gadget_matrix`](decomposition.md#gadget_matrix), [`MAX_SIGNED_REPRESENTABLE`](decomposition_test.md#MAX_SIGNED_REPRESENTABLE), [`BASE`](decomposition_test.md#BASE), [`decompose`](decomposition.md#decompose), [`BASE_LOG`](decomposition_test.md#BASE_LOG), [`recompose`](decomposition.md#recompose), [`signed_decomposition`](decomposition.md#signed_decomposition), [`recomposition_summands`](decomposition.md#recomposition_summands), [`total_bit_length`](decomposition.md#DecompositionParameters.total_bit_length)

## Module values
- `BASE` — [`L12`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/decomposition_test.py#L12)
- `BASE_LOG` — [`L11`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/decomposition_test.py#L11)
- `MAX_SIGNED_REPRESENTABLE` — [`L14`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/decomposition_test.py#L14)
- `NUM_BITS` — [`L10`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/decomposition_test.py#L10)


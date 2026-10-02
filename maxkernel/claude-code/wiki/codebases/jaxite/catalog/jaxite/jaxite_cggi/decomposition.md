---
title: 'Module: jaxite/jaxite_cggi/decomposition.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_cggi/decomposition.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_cggi.decomposition`/
symbols:
  DecompositionParameters: DecompositionParameters#
  DecompositionParameters.level_count: DecompositionParameters#level_count.
  DecompositionParameters.log_base: DecompositionParameters#log_base.
  decompose_rlwe_ciphertext: decompose_rlwe_ciphertext().
  inverse_gadget: inverse_gadget().
  gadget_matrix: gadget_matrix().
  decompose: decompose().
  recompose: recompose().
  signed_decomposition_polynomial: signed_decomposition_polynomial.
  signed_decomposition_polynomial_list: signed_decomposition_polynomial_list.
  signed_decomposition: signed_decomposition().
  DecomposedInt: DecomposedInt.
  DecompositionParameters.total_bit_length: DecompositionParameters#total_bit_length.
  recomposition_summands: recomposition_summands().
  GadgetMatrix: GadgetMatrix.
  GadgetDecomp: GadgetDecomp.
---
# Module: [`jaxite/jaxite_cggi/decomposition.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/decomposition.py)

## Classes
### `DecompositionParameters`
- def: [`jaxite/jaxite_cggi/decomposition.py:15`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/decomposition.py#L15) — documented in [jaxite-jaxite_bool-bool_params](../../../concepts/jaxite-jaxite_bool-bool_params.md)
- doc: The parameters to a bit decomposition subroutine.
- signature: `class DecompositionParameters:`
- members:
  - `level_count` — [`L25`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/decomposition.py#L25) — documented in [jaxite-jaxite_cggi-decomposition](../../../concepts/jaxite-jaxite_cggi-decomposition.md)
  - `log_base` — [`L24`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/decomposition.py#L24) — documented in [jaxite-jaxite_cggi-decomposition](../../../concepts/jaxite-jaxite_cggi-decomposition.md)
  - `total_bit_length` — [`L26`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/decomposition.py#L26) — documented in [jaxite-jaxite_cggi-decomposition](../../../concepts/jaxite-jaxite_cggi-decomposition.md)
- used by: [`test_key_switch_128_bit_security`](key_switch_test.md#KeySwitchTest.test_key_switch_128_bit_security), [`bootstrap`](bootstrap.md#bootstrap), [`encrypt`](rgsw.md#encrypt), [`gen_bootstrapping_key`](bootstrap.md#gen_bootstrapping_key), [`gen_key`](key_switch.md#gen_key), [`decrypt`](rgsw.md#decrypt), [`jit_bootstrap`](bootstrap.md#jit_bootstrap), [`test_bsk_decomposition_params`](../jaxite_bool/jaxite_bool_test.md#BoolBasicOperationsTest.test_bsk_decomposition_params), [`test_encrypt_diagonal`](rgsw_test.md#RgswTest.test_encrypt_diagonal), [`test_ksk_decomposition_params`](../jaxite_bool/jaxite_bool_test.md#BoolBasicOperationsTest.test_ksk_decomposition_params), [`blind_rotate`](bootstrap.md#blind_rotate), [`test_encrypt_dimension`](rgsw_test.md#RgswTest.test_encrypt_dimension), [`cmux`](bootstrap.md#cmux), [`decomposition_params`](blind_rotate_test.md#BlindRotateTest.decomposition_params), [`test_key_switch_gen`](key_switch_test.md#KeySwitchTest.test_key_switch_gen), [`external_product`](bootstrap.md#external_product), [`decomposition_params`](rgsw_test.md#RgswTest.decomposition_params), [`bs_decomp_params`](../jaxite_bool/bool_params.md#Parameters.bs_decomp_params), [`ks_decomp_params`](../jaxite_bool/bool_params.md#Parameters.ks_decomp_params), [`jit_external_product`](bootstrap.md#jit_external_product), [`test_gadget_inverse_dot_gadget`](decomposition_test.md#DecomposeTest.test_gadget_inverse_dot_gadget), [`jit_blind_rotate`](bootstrap.md#jit_blind_rotate), [`test_gadget_matrix_32_bit`](decomposition_test.md#DecomposeTest.test_gadget_matrix_32_bit), [`decompose_rlwe_ciphertext`](decomposition.md#decompose_rlwe_ciphertext), [`jit_encrypt`](rgsw.md#jit_encrypt), [`inverse_gadget`](decomposition.md#inverse_gadget), [`gadget_matrix`](decomposition.md#gadget_matrix), [`test_decompose_rlwe_ciphertext_vmap_compatibility`](decomposition_test.md#DecomposeTest.test_decompose_rlwe_ciphertext_vmap_compatibility), [`test_gadget_matrix`](decomposition_test.md#DecomposeTest.test_gadget_matrix), [`decomposition_params`](key_switch_test.md#KeySwitchTest.decomposition_params), [`decompose_and_encrypt`](key_switch.md#gen_key.decompose_and_encrypt), [`BSK_DECOMP_PARAMS_128_BIT_SECURITY`](../jaxite_bool/bool_params.md#BSK_DECOMP_PARAMS_128_BIT_SECURITY), [`KSK_DECOMP_PARAMS_128_BIT_SECURITY`](../jaxite_bool/bool_params.md#KSK_DECOMP_PARAMS_128_BIT_SECURITY), [`TEST_BSK_DECOMP_PARAMS`](../jaxite_bool/bool_params.md#TEST_BSK_DECOMP_PARAMS), [`jit_cmux`](bootstrap.md#jit_cmux), [`__init__`](../jaxite_bool/bool_params.md#Parameters.__init__)

## Functions
- `decompose(x: jnp.uint32, base_log: int = 4, num_levels: int = 3, total_bit_length: int = 32)` — [`L34`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/decomposition.py#L34) — Decompose a number in a given power-of-2 base, up to a given level.
- `decompose_rlwe_ciphertext(rlwe_ct: jnp.ndarray, decomposition_params: DecompositionParameters)` — [`L235`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/decomposition.py#L235) — Bit-decompose and rearrange an RLWE ciphertext for external product. — documented in [jaxite-jaxite_cggi-decomposition](../../../concepts/jaxite-jaxite_cggi-decomposition.md)
- `gadget_matrix(decomp_params: DecompositionParameters, vector_length: int, total_bit_length: int = 32)` — [`L270`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/decomposition.py#L270) — Construct a gadget matrix for the decomposition parameters. — documented in [jaxite-jaxite_cggi-decomposition](../../../concepts/jaxite-jaxite_cggi-decomposition.md)
- `inverse_gadget(vector: jnp.ndarray, decomp_params: DecompositionParameters, total_bit_length: int = 32)` — [`L324`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/decomposition.py#L324) — Compute the inverse gadget decomposition for a vector. — documented in [jaxite-jaxite_cggi-decomposition](../../../concepts/jaxite-jaxite_cggi-decomposition.md)
- `recompose(digits: DecomposedInt, base_log: int = 4, total_bit_length: int = 32)` — [`L131`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/decomposition.py#L131) — The inverse of decompose.
- `recomposition_summands(x: jnp.uint32, base_log: int = 4, num_levels: int = 3, total_bit_length: int = 32)` — [`L94`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/decomposition.py#L94) — Outputs a list of summands of the digit decomposition.
- `signed_decomposition(x: jnp.uint32, base_log: jnp.uint32, num_levels: int, total_bit_length: int = 32)` — [`L155`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/decomposition.py#L155) — Compute the signed base-B digit decomposition of a number. — documented in [jaxite-jaxite_cggi-decomposition](../../../concepts/jaxite-jaxite_cggi-decomposition.md)

## Module values
- `DecomposedInt` — [`L9`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/decomposition.py#L9)
- `GadgetDecomp` — [`L11`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/decomposition.py#L11)
- `GadgetMatrix` — [`L10`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/decomposition.py#L10)
- `signed_decomposition_polynomial` — [`L220`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/decomposition.py#L220) — documented in [jaxite-jaxite_cggi-decomposition](../../../concepts/jaxite-jaxite_cggi-decomposition.md)
- `signed_decomposition_polynomial_list` — [`L228`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/decomposition.py#L228) — documented in [jaxite-jaxite_cggi-decomposition](../../../concepts/jaxite-jaxite_cggi-decomposition.md)


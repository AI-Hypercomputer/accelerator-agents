---
title: 'Module: jaxite/jaxite_ckks/blind_rotate_utils_test.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_ckks/blind_rotate_utils_test.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_ckks.blind_rotate_utils_test`/BlindRotateUtilsTest#
symbols:
  BlindRotateUtilsTest.test_lift_ciphertext: test_lift_ciphertext().
  BlindRotateUtilsTest.test_apply_automorphism_ntt_non_trivial: test_apply_automorphism_ntt_non_trivial().
  BlindRotateUtilsTest.test_apply_automorphism_ntt: test_apply_automorphism_ntt().
  BlindRotateUtilsTest: ''
---
# Module: [`jaxite/jaxite_ckks/blind_rotate_utils_test.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate_utils_test.py)

## Classes
### `BlindRotateUtilsTest`  ·  implements/extends TestCase
- def: [`jaxite/jaxite_ckks/blind_rotate_utils_test.py:18`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate_utils_test.py#L18)
- signature: `class BlindRotateUtilsTest(parameterized.TestCase):`
- members:
  - `test_apply_automorphism_ntt(self)` — [`L20`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate_utils_test.py#L20)
  - `test_apply_automorphism_ntt_non_trivial(self, g)` — [`L69`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate_utils_test.py#L69)
  - `test_lift_ciphertext(self)` — [`L27`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate_utils_test.py#L27)
- uses (calls/refs, reference-scoped): [`data`](types.md#Ciphertext.data), [`precompute_constants`](ntt.md#NTTBarrett.precompute_constants), [`moduli`](types.md#Ciphertext.moduli), [`Ciphertext`](types.md#Ciphertext), [`NTTBarrett`](ntt.md#NTTBarrett), [`precompute_constants`](basis_conversion.md#BasisConversionBarrett.precompute_constants), [`ntt_negacyclic_poly`](ntt_cpu.md#ntt_negacyclic_poly), [`lift_ciphertext`](blind_rotate_utils.md#lift_ciphertext), [`BasisConversionBarrett`](basis_conversion.md#BasisConversionBarrett), [`apply_automorphism_ntt`](blind_rotate_utils.md#apply_automorphism_ntt)


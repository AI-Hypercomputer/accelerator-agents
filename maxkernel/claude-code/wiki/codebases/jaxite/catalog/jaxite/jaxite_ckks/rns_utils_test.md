---
title: 'Module: jaxite/jaxite_ckks/rns_utils_test.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_ckks/rns_utils_test.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_ckks.rns_utils_test`/
symbols:
  RnsUtilsTest.test_reconstruct_crt_hypothesis: RnsUtilsTest#test_reconstruct_crt_hypothesis().
  RnsUtilsTest.test_reconstruct_crt_basic: RnsUtilsTest#test_reconstruct_crt_basic().
  moduli_and_values: moduli_and_values().
  RnsUtilsTest: RnsUtilsTest#
---
# Module: [`jaxite/jaxite_ckks/rns_utils_test.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns_utils_test.py)

## Classes
### `RnsUtilsTest`  ·  implements/extends TestCase
- def: [`jaxite/jaxite_ckks/rns_utils_test.py:40`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns_utils_test.py#L40)
- signature: `class RnsUtilsTest(parameterized.TestCase):`
- members:
  - `test_reconstruct_crt_basic(self, moduli, values)` — [`L46`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns_utils_test.py#L46)
  - `test_reconstruct_crt_hypothesis(self, data)` — [`L62`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns_utils_test.py#L62)
- uses (calls/refs, reference-scoped): [`reconstruct_crt`](rns_utils.md#reconstruct_crt), [`moduli_and_values`](rns_utils_test.md#moduli_and_values)

## Functions
- `moduli_and_values(draw)` — [`L13`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns_utils_test.py#L13) — Strategy to generate a batch of moduli and original integers.


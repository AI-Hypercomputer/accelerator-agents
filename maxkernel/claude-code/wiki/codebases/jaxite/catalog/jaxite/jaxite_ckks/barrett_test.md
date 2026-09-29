---
title: 'Module: jaxite/jaxite_ckks/barrett_test.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_ckks/barrett_test.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_ckks.barrett_test`/
symbols:
  BarrettTest.test_modular_reduction_hypothesis: BarrettTest#test_modular_reduction_hypothesis().
  BarrettTest.test_modular_reduction_basic: BarrettTest#test_modular_reduction_basic().
  BarrettTest.test_modular_reduction_simulation_data: BarrettTest#test_modular_reduction_simulation_data().
  moduli_and_z: moduli_and_z().
  BATCH_SIZE: BATCH_SIZE.
  BarrettTest: BarrettTest#
---
# Module: [`jaxite/jaxite_ckks/barrett_test.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/barrett_test.py)

## Classes
### `BarrettTest`  ·  implements/extends TestCase
- def: [`jaxite/jaxite_ckks/barrett_test.py:40`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/barrett_test.py#L40)
- signature: `class BarrettTest(parameterized.TestCase):`
- members:
  - `test_modular_reduction_basic(self, modulus, inputs)` — [`L47`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/barrett_test.py#L47)
  - `test_modular_reduction_hypothesis(self, moduli_and_input)` — [`L69`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/barrett_test.py#L69)
  - `test_modular_reduction_simulation_data(self)` — [`L55`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/barrett_test.py#L55)
- uses (calls/refs, reference-scoped): [`modular_reduction`](barrett.md#modular_reduction), [`precompute_barrett_constants`](barrett.md#precompute_barrett_constants), [`moduli_and_z`](barrett_test.md#moduli_and_z)

## Functions
- `moduli_and_z(draw)` — [`L19`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/barrett_test.py#L19) — Strategy to generate a batch of moduli and corresponding inputs.

## Module values
- `BATCH_SIZE` — [`L15`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/barrett_test.py#L15)


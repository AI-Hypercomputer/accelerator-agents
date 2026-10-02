---
title: 'Module: jaxite/jaxite_ckks/basis_conversion_test.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_ckks/basis_conversion_test.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_ckks.basis_conversion_test`/
symbols:
  BasisConversionBarrettTest.test_hypothesis: BasisConversionBarrettTest#test_hypothesis().
  BasisConversionBarrettTest.test_basis_conversion: BasisConversionBarrettTest#test_basis_conversion().
  verify_approximate_basis_conversion: verify_approximate_basis_conversion().
  generate_rns_value: generate_rns_value().
  TEST_PRIMES: TEST_PRIMES.
  BasisConversionBarrettTest: BasisConversionBarrettTest#
---
# Module: [`jaxite/jaxite_ckks/basis_conversion_test.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/basis_conversion_test.py)

## Classes
### `BasisConversionBarrettTest`  ·  implements/extends TestCase
- def: [`jaxite/jaxite_ckks/basis_conversion_test.py:138`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/basis_conversion_test.py#L138)
- signature: `class BasisConversionBarrettTest(parameterized.TestCase):`
- members:
  - `test_basis_conversion(self, moduli, ring_dim)` — [`L145`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/basis_conversion_test.py#L145)
  - `test_hypothesis(self, num_total_moduli, log2_ring_dim, seed)` — [`L178`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/basis_conversion_test.py#L178)
- uses (calls/refs, reference-scoped): [`precompute_constants`](basis_conversion.md#BasisConversionBarrett.precompute_constants), [`basis_change`](basis_conversion.md#BasisConversionBarrett.basis_change), [`BasisConversionBarrett`](basis_conversion.md#BasisConversionBarrett), [`verify_approximate_basis_conversion`](basis_conversion_test.md#verify_approximate_basis_conversion), [`generate_rns_value`](basis_conversion_test.md#generate_rns_value), [`TEST_PRIMES`](basis_conversion_test.md#TEST_PRIMES)

## Functions
- `generate_rns_value(ring_dim, moduli, key)` — [`L80`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/basis_conversion_test.py#L80) — Generate an RNS value for the given ring_dim and moduli.
- `verify_approximate_basis_conversion(in_tower: jax.Array, out_tower: jax.Array, orig_moduli: list[int], target_moduli: list[int])` — [`L18`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/basis_conversion_test.py#L18) — Verifies the mathematical correctness of approximate basis conversion.

## Module values
- `TEST_PRIMES` — [`L95`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/basis_conversion_test.py#L95)


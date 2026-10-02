---
title: 'Module: jaxite/jaxite_ckks/ntt_test.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_ckks/ntt_test.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_ckks.ntt_test`/
symbols:
  NTTTest.test_ntt_constants_slicing: NTTTest#test_ntt_constants_slicing().
  NTTTest.test_ntt_intt_identity: NTTTest#test_ntt_intt_identity().
  NTTTest.test_ntt_cpu_equivalence: NTTTest#test_ntt_cpu_equivalence().
  NTTTest.test_ntt_cpu_equivalence_property: NTTTest#test_ntt_cpu_equivalence_property().
  NTTTest.test_ntt_intt_identity_property: NTTTest#test_ntt_intt_identity_property().
  NTTTest.test_ntt_linearity: NTTTest#test_ntt_linearity().
  TEST_PRIMES: TEST_PRIMES.
  ntt_reference: ntt_reference().
  NTTTest: NTTTest#
---
# Module: [`jaxite/jaxite_ckks/ntt_test.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt_test.py)

## Classes
### `NTTTest`  ·  implements/extends TestCase
- def: [`jaxite/jaxite_ckks/ntt_test.py:50`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt_test.py#L50)
- signature: `class NTTTest(parameterized.TestCase):`
- members:
  - `test_ntt_constants_slicing(self)` — [`L166`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt_test.py#L166)
  - `test_ntt_cpu_equivalence(self)` — [`L70`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt_test.py#L70)
  - `test_ntt_cpu_equivalence_property(self, r_log2, c_log2)` — [`L96`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt_test.py#L96)
  - `test_ntt_intt_identity(self, r, c, moduli)` — [`L56`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt_test.py#L56)
  - `test_ntt_intt_identity_property(self, r_log2, c_log2)` — [`L123`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt_test.py#L123)
  - `test_ntt_linearity(self)` — [`L141`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt_test.py#L141)
- uses (calls/refs, reference-scoped): [`precompute_constants`](ntt.md#NTTBarrett.precompute_constants), [`NTTBarrett`](ntt.md#NTTBarrett), [`slice_moduli`](ntt.md#NTTBarrettConstants.slice_moduli), [`constants`](ntt.md#NTTBarrett.constants), [`ntt`](ntt.md#NTTBarrett.ntt), [`ntt_negacyclic_poly`](ntt_cpu.md#ntt_negacyclic_poly), [`intt`](ntt.md#NTTBarrett.intt), [`barrett_constants`](ntt.md#NTTBarrettConstants.barrett_constants), [`TEST_PRIMES`](ntt_test.md#TEST_PRIMES), [`intt_bat_tf_step1`](ntt.md#NTTBarrettConstants.intt_bat_tf_step1), [`intt_bat_tf_step3`](ntt.md#NTTBarrettConstants.intt_bat_tf_step3), [`intt_tf_step2`](ntt.md#NTTBarrettConstants.intt_tf_step2), [`m`](barrett.md#BarrettConstants.m), [`moduli`](barrett.md#BarrettConstants.moduli), [`ntt_bat_tf_step1`](ntt.md#NTTBarrettConstants.ntt_bat_tf_step1), [`ntt_bat_tf_step3`](ntt.md#NTTBarrettConstants.ntt_bat_tf_step3), [`ntt_tf_step2`](ntt.md#NTTBarrettConstants.ntt_tf_step2), [`s_w`](barrett.md#BarrettConstants.s_w), [`w`](barrett.md#BarrettConstants.w), [`moduli`](ntt.md#NTTBarrettConstants.moduli)

## Functions
- `ntt_reference(v, q, omega, r, c)` — [`L25`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt_test.py#L25) — Naive 4-step NTT for reference.

## Module values
- `TEST_PRIMES` — [`L17`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt_test.py#L17)


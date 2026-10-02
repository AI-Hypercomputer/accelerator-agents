---
title: 'Module: jaxite_ec/finite_field_test.py'
type: catalog
provenance: extracted
module: jaxite_ec/finite_field_test.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite_ec.finite_field_test`/
symbols:
  FiniteFieldTest.test_add_three: FiniteFieldTest#test_add_three().
  FiniteFieldTest.test_cond_sub_1: FiniteFieldTest#test_cond_sub_1().
  FiniteFieldTest.test_cond_sub_2: FiniteFieldTest#test_cond_sub_2().
  FiniteFieldTest.test_mod_mul_barrett_1: FiniteFieldTest#test_mod_mul_barrett_1().
  FiniteFieldTest.test_add_two: FiniteFieldTest#test_add_two().
  FiniteFieldTest.test_mul_1: FiniteFieldTest#test_mul_1().
  FiniteFieldTest.value_a_jax: FiniteFieldTest#value_a_jax.
  FiniteFieldTest.value_b_jax: FiniteFieldTest#value_b_jax.
  list_operation: list_operation().
  FiniteFieldTest.value_a: FiniteFieldTest#value_a.
  FiniteFieldTest.value_b: FiniteFieldTest#value_b.
  randint: randint.
  FiniteFieldTest.value_c_jax: FiniteFieldTest#value_c_jax.
  FiniteFieldTest.test_cond_sub_mod_1: FiniteFieldTest#test_cond_sub_mod_1().
  FiniteFieldTest.test_jax_mod_mul_lazy_reduction: FiniteFieldTest#test_jax_mod_mul_lazy_reduction().
  FiniteFieldTest.test_jax_mod_mul_rns_reduction: FiniteFieldTest#test_jax_mod_mul_rns_reduction().
  FiniteFieldTest.test_jax_sub_rns: FiniteFieldTest#test_jax_sub_rns().
  FiniteFieldTest.value_c: FiniteFieldTest#value_c.
  FiniteFieldTest.cond_sub: FiniteFieldTest#cond_sub().
  FiniteFieldTest.cond_sub_mod: FiniteFieldTest#cond_sub_mod().
  FiniteFieldTest.mod_mul_barrett: FiniteFieldTest#mod_mul_barrett().
  list_operation_three: list_operation_three().
  FiniteFieldTest: FiniteFieldTest#
  FiniteFieldTest.setUp: FiniteFieldTest#setUp().
  FiniteFieldTest.test_generate_rns_precompute_matrix: FiniteFieldTest#test_generate_rns_precompute_matrix().
  FiniteFieldTest.test_jax_add_rns: FiniteFieldTest#test_jax_add_rns().
  FiniteFieldTest.test_jax_add_rns_specific_case: FiniteFieldTest#test_jax_add_rns_specific_case().
---
# Module: [`jaxite_ec/finite_field_test.py`](../../../../../raw/code/jaxite/jaxite_ec/finite_field_test.py)

## Classes
### `FiniteFieldTest`  ·  implements/extends TestCase
- def: [`jaxite_ec/finite_field_test.py:26`](../../../../../raw/code/jaxite/jaxite_ec/finite_field_test.py#L26)
- signature: `class FiniteFieldTest(absltest.TestCase):`
- members:
  - `cond_sub(a, b)` — [`L107`](../../../../../raw/code/jaxite/jaxite_ec/finite_field_test.py#L107)
  - `cond_sub_mod(a)` — [`L138`](../../../../../raw/code/jaxite/jaxite_ec/finite_field_test.py#L138)
  - `mod_mul_barrett(a, b)` — [`L158`](../../../../../raw/code/jaxite/jaxite_ec/finite_field_test.py#L158)
  - `setUp(self)` — [`L28`](../../../../../raw/code/jaxite/jaxite_ec/finite_field_test.py#L28)
  - `test_add_three(self)` — [`L90`](../../../../../raw/code/jaxite/jaxite_ec/finite_field_test.py#L90)
  - `test_add_two(self)` — [`L82`](../../../../../raw/code/jaxite/jaxite_ec/finite_field_test.py#L82)
  - `test_cond_sub_1(self)` — [`L103`](../../../../../raw/code/jaxite/jaxite_ec/finite_field_test.py#L103)
  - `test_cond_sub_2(self)` — [`L116`](../../../../../raw/code/jaxite/jaxite_ec/finite_field_test.py#L116)
  - `test_cond_sub_mod_1(self)` — [`L130`](../../../../../raw/code/jaxite/jaxite_ec/finite_field_test.py#L130)
  - `test_generate_rns_precompute_matrix(self)` — [`L53`](../../../../../raw/code/jaxite/jaxite_ec/finite_field_test.py#L53)
  - `test_jax_add_rns(self)` — [`L204`](../../../../../raw/code/jaxite/jaxite_ec/finite_field_test.py#L204)
  - `test_jax_add_rns_specific_case(self)` — [`L243`](../../../../../raw/code/jaxite/jaxite_ec/finite_field_test.py#L243)
  - `test_jax_mod_mul_lazy_reduction(self)` — [`L167`](../../../../../raw/code/jaxite/jaxite_ec/finite_field_test.py#L167) — This test case check the jax version (TPU deployment) of the lazy reduction based modular multiplication algorithm.
  - `test_jax_mod_mul_rns_reduction(self)` — [`L187`](../../../../../raw/code/jaxite/jaxite_ec/finite_field_test.py#L187) — This test case check the jax version (TPU deployment) of the rns reduction based modular multiplication algorithm.
  - `test_jax_sub_rns(self)` — [`L227`](../../../../../raw/code/jaxite/jaxite_ec/finite_field_test.py#L227)
  - `test_mod_mul_barrett_1(self)` — [`L154`](../../../../../raw/code/jaxite/jaxite_ec/finite_field_test.py#L154)
  - `test_mul_1(self)` — [`L148`](../../../../../raw/code/jaxite/jaxite_ec/finite_field_test.py#L148)
  - `value_a` — [`L30`](../../../../../raw/code/jaxite/jaxite_ec/finite_field_test.py#L30)
  - `value_a_jax` — [`L42`](../../../../../raw/code/jaxite/jaxite_ec/finite_field_test.py#L42)
  - `value_b` — [`L34`](../../../../../raw/code/jaxite/jaxite_ec/finite_field_test.py#L34)
  - `value_b_jax` — [`L45`](../../../../../raw/code/jaxite/jaxite_ec/finite_field_test.py#L45)
  - `value_c` — [`L38`](../../../../../raw/code/jaxite/jaxite_ec/finite_field_test.py#L38)
  - `value_c_jax` — [`L48`](../../../../../raw/code/jaxite/jaxite_ec/finite_field_test.py#L48)
- uses (calls/refs, reference-scoped): [`list_operation`](finite_field_test.md#list_operation), [`randint`](finite_field_test.md#randint)

## Functions
- `list_operation(a, b, func)` — [`L18`](../../../../../raw/code/jaxite/jaxite_ec/finite_field_test.py#L18)
- `list_operation_three(a, b, c, func)` — [`L22`](../../../../../raw/code/jaxite/jaxite_ec/finite_field_test.py#L22)

## Module values
- `randint` — [`L15`](../../../../../raw/code/jaxite/jaxite_ec/finite_field_test.py#L15)


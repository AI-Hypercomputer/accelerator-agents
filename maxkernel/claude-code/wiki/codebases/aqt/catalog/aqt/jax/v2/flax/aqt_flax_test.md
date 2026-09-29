---
title: 'Module: aqt/jax/v2/flax/aqt_flax_test.py'
type: catalog
provenance: extracted
module: aqt/jax/v2/flax/aqt_flax_test.py
status: fresh
symbol_base: scip-python python aqt 0.0.0 `aqt.jax.v2.flax.aqt_flax_test`/AqtFlaxTest#
symbols:
  AqtFlaxTest: ''
  AqtFlaxTest.test_aqt_promote_dtype: test_aqt_promote_dtype().
  AqtFlaxTest.test_aqt_einsum: test_aqt_einsum().
  AqtFlaxTest.test_einsum_grad_leak: test_einsum_grad_leak().
  AqtFlaxTest.test_freezer: test_freezer().
  AqtFlaxTest.test_dot_general_tiling_fn: test_dot_general_tiling_fn().
  AqtFlaxTest.test_einsum_tiling_fn: test_einsum_tiling_fn().
---
# Module: [`aqt/jax/v2/flax/aqt_flax_test.py`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax_test.py)

## Classes
### `AqtFlaxTest`
- def: [`aqt/jax/v2/flax/aqt_flax_test.py:35`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax_test.py#L35)
- signature: `class AqtFlaxTest(parameterized.TestCase):`
- members:
  - `test_aqt_einsum(self)` — [`L81`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax_test.py#L81)
  - `test_aqt_promote_dtype(self, lhs_dtype, rhs_dtype, expected_dtype)` — [`L66`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax_test.py#L66)
  - `test_dot_general_tiling_fn(self)` — [`L288`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax_test.py#L288)
  - `test_einsum_grad_leak(self)` — [`L160`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax_test.py#L160)
  - `test_einsum_tiling_fn(self)` — [`L381`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax_test.py#L381)
  - `test_freezer(self, use_legacy_freezer: bool)` — [`L191`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax_test.py#L191)


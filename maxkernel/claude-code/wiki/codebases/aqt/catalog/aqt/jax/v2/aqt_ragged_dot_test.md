---
title: 'Module: aqt/jax/v2/aqt_ragged_dot_test.py'
type: catalog
provenance: extracted
module: aqt/jax/v2/aqt_ragged_dot_test.py
status: fresh
symbol_base: scip-python python aqt 0.0.0 `aqt.jax.v2.aqt_ragged_dot_test`/
symbols:
  sample_groups: sample_groups().
  random_dense: random_dense().
  AqtRaggedDotTest: AqtRaggedDotTest#
  AqtRaggedDotTest.assert_allclose: AqtRaggedDotTest#assert_allclose().
  AqtRaggedDotTest.make_args: AqtRaggedDotTest#make_args().
  AqtRaggedDotTest.test_numeric_correctness: AqtRaggedDotTest#test_numeric_correctness().
  AqtRaggedDotTest.test_ragged_dot_with_precision: AqtRaggedDotTest#test_ragged_dot_with_precision().
---
# Module: [`aqt/jax/v2/aqt_ragged_dot_test.py`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_ragged_dot_test.py)

## Classes
### `AqtRaggedDotTest`
- def: [`aqt/jax/v2/aqt_ragged_dot_test.py:54`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_ragged_dot_test.py#L54)
- signature: `class AqtRaggedDotTest(parameterized.TestCase):`
- members:
  - `assert_allclose(self, out: jnp.ndarray, expected_out: jnp.ndarray, *, rtol: float=1e-05, atol: float=1e-05)` — [`L56`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_ragged_dot_test.py#L56) — Asserts that two arrays are close.
  - `make_args(self, m: int, k: int, n: int, num_groups: int, in_dtype: jnp.dtype, balanced_groups: bool)` — [`L84`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_ragged_dot_test.py#L84) — Makes arguments for ragged_dot.
  - `test_numeric_correctness(self, in_dtype, balanced_groups)` — [`L114`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_ragged_dot_test.py#L114)
  - `test_ragged_dot_with_precision(self, precision)` — [`L155`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_ragged_dot_test.py#L155)

## Functions
- `random_dense(shape: tuple[int, ...], key: jax.Array, dtype: jnp.dtype)` — [`L45`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_ragged_dot_test.py#L45)
- `sample_groups(m: int, num_groups: int, key: jax.Array)` — [`L31`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_ragged_dot_test.py#L31)


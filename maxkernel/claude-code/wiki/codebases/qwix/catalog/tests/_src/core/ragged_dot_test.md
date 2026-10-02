---
title: 'Module: tests/_src/core/ragged_dot_test.py'
type: catalog
provenance: extracted
module: tests/_src/core/ragged_dot_test.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `tests._src.core.ragged_dot_test`/
symbols:
  RaggedDotTest.test_ragged_dot_general: RaggedDotTest#test_ragged_dot_general().
  RaggedDotTest.test_ragged_dot_general_implementation: RaggedDotTest#test_ragged_dot_general_implementation().
  RaggedDotTest.test_ragged_dot: RaggedDotTest#test_ragged_dot().
  RaggedDotTest._jitted_ragged_dot_general: RaggedDotTest#_jitted_ragged_dot_general().
  RaggedDotTest._make_array: RaggedDotTest#_make_array().
  RaggedDotTest._jitted_ragged_dot: RaggedDotTest#_jitted_ragged_dot().
  rel_mae: rel_mae().
  RaggedDotTest._random_key: RaggedDotTest#_random_key.
  RaggedDotTest: RaggedDotTest#
  RaggedDotTest.setUp: RaggedDotTest#setUp().
---
# Module: [`tests/_src/core/ragged_dot_test.py`](../../../../../../../raw/code/qwix/tests/_src/core/ragged_dot_test.py)

## Classes
### `RaggedDotTest`  ·  implements/extends TestCase
- def: [`tests/_src/core/ragged_dot_test.py:31`](../../../../../../../raw/code/qwix/tests/_src/core/ragged_dot_test.py#L31)
- signature: `class RaggedDotTest(parameterized.TestCase):`
- members:
  - `setUp(self)` — [`L33`](../../../../../../../raw/code/qwix/tests/_src/core/ragged_dot_test.py#L33)
  - `test_ragged_dot(self, *, lhs_shape: tuple[int, ...], lhs_how: qarray.HowToQuantize | None, rhs_shape: tuple[int, ...], rhs_how: qarray.HowToQuantize | None, group_sizes: tuple[int, ...], expected_mae: float)` — [`L84`](../../../../../../../raw/code/qwix/tests/_src/core/ragged_dot_test.py#L84)
  - `test_ragged_dot_general(self, *, lhs_shape: tuple[int, ...], lhs_how: qarray.HowToQuantize | None, rhs_shape: tuple[int, ...], rhs_how: qarray.HowToQuantize | None, group_sizes: tuple[int, ...], expected_mae: float, disable_fast_ragged_dot: bool = False, dot_dimension_numbers: jax.lax.RaggedDotDimensionNumbers = ragged_dot._BASIC_RAGGED_DOT_DIMENSION_NUMBERS)` — [`L277`](../../../../../../../raw/code/qwix/tests/_src/core/ragged_dot_test.py#L277)
  - `test_ragged_dot_general_implementation(self, mock_fast, mock_slow, *, lhs_how: qarray.HowToQuantize | None, rhs_how: qarray.HowToQuantize | None, expect_fast: bool)` — [`L420`](../../../../../../../raw/code/qwix/tests/_src/core/ragged_dot_test.py#L420)
- protocol/private: `_jitted_ragged_dot`[`L101`](../../../../../../../raw/code/qwix/tests/_src/core/ragged_dot_test.py#L101), `_jitted_ragged_dot_general`[`L310`](../../../../../../../raw/code/qwix/tests/_src/core/ragged_dot_test.py#L310), `_make_array`[`L37`](../../../../../../../raw/code/qwix/tests/_src/core/ragged_dot_test.py#L37), `_random_key`[`L35`](../../../../../../../raw/code/qwix/tests/_src/core/ragged_dot_test.py#L35)
- uses (calls/refs, reference-scoped): [`HowToQuantize`](../../../qwix/_src/core/qarray.md#HowToQuantize), [`qtype`](../../../qwix/_src/core/qarray.md#HowToQuantize.qtype), [`quantize`](../../../qwix/_src/core/qarray.md#quantize), [`tiled_axes`](../../../qwix/_src/core/qarray.md#HowToQuantize.tiled_axes), [`channelwise_axes`](../../../qwix/_src/core/qarray.md#HowToQuantize.channelwise_axes), [`calibration_method`](../../../qwix/_src/core/qarray.md#HowToQuantize.calibration_method), [`_fast_ragged_dot_general`](../../../qwix/_src/core/ragged_dot.md#_fast_ragged_dot_general), [`ragged_dot_general`](../../../qwix/_src/core/ragged_dot.md#ragged_dot_general), [`ragged_dot`](../../../qwix/_src/core/ragged_dot.md#ragged_dot), [`_slow_ragged_dot_general`](../../../qwix/_src/core/ragged_dot.md#_slow_ragged_dot_general), [`_BASIC_RAGGED_DOT_DIMENSION_NUMBERS`](../../../qwix/_src/core/ragged_dot.md#_BASIC_RAGGED_DOT_DIMENSION_NUMBERS)  (1 test-only)

## Functions
- `rel_mae(a, b)` — [`L26`](../../../../../../../raw/code/qwix/tests/_src/core/ragged_dot_test.py#L26)


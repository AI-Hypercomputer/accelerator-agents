---
title: 'Module: tests/_src/core/dot_general_tpu_test.py'
type: catalog
provenance: extracted
module: tests/_src/core/dot_general_tpu_test.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `tests._src.core.dot_general_tpu_test`/
symbols:
  DotGeneralTest.test_dot_general: DotGeneralTest#test_dot_general().
  DotGeneralTest._multi_dot_general: DotGeneralTest#_multi_dot_general().
  DotGeneralTest._make_array: DotGeneralTest#_make_array().
  rel_mae: rel_mae().
  DotGeneralTest._random_key: DotGeneralTest#_random_key.
  DotGeneralTest: DotGeneralTest#
  DotGeneralTest.setUp: DotGeneralTest#setUp().
---
# Module: [`tests/_src/core/dot_general_tpu_test.py`](../../../../../../../raw/code/qwix/tests/_src/core/dot_general_tpu_test.py)

## Classes
### `DotGeneralTest`  ·  implements/extends TestCase
- def: [`tests/_src/core/dot_general_tpu_test.py:29`](../../../../../../../raw/code/qwix/tests/_src/core/dot_general_tpu_test.py#L29)
- doc: More expensive TPU tests for dot_general, mainly on numerics.
- signature: `class DotGeneralTest(parameterized.TestCase):`
- members:
  - `setUp(self)` — [`L32`](../../../../../../../raw/code/qwix/tests/_src/core/dot_general_tpu_test.py#L32)
  - `test_dot_general(self, *, lhs_shape: tuple[int, ...], lhs_qtype: jax.typing.DTypeLike | None, lhs_tile_sizes: tuple[int | None, ...] = (), lhs_asymmetric: bool = False, rhs_shape: tuple[int, ...], rhs_qtype: jax.typing.DTypeLike | None, rhs_tile_sizes: tuple[int | None, ...] = (), rhs_asymmetric: bool = False, dimension_numbers: jax.lax.DotDimensionNumbers, expected_mae: float, disable_fast_dot_general: bool = False, disable_loop_dot_general: bool = False)` — [`L149`](../../../../../../../raw/code/qwix/tests/_src/core/dot_general_tpu_test.py#L149)
- protocol/private: `_make_array`[`L36`](../../../../../../../raw/code/qwix/tests/_src/core/dot_general_tpu_test.py#L36), `_multi_dot_general`[`L193`](../../../../../../../raw/code/qwix/tests/_src/core/dot_general_tpu_test.py#L193), `_random_key`[`L34`](../../../../../../../raw/code/qwix/tests/_src/core/dot_general_tpu_test.py#L34)
- uses (calls/refs, reference-scoped): [`HowToQuantize`](../../../qwix/_src/core/qarray.md#HowToQuantize), [`qtype`](../../../qwix/_src/core/qarray.md#HowToQuantize.qtype), [`quantize`](../../../qwix/_src/core/qarray.md#quantize), [`tiled_axes`](../../../qwix/_src/core/qarray.md#HowToQuantize.tiled_axes), [`channelwise_axes`](../../../qwix/_src/core/qarray.md#HowToQuantize.channelwise_axes), [`calibration_method`](../../../qwix/_src/core/qarray.md#HowToQuantize.calibration_method), [`_fast_dot_general`](../../../qwix/_src/core/dot_general.md#_fast_dot_general), [`loop_dot_general`](../../../qwix/_src/core/dot_general.md#loop_dot_general), [`_slow_dot_general`](../../../qwix/_src/core/dot_general.md#_slow_dot_general)  (1 test-only)

## Functions
- `rel_mae(x, y)` — [`L24`](../../../../../../../raw/code/qwix/tests/_src/core/dot_general_tpu_test.py#L24)


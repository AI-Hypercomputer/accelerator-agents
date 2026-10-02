---
title: 'Module: tests/_src/core/dot_general_test.py'
type: catalog
provenance: extracted
module: tests/_src/core/dot_general_test.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `tests._src.core.dot_general_test`/DotGeneralTest#
symbols:
  DotGeneralTest.test_output_dtype: test_output_dtype().
  DotGeneralTest.test_outer_product: test_outer_product().
  DotGeneralTest.test_innermost_tiling_heuristic: test_innermost_tiling_heuristic().
  DotGeneralTest: ''
---
# Module: [`tests/_src/core/dot_general_test.py`](../../../../../../../raw/code/qwix/tests/_src/core/dot_general_test.py)

## Classes
### `DotGeneralTest`  ·  implements/extends TestCase
- def: [`tests/_src/core/dot_general_test.py:26`](../../../../../../../raw/code/qwix/tests/_src/core/dot_general_test.py#L26)
- doc: Small-scale CPU tests for dot_general which doesn't cover numerics.
- signature: `class DotGeneralTest(parameterized.TestCase):`
- members:
  - `test_innermost_tiling_heuristic(self)` — [`L185`](../../../../../../../raw/code/qwix/tests/_src/core/dot_general_test.py#L185) — Verifies that multi-dimensional dot_general picks the innermost contracting reduction axis.
  - `test_outer_product(self, mock_devices, mock_scaled_matmul)` — [`L151`](../../../../../../../raw/code/qwix/tests/_src/core/dot_general_test.py#L151)
  - `test_output_dtype(self, lhs_dtype, rhs_dtype, expected_output_dtype)` — [`L73`](../../../../../../../raw/code/qwix/tests/_src/core/dot_general_test.py#L73)
- uses (calls/refs, reference-scoped): [`QArray`](../../../qwix/_src/core/qarray.md#QArray), [`tiled_axes`](../../../qwix/_src/core/qarray.md#HowToQuantize.tiled_axes), [`dot_general`](../../../qwix/_src/core/dot_general.md#dot_general), [`_fast_dot_general`](../../../qwix/_src/core/dot_general.md#_fast_dot_general), [`loop_dot_general`](../../../qwix/_src/core/dot_general.md#loop_dot_general), [`einsum`](../../../qwix/_src/core/einsum.md#einsum), [`get_how_to_quantize`](../../../qwix/_src/core/dot_general.md#get_how_to_quantize), [`qtype`](../../../qwix/_src/core/qarray.md#QArray.qtype), [`_slow_dot_general`](../../../qwix/_src/core/dot_general.md#_slow_dot_general), [`_get_primary_platform`](../../../qwix/_src/core/mxfp_dot.md#_get_primary_platform)


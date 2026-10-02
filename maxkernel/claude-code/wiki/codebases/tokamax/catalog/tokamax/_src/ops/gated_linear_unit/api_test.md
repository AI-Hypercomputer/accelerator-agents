---
title: 'Module: tokamax/_src/ops/gated_linear_unit/api_test.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/gated_linear_unit/api_test.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.gated_linear_unit.api_test`/
symbols:
  GatedLinearUnitTest.test_basic_api: GatedLinearUnitTest#test_basic_api().
  GatedLinearUnitTritonTest.__init__: GatedLinearUnitTritonTest#__init__().
  GatedLinearUnitXlaTest.__init__: GatedLinearUnitXlaTest#__init__().
  GatedLinearUnitTest.f: GatedLinearUnitTest#f().
  GatedLinearUnitTest.f_xla: GatedLinearUnitTest#f_xla().
  _IMPLEMENTATIONS._IMPLEMENTATIONS: _IMPLEMENTATIONS._IMPLEMENTATIONS.
  GatedLinearUnitTritonTest: GatedLinearUnitTritonTest#
  GatedLinearUnitXlaTest: GatedLinearUnitXlaTest#
  _get_input_data: _get_input_data().
  GatedLinearUnitTest: GatedLinearUnitTest#
---
# Module: [`tokamax/_src/ops/gated_linear_unit/api_test.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/api_test.py)

## Classes
### `GatedLinearUnitTest`  ·  implements/extends TestCase
- def: [`tokamax/_src/ops/gated_linear_unit/api_test.py:40`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/api_test.py#L40)
- signature: `class GatedLinearUnitTest(parameterized.TestCase):`
- members:
  - `f(x, weights)` — [`L57`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/api_test.py#L57)
  - `f_xla(x, weights)` — [`L64`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/api_test.py#L64)
  - `test_basic_api(self, implementation, use_tuple_weights)` — [`L46`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/api_test.py#L46)
- uses (calls/refs, reference-scoped): [`op`](../op.md#BoundArguments.op), [`arguments`](../op.md#BoundArguments.arguments), [`get_bound_args`](../../autotuning/api.md#get_bound_args), [`gated_linear_unit`](api.md#gated_linear_unit), [`IMPLEMENTATIONS`](api.md#IMPLEMENTATIONS.IMPLEMENTATIONS), [`_IMPLEMENTATIONS`](api_test.md#_IMPLEMENTATIONS._IMPLEMENTATIONS), [`_get_input_data`](api_test.md#_get_input_data)

### `GatedLinearUnitTritonTest`  ·  implements/extends GatedLinearUnitTestBase
- def: [`tokamax/_src/ops/gated_linear_unit/api_test.py:101`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/api_test.py#L101)
- signature: `class GatedLinearUnitTritonTest(test_base.GatedLinearUnitTestBase):`
- protocol/private: `__init__`[`L103`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/api_test.py#L103)
- uses (calls/refs, reference-scoped): [`gated_linear_unit`](api.md#gated_linear_unit)  (2 test-only)
- used by: (1 test-only callers)

### `GatedLinearUnitXlaTest`  ·  implements/extends GatedLinearUnitTestBase
- def: [`tokamax/_src/ops/gated_linear_unit/api_test.py:108`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/api_test.py#L108)
- signature: `class GatedLinearUnitXlaTest(test_base.GatedLinearUnitTestBase):`
- protocol/private: `__init__`[`L110`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/api_test.py#L110)
- uses (calls/refs, reference-scoped): [`gated_linear_unit`](api.md#gated_linear_unit)  (2 test-only)
- used by: (1 test-only callers)

## Functions
- `_get_input_data(m, k, n, dtype=jnp.bfloat16)` — [`L33`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/api_test.py#L33)

## Module values
- `_IMPLEMENTATIONS` — [`L28`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/api_test.py#L28)


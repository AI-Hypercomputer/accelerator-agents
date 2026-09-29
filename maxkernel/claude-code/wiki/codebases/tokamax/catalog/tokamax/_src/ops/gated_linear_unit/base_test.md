---
title: 'Module: tokamax/_src/ops/gated_linear_unit/base_test.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/gated_linear_unit/base_test.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.gated_linear_unit.base_test`/GatedLinearUnitTest#
symbols:
  GatedLinearUnitTest.test_explicit_vjp: test_explicit_vjp().
  GatedLinearUnitTest.__init__: __init__().
  GatedLinearUnitTest: ''
  GatedLinearUnitTest.f_vjp: f_vjp().
  GatedLinearUnitTest.f_vjp_ref: f_vjp_ref().
---
# Module: [`tokamax/_src/ops/gated_linear_unit/base_test.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/base_test.py)

## Classes
### `GatedLinearUnitTest`  ·  implements/extends GatedLinearUnitTestBase
- def: [`tokamax/_src/ops/gated_linear_unit/base_test.py:29`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/base_test.py#L29)
- signature: `class GatedLinearUnitTest(test_base.GatedLinearUnitTestBase):`
- members:
  - `f_vjp(x, w, do)` — [`L62`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/base_test.py#L62)
  - `f_vjp_ref(x, w, do)` — [`L67`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/base_test.py#L67)
  - `test_explicit_vjp(self, dtype, batch_dim, precision)` — [`L45`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/base_test.py#L45)
- protocol/private: `__init__`[`L31`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/base_test.py#L31)
- uses (calls/refs, reference-scoped): [`GatedLinearUnit`](base.md#GatedLinearUnit), [`GatedLinearUnitVjp`](base.md#GatedLinearUnitVjp)  (2 test-only)
- used by: (1 test-only callers)


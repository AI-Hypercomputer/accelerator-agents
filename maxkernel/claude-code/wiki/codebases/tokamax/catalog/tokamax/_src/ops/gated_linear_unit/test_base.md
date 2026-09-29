---
title: 'Module: tokamax/_src/ops/gated_linear_unit/test_base.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/gated_linear_unit/test_base.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.gated_linear_unit.test_base`/GatedLinearUnitTestBase#
symbols:
  GatedLinearUnitTestBase: ''
  GatedLinearUnitTestBase._glu_fn: _glu_fn.
  GatedLinearUnitTestBase.__init__: __init__().
  GatedLinearUnitTestBase.test_gated_linear_unit: test_gated_linear_unit().
  GatedLinearUnitTestBase.f_ref: f_ref().
---
# Module: [`tokamax/_src/ops/gated_linear_unit/test_base.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/test_base.py)

## Classes
### `GatedLinearUnitTestBase`  ·  implements/extends TestCase
- def: [`tokamax/_src/ops/gated_linear_unit/test_base.py:27`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/test_base.py#L27)
- doc: Base class for Gated Linear Unit op tests.
- signature: `class GatedLinearUnitTestBase(parameterized.TestCase):`
- members:
  - `f_ref(x, weights)` — [`L81`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/test_base.py#L81)
  - `test_gated_linear_unit(self, *, batch=(), m, n, k, activation=None, use_tuple_weights=False)` — [`L40`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/test_base.py#L40)
- protocol/private: `__init__`[`L30`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/test_base.py#L30), `_glu_fn`[`L32`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/test_base.py#L32)
- uses (calls/refs, reference-scoped): [`GatedLinearUnitTest`](base_test.md#GatedLinearUnitTest), [`GatedLinearUnitTritonTest`](api_test.md#GatedLinearUnitTritonTest), [`GatedLinearUnitXlaTest`](api_test.md#GatedLinearUnitXlaTest), [`PallasMosaicGpuGatedLinearUnitTest`](pallas_mosaic_gpu_test.md#PallasMosaicGpuGatedLinearUnitTest), [`PallasTritonGatedLinearUnitTest`](pallas_triton_test.md#PallasTritonGatedLinearUnitTest)
- used by: [`__init__`](api_test.md#GatedLinearUnitTritonTest.__init__), [`__init__`](api_test.md#GatedLinearUnitXlaTest.__init__), [`__init__`](base_test.md#GatedLinearUnitTest.__init__), [`__init__`](pallas_mosaic_gpu_test.md#PallasMosaicGpuGatedLinearUnitTest.__init__), [`__init__`](pallas_triton_test.md#PallasTritonGatedLinearUnitTest.__init__), [`GatedLinearUnitTest`](base_test.md#GatedLinearUnitTest), [`GatedLinearUnitTritonTest`](api_test.md#GatedLinearUnitTritonTest), [`GatedLinearUnitXlaTest`](api_test.md#GatedLinearUnitXlaTest), [`PallasMosaicGpuGatedLinearUnitTest`](pallas_mosaic_gpu_test.md#PallasMosaicGpuGatedLinearUnitTest), [`PallasTritonGatedLinearUnitTest`](pallas_triton_test.md#PallasTritonGatedLinearUnitTest), [`test_autotuning_configs`](pallas_mosaic_gpu_test.md#PallasMosaicGpuGatedLinearUnitTest.test_autotuning_configs)


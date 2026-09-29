---
title: 'Module: tokamax/_src/ops/normalization/test_base.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/normalization/test_base.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.normalization.test_base`/N
symbols:
  NormalizationTestBase: ormalizationTestBase#
  NormalizationTestBase._run_test: ormalizationTestBase#_run_test().
  NormalizationTestBase.test_bench: ormalizationTestBase#test_bench().
  NormalizationTestBase._test_layer_norm_vmap: ormalizationTestBase#_test_layer_norm_vmap().
  NormalizationTestBase.__init__: ormalizationTestBase#__init__().
  NormalizationTestBase._norm_fn: ormalizationTestBase#_norm_fn.
  NormalizationTestBase.f_ref: ormalizationTestBase#f_ref().
  NAMED_ARG_SPECS: AMED_ARG_SPECS.
  NormalizationTestBase.test_layer_norm: ormalizationTestBase#test_layer_norm().
  NormalizationTestBase.test_layer_norm_axis: ormalizationTestBase#test_layer_norm_axis().
  NormalizationTestBase.test_layer_norm_vmap: ormalizationTestBase#test_layer_norm_vmap().
  NormalizationTestBase.test_rms_norm: ormalizationTestBase#test_rms_norm().
  NormalizationTestBase._supports_vjp: ormalizationTestBase#_supports_vjp.
---
# Module: [`tokamax/_src/ops/normalization/test_base.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/test_base.py)

## Classes
### `NormalizationTestBase`  ·  implements/extends TestCase
- def: [`tokamax/_src/ops/normalization/test_base.py:35`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/test_base.py#L35)
- doc: Base class for normalization op tests.
- signature: `class NormalizationTestBase(parameterized.TestCase):`
- members:
  - `f_ref(x, scale, offset)` — [`L147`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/test_base.py#L147)
  - `test_bench(self, kwargs)` — [`L192`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/test_base.py#L192)
  - `test_layer_norm(self, shape, use_scale_and_offset, scale_offset)` — [`L51`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/test_base.py#L51)
  - `test_layer_norm_axis(self, axis)` — [`L69`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/test_base.py#L69)
  - `test_layer_norm_vmap(self, axis, vmap_in_axes)` — [`L87`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/test_base.py#L87)
  - `test_rms_norm(self, shape)` — [`L112`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/test_base.py#L112)
- protocol/private: `__init__`[`L38`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/test_base.py#L38), `_norm_fn`[`L40`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/test_base.py#L40), `_run_test`[`L120`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/test_base.py#L120), `_supports_vjp`[`L41`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/test_base.py#L41), `_test_layer_norm_vmap`[`L90`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/test_base.py#L90)
- uses (calls/refs, reference-scoped): [`Normalization`](base.md#Normalization), [`__call__`](../op.md#Op.__call__), [`_test_layer_norm_vmap`](pallas_triton_test.md#PallasTritonNormalizationTest._test_layer_norm_vmap), [`LayerNormTritonTest`](api_test.md#LayerNormTritonTest), [`LayerNormXlATest`](api_test.md#LayerNormXlATest), [`NormalizationTest`](base_test.md#NormalizationTest), [`PallasTritonNormalizationTest`](pallas_triton_test.md#PallasTritonNormalizationTest)  (1 test-only)
- used by: [`__init__`](api_test.md#LayerNormTritonTest.__init__), [`__init__`](api_test.md#LayerNormXlATest.__init__), [`_test_layer_norm_vmap`](pallas_triton_test.md#PallasTritonNormalizationTest._test_layer_norm_vmap), [`__init__`](base_test.md#NormalizationTest.__init__), [`__init__`](pallas_triton_test.md#PallasTritonNormalizationTest.__init__), [`f`](pallas_triton_test.md#PallasTritonNormalizationTest.f), [`LayerNormTritonTest`](api_test.md#LayerNormTritonTest), [`LayerNormXlATest`](api_test.md#LayerNormXlATest), [`NormalizationTest`](base_test.md#NormalizationTest), [`PallasTritonNormalizationTest`](pallas_triton_test.md#PallasTritonNormalizationTest), [`test_layer_norm_with_pre_scale`](pallas_triton_test.md#PallasTritonNormalizationTest.test_layer_norm_with_pre_scale), [`test_remat`](pallas_triton_test.md#PallasTritonNormalizationTest.test_remat)

## Module values
- `NAMED_ARG_SPECS` — [`L29`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/test_base.py#L29)


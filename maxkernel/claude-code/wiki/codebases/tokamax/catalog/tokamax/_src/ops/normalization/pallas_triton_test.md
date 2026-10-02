---
title: 'Module: tokamax/_src/ops/normalization/pallas_triton_test.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/normalization/pallas_triton_test.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.normalization.pallas_triton_test`/PallasTritonNormalizationTest#
symbols:
  PallasTritonNormalizationTest._test_layer_norm_vmap: _test_layer_norm_vmap().
  PallasTritonNormalizationTest.__init__: __init__().
  PallasTritonNormalizationTest.f: f().
  PallasTritonNormalizationTest: ''
  PallasTritonNormalizationTest.test_layer_norm_with_pre_scale: test_layer_norm_with_pre_scale().
  PallasTritonNormalizationTest.test_remat: test_remat().
  PallasTritonNormalizationTest.test_remat_with_vmap: test_remat_with_vmap().
  PallasTritonNormalizationTest.my_heuristics_config: my_heuristics_config().
  PallasTritonNormalizationTest.setUp: setUp().
---
# Module: [`tokamax/_src/ops/normalization/pallas_triton_test.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton_test.py)

## Classes
### `PallasTritonNormalizationTest`  ·  implements/extends NormalizationTestBase
- def: [`tokamax/_src/ops/normalization/pallas_triton_test.py:28`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton_test.py#L28)
- signature: `class PallasTritonNormalizationTest(test_base.NormalizationTestBase):`
- members:
  - `f(x, scale, offset)` — [`L110`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton_test.py#L110)
  - `my_heuristics_config(*args, **kwargs)` — [`L65`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton_test.py#L65)
  - `setUp(self)` — [`L33`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton_test.py#L33)
  - `test_layer_norm_with_pre_scale(self)` — [`L38`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton_test.py#L38)
  - `test_remat(self)` — [`L79`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton_test.py#L79)
  - `test_remat_with_vmap(self)` — [`L101`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton_test.py#L101)
- protocol/private: `__init__`[`L30`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton_test.py#L30), `_test_layer_norm_vmap`[`L56`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton_test.py#L56)
- uses (calls/refs, reference-scoped): [`get_heuristics_config`](pallas_triton_config.md#get_heuristics_config), [`PallasTritonNormalization`](pallas_triton.md#PallasTritonNormalization)  (4 test-only)
- used by: (2 test-only callers)


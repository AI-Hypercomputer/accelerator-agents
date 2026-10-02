---
title: 'Module: tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_test.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_test.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.ragged_dot.pallas_mosaic_gpu_test`/
symbols:
  PallasMosaicGpuRaggedDotTest.fn: PallasMosaicGpuRaggedDotTest#fn().
  _CONFIG: _CONFIG.
  PallasMosaicGpuRaggedDotTest.__init__: PallasMosaicGpuRaggedDotTest#__init__().
  PallasMosaicGpuRaggedDotTest: PallasMosaicGpuRaggedDotTest#
  PallasMosaicGpuRaggedDotTest.setUp: PallasMosaicGpuRaggedDotTest#setUp().
---
# Module: [`tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_test.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_test.py)

## Classes
### `PallasMosaicGpuRaggedDotTest`  ·  implements/extends RaggedDotTestBase
- def: [`tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_test.py:38`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_test.py#L38)
- signature: `class PallasMosaicGpuRaggedDotTest(test_base.RaggedDotTestBase):`
- members:
  - `fn(lhs, rhs, *, config=None, **kwargs)` — [`L43`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_test.py#L43)
  - `setUp(self)` — [`L97`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_test.py#L97)
- protocol/private: `__init__`[`L40`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_test.py#L40)
- uses (calls/refs, reference-scoped): [`block_k`](pallas_mosaic_gpu_common.md#Config.block_k), [`block_m`](pallas_mosaic_gpu_common.md#Config.block_m), [`Config`](pallas_mosaic_gpu.md#Config), [`block_n`](pallas_mosaic_gpu_common.md#Config.block_n), [`num_stages`](pallas_mosaic_gpu_common.md#Config.num_stages), [`_CONFIG`](pallas_mosaic_gpu_test.md#_CONFIG), [`replace`](../op.md#Op.replace), [`split_k`](pallas_mosaic_gpu_common.md#Config.split_k), [`PallasMosaicGpuRaggedDot`](pallas_mosaic_gpu.md#PallasMosaicGpuRaggedDot)  (2 test-only)
- used by: (1 test-only callers)

## Module values
- `_CONFIG` — [`L29`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_test.py#L29)


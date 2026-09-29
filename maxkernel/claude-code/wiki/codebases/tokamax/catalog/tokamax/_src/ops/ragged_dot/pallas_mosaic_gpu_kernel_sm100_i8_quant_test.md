---
title: 'Module: tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant_test.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant_test.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.ragged_dot.pallas_mosaic_gpu_kernel_sm100_i8_quant_test`/
symbols:
  _CONFIG: _CONFIG.
  PallasMosaicGpuKernelSm100I8QuantTest.test_group_sizes: PallasMosaicGpuKernelSm100I8QuantTest#test_group_sizes().
  PallasMosaicGpuKernelSm100I8QuantTest.fn: PallasMosaicGpuKernelSm100I8QuantTest#fn().
  PallasMosaicGpuKernelSm100I8QuantTest.__init__: PallasMosaicGpuKernelSm100I8QuantTest#__init__().
  PallasMosaicGpuKernelSm100I8QuantTest.test_padded: PallasMosaicGpuKernelSm100I8QuantTest#test_padded().
  PallasMosaicGpuKernelSm100I8QuantTest.test_zero_group_sizes: PallasMosaicGpuKernelSm100I8QuantTest#test_zero_group_sizes().
  PallasMosaicGpuKernelSm100I8QuantTest.test_wi4_ai8_quantized: PallasMosaicGpuKernelSm100I8QuantTest#test_wi4_ai8_quantized().
  PallasMosaicGpuKernelSm100I8QuantTest: PallasMosaicGpuKernelSm100I8QuantTest#
  PallasMosaicGpuKernelSm100I8QuantTest._test_quantized: PallasMosaicGpuKernelSm100I8QuantTest#_test_quantized().
  PallasMosaicGpuKernelSm100I8QuantTest._test_preferred_element_type: PallasMosaicGpuKernelSm100I8QuantTest#_test_preferred_element_type().
  PallasMosaicGpuKernelSm100I8QuantTest._test_vjp: PallasMosaicGpuKernelSm100I8QuantTest#_test_vjp().
  PallasMosaicGpuKernelSm100I8QuantTest._test_bench: PallasMosaicGpuKernelSm100I8QuantTest#_test_bench().
  PallasMosaicGpuKernelSm100I8QuantTest._test_simple: PallasMosaicGpuKernelSm100I8QuantTest#_test_simple().
  PallasMosaicGpuKernelSm100I8QuantTest.setUp: PallasMosaicGpuKernelSm100I8QuantTest#setUp().
---
# Module: [`tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant_test.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant_test.py)

## Classes
### `PallasMosaicGpuKernelSm100I8QuantTest`  ·  implements/extends RaggedDotTestBase
- def: [`tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant_test.py:44`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant_test.py#L44)
- signature: `class PallasMosaicGpuKernelSm100I8QuantTest(test_base.RaggedDotTestBase):`
- members:
  - `fn(lhs, rhs, *, config=None, **kwargs)` — [`L49`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant_test.py#L49)
  - `setUp(self)` — [`L209`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant_test.py#L209)
  - `test_group_sizes(self)` — [`L161`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant_test.py#L161)
  - `test_padded(self)` — [`L138`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant_test.py#L138)
  - `test_wi4_ai8_quantized(self, tile_shape, use_as_qarray, activation, task)` — [`L110`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant_test.py#L110)
  - `test_zero_group_sizes(self)` — [`L182`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant_test.py#L182)
- protocol/private: `__init__`[`L46`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant_test.py#L46), `_test_bench`[`L130`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant_test.py#L130), `_test_preferred_element_type`[`L122`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant_test.py#L122), `_test_quantized`[`L88`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant_test.py#L88), `_test_simple`[`L134`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant_test.py#L134), `_test_vjp`[`L126`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant_test.py#L126)
- uses (calls/refs, reference-scoped): [`_CONFIG`](pallas_mosaic_gpu_kernel_sm100_i8_quant_test.md#_CONFIG), [`block_k`](pallas_mosaic_gpu_common.md#Config.block_k), [`GroupSizes`](base.md#GroupSizes), [`replace`](../op.md#Op.replace), [`PallasMosaicGpuRaggedDot`](pallas_mosaic_gpu.md#PallasMosaicGpuRaggedDot)  (7 test-only)
- used by: (9 test-only callers)

## Module values
- `_CONFIG` — [`L31`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_i8_quant_test.py#L31)


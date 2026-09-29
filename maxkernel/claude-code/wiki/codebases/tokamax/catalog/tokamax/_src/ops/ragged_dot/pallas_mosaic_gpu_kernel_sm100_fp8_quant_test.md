---
title: 'Module: tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_fp8_quant_test.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_fp8_quant_test.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.ragged_dot.pallas_mosaic_gpu_kernel_sm100_fp8_quant_test`/
symbols:
  _CONFIG: _CONFIG.
  PallasMosaicGpuKernelSm100FP8QuantTest.fn: PallasMosaicGpuKernelSm100FP8QuantTest#fn().
  PallasMosaicGpuKernelSm100FP8QuantTest.test_wi4_afp8_quantized: PallasMosaicGpuKernelSm100FP8QuantTest#test_wi4_afp8_quantized().
  PallasMosaicGpuKernelSm100FP8QuantTest.test_group_sizes: PallasMosaicGpuKernelSm100FP8QuantTest#test_group_sizes().
  PallasMosaicGpuKernelSm100FP8QuantTest.__init__: PallasMosaicGpuKernelSm100FP8QuantTest#__init__().
  PallasMosaicGpuKernelSm100FP8QuantTest.test_padded: PallasMosaicGpuKernelSm100FP8QuantTest#test_padded().
  PallasMosaicGpuKernelSm100FP8QuantTest.test_zero_group_sizes: PallasMosaicGpuKernelSm100FP8QuantTest#test_zero_group_sizes().
  PallasMosaicGpuKernelSm100FP8QuantTest: PallasMosaicGpuKernelSm100FP8QuantTest#
  PallasMosaicGpuKernelSm100FP8QuantTest._test_quantized: PallasMosaicGpuKernelSm100FP8QuantTest#_test_quantized().
  PallasMosaicGpuKernelSm100FP8QuantTest._test_preferred_element_type: PallasMosaicGpuKernelSm100FP8QuantTest#_test_preferred_element_type().
  PallasMosaicGpuKernelSm100FP8QuantTest._test_vjp: PallasMosaicGpuKernelSm100FP8QuantTest#_test_vjp().
  PallasMosaicGpuKernelSm100FP8QuantTest._test_bench: PallasMosaicGpuKernelSm100FP8QuantTest#_test_bench().
  PallasMosaicGpuKernelSm100FP8QuantTest._test_simple: PallasMosaicGpuKernelSm100FP8QuantTest#_test_simple().
  PallasMosaicGpuKernelSm100FP8QuantTest.setUp: PallasMosaicGpuKernelSm100FP8QuantTest#setUp().
---
# Module: [`tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_fp8_quant_test.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_fp8_quant_test.py)

## Classes
### `PallasMosaicGpuKernelSm100FP8QuantTest`  ·  implements/extends RaggedDotTestBase
- def: [`tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_fp8_quant_test.py:44`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_fp8_quant_test.py#L44)
- doc: Tests for Pallas Mosaic GPU kernel with fp8xi4 quantization.
- signature: `class PallasMosaicGpuKernelSm100FP8QuantTest(test_base.RaggedDotTestBase):`
- members:
  - `fn(lhs, rhs, **kwargs)` — [`L50`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_fp8_quant_test.py#L50)
  - `setUp(self)` — [`L224`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_fp8_quant_test.py#L224)
  - `test_group_sizes(self)` — [`L176`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_fp8_quant_test.py#L176)
  - `test_padded(self)` — [`L153`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_fp8_quant_test.py#L153)
  - `test_wi4_afp8_quantized(self, subchannels, use_as_qarray, activation, task, block_m, block_k)` — [`L117`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_fp8_quant_test.py#L117)
  - `test_zero_group_sizes(self)` — [`L197`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_fp8_quant_test.py#L197)
- protocol/private: `__init__`[`L47`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_fp8_quant_test.py#L47), `_test_bench`[`L145`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_fp8_quant_test.py#L145), `_test_preferred_element_type`[`L137`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_fp8_quant_test.py#L137), `_test_quantized`[`L90`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_fp8_quant_test.py#L90), `_test_simple`[`L149`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_fp8_quant_test.py#L149), `_test_vjp`[`L141`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_fp8_quant_test.py#L141)
- uses (calls/refs, reference-scoped): [`_CONFIG`](pallas_mosaic_gpu_kernel_sm100_fp8_quant_test.md#_CONFIG), [`block_k`](pallas_mosaic_gpu_common.md#Config.block_k), [`GroupSizes`](base.md#GroupSizes), [`replace`](../op.md#Op.replace), [`PallasMosaicGpuRaggedDot`](pallas_mosaic_gpu.md#PallasMosaicGpuRaggedDot)  (9 test-only)
- used by: (9 test-only callers)

## Module values
- `_CONFIG` — [`L31`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm100_fp8_quant_test.py#L31)


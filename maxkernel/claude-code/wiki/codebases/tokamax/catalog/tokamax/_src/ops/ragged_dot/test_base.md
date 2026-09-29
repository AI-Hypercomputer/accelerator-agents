---
title: 'Module: tokamax/_src/ops/ragged_dot/test_base.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/ragged_dot/test_base.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.ragged_dot.test_base`/
symbols:
  RaggedDotTestBase: RaggedDotTestBase#
  RaggedDotTestBase._test_bench: RaggedDotTestBase#_test_bench().
  RaggedDotTestBase._test_quantized: RaggedDotTestBase#_test_quantized().
  RaggedDotTestBase._test_simple: RaggedDotTestBase#_test_simple().
  ref: ref().
  RaggedDotTestBase._create_inputs: RaggedDotTestBase#_create_inputs().
  RaggedDotTestBase.test_group_sizes: RaggedDotTestBase#test_group_sizes().
  RaggedDotTestBase._test_preferred_element_type: RaggedDotTestBase#_test_preferred_element_type().
  RaggedDotTestBase._test_vjp: RaggedDotTestBase#_test_vjp().
  RaggedDotTestBase.test_padded: RaggedDotTestBase#test_padded().
  RaggedDotTestBase.test_zero_group_sizes: RaggedDotTestBase#test_zero_group_sizes().
  RaggedDotTestBase._dot_fn: RaggedDotTestBase#_dot_fn.
  RaggedDotTestBase._dot_fn_f32: RaggedDotTestBase#_dot_fn_f32().
  RaggedDotTestBase.__init__: RaggedDotTestBase#__init__().
  ConfigManager.__enter__: ConfigManager#__enter__().
  RaggedDotTestBase.test_invalid_group_sizes: RaggedDotTestBase#test_invalid_group_sizes().
  override_chex_args: override_chex_args().
  ConfigManager.__exit__: ConfigManager#__exit__().
  RaggedDotTestBase.test_quantized: RaggedDotTestBase#test_quantized().
  RaggedDotTestBase.test_vjp: RaggedDotTestBase#test_vjp().
  RaggedDotTestBase.test_bench: RaggedDotTestBase#test_bench().
  _dot_fn_f32: _dot_fn_f32().
  relu: relu().
  test_config: test_config.
  ARG_SPECS: ARG_SPECS.
  _jax_ragged_dot_f32: _jax_ragged_dot_f32.
  NAMED_ARG_SPECS: NAMED_ARG_SPECS.
  RaggedDotTestBase.test_simple: RaggedDotTestBase#test_simple().
  RaggedDotTestBase.test_preferred_element_type: RaggedDotTestBase#test_preferred_element_type().
  ConfigManager.prev_config: ConfigManager#prev_config.
  RaggedDotTestBase.quantize: RaggedDotTestBase#quantize().
  _dot_fn_f32.wrapped: _dot_fn_f32().wrapped().
  ConfigManager: ConfigManager#
  ConfigManager.config: ConfigManager#config.
  ConfigManager.__init__: ConfigManager#__init__().
---
# Module: [`tokamax/_src/ops/ragged_dot/test_base.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/test_base.py)

## Classes
### `ConfigManager`
- def: [`tokamax/_src/ops/ragged_dot/test_base.py:86`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/test_base.py#L86)
- signature: `class ConfigManager:`
- members:
  - `config` — [`L90`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/test_base.py#L90)
  - `prev_config` — [`L89`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/test_base.py#L89)
- protocol/private: `__enter__`[`L92`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/test_base.py#L92), `__exit__`[`L96`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/test_base.py#L96), `__init__`[`L88`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/test_base.py#L88)
- uses (calls/refs, reference-scoped): (1 test-only callers)
- used by: [`test_wi4_afp8_quantized`](pallas_mosaic_gpu_kernel_sm100_fp8_quant_test.md#PallasMosaicGpuKernelSm100FP8QuantTest.test_wi4_afp8_quantized)

### `RaggedDotTestBase`  ·  implements/extends TestCase
- def: [`tokamax/_src/ops/ragged_dot/test_base.py:101`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/test_base.py#L101)
- doc: Base class for ragged dot op tests.
- signature: `class RaggedDotTestBase(parameterized.TestCase):`
- members:
  - `quantize(x, dtype, tile_shape)` — [`L143`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/test_base.py#L143)
  - `test_bench(self, spec)` — [`L321`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/test_base.py#L321)
  - `test_group_sizes(self)` — [`L299`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/test_base.py#L299)
  - `test_invalid_group_sizes(self, group_sizes)` — [`L308`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/test_base.py#L308)
  - `test_padded(self)` — [`L166`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/test_base.py#L166)
  - `test_preferred_element_type(self, out_type)` — [`L249`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/test_base.py#L249)
  - `test_quantized(self, dtype, a_tile_shape, b_tile_shape, use_as_qarray, activation)` — [`L198`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/test_base.py#L198)
  - `test_simple(self, dtype)` — [`L163`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/test_base.py#L163)
  - `test_vjp(self, num_groups, m, k, n, activation=None)` — [`L270`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/test_base.py#L270)
  - `test_zero_group_sizes(self)` — [`L176`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/test_base.py#L176)
- protocol/private: `__init__`[`L104`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/test_base.py#L104), `_create_inputs`[`L112`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/test_base.py#L112), `_dot_fn`[`L106`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/test_base.py#L106), `_dot_fn_f32`[`L109`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/test_base.py#L109), `_test_bench`[`L324`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/test_base.py#L324), `_test_preferred_element_type`[`L252`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/test_base.py#L252), `_test_quantized`[`L210`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/test_base.py#L210), `_test_simple`[`L156`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/test_base.py#L156), `_test_vjp`[`L273`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/test_base.py#L273)
- uses (calls/refs, reference-scoped): [`RaggedDot`](base.md#RaggedDot), [`GroupSizes`](base.md#GroupSizes), [`RaggedDotImplementationTest`](api_test.md#RaggedDotImplementationTest), [`test_group_sizes`](pallas_mosaic_gpu_kernel_sm100_fp8_quant_test.md#PallasMosaicGpuKernelSm100FP8QuantTest.test_group_sizes), [`test_group_sizes`](pallas_mosaic_gpu_kernel_sm100_i8_quant_test.md#PallasMosaicGpuKernelSm100I8QuantTest.test_group_sizes), [`test_padded`](pallas_mosaic_gpu_kernel_sm100_fp8_quant_test.md#PallasMosaicGpuKernelSm100FP8QuantTest.test_padded), [`test_padded`](pallas_mosaic_gpu_kernel_sm100_i8_quant_test.md#PallasMosaicGpuKernelSm100I8QuantTest.test_padded), [`test_zero_group_sizes`](pallas_mosaic_gpu_kernel_sm100_fp8_quant_test.md#PallasMosaicGpuKernelSm100FP8QuantTest.test_zero_group_sizes), [`test_zero_group_sizes`](pallas_mosaic_gpu_kernel_sm100_i8_quant_test.md#PallasMosaicGpuKernelSm100I8QuantTest.test_zero_group_sizes), [`_test_bench`](pallas_mosaic_tpu_test.md#PallasMosaicTpuRaggedDotTest._test_bench), [`_test_quantized`](pallas_mosaic_tpu_test.md#PallasMosaicTpuRaggedDotTest._test_quantized), [`_test_simple`](pallas_triton_test.md#PallasTritonRaggedDotTest._test_simple), [`PallasMosaicGpuKernelSm100FP8QuantTest`](pallas_mosaic_gpu_kernel_sm100_fp8_quant_test.md#PallasMosaicGpuKernelSm100FP8QuantTest), [`PallasMosaicGpuKernelSm100I8QuantTest`](pallas_mosaic_gpu_kernel_sm100_i8_quant_test.md#PallasMosaicGpuKernelSm100I8QuantTest), [`PallasMosaicGpuRaggedDotTest`](pallas_mosaic_gpu_test.md#PallasMosaicGpuRaggedDotTest), [`PallasMosaicTpuRaggedDotTest`](pallas_mosaic_tpu_test.md#PallasMosaicTpuRaggedDotTest), [`PallasTritonRaggedDotTest`](pallas_triton_test.md#PallasTritonRaggedDotTest), [`RaggedDotTest`](base_test.md#RaggedDotTest), [`RaggedDotWithExplicitVjpTest`](base_test.md#RaggedDotWithExplicitVjpTest), [`_test_bench`](pallas_triton_test.md#PallasTritonRaggedDotTest._test_bench), [`_test_bench`](api_test.md#RaggedDotTritonTest._test_bench), [`_test_bench`](pallas_mosaic_gpu_kernel_sm100_fp8_quant_test.md#PallasMosaicGpuKernelSm100FP8QuantTest._test_bench), [`_test_bench`](pallas_mosaic_gpu_kernel_sm100_i8_quant_test.md#PallasMosaicGpuKernelSm100I8QuantTest._test_bench), [`_test_preferred_element_type`](pallas_mosaic_gpu_kernel_sm100_fp8_quant_test.md#PallasMosaicGpuKernelSm100FP8QuantTest._test_preferred_element_type), [`_test_preferred_element_type`](pallas_mosaic_gpu_kernel_sm100_i8_quant_test.md#PallasMosaicGpuKernelSm100I8QuantTest._test_preferred_element_type), [`_test_quantized`](pallas_mosaic_gpu_kernel_sm100_fp8_quant_test.md#PallasMosaicGpuKernelSm100FP8QuantTest._test_quantized), [`_test_quantized`](pallas_mosaic_gpu_kernel_sm100_i8_quant_test.md#PallasMosaicGpuKernelSm100I8QuantTest._test_quantized), [`_test_simple`](pallas_mosaic_gpu_kernel_sm100_fp8_quant_test.md#PallasMosaicGpuKernelSm100FP8QuantTest._test_simple), [`_test_simple`](pallas_mosaic_gpu_kernel_sm100_i8_quant_test.md#PallasMosaicGpuKernelSm100I8QuantTest._test_simple), [`_test_vjp`](pallas_mosaic_gpu_kernel_sm100_fp8_quant_test.md#PallasMosaicGpuKernelSm100FP8QuantTest._test_vjp), [`_test_vjp`](pallas_mosaic_gpu_kernel_sm100_i8_quant_test.md#PallasMosaicGpuKernelSm100I8QuantTest._test_vjp)  (4 test-only)
- used by: [`RaggedDotImplementationTest`](api_test.md#RaggedDotImplementationTest), [`__init__`](base_test.md#RaggedDotWithExplicitVjpTest.__init__), [`test_group_sizes`](pallas_mosaic_gpu_kernel_sm100_fp8_quant_test.md#PallasMosaicGpuKernelSm100FP8QuantTest.test_group_sizes), [`test_group_sizes`](pallas_mosaic_gpu_kernel_sm100_i8_quant_test.md#PallasMosaicGpuKernelSm100I8QuantTest.test_group_sizes), [`test_wi4_afp8_quantized`](pallas_mosaic_gpu_kernel_sm100_fp8_quant_test.md#PallasMosaicGpuKernelSm100FP8QuantTest.test_wi4_afp8_quantized), [`__init__`](api_test.md#RaggedDotImplementationTest.__init__), [`__init__`](pallas_mosaic_gpu_kernel_sm100_fp8_quant_test.md#PallasMosaicGpuKernelSm100FP8QuantTest.__init__), [`__init__`](pallas_mosaic_gpu_kernel_sm100_i8_quant_test.md#PallasMosaicGpuKernelSm100I8QuantTest.__init__), [`__init__`](pallas_mosaic_gpu_test.md#PallasMosaicGpuRaggedDotTest.__init__), [`test_padded`](pallas_mosaic_gpu_kernel_sm100_fp8_quant_test.md#PallasMosaicGpuKernelSm100FP8QuantTest.test_padded), [`test_padded`](pallas_mosaic_gpu_kernel_sm100_i8_quant_test.md#PallasMosaicGpuKernelSm100I8QuantTest.test_padded), [`test_zero_group_sizes`](pallas_mosaic_gpu_kernel_sm100_fp8_quant_test.md#PallasMosaicGpuKernelSm100FP8QuantTest.test_zero_group_sizes), [`test_zero_group_sizes`](pallas_mosaic_gpu_kernel_sm100_i8_quant_test.md#PallasMosaicGpuKernelSm100I8QuantTest.test_zero_group_sizes), [`__init__`](base_test.md#RaggedDotTest.__init__), [`__init__`](pallas_mosaic_tpu_test.md#PallasMosaicTpuRaggedDotTest.__init__), [`__init__`](pallas_triton_test.md#PallasTritonRaggedDotTest.__init__), [`_test_bench`](pallas_mosaic_tpu_test.md#PallasMosaicTpuRaggedDotTest._test_bench), [`_test_quantized`](pallas_mosaic_tpu_test.md#PallasMosaicTpuRaggedDotTest._test_quantized), [`_test_simple`](pallas_triton_test.md#PallasTritonRaggedDotTest._test_simple), [`test_wi4_ai8_quantized`](pallas_mosaic_gpu_kernel_sm100_i8_quant_test.md#PallasMosaicGpuKernelSm100I8QuantTest.test_wi4_ai8_quantized), [`PallasMosaicGpuKernelSm100FP8QuantTest`](pallas_mosaic_gpu_kernel_sm100_fp8_quant_test.md#PallasMosaicGpuKernelSm100FP8QuantTest), [`PallasMosaicGpuKernelSm100I8QuantTest`](pallas_mosaic_gpu_kernel_sm100_i8_quant_test.md#PallasMosaicGpuKernelSm100I8QuantTest), [`PallasMosaicGpuRaggedDotTest`](pallas_mosaic_gpu_test.md#PallasMosaicGpuRaggedDotTest), [`PallasMosaicTpuRaggedDotTest`](pallas_mosaic_tpu_test.md#PallasMosaicTpuRaggedDotTest), [`PallasTritonRaggedDotTest`](pallas_triton_test.md#PallasTritonRaggedDotTest), [`RaggedDotTest`](base_test.md#RaggedDotTest), [`RaggedDotWithExplicitVjpTest`](base_test.md#RaggedDotWithExplicitVjpTest), [`_test_bench`](pallas_triton_test.md#PallasTritonRaggedDotTest._test_bench)

## Functions
- `_dot_fn_f32(dot_fn)` — [`L36`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/test_base.py#L36) — Wraps a dot_fn to ensure that the output is always f32.
- `override_chex_args(**kwargs)` — [`L65`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/test_base.py#L65)
- `ref(lhs, rhs, group_sizes, activation=None)` — [`L51`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/test_base.py#L51) — Reference implementation of ragged dot.
- `relu(x)` — [`L79`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/test_base.py#L79)
- `wrapped(lhs, rhs, group_sizes, **kwargs)` — [`L39`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/test_base.py#L39)

## Module values
- `ARG_SPECS` — [`L33`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/test_base.py#L33)
- `NAMED_ARG_SPECS` — [`L71`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/test_base.py#L71)
- `_jax_ragged_dot_f32` — [`L48`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/test_base.py#L48)
- `test_config` — [`L83`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/test_base.py#L83)


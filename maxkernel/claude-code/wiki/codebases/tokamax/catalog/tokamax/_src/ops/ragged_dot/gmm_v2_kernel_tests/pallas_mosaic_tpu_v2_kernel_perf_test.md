---
title: 'Module: tokamax/_src/ops/ragged_dot/gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_perf_test.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/ragged_dot/gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_perf_test.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.ragged_dot.gmm_v2_kernel_tests.pallas_mosaic_tpu_v2_kernel_perf_test`/GmmPerfTest#
symbols:
  GmmPerfTest.test_tgmm_perf_regression: test_tgmm_perf_regression().
  GmmPerfTest.test_gmm_perf_regression: test_gmm_perf_regression().
  GmmPerfTest: ''
  GmmPerfTest.setUp: setUp().
---
# Module: [`tokamax/_src/ops/ragged_dot/gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_perf_test.py`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_perf_test.py)

## Classes
### `GmmPerfTest`  ·  implements/extends TestCase
- def: [`tokamax/_src/ops/ragged_dot/gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_perf_test.py:29`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_perf_test.py#L29)
- signature: `class GmmPerfTest(parameterized.TestCase):`
- members:
  - `setUp(self)` — [`L31`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_perf_test.py#L31)
  - `test_gmm_perf_regression(self)` — [`L36`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_perf_test.py#L36)
  - `test_tgmm_perf_regression(self)` — [`L79`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_perf_test.py#L79)
- uses (calls/refs, reference-scoped): [`standardize_function`](../../../benchmarking.md#standardize_function), [`PallasMosaicTpuV2RaggedDot`](../pallas_mosaic_tpu_v2.md#PallasMosaicTpuV2RaggedDot), [`benchmark`](../../../benchmarking.md#benchmark), [`median_evaluation_time_ms`](../../../benchmarking.md#BenchmarkData.median_evaluation_time_ms), [`DRHS_RAGGED_DOT_DIM_NUMS`](../pallas_mosaic_tpu_v2.md#DRHS_RAGGED_DOT_DIM_NUMS), [`num_actual_groups`](../pallas_mosaic_tpu_v2.md#PallasMosaicTpuV2RaggedDot.num_actual_groups), [`validate_tgmm_inputs`](../pallas_mosaic_tpu_v2_tgmm_kernel.md#validate_tgmm_inputs)


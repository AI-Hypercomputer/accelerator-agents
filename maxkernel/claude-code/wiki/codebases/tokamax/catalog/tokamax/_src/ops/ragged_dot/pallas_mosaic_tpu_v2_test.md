---
title: 'Module: tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_test.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_test.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.ragged_dot.pallas_mosaic_tpu_v2_test`/PallasMosaicTpuV2OpParameterPipingTest#
symbols:
  PallasMosaicTpuV2OpParameterPipingTest.test_tgmm_drhs_with_tile_info_pipes: test_tgmm_drhs_with_tile_info_pipes().
  PallasMosaicTpuV2OpParameterPipingTest.test_gmm_maxtext: test_gmm_maxtext().
  PallasMosaicTpuV2OpParameterPipingTest._assert_gmm_api_matches_kernel: _assert_gmm_api_matches_kernel().
  PallasMosaicTpuV2OpParameterPipingTest.test_tgmm_drhs_pipes: test_tgmm_drhs_pipes().
  PallasMosaicTpuV2OpParameterPipingTest.test_tgmm_drhs_with_rhs_scale_pipes: test_tgmm_drhs_with_rhs_scale_pipes().
  PallasMosaicTpuV2OpParameterPipingTest.test_gmm_preferred_element_type_pipes: test_gmm_preferred_element_type_pipes().
  PallasMosaicTpuV2OpParameterPipingTest.test_gmm_basic_pipes: test_gmm_basic_pipes().
  PallasMosaicTpuV2OpParameterPipingTest.test_gmm_weight_quantized_pipes: test_gmm_weight_quantized_pipes().
  PallasMosaicTpuV2OpParameterPipingTest.test_gmm_activation_weight_quantized_pipes: test_gmm_activation_weight_quantized_pipes().
  PallasMosaicTpuV2OpParameterPipingTest.test_gmm_implicit_padding_pipes: test_gmm_implicit_padding_pipes().
  PallasMosaicTpuV2OpParameterPipingTest.test_gmm_weight_quantized_padding_pipes: test_gmm_weight_quantized_padding_pipes().
  PallasMosaicTpuV2OpParameterPipingTest.test_gmm_nonlocal_groups_produce_zeros_pipes: test_gmm_nonlocal_groups_produce_zeros_pipes().
  PallasMosaicTpuV2OpParameterPipingTest.test_gmm_fused_activation_pipes: test_gmm_fused_activation_pipes().
  PallasMosaicTpuV2OpParameterPipingTest.test_gmm_precision_pipes: test_gmm_precision_pipes().
  PallasMosaicTpuV2OpParameterPipingTest.test_gmm_tpu_inference: test_gmm_tpu_inference().
  PallasMosaicTpuV2OpParameterPipingTest: ''
  PallasMosaicTpuV2OpParameterPipingTest.setUp: setUp().
---
# Module: [`tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_test.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_test.py)

## Classes
### `PallasMosaicTpuV2OpParameterPipingTest`  ·  implements/extends TestCase
- def: [`tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_test.py:29`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_test.py#L29)
- doc: Verifies PallasMosaicTpuV2RaggedDot pipes kwargs correctly to the kernel.
- signature: `class PallasMosaicTpuV2OpParameterPipingTest(parameterized.TestCase):`
- members:
  - `setUp(self)` — [`L40`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_test.py#L40)
  - `test_gmm_activation_weight_quantized_pipes(self)` — [`L249`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_test.py#L249)
  - `test_gmm_basic_pipes(self)` — [`L64`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_test.py#L64)
  - `test_gmm_fused_activation_pipes(self)` — [`L352`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_test.py#L352)
  - `test_gmm_implicit_padding_pipes(self)` — [`L276`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_test.py#L276)
  - `test_gmm_maxtext(self)` — [`L466`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_test.py#L466)
  - `test_gmm_nonlocal_groups_produce_zeros_pipes(self)` — [`L321`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_test.py#L321)
  - `test_gmm_precision_pipes(self)` — [`L403`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_test.py#L403) — `precision` must thread through the API to the kernel without error.
  - `test_gmm_preferred_element_type_pipes(self)` — [`L377`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_test.py#L377) — `preferred_element_type` must reach the kernel and set the output dtype.
  - `test_gmm_tpu_inference(self)` — [`L428`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_test.py#L428) — Mirrors how tpu-inference's fused MoE invokes gmm_v2.
  - `test_gmm_weight_quantized_padding_pipes(self)` — [`L293`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_test.py#L293)
  - `test_gmm_weight_quantized_pipes(self)` — [`L213`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_test.py#L213)
  - `test_tgmm_drhs_pipes(self)` — [`L92`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_test.py#L92) — Exercises the drhs (tgmm) path: the op must match a direct `tgmm_v2`.
  - `test_tgmm_drhs_with_rhs_scale_pipes(self)` — [`L167`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_test.py#L167) — Mirrors `test_tgmm_with_tile_info`.
  - `test_tgmm_drhs_with_tile_info_pipes(self)` — [`L127`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_test.py#L127) — Mirrors `test_tgmm_with_tile_info`.
- protocol/private: `_assert_gmm_api_matches_kernel`[`L47`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_test.py#L47)
- uses (calls/refs, reference-scoped): [`gmm_v2`](pallas_mosaic_tpu_v2_gmm_kernel.md#gmm_v2), [`tgmm_v2`](pallas_mosaic_tpu_v2_tgmm_kernel.md#tgmm_v2), [`tile_k`](pallas_mosaic_tpu_v2_gmm_kernel.md#TileSizes.tile_k), [`tile_n`](pallas_mosaic_tpu_v2_gmm_kernel.md#TileSizes.tile_n), [`TileSizes`](pallas_mosaic_tpu_v2_gmm_kernel.md#TileSizes), [`tile_m`](pallas_mosaic_tpu_v2_gmm_kernel.md#TileSizes.tile_m), [`config`](../op.md#Op.config), [`PallasMosaicTpuV2RaggedDot`](pallas_mosaic_tpu_v2.md#PallasMosaicTpuV2RaggedDot), [`DRHS_RAGGED_DOT_DIM_NUMS`](pallas_mosaic_tpu_v2.md#DRHS_RAGGED_DOT_DIM_NUMS), [`num_actual_groups`](pallas_mosaic_tpu_v2.md#PallasMosaicTpuV2RaggedDot.num_actual_groups), [`Config`](pallas_mosaic_tpu_v2.md#Config), [`DLHS_RAGGED_DOT_DIM_NUMS`](pallas_mosaic_tpu_v2.md#DLHS_RAGGED_DOT_DIM_NUMS), [`tile_k`](pallas_mosaic_tpu_v2.md#Config.tile_k), [`tile_m`](pallas_mosaic_tpu_v2.md#Config.tile_m), [`tile_n`](pallas_mosaic_tpu_v2.md#Config.tile_n)


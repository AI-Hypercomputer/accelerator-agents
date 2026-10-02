---
title: 'Module: tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.ragged_dot.pallas_mosaic_tpu_v2`/
symbols:
  PallasMosaicTpuV2RaggedDot._fwd: PallasMosaicTpuV2RaggedDot#_fwd().
  PallasMosaicTpuV2RaggedDot: PallasMosaicTpuV2RaggedDot#
  DRHS_RAGGED_DOT_DIM_NUMS: DRHS_RAGGED_DOT_DIM_NUMS.
  PallasMosaicTpuV2RaggedDot.make_fn: PallasMosaicTpuV2RaggedDot#make_fn().
  UNSUPPORTED_DIMENSIONS_MSG: UNSUPPORTED_DIMENSIONS_MSG.
  PallasMosaicTpuV2RaggedDot.num_actual_groups: PallasMosaicTpuV2RaggedDot#num_actual_groups.
  PallasMosaicTpuV2RaggedDot.__post_init__: PallasMosaicTpuV2RaggedDot#__post_init__().
  Config: Config#
  QArray: QArray.
  AsQArray: AsQArray.
  PallasMosaicTpuV2RaggedDot._vjp: PallasMosaicTpuV2RaggedDot#_vjp().
  DLHS_RAGGED_DOT_DIM_NUMS: DLHS_RAGGED_DOT_DIM_NUMS.
  PallasMosaicTpuV2RaggedDot._get_heuristics_config: PallasMosaicTpuV2RaggedDot#_get_heuristics_config().
  DEFAULT_RAGGED_DOT_DIM_NUMS: DEFAULT_RAGGED_DOT_DIM_NUMS.
  PallasMosaicTpuV2RaggedDot.qdtype: PallasMosaicTpuV2RaggedDot#qdtype.
  Config.tile_m: Config#tile_m.
  Config.tile_k: Config#tile_k.
  Config.tile_n: Config#tile_n.
  PallasMosaicTpuV2RaggedDot.config_cls: PallasMosaicTpuV2RaggedDot#config_cls.
  _has_manual_axes: _has_manual_axes().
  Residuals: Residuals.
  PallasMosaicTpuV2RaggedDot.supported_on: PallasMosaicTpuV2RaggedDot#supported_on().
---
# Module: [`tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2.py)

## Classes
### `Config`
- def: [`tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2.py:38`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2.py#L38)
- doc: Pallas Mosaic TPU Ragged Dot config holding the kernel tuning parameters.
- signature: `class Config:`
- members:
  - `tile_k` — [`L47`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2.py#L47)
  - `tile_m` — [`L46`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2.py#L46)
  - `tile_n` — [`L48`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2.py#L48)
- used by: [`_fwd`](pallas_mosaic_tpu_v2.md#PallasMosaicTpuV2RaggedDot._fwd), [`test_tgmm_drhs_with_tile_info_pipes`](pallas_mosaic_tpu_v2_test.md#PallasMosaicTpuV2OpParameterPipingTest.test_tgmm_drhs_with_tile_info_pipes), [`PallasMosaicTpuV2RaggedDot`](pallas_mosaic_tpu_v2.md#PallasMosaicTpuV2RaggedDot), [`_get_heuristics_config`](pallas_mosaic_tpu_v2.md#PallasMosaicTpuV2RaggedDot._get_heuristics_config), [`config_cls`](pallas_mosaic_tpu_v2.md#PallasMosaicTpuV2RaggedDot.config_cls)

### `PallasMosaicTpuV2RaggedDot`  ·  implements/extends RaggedDot
- def: [`tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2.py:78`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2.py#L78)
- doc: Pallas-Mosaic-TPU ragged dot implementation v2.
- signature: `class PallasMosaicTpuV2RaggedDot(base.RaggedDot[Config, None]):`
- members:
  - `make_fn(num_actual_groups=None)` — [`L104`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2.py#L104)
  - `supported_on(self, device: jax.Device)` — [`L282`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2.py#L282)
  - `config_cls` — [`L88`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2.py#L88) — ---
  - `num_actual_groups` — [`L94`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2.py#L94)
  - `qdtype` — [`L89`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2.py#L89)
- protocol/private: `__post_init__`[`L96`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2.py#L96), `_fwd`[`L130`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2.py#L130), `_get_heuristics_config`[`L273`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2.py#L273), `_vjp`[`L110`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2.py#L110)
- uses (calls/refs, reference-scoped): [`BoundArguments`](../op.md#BoundArguments), [`gmm_v2`](pallas_mosaic_tpu_v2_gmm_kernel.md#gmm_v2), [`tgmm_v2`](pallas_mosaic_tpu_v2_tgmm_kernel.md#tgmm_v2), [`calculate_tiling`](pallas_mosaic_tpu_v2_gmm_kernel.md#calculate_tiling), [`RaggedDot`](base.md#RaggedDot), [`calculate_tgmm_tiling`](pallas_mosaic_tpu_v2_tgmm_kernel.md#calculate_tgmm_tiling), [`tile_k`](pallas_mosaic_tpu_v2_gmm_kernel.md#TileSizes.tile_k), [`vjp`](../op.md#Op.vjp), [`tile_n`](pallas_mosaic_tpu_v2_gmm_kernel.md#TileSizes.tile_n), [`GroupSizes`](base.md#GroupSizes), [`TileSizes`](pallas_mosaic_tpu_v2_gmm_kernel.md#TileSizes), [`ActivationFunction`](base.md#ActivationFunction), [`tile_m`](pallas_mosaic_tpu_v2_gmm_kernel.md#TileSizes.tile_m), [`vjp`](base.md#vjp), [`DRHS_RAGGED_DOT_DIM_NUMS`](pallas_mosaic_tpu_v2.md#DRHS_RAGGED_DOT_DIM_NUMS), [`Residuals`](base.md#Residuals), [`UNSUPPORTED_DIMENSIONS_MSG`](pallas_mosaic_tpu_v2.md#UNSUPPORTED_DIMENSIONS_MSG), [`CanonicalPrecision`](base.md#CanonicalPrecision), [`Config`](pallas_mosaic_tpu_v2.md#Config), [`AsQArray`](pallas_mosaic_tpu_v2.md#AsQArray), [`QArray`](pallas_mosaic_tpu_v2.md#QArray), [`DLHS_RAGGED_DOT_DIM_NUMS`](pallas_mosaic_tpu_v2.md#DLHS_RAGGED_DOT_DIM_NUMS), [`DEFAULT_RAGGED_DOT_DIM_NUMS`](pallas_mosaic_tpu_v2.md#DEFAULT_RAGGED_DOT_DIM_NUMS), [`tile_k`](pallas_mosaic_tpu_v2.md#Config.tile_k), [`tile_m`](pallas_mosaic_tpu_v2.md#Config.tile_m), [`tile_n`](pallas_mosaic_tpu_v2.md#Config.tile_n), [`_has_manual_axes`](pallas_mosaic_tpu_v2.md#_has_manual_axes)
- used by: [`_get_heuristics_config`](../op.md#Op._get_heuristics_config), [`supported_on`](../op.md#Op.supported_on), [`RaggedDot`](base.md#RaggedDot), [`_fwd`](base.md#RaggedDot._fwd), [`test_tgmm_drhs_with_tile_info_pipes`](pallas_mosaic_tpu_v2_test.md#PallasMosaicTpuV2OpParameterPipingTest.test_tgmm_drhs_with_tile_info_pipes), [`test_tgmm_perf_regression`](gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_perf_test.md#GmmPerfTest.test_tgmm_perf_regression), [`test_gmm_maxtext`](pallas_mosaic_tpu_v2_test.md#PallasMosaicTpuV2OpParameterPipingTest.test_gmm_maxtext), [`_assert_gmm_api_matches_kernel`](pallas_mosaic_tpu_v2_test.md#PallasMosaicTpuV2OpParameterPipingTest._assert_gmm_api_matches_kernel), [`test_gmm_perf_regression`](gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_perf_test.md#GmmPerfTest.test_gmm_perf_regression), [`test_tgmm_drhs_pipes`](pallas_mosaic_tpu_v2_test.md#PallasMosaicTpuV2OpParameterPipingTest.test_tgmm_drhs_pipes), [`test_tgmm_drhs_with_rhs_scale_pipes`](pallas_mosaic_tpu_v2_test.md#PallasMosaicTpuV2OpParameterPipingTest.test_tgmm_drhs_with_rhs_scale_pipes), [`test_gmm_preferred_element_type_pipes`](pallas_mosaic_tpu_v2_test.md#PallasMosaicTpuV2OpParameterPipingTest.test_gmm_preferred_element_type_pipes)

## Functions
- `_has_manual_axes(manual_axis_type)` — [`L61`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2.py#L61) — Returns whether `manual_axis_type` names any manual axes.

## Module values
- `AsQArray` — [`L34`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2.py#L34)
- `DEFAULT_RAGGED_DOT_DIM_NUMS` — [`L50`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2.py#L50)
- `DLHS_RAGGED_DOT_DIM_NUMS` — [`L51`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2.py#L51)
- `DRHS_RAGGED_DOT_DIM_NUMS` — [`L52`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2.py#L52)
- `QArray` — [`L33`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2.py#L33)
- `Residuals` — [`L35`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2.py#L35)
- `UNSUPPORTED_DIMENSIONS_MSG` — [`L54`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2.py#L54)


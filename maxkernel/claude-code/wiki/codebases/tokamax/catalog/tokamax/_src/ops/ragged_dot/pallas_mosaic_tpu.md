---
title: 'Module: tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.ragged_dot.pallas_mosaic_tpu`/
symbols:
  PallasMosaicTpuRaggedDot._fwd: PallasMosaicTpuRaggedDot#_fwd().
  PallasMosaicTpuRaggedDot._get_heuristics_config: PallasMosaicTpuRaggedDot#_get_heuristics_config().
  PallasMosaicTpuRaggedDot._get_autotuning_configs: PallasMosaicTpuRaggedDot#_get_autotuning_configs().
  PallasMosaicTpuRaggedDot._tgmm_heuristics_config: PallasMosaicTpuRaggedDot#_tgmm_heuristics_config().
  Config: Config#
  Config.tile_n: Config#tile_n.
  Config.tile_m: Config#tile_m.
  Config.tile_k: Config#tile_k.
  PallasMosaicTpuRaggedDot: PallasMosaicTpuRaggedDot#
  PallasMosaicTpuRaggedDot.maybe_quantize: PallasMosaicTpuRaggedDot#maybe_quantize().
  Config.input_buffer_count: Config#input_buffer_count.
  PallasMosaicTpuRaggedDot.__post_init__: PallasMosaicTpuRaggedDot#__post_init__().
  UNSUPPORTED_DIMENSIONS_MSG: UNSUPPORTED_DIMENSIONS_MSG.
  DEFAULT_RAGGED_DOT_DIM_NUMS: DEFAULT_RAGGED_DOT_DIM_NUMS.
  QArray: QArray.
  PallasMosaicTpuRaggedDot.qdtype: PallasMosaicTpuRaggedDot#qdtype.
  PallasMosaicTpuRaggedDot._fit_within_tpu_vmem: PallasMosaicTpuRaggedDot#_fit_within_tpu_vmem().
  PallasMosaicTpuRaggedDot._deflate_tile: PallasMosaicTpuRaggedDot#_deflate_tile().
  DLHS_RAGGED_DOT_DIM_NUMS: DLHS_RAGGED_DOT_DIM_NUMS.
  DRHS_RAGGED_DOT_DIM_NUMS: DRHS_RAGGED_DOT_DIM_NUMS.
  PallasMosaicTpuRaggedDot.interpret: PallasMosaicTpuRaggedDot#interpret.
  AsQArray: AsQArray.
  Config.combine_scopes: Config#combine_scopes.
  PallasMosaicTpuRaggedDot.config_cls: PallasMosaicTpuRaggedDot#config_cls.
  InputBufferCount: InputBufferCount.
  _group_sizes_to_indices: _group_sizes_to_indices().
  TilingTuple: TilingTuple.
  Residuals: Residuals.
  PallasMosaicTpuRaggedDot.supported_on: PallasMosaicTpuRaggedDot#supported_on().
---
# Module: [`tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py)

## Classes
### `Config`
- def: [`tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py:65`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py#L65) — documented in [tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu](../../../../../concepts/tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu.md)
- doc: Pallas Mosaic TPU Ragged Dot config.
- signature: `class Config:`
- members:
  - `combine_scopes` — [`L72`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py#L72) — documented in [tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu](../../../../../concepts/tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu.md)
  - `input_buffer_count` — [`L71`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py#L71) — documented in [tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu](../../../../../concepts/tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu.md)
  - `tile_k` — [`L69`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py#L69) — documented in [tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu](../../../../../concepts/tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu.md)
  - `tile_m` — [`L68`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py#L68) — documented in [tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu](../../../../../concepts/tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu.md)
  - `tile_n` — [`L70`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py#L70) — documented in [tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu](../../../../../concepts/tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu.md)
- uses (calls/refs, reference-scoped): [`InputBufferCount`](pallas_mosaic_tpu.md#InputBufferCount)
- used by: [`_fwd`](pallas_mosaic_tpu.md#PallasMosaicTpuRaggedDot._fwd), [`_get_heuristics_config`](pallas_mosaic_tpu.md#PallasMosaicTpuRaggedDot._get_heuristics_config), [`test_heuristics_config`](pallas_mosaic_tpu_test.md#PallasMosaicTpuRaggedDotTest.test_heuristics_config), [`_get_autotuning_configs`](pallas_mosaic_tpu.md#PallasMosaicTpuRaggedDot._get_autotuning_configs), [`_tgmm_heuristics_config`](pallas_mosaic_tpu.md#PallasMosaicTpuRaggedDot._tgmm_heuristics_config), [`test_heuristics_monkey_patch`](pallas_mosaic_tpu_test.md#PallasMosaicTpuRaggedDotTest.test_heuristics_monkey_patch), [`_monkey_patch_heuristics_config`](pallas_mosaic_tpu_test.md#PallasMosaicTpuRaggedDotTest._monkey_patch_heuristics_config), [`test_shard_map_manual_axis_type_unreduced`](pallas_mosaic_tpu_test.md#PallasMosaicTpuRaggedDotTest.test_shard_map_manual_axis_type_unreduced), [`test_shard_map_manual_axis_type_varying`](pallas_mosaic_tpu_test.md#PallasMosaicTpuRaggedDotTest.test_shard_map_manual_axis_type_varying), [`PallasMosaicTpuRaggedDot`](pallas_mosaic_tpu.md#PallasMosaicTpuRaggedDot), [`_is_config_supported`](pallas_mosaic_tpu_test.md#_is_config_supported), [`fn`](pallas_mosaic_tpu_test.md#PallasMosaicTpuRaggedDotTest.fn), [`config_cls`](pallas_mosaic_tpu.md#PallasMosaicTpuRaggedDot.config_cls)

### `PallasMosaicTpuRaggedDot`  ·  implements/extends RaggedDot
- def: [`tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py:97`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py#L97)
- doc: Pallas-Mosaic-TPU ragged dot implementation.
- signature: `class PallasMosaicTpuRaggedDot(base.RaggedDot[Config, None]):`
- members:
  - `_fit_within_tpu_vmem(self, input_tiles: list[tuple[int, int, Any]], output_tile: tuple[int, int, Any], input_buffer_count: int, utilize_ratio: float = 0.75)` — [`L277`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py#L277) — Returns whether the given tiling fits within TPU VMEM.
  - `_tgmm_heuristics_config(self, ba: op.BoundArguments, m, n, k, input_buffer_count)` — [`L398`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py#L398) — Heuristics config for tgmm.
  - `maybe_quantize(x, tile_shape)` — [`L177`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py#L177)
  - `supported_on(self, device: jax.Device)` — [`L513`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py#L513)
  - `config_cls` — [`L103`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py#L103) — ---
  - `interpret` — [`L105`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py#L105)
  - `qdtype` — [`L104`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py#L104)
- protocol/private: `__post_init__`[`L107`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py#L107), `_deflate_tile`[`L317`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py#L317), `_fwd`[`L126`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py#L126), `_get_autotuning_configs`[`L452`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py#L452), `_get_heuristics_config`[`L325`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py#L325)
- uses (calls/refs, reference-scoped): [`BoundArguments`](../op.md#BoundArguments), [`RaggedDot`](base.md#RaggedDot), [`args`](../op.md#BoundArguments.args), [`arguments`](../op.md#BoundArguments.arguments), [`gmm`](pallas_mosaic_tpu_kernel.md#gmm), [`tgmm`](pallas_mosaic_tpu_kernel.md#tgmm), [`vjp`](../op.md#Op.vjp), [`GroupSizes`](base.md#GroupSizes), [`Config`](pallas_mosaic_tpu.md#Config), [`ActivationFunction`](base.md#ActivationFunction), [`tile_n`](pallas_mosaic_tpu.md#Config.tile_n), [`vjp`](base.md#vjp), [`tile_k`](pallas_mosaic_tpu.md#Config.tile_k), [`tile_m`](pallas_mosaic_tpu.md#Config.tile_m), [`input_buffer_count`](pallas_mosaic_tpu.md#Config.input_buffer_count), [`UNSUPPORTED_DIMENSIONS_MSG`](pallas_mosaic_tpu.md#UNSUPPORTED_DIMENSIONS_MSG), [`Residuals`](base.md#Residuals), [`DEFAULT_RAGGED_DOT_DIM_NUMS`](pallas_mosaic_tpu.md#DEFAULT_RAGGED_DOT_DIM_NUMS), [`CanonicalPrecision`](base.md#CanonicalPrecision), [`QArray`](pallas_mosaic_tpu.md#QArray), [`DLHS_RAGGED_DOT_DIM_NUMS`](pallas_mosaic_tpu.md#DLHS_RAGGED_DOT_DIM_NUMS), [`DRHS_RAGGED_DOT_DIM_NUMS`](pallas_mosaic_tpu.md#DRHS_RAGGED_DOT_DIM_NUMS), [`AsQArray`](pallas_mosaic_tpu.md#AsQArray), [`combine_scopes`](pallas_mosaic_tpu.md#Config.combine_scopes), [`_group_sizes_to_indices`](pallas_mosaic_tpu.md#_group_sizes_to_indices)
- used by: [`_get_heuristics_config`](../op.md#Op._get_heuristics_config), [`_get_autotuning_configs`](../op.md#Op._get_autotuning_configs), [`supported_on`](../op.md#Op.supported_on), [`RaggedDot`](base.md#RaggedDot), [`_fwd`](base.md#RaggedDot._fwd), [`test_heuristics_config`](pallas_mosaic_tpu_test.md#PallasMosaicTpuRaggedDotTest.test_heuristics_config), [`_DEFAULT_IMPLEMENTATIONS`](api.md#_DEFAULT_IMPLEMENTATIONS), [`test_heuristics_monkey_patch`](pallas_mosaic_tpu_test.md#PallasMosaicTpuRaggedDotTest.test_heuristics_monkey_patch), [`test_shard_map_manual_axis_type_unreduced`](pallas_mosaic_tpu_test.md#PallasMosaicTpuRaggedDotTest.test_shard_map_manual_axis_type_unreduced), [`test_shard_map_manual_axis_type_varying`](pallas_mosaic_tpu_test.md#PallasMosaicTpuRaggedDotTest.test_shard_map_manual_axis_type_varying), [`test_autotuning_configs`](pallas_mosaic_tpu_test.md#PallasMosaicTpuRaggedDotTest.test_autotuning_configs), [`fn`](pallas_mosaic_tpu_test.md#PallasMosaicTpuRaggedDotTest.fn)

## Functions
- `_group_sizes_to_indices(gs: jax.Array, *, m: int)` — [`L56`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py#L56)

## Module values
- `AsQArray` — [`L52`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py#L52)
- `DEFAULT_RAGGED_DOT_DIM_NUMS` — [`L75`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py#L75) — documented in [tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu](../../../../../concepts/tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu.md)
- `DLHS_RAGGED_DOT_DIM_NUMS` — [`L77`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py#L77) — documented in [tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu](../../../../../concepts/tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu.md)
- `DRHS_RAGGED_DOT_DIM_NUMS` — [`L83`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py#L83) — documented in [tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu](../../../../../concepts/tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu.md)
- `InputBufferCount` — [`L49`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py#L49)
- `QArray` — [`L51`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py#L51)
- `Residuals` — [`L53`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py#L53)
- `TilingTuple` — [`L44`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py#L44)
- `UNSUPPORTED_DIMENSIONS_MSG` — [`L89`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py#L89) — documented in [tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu](../../../../../concepts/tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu.md)


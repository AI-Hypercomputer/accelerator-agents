---
title: 'Module: tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_test.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_test.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.ragged_dot.pallas_mosaic_tpu_test`/
symbols:
  PallasMosaicTpuRaggedDotTest.test_heuristics_config: PallasMosaicTpuRaggedDotTest#test_heuristics_config().
  PallasMosaicTpuRaggedDotTest.test_heuristics_monkey_patch: PallasMosaicTpuRaggedDotTest#test_heuristics_monkey_patch().
  PallasMosaicTpuRaggedDotTest._monkey_patch_heuristics_config: PallasMosaicTpuRaggedDotTest#_monkey_patch_heuristics_config().
  PallasMosaicTpuRaggedDotTest.test_shard_map_manual_axis_type_varying: PallasMosaicTpuRaggedDotTest#test_shard_map_manual_axis_type_varying().
  PallasMosaicTpuRaggedDotTest.test_shard_map_manual_axis_type_unreduced: PallasMosaicTpuRaggedDotTest#test_shard_map_manual_axis_type_unreduced().
  _is_config_supported: _is_config_supported().
  PallasMosaicTpuRaggedDotTest.test_autotuning_configs: PallasMosaicTpuRaggedDotTest#test_autotuning_configs().
  PallasMosaicTpuRaggedDotTest.fn: PallasMosaicTpuRaggedDotTest#fn().
  PallasMosaicTpuRaggedDotTest.__init__: PallasMosaicTpuRaggedDotTest#__init__().
  PallasMosaicTpuRaggedDotTest._test_quantized: PallasMosaicTpuRaggedDotTest#_test_quantized().
  PallasMosaicTpuRaggedDotTest._test_bench: PallasMosaicTpuRaggedDotTest#_test_bench().
  PallasMosaicTpuRaggedDotTest: PallasMosaicTpuRaggedDotTest#
  PallasMosaicTpuRaggedDotTest.test_vjp0: PallasMosaicTpuRaggedDotTest#test_vjp0().
  AsQArray: AsQArray.
  _is_scale_tiling_supported: _is_scale_tiling_supported().
  PallasMosaicTpuRaggedDotTest.f: PallasMosaicTpuRaggedDotTest#f().
  PallasMosaicTpuRaggedDotTest.setUp: PallasMosaicTpuRaggedDotTest#setUp().
---
# Module: [`tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_test.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_test.py)

## Classes
### `PallasMosaicTpuRaggedDotTest`  ·  implements/extends RaggedDotTestBase
- def: [`tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_test.py:81`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_test.py#L81)
- doc: Pallas Mosaic TPU Ragged Dot tests.
- signature: `class PallasMosaicTpuRaggedDotTest(test_base.RaggedDotTestBase):`
- members:
  - `f(lhs, rhs, sizes)` — [`L147`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_test.py#L147)
  - `fn(lhs, rhs, *, config=None, **kwargs)` — [`L86`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_test.py#L86)
  - `setUp(self)` — [`L100`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_test.py#L100)
  - `test_autotuning_configs(self)` — [`L248`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_test.py#L248)
  - `test_heuristics_config(self, m, k, n)` — [`L268`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_test.py#L268)
  - `test_heuristics_monkey_patch(self)` — [`L350`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_test.py#L350) — Tests that the heuristics config is monkey-patched correctly.
  - `test_shard_map_manual_axis_type_unreduced(self)` — [`L167`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_test.py#L167)
  - `test_shard_map_manual_axis_type_varying(self)` — [`L109`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_test.py#L109)
  - `test_vjp0(self)` — [`L105`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_test.py#L105)
- protocol/private: `__init__`[`L84`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_test.py#L84), `_monkey_patch_heuristics_config`[`L362`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_test.py#L362), `_test_bench`[`L238`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_test.py#L238), `_test_quantized`[`L215`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_test.py#L215)
- uses (calls/refs, reference-scoped): [`BoundArguments`](../op.md#BoundArguments), [`op`](../op.md#BoundArguments.op), [`_get_heuristics_config`](pallas_mosaic_tpu.md#PallasMosaicTpuRaggedDot._get_heuristics_config), [`arguments`](../op.md#BoundArguments.arguments), [`Config`](pallas_mosaic_tpu.md#Config), [`tile_n`](pallas_mosaic_tpu.md#Config.tile_n), [`config`](../op.md#Op.config), [`tile_k`](pallas_mosaic_tpu.md#Config.tile_k), [`tile_m`](pallas_mosaic_tpu.md#Config.tile_m), [`PallasMosaicTpuRaggedDot`](pallas_mosaic_tpu.md#PallasMosaicTpuRaggedDot), [`_is_config_supported`](pallas_mosaic_tpu_test.md#_is_config_supported), [`autotuning_configs`](../op.md#BoundArguments.autotuning_configs), [`heuristics_config`](../op.md#BoundArguments.heuristics_config), [`input_buffer_count`](pallas_mosaic_tpu.md#Config.input_buffer_count), [`generate_group_sizes`](base.md#generate_group_sizes), [`DLHS_RAGGED_DOT_DIM_NUMS`](pallas_mosaic_tpu.md#DLHS_RAGGED_DOT_DIM_NUMS), [`DRHS_RAGGED_DOT_DIM_NUMS`](pallas_mosaic_tpu.md#DRHS_RAGGED_DOT_DIM_NUMS), [`combine_scopes`](pallas_mosaic_tpu.md#Config.combine_scopes)  (5 test-only)
- used by: (3 test-only callers)

## Functions
- `_is_config_supported(lhs: jax.Array | qwix.QArray | AsQArray, rhs: jax.Array | qwix.QArray | AsQArray, config: pallas_mosaic_tpu.Config)` — [`L59`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_test.py#L59)
- `_is_scale_tiling_supported(x: qwix.QArray, axis: int)` — [`L44`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_test.py#L44)

## Module values
- `AsQArray` — [`L41`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_test.py#L41)


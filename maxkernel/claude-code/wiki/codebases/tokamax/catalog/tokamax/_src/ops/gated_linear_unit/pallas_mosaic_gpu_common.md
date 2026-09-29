---
title: 'Module: tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_common.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_common.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.gated_linear_unit.pallas_mosaic_gpu_common`/
symbols:
  Config.__post_init__: Config#__post_init__().
  Config.cluster_size_m: Config#cluster_size_m.
  Config.cluster_size_n: Config#cluster_size_n.
  MatmulDimension: MatmulDimension#
  Config: Config#
  Config.wg_dimension: Config#wg_dimension.
  Config.grid_minor_dim: Config#grid_minor_dim.
  Config.tile_m: Config#tile_m.
  Config.tile_n: Config#tile_n.
  Config.epi_tile_n: Config#epi_tile_n.
  MatmulDimension.N: MatmulDimension#N.
  Config.tile_k: Config#tile_k.
  Config.num_stages: Config#num_stages.
  Config.epi_tile_m: Config#epi_tile_m.
  MatmulDimension.M: MatmulDimension#M.
  Config.grid_tile_width: Config#grid_tile_width.
  MatmulDimension.__str__: MatmulDimension#__str__().
  MatmulDimension.__repr__: MatmulDimension#__repr__().
---
# Module: [`tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_common.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_common.py)

## Classes
### `Config`
- def: [`tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_common.py:33`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_common.py#L33) — documented in [tokamax-_src-ops-gated_linear_unit-pallas_mosaic_gpu_common](../../../../../concepts/tokamax-_src-ops-gated_linear_unit-pallas_mosaic_gpu_common.md)
- doc: Configuration for the Pallas-Mosaic-GPU Gated Linear Unit.
- signature: `class Config:`
- members:
  - `cluster_size_m` — [`L59`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_common.py#L59) — documented in [tokamax-_src-ops-gated_linear_unit-pallas_mosaic_gpu_common](../../../../../concepts/tokamax-_src-ops-gated_linear_unit-pallas_mosaic_gpu_common.md)
  - `cluster_size_n` — [`L60`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_common.py#L60) — documented in [tokamax-_src-ops-gated_linear_unit-pallas_mosaic_gpu_common](../../../../../concepts/tokamax-_src-ops-gated_linear_unit-pallas_mosaic_gpu_common.md)
  - `epi_tile_m` — [`L55`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_common.py#L55) — documented in [tokamax-_src-ops-gated_linear_unit-pallas_mosaic_gpu_common](../../../../../concepts/tokamax-_src-ops-gated_linear_unit-pallas_mosaic_gpu_common.md)
  - `epi_tile_n` — [`L54`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_common.py#L54) — documented in [tokamax-_src-ops-gated_linear_unit-pallas_mosaic_gpu_common](../../../../../concepts/tokamax-_src-ops-gated_linear_unit-pallas_mosaic_gpu_common.md)
  - `grid_minor_dim` — [`L56`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_common.py#L56)
  - `grid_tile_width` — [`L57`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_common.py#L57)
  - `num_stages` — [`L53`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_common.py#L53)
  - `tile_k` — [`L52`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_common.py#L52)
  - `tile_m` — [`L50`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_common.py#L50) — documented in [tokamax-_src-ops-gated_linear_unit-pallas_mosaic_gpu_common](../../../../../concepts/tokamax-_src-ops-gated_linear_unit-pallas_mosaic_gpu_common.md)
  - `tile_n` — [`L51`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_common.py#L51) — documented in [tokamax-_src-ops-gated_linear_unit-pallas_mosaic_gpu_common](../../../../../concepts/tokamax-_src-ops-gated_linear_unit-pallas_mosaic_gpu_common.md)
  - `wg_dimension` — [`L58`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_common.py#L58)
- protocol/private: `__post_init__`[`L62`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_common.py#L62)
- uses (calls/refs, reference-scoped): [`MatmulDimension`](pallas_mosaic_gpu_common.md#MatmulDimension), [`N`](pallas_mosaic_gpu_common.md#MatmulDimension.N)
- used by: [`get_autotuning_configs`](pallas_mosaic_gpu_kernel_sm90.md#get_autotuning_configs), [`gated_linear_unit`](pallas_mosaic_gpu_kernel_sm90.md#gated_linear_unit), [`gated_linear_unit`](pallas_mosaic_gpu_kernel_sm100.md#gated_linear_unit), [`_mn_loop`](pallas_mosaic_gpu_kernel_sm90.md#gated_linear_unit.kernel._pipeline_scope._mn_loop), [`get_heuristics_config`](pallas_mosaic_gpu_kernel_sm100.md#get_heuristics_config), [`get_heuristics_config`](pallas_mosaic_gpu_kernel_sm90.md#get_heuristics_config), [`mn_loop`](pallas_mosaic_gpu_kernel_sm100.md#gated_linear_unit.kernel.mn_loop), [`Config`](pallas_mosaic_gpu.md#Config), [`_loop_body`](pallas_mosaic_gpu_kernel_sm100.md#gated_linear_unit.kernel.mn_loop._compute_wg._per_warp._tma_warp._loop_body), [`get_pipeline`](pallas_mosaic_gpu_kernel_sm90.md#gated_linear_unit.kernel.get_pipeline), [`get_autotuning_configs`](pallas_mosaic_gpu_kernel_sm100.md#get_autotuning_configs)

### `MatmulDimension`  ·  implements/extends IntEnum
- def: [`tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_common.py:21`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_common.py#L21)
- signature: `class MatmulDimension(enum.IntEnum):`
- members:
  - `M` — [`L22`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_common.py#L22)
  - `N` — [`L23`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_common.py#L23)
- protocol/private: `__repr__`[`L28`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_common.py#L28), `__str__`[`L25`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu_common.py#L25)
- used by: [`get_autotuning_configs`](pallas_mosaic_gpu_kernel_sm90.md#get_autotuning_configs), [`gated_linear_unit`](pallas_mosaic_gpu_kernel_sm90.md#gated_linear_unit), [`_mn_loop`](pallas_mosaic_gpu_kernel_sm90.md#gated_linear_unit.kernel._pipeline_scope._mn_loop), [`wg_dimension`](pallas_mosaic_gpu_common.md#Config.wg_dimension), [`grid_minor_dim`](pallas_mosaic_gpu_common.md#Config.grid_minor_dim)


---
title: 'Module: tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.gated_linear_unit.pallas_mosaic_gpu`/
symbols:
  PallasMosaicGpuGatedLinearUnit._fwd: PallasMosaicGpuGatedLinearUnit#_fwd().
  PallasMosaicGpuGatedLinearUnit._get_heuristics_config: PallasMosaicGpuGatedLinearUnit#_get_heuristics_config().
  PallasMosaicGpuGatedLinearUnit._get_autotuning_configs: PallasMosaicGpuGatedLinearUnit#_get_autotuning_configs().
  Config: Config.
  PallasMosaicGpuGatedLinearUnit: PallasMosaicGpuGatedLinearUnit#
  Residuals: Residuals.
  _get_kernel_module: _get_kernel_module().
  PallasMosaicGpuGatedLinearUnit.config_cls: PallasMosaicGpuGatedLinearUnit#config_cls.
  PallasMosaicGpuGatedLinearUnit.supported_on: PallasMosaicGpuGatedLinearUnit#supported_on().
---
# Module: [`tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu.py)

## Classes
### `PallasMosaicGpuGatedLinearUnit`  ·  implements/extends GatedLinearUnit
- def: [`tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu.py:47`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu.py#L47)
- doc: Pallas-Mosaic-GPU gated linear unit.
- signature: `class PallasMosaicGpuGatedLinearUnit(base.GatedLinearUnit[Config, None]):`
- members:
  - `supported_on(self, device: jax.Device)` — [`L53`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu.py#L53)
  - `config_cls` — [`L50`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu.py#L50) — ---
- protocol/private: `_fwd`[`L57`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu.py#L57), `_get_autotuning_configs`[`L97`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu.py#L97), `_get_heuristics_config`[`L93`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu.py#L93)
- uses (calls/refs, reference-scoped): [`BoundArguments`](../op.md#BoundArguments), [`GatedLinearUnit`](base.md#GatedLinearUnit), [`gated_linear_unit`](pallas_mosaic_gpu_kernel_sm100.md#gated_linear_unit), [`_fwd`](base.md#GatedLinearUnit._fwd), [`get_heuristics_config`](pallas_mosaic_gpu_kernel_sm100.md#get_heuristics_config), [`FusedWeights`](base.md#FusedWeights.FusedWeights), [`UnfusedWeights`](base.md#UnfusedWeights.UnfusedWeights), [`Config`](pallas_mosaic_gpu.md#Config), [`_with_vmap`](base.md#GatedLinearUnit._with_vmap), [`get_autotuning_configs`](pallas_mosaic_gpu_kernel_sm100.md#get_autotuning_configs), [`CanonicalPrecision`](base.md#CanonicalPrecision), [`Residuals`](pallas_mosaic_gpu.md#Residuals), [`_get_kernel_module`](pallas_mosaic_gpu.md#_get_kernel_module)
- used by: [`_get_heuristics_config`](../op.md#Op._get_heuristics_config), [`_get_autotuning_configs`](../op.md#Op._get_autotuning_configs), [`supported_on`](../op.md#Op.supported_on), [`GatedLinearUnit`](base.md#GatedLinearUnit), [`_fwd`](base.md#GatedLinearUnit._fwd), [`_DEFAULT_IMPLEMENTATIONS`](api.md#_DEFAULT_IMPLEMENTATIONS), [`__init__`](pallas_mosaic_gpu_test.md#PallasMosaicGpuGatedLinearUnitTest.__init__)

## Functions
- `_get_kernel_module()` — [`L38`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu.py#L38)

## Module values
- `Config` — [`L35`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu.py#L35)
- `Residuals` — [`L34`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu.py#L34)


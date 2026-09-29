---
title: 'Module: tokamax/_src/ops/gated_linear_unit/pallas_triton.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/gated_linear_unit/pallas_triton.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.gated_linear_unit.pallas_triton`/
symbols:
  PallasTritonGatedLinearUnit._fwd: PallasTritonGatedLinearUnit#_fwd().
  PallasTritonGatedLinearUnit._get_heuristics_config: PallasTritonGatedLinearUnit#_get_heuristics_config().
  PallasTritonGatedLinearUnit._get_autotuning_configs: PallasTritonGatedLinearUnit#_get_autotuning_configs().
  PallasTritonGatedLinearUnit.fn: PallasTritonGatedLinearUnit#fn().
  Config: Config#
  PallasTritonGatedLinearUnit: PallasTritonGatedLinearUnit#
  Residuals: Residuals.
  _gated_linear_unit_kernel: _gated_linear_unit_kernel().
  _gated_linear_unit_kernel.body: _gated_linear_unit_kernel().body().
  Config.block_m: Config#block_m.
  Config.block_n: Config#block_n.
  Config.block_k: Config#block_k.
  Config.num_warps: Config#num_warps.
  Config.num_stages: Config#num_stages.
  PallasTritonGatedLinearUnit.config_cls: PallasTritonGatedLinearUnit#config_cls.
  _get_best_block_size: _get_best_block_size().
  PallasTritonGatedLinearUnit.supports_symbolic_shapes: PallasTritonGatedLinearUnit#supports_symbolic_shapes.
  PallasTritonGatedLinearUnit.supported_on: PallasTritonGatedLinearUnit#supported_on().
---
# Module: [`tokamax/_src/ops/gated_linear_unit/pallas_triton.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_triton.py)

## Classes
### `Config`
- def: [`tokamax/_src/ops/gated_linear_unit/pallas_triton.py:81`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_triton.py#L81)
- signature: `class Config:`
- members:
  - `block_k` — [`L84`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_triton.py#L84)
  - `block_m` — [`L82`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_triton.py#L82)
  - `block_n` — [`L83`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_triton.py#L83)
  - `num_stages` — [`L86`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_triton.py#L86)
  - `num_warps` — [`L85`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_triton.py#L85)
- used by: [`_fwd`](pallas_triton.md#PallasTritonGatedLinearUnit._fwd), [`_get_heuristics_config`](pallas_triton.md#PallasTritonGatedLinearUnit._get_heuristics_config), [`_get_autotuning_configs`](pallas_triton.md#PallasTritonGatedLinearUnit._get_autotuning_configs), [`fn`](pallas_triton.md#PallasTritonGatedLinearUnit.fn), [`PallasTritonGatedLinearUnit`](pallas_triton.md#PallasTritonGatedLinearUnit), [`config_cls`](pallas_triton.md#PallasTritonGatedLinearUnit.config_cls)

### `PallasTritonGatedLinearUnit`  ·  implements/extends GatedLinearUnit
- def: [`tokamax/_src/ops/gated_linear_unit/pallas_triton.py:106`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_triton.py#L106)
- doc: Pallas-Triton gated linear unit.
- signature: `class PallasTritonGatedLinearUnit(base.GatedLinearUnit[Config, None]):`
- members:
  - `fn(x, weights)` — [`L139`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_triton.py#L139)
  - `supported_on(self, device: jax.Device)` — [`L243`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_triton.py#L243)
  - `config_cls` — [`L109`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_triton.py#L109)
  - `supports_symbolic_shapes` — [`L110`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_triton.py#L110)
- protocol/private: `_fwd`[`L113`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_triton.py#L113), `_get_autotuning_configs`[`L221`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_triton.py#L221), `_get_heuristics_config`[`L205`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_triton.py#L205)
- uses (calls/refs, reference-scoped): [`BoundArguments`](../op.md#BoundArguments), [`args`](../op.md#BoundArguments.args), [`GatedLinearUnit`](base.md#GatedLinearUnit), [`pallas_call`](../../pallas/block.md#pallas_call), [`get_cheapest_grid_pids`](../../pallas/grid.md#get_cheapest_grid_pids), [`FusedWeights`](base.md#FusedWeights.FusedWeights), [`UnfusedWeights`](base.md#UnfusedWeights.UnfusedWeights), [`Config`](pallas_triton.md#Config), [`_with_vmap`](base.md#GatedLinearUnit._with_vmap), [`CanonicalPrecision`](base.md#CanonicalPrecision), [`Residuals`](pallas_triton.md#Residuals), [`_gated_linear_unit_kernel`](pallas_triton.md#_gated_linear_unit_kernel), [`block_k`](pallas_triton.md#Config.block_k), [`block_m`](pallas_triton.md#Config.block_m), [`block_n`](pallas_triton.md#Config.block_n), [`num_stages`](pallas_triton.md#Config.num_stages), [`num_warps`](pallas_triton.md#Config.num_warps), [`_get_best_block_size`](pallas_triton.md#_get_best_block_size)
- used by: [`_get_heuristics_config`](../op.md#Op._get_heuristics_config), [`_get_autotuning_configs`](../op.md#Op._get_autotuning_configs), [`supported_on`](../op.md#Op.supported_on), [`GatedLinearUnit`](base.md#GatedLinearUnit), [`_fwd`](base.md#GatedLinearUnit._fwd), [`_DEFAULT_IMPLEMENTATIONS`](api.md#_DEFAULT_IMPLEMENTATIONS), [`test_autotuning_search_space`](pallas_triton_test.md#PallasTritonGatedLinearUnitTest.test_autotuning_search_space), [`__init__`](pallas_triton_test.md#PallasTritonGatedLinearUnitTest.__init__)

## Functions
- `_gated_linear_unit_kernel(x_ref, weights_ref, out_ref, residuals_ref, *, block_m, block_n, block_k, activation, precision)` — [`L40`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_triton.py#L40) — Pallas GLU kernel.
- `_get_best_block_size(m: int, n: int)` — [`L89`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_triton.py#L89) — Returns the best block size for the given shape.
- `body(i, acc)` — [`L54`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_triton.py#L54)

## Module values
- `Residuals` — [`L37`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/pallas_triton.py#L37)


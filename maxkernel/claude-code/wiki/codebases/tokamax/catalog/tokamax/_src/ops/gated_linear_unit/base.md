---
title: 'Module: tokamax/_src/ops/gated_linear_unit/base.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/gated_linear_unit/base.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.gated_linear_unit.base`/
symbols:
  GatedLinearUnit: GatedLinearUnit#
  GatedLinearUnit._fwd: GatedLinearUnit#_fwd().
  GatedLinearUnitVjp._fwd: GatedLinearUnitVjp#_fwd().
  GatedLinearUnitVjp: GatedLinearUnitVjp#
  GatedLinearUnit.bind: GatedLinearUnit#bind().
  FusedWeights.FusedWeights: FusedWeights.FusedWeights.
  UnfusedWeights.UnfusedWeights: UnfusedWeights.UnfusedWeights.
  GatedLinearUnit.__post_init__: GatedLinearUnit#__post_init__().
  GatedLinearUnit._with_vmap: GatedLinearUnit#_with_vmap().
  Residuals.Residuals: Residuals.Residuals.
  _FwdFn: _FwdFn.
  _Config: _Config.
  CanonicalPrecision: CanonicalPrecision.
  GatedLinearUnitVjp.fn: GatedLinearUnitVjp#fn().
  _Key: _Key.
  _T: _T.
  GatedLinearUnit.vmap_rule: GatedLinearUnit#vmap_rule().
---
# Module: [`tokamax/_src/ops/gated_linear_unit/base.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/base.py)

## Classes
### `GatedLinearUnit`  ·  implements/extends Op
- def: [`tokamax/_src/ops/gated_linear_unit/base.py:40`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/base.py#L40) — documented in [tokamax-_src-ops-gated_linear_unit-base](../../../../../concepts/tokamax-_src-ops-gated_linear_unit-base.md)
- doc: Gated linear unit.
- signature: `class GatedLinearUnit(op.Op[Any, jax.Array, Residuals, _Config, _Key]):`
- members:
  - `_fwd(self, x: Float[Array, '*B M K'], weights: FusedWeights | UnfusedWeights, *, activation: Callable[[jax.Array], jax.Array] | None, precision: CanonicalPrecision, return_residuals: bool, config: _Config)` — [`L69`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/base.py#L69) — Applies a gated linear unit (https://arxiv.org/abs/1612.08083). — documented in [tokamax-_src-ops-gated_linear_unit-base](../../../../../concepts/tokamax-_src-ops-gated_linear_unit-base.md)
  - `_with_vmap(self, fwd: _FwdFn, *, fallback_to_sequential: bool = True)` — [`L127`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/base.py#L127) — Applies a vmap rule to a fwd function.
  - `bind(self, x: Float[Array, '*B M K'], weights: FusedWeights | UnfusedWeights, *, activation: Callable[[jax.Array], jax.Array] | None = None, precision: jax.lax.PrecisionLike = None, return_residuals: bool = False)` — [`L48`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/base.py#L48) — Binds the arguments for the gated linear unit function.
  - `vmap_rule(axis_size, in_batched, x, weights)` — [`L133`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/base.py#L133)
- protocol/private: `__post_init__`[`L43`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/base.py#L43)
- uses (calls/refs, reference-scoped): [`BoundArguments`](../op.md#BoundArguments), [`Op`](../op.md#Op), [`bind`](../op.md#Op.bind), [`jaxtyped`](../../jaxtyping.md#jaxtyped), [`vjp`](../op.md#Op.vjp), [`_fwd`](pallas_mosaic_gpu.md#PallasMosaicGpuGatedLinearUnit._fwd), [`_fwd`](pallas_triton.md#PallasTritonGatedLinearUnit._fwd), [`GatedLinearUnitVjp`](base.md#GatedLinearUnitVjp), [`FusedWeights`](base.md#FusedWeights.FusedWeights), [`UnfusedWeights`](base.md#UnfusedWeights.UnfusedWeights), [`PallasTritonGatedLinearUnit`](pallas_triton.md#PallasTritonGatedLinearUnit), [`PallasMosaicGpuGatedLinearUnit`](pallas_mosaic_gpu.md#PallasMosaicGpuGatedLinearUnit), [`Residuals`](base.md#Residuals.Residuals), [`_FwdFn`](base.md#_FwdFn), [`CanonicalPrecision`](base.md#CanonicalPrecision), [`_Config`](base.md#_Config), [`_Key`](base.md#_Key)
- used by: [`Op`](../op.md#Op), [`bind`](../op.md#Op.bind), [`test_roundtrip`](../op_test.md#BoundArgumentsTest.test_roundtrip), [`_fwd`](../op.md#Op._fwd), [`_API_IMPLEMENTATIONS`](../../autotuning/api.md#_API_IMPLEMENTATIONS._API_IMPLEMENTATIONS), [`_fwd`](pallas_mosaic_gpu.md#PallasMosaicGpuGatedLinearUnit._fwd), [`_fwd`](pallas_triton.md#PallasTritonGatedLinearUnit._fwd), [`gated_linear_unit`](api.md#gated_linear_unit), [`_fwd`](../triangle_multiplication/base.md#TriangleMultiplication._fwd), [`test_explicit_vjp`](base_test.md#GatedLinearUnitTest.test_explicit_vjp), [`PallasTritonGatedLinearUnit`](pallas_triton.md#PallasTritonGatedLinearUnit), [`test_autotuning_search_space`](pallas_triton_test.md#PallasTritonGatedLinearUnitTest.test_autotuning_search_space), [`PallasMosaicGpuGatedLinearUnit`](pallas_mosaic_gpu.md#PallasMosaicGpuGatedLinearUnit), [`_IMPLEMENTATIONS`](api.md#_IMPLEMENTATIONS), [`__init__`](base_test.md#GatedLinearUnitTest.__init__), [`fn`](base.md#GatedLinearUnitVjp.fn)

### `GatedLinearUnitVjp`
- def: [`tokamax/_src/ops/gated_linear_unit/base.py:152`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/base.py#L152) — documented in [tokamax-_src-ops-gated_linear_unit-base](../../../../../concepts/tokamax-_src-ops-gated_linear_unit-base.md)
- members:
  - `_fwd(self, residuals: Residuals, out: Float[Array, '*B M N'], dout: Float[Array, '*B M N'], x: Float[Array, '*B M K'], weights: FusedWeights | UnfusedWeights, *, activation: Callable[[jax.Array], jax.Array] | None, precision: CanonicalPrecision, return_residuals: bool, config: _Config)` — [`L167`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/base.py#L167) — Gated linear unit VJP. — documented in [tokamax-_src-ops-gated_linear_unit-base](../../../../../concepts/tokamax-_src-ops-gated_linear_unit-base.md)
  - `fn(x, weights)` — [`L190`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/base.py#L190)
- uses (calls/refs, reference-scoped): [`Op`](../op.md#Op), [`GatedLinearUnit`](base.md#GatedLinearUnit), [`FusedWeights`](base.md#FusedWeights.FusedWeights), [`UnfusedWeights`](base.md#UnfusedWeights.UnfusedWeights), [`Residuals`](base.md#Residuals.Residuals), [`CanonicalPrecision`](base.md#CanonicalPrecision), [`_Config`](base.md#_Config), [`_Key`](base.md#_Key)
- used by: [`_fwd`](../op.md#Op._fwd), [`test_explicit_vjp`](base_test.md#GatedLinearUnitTest.test_explicit_vjp), [`__post_init__`](base.md#GatedLinearUnit.__post_init__)

## Module values
- `CanonicalPrecision` — [`L37`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/base.py#L37)
- `FusedWeights` — [`L34`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/base.py#L34)
- `Residuals` — [`L36`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/base.py#L36)
- `UnfusedWeights` — [`L35`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/base.py#L35)
- `_Config` — [`L31`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/base.py#L31)
- `_FwdFn` — [`L33`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/base.py#L33)
- `_Key` — [`L32`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/base.py#L32)
- `_T` — [`L30`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/gated_linear_unit/base.py#L30)


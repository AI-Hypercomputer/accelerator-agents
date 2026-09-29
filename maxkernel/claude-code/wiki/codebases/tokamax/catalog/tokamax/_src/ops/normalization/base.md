---
title: 'Module: tokamax/_src/ops/normalization/base.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/normalization/base.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.normalization.base`/
symbols:
  Normalization: Normalization#
  NormalizationVjp: NormalizationVjp#
  Normalization._fwd: Normalization#_fwd().
  NormalizationVjp._fwd: NormalizationVjp#_fwd().
  Residuals.Residuals: Residuals.Residuals.
  Normalization.bind: Normalization#bind().
  _Config: _Config.
  _Key: _Key.
  FusedInputArray.FusedInputArray: FusedInputArray.FusedInputArray.
---
# Module: [`tokamax/_src/ops/normalization/base.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/base.py)

## Classes
### `Normalization`  ·  implements/extends Op
- def: [`tokamax/_src/ops/normalization/base.py:34`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/base.py#L34)
- doc: Normalization op.
- signature: `class Normalization(op.Op[Any, jax.Array, Residuals, _Config, _Key]):`
- members:
  - `bind(self, x: jax.Array | Callable[[], jax.Array], scale: jax.Array | None, offset: jax.Array | None, *, axis: int = -1, epsilon: float = 0.000001, scale_offset: float = 0, subtract_mean: bool = True, return_residuals: bool = False)` — [`L38`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/base.py#L38) — Binds normalization op to the given arguments.
- protocol/private: `_fwd`[`L101`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/base.py#L101)
- uses (calls/refs, reference-scoped): [`BoundArguments`](../op.md#BoundArguments), [`Op`](../op.md#Op), [`bind`](../op.md#Op.bind), [`_fwd`](pallas_triton.md#PallasTritonNormalization._fwd), [`PallasTritonNormalization`](pallas_triton.md#PallasTritonNormalization), [`Residuals`](base.md#Residuals.Residuals), [`_Config`](base.md#_Config), [`FusedInputArray`](base.md#FusedInputArray.FusedInputArray), [`_Key`](base.md#_Key)
- used by: [`Op`](../op.md#Op), [`bind`](../op.md#Op.bind), [`test_roundtrip`](../op_test.md#BoundArgumentsTest.test_roundtrip), [`_fwd`](../op.md#Op._fwd), [`_API_IMPLEMENTATIONS`](../../autotuning/api.md#_API_IMPLEMENTATIONS._API_IMPLEMENTATIONS), [`layer_norm`](api.md#layer_norm), [`_fwd`](../triangle_multiplication/base.md#TriangleMultiplication._fwd), [`PallasTritonNormalization`](pallas_triton.md#PallasTritonNormalization), [`_IMPLS`](bench.md#_IMPLS), [`test_ignore_cache_overlay`](../op_test.md#BoundArgumentsTest.test_ignore_cache_overlay), [`_IMPLEMENTATIONS`](api.md#_IMPLEMENTATIONS), [`__init__`](base_test.md#NormalizationTest.__init__)  (1 test-only)

### `NormalizationVjp`  ·  implements/extends Op
- def: [`tokamax/_src/ops/normalization/base.py:140`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/base.py#L140)
- doc: Normalization VJP.
- signature: `class NormalizationVjp(op.Op[Any, Any, types.NoneType, _Config, _Key]):`
- members:
  - `_fwd(self, residuals: Residuals, out: jax.Array, dout: jax.Array, x: jax.Array, scale: jax.Array | None, offset: jax.Array | None, *, axis: int, epsilon: float, scale_offset: float, subtract_mean: bool, return_residuals: bool, config: _Config)` — [`L144`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/base.py#L144) — Computes normalization VJP `(dx, dscale, doffset)`.
- uses (calls/refs, reference-scoped): [`Op`](../op.md#Op), [`_fwd`](pallas_triton_vjp.md#PallasTritonNormalizationVjp._fwd), [`PallasTritonNormalizationVjp`](pallas_triton_vjp.md#PallasTritonNormalizationVjp), [`Residuals`](base.md#Residuals.Residuals), [`_Config`](base.md#_Config), [`_Key`](base.md#_Key)
- used by: [`Op`](../op.md#Op), [`_fwd`](../op.md#Op._fwd), [`PallasTritonNormalizationVjp`](pallas_triton_vjp.md#PallasTritonNormalizationVjp), [`_IMPLEMENTATIONS_VJP`](api_test.md#_IMPLEMENTATIONS_VJP._IMPLEMENTATIONS_VJP)

## Module values
- `FusedInputArray` — [`L30`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/base.py#L30)
- `Residuals` — [`L31`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/base.py#L31)
- `_Config` — [`L28`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/base.py#L28)
- `_Key` — [`L29`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/base.py#L29)


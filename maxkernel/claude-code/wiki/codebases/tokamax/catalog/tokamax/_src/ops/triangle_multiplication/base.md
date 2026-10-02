---
title: 'Module: tokamax/_src/ops/triangle_multiplication/base.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/triangle_multiplication/base.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.triangle_multiplication.base`/
symbols:
  TriangleMultiplication._fwd: TriangleMultiplication#_fwd().
  TriangleMultiplication: TriangleMultiplication#
  TriangleMultiplication.bind: TriangleMultiplication#bind().
  _Config: _Config.
  Residuals.Residuals: Residuals.Residuals.
  _Key: _Key.
  CanonicalPrecision: CanonicalPrecision.
---
# Module: [`tokamax/_src/ops/triangle_multiplication/base.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/triangle_multiplication/base.py)

## Classes
### `TriangleMultiplication`  ·  implements/extends Op
- def: [`tokamax/_src/ops/triangle_multiplication/base.py:37`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/triangle_multiplication/base.py#L37)
- doc: Triangle multiplicative update.
- signature: `class TriangleMultiplication(op.Op[Any, jax.Array, Residuals, _Config, _Key]):`
- members:
  - `_fwd(self, x: Float[Array, "N N C"], mask: Bool[Array, "N N"], projection_in_weights: Float[Array, "C 2 H"], gate_in_weights: Float[Array, "C 2 H"], projection_out_weights: Float[Array, "H D"], gate_out_weights: Float[Array, "C D"], layernorm_in_scale: Float[Array, C], layernorm_in_offset: Float[Array, C], layernorm_out_scale: Float[Array, H], layernorm_out_offset: Float[Array, H], triangle_type: Literal["incoming", "outgoing"], *, precision: CanonicalPrecision, epsilon: float, return_residuals: bool, config: _Config)` — [`L79`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/triangle_multiplication/base.py#L79) — Triangle multiplicative update.
  - `bind(self, x: Float[Array, "N N C"], mask: Bool[Array, "N N"], projection_in_weights: Float[Array, "C 2 H"], gate_in_weights: Float[Array, "C 2 H"], projection_out_weights: Float[Array, "H D"], gate_out_weights: Float[Array, "C D"], layernorm_in_scale: Float[Array, C], layernorm_in_offset: Float[Array, C], layernorm_out_scale: Float[Array, H], layernorm_out_offset: Float[Array, H], triangle_type: Literal["incoming", "outgoing"], *, precision: jax.lax.PrecisionLike = None, epsilon: float = 0.000001, return_residuals: bool = False)` — [`L41`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/triangle_multiplication/base.py#L41) — Binds the arguments for the triangle multiplication function.
- uses (calls/refs, reference-scoped): [`BoundArguments`](../op.md#BoundArguments), [`Op`](../op.md#Op), [`bind`](../op.md#Op.bind), [`jaxtyped`](../../jaxtyping.md#jaxtyped), [`GatedLinearUnit`](../gated_linear_unit/base.md#GatedLinearUnit), [`Normalization`](../normalization/base.md#Normalization), [`Residuals`](base.md#Residuals.Residuals), [`_Config`](base.md#_Config), [`CanonicalPrecision`](base.md#CanonicalPrecision), [`_Key`](base.md#_Key)
- used by: [`Op`](../op.md#Op), [`bind`](../op.md#Op.bind), [`_fwd`](../op.md#Op._fwd), [`f`](base_test.md#TriangleMultiplicationTest.f), [`IMPLEMENTATIONS`](api.md#IMPLEMENTATIONS)

## Module values
- `CanonicalPrecision` — [`L34`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/triangle_multiplication/base.py#L34)
- `Residuals` — [`L33`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/triangle_multiplication/base.py#L33)
- `_Config` — [`L31`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/triangle_multiplication/base.py#L31)
- `_Key` — [`L32`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/triangle_multiplication/base.py#L32)


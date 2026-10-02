---
title: 'Module: tokamax/_src/ops/flex_attention/base.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/flex_attention/base.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.flex_attention.base`/
symbols:
  FlexAttention: FlexAttention#
  FlexAttention._fwd: FlexAttention#_fwd().
  QArray: QArray.
  FlexAttention.bind: FlexAttention#bind().
  FlexAttention.__call__: FlexAttention#__call__().
  CanonicalPrecision: CanonicalPrecision.
  Residuals: Residuals.
  ScoreMod: ScoreMod.
  MaskMod: MaskMod.
  FlexAttention.fwd_closed: FlexAttention#fwd_closed().
  _softmax: _softmax().
  _softmax_jvp: _softmax_jvp().
  _Config: _Config.
  _Key: _Key.
  FlexAttention.supports_batched_args_capture: FlexAttention#supports_batched_args_capture.
---
# Module: [`tokamax/_src/ops/flex_attention/base.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/base.py)

## Classes
### `FlexAttention`  ·  implements/extends Op
- def: [`tokamax/_src/ops/flex_attention/base.py:45`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/base.py#L45)
- doc: FlexAttention function.
- signature: `class FlexAttention(op.Op[Any, Float[Array, "*B T H d"], Residuals, _Config, _Key]):`
- members:
  - `bind(self, q: Float[Array | QArray, "*B T H D"], k: Float[Array | QArray, "*B t h D"], v: Float[Array | QArray, "*B t h d"], *, precision: (jax.lax.PrecisionLike | tuple[jax.lax.PrecisionLike, jax.lax.PrecisionLike]) = None, score_mod: ScoreMod | None = None, mask_mod: MaskMod | None = None, dropout_mask: Bool[Array, "*#B #H #T #t"] | None = None, dropout_rate: float = 0, q_sharding: jax.sharding.NamedSharding | None = None, k_sharding: jax.sharding.NamedSharding | None = None, normalize_output: bool = True, return_residuals: bool = False)` — [`L191`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/base.py#L191) — Binds the arguments for the attention function.
  - `fwd_closed(q, k, v, dropout_mask)` — [`L169`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/base.py#L169)
  - `supports_batched_args_capture` — [`L50`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/base.py#L50)
- protocol/private: `__call__`[`L54`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/base.py#L54), `_fwd`[`L235`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/base.py#L235)
- uses (calls/refs, reference-scoped): [`BoundArguments`](../op.md#BoundArguments), [`Op`](../op.md#Op), [`bind`](../op.md#Op.bind), [`jaxtyped`](../../jaxtyping.md#jaxtyped), [`_fwd`](pallas_triton.md#PallasTritonFlexAttention._fwd), [`QArray`](base.md#QArray), [`__call__`](../op.md#Op.__call__), [`PallasTritonFlexAttention`](pallas_triton.md#PallasTritonFlexAttention), [`CanonicalPrecision`](base.md#CanonicalPrecision), [`MaskMod`](base.md#MaskMod), [`Residuals`](base.md#Residuals), [`ScoreMod`](base.md#ScoreMod), [`_softmax`](base.md#_softmax), [`_Config`](base.md#_Config), [`_Key`](base.md#_Key)
- used by: [`Op`](../op.md#Op), [`bind`](../op.md#Op.bind), [`_fwd`](../op.md#Op._fwd), [`__init__`](wrapper_test_base.md#WrappedFlexAttentionTestBase.__init__), [`PallasTritonFlexAttention`](pallas_triton.md#PallasTritonFlexAttention), [`impl`](wrapper.md#WrappedFlexAttention.impl), [`IMPLEMENTATIONS`](api.md#IMPLEMENTATIONS.IMPLEMENTATIONS), [`__init__`](base_test.md#FlexAttentionTest.__init__), [`__init__`](base_test.md#WrappedFlexAttentionTest.__init__)

## Functions
- `_softmax(x: jax.Array, normalize: bool)` — [`L302`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/base.py#L302) — Computes softmax, possibly using residual from the forward pass.
- `_softmax_jvp(normalize, primals, tangents)` — [`L314`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/base.py#L314)

## Module values
- `CanonicalPrecision` — [`L32`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/base.py#L32)
- `MaskMod` — [`L42`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/base.py#L42)
- `QArray` — [`L33`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/base.py#L33)
- `Residuals` — [`L40`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/base.py#L40)
- `ScoreMod` — [`L41`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/base.py#L41)
- `_Config` — [`L36`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/base.py#L36)
- `_Key` — [`L37`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/base.py#L37)


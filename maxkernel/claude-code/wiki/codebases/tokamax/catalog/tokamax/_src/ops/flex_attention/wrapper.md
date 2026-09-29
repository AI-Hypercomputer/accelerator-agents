---
title: 'Module: tokamax/_src/ops/flex_attention/wrapper.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/flex_attention/wrapper.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.flex_attention.wrapper`/
symbols:
  WrappedFlexAttention._fwd: WrappedFlexAttention#_fwd().
  WrappedFlexAttention.impl: WrappedFlexAttention#impl.
  WrappedFlexAttention: WrappedFlexAttention#
  Mask: Mask.
  Residuals: Residuals.
  PagingInfo: PagingInfo.
  WrappedFlexAttention.mask_mod: WrappedFlexAttention#mask_mod().
  QArray: QArray.
  WrappedFlexAttention.score_mod: WrappedFlexAttention#score_mod().
---
# Module: [`tokamax/_src/ops/flex_attention/wrapper.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/wrapper.py)

## Classes
### `WrappedFlexAttention`  ·  implements/extends DotProductAttention
- def: [`tokamax/_src/ops/flex_attention/wrapper.py:38`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/wrapper.py#L38)
- doc: `DotProductAttention` implementation wrapping a `FlexAttention`.
- signature: `class WrappedFlexAttention(attn_base.DotProductAttention[op.NullConfig, None]):`
- members:
  - `mask_mod(shape, *, mask=mask, q_indices=q_indices, k_indices=k_indices)` — [`L79`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/wrapper.py#L79)
  - `score_mod(logits)` — [`L71`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/wrapper.py#L71)
  - `impl` — [`L41`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/wrapper.py#L41)
- protocol/private: `_fwd`[`L45`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/wrapper.py#L45)
- uses (calls/refs, reference-scoped): [`DotProductAttention`](../attention/base.md#DotProductAttention), [`jaxtyped`](../../jaxtyping.md#jaxtyped), [`FlexAttention`](base.md#FlexAttention), [`as_array`](../attention/base.md#Mask.as_array), [`NullConfig`](../op.md#NullConfig), [`CanonicalPrecision`](base.md#CanonicalPrecision), [`Mask`](wrapper.md#Mask), [`PagingInfo`](wrapper.md#PagingInfo), [`Residuals`](wrapper.md#Residuals), [`QArray`](wrapper.md#QArray)
- used by: [`DotProductAttention`](../attention/base.md#DotProductAttention), [`_fwd`](../attention/base.md#DotProductAttention._fwd), [`_IMPLS`](../attention/bench.md#_IMPLS), [`__init__`](wrapper_test_base.md#WrappedFlexAttentionTestBase.__init__)

## Module values
- `Mask` — [`L31`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/wrapper.py#L31)
- `PagingInfo` — [`L34`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/wrapper.py#L34)
- `QArray` — [`L33`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/wrapper.py#L33)
- `Residuals` — [`L32`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/wrapper.py#L32)


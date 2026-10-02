---
title: 'Module: tokamax/_src/ops/attention/jax_nn.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/attention/jax_nn.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.attention.jax_nn`/
symbols:
  JaxNnDotProductAttention._fwd: JaxNnDotProductAttention#_fwd().
  JaxNnDotProductAttention.implementation: JaxNnDotProductAttention#implementation.
  JaxNnDotProductAttention: JaxNnDotProductAttention#
  QArray: QArray.
  Mask: Mask.
  PagingInfo: PagingInfo.
  JaxNnDotProductAttention.flatten_batch: JaxNnDotProductAttention#flatten_batch().
  JaxNnDotProductAttention.supported_on: JaxNnDotProductAttention#supported_on().
---
# Module: [`tokamax/_src/ops/attention/jax_nn.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/jax_nn.py)

## Classes
### `JaxNnDotProductAttention`  ·  implements/extends DotProductAttention
- def: [`tokamax/_src/ops/attention/jax_nn.py:40`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/jax_nn.py#L40)
- doc: `jax.nn.dot_product_attention` implementation.
- signature: `class JaxNnDotProductAttention(base.DotProductAttention[op.NullConfig, None]):`
- members:
  - `flatten_batch(x, rank=q.ndim, always_bcast=False)` — [`L119`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/jax_nn.py#L119)
  - `supported_on(self, device: jax.Device)` — [`L155`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/jax_nn.py#L155)
  - `implementation` — [`L43`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/jax_nn.py#L43)
- protocol/private: `_fwd`[`L47`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/jax_nn.py#L47)
- uses (calls/refs, reference-scoped): [`DotProductAttention`](base.md#DotProductAttention), [`jaxtyped`](../../jaxtyping.md#jaxtyped), [`as_array`](base.md#Mask.as_array), [`CanonicalPrecision`](base.md#CanonicalPrecision), [`q_end`](base.md#Mask.q_end), [`k_end`](base.md#Mask.k_end), [`NullConfig`](../op.md#NullConfig), [`take`](base.md#Mask.take), [`QArray`](jax_nn.md#QArray), [`Mask`](jax_nn.md#Mask), [`PagingInfo`](jax_nn.md#PagingInfo)
- used by: [`DotProductAttention`](base.md#DotProductAttention), [`supported_on`](../op.md#Op.supported_on), [`_fwd`](base.md#DotProductAttention._fwd), [`_IMPLS`](bench.md#_IMPLS), [`_IMPLEMENTATIONS`](api.md#_IMPLEMENTATIONS), [`__init__`](jax_nn_test.md#JaxNnDotProductAttentionTest.__init__)

## Module values
- `Mask` — [`L34`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/jax_nn.py#L34)
- `PagingInfo` — [`L36`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/jax_nn.py#L36)
- `QArray` — [`L35`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/jax_nn.py#L35)


---
title: 'Module: tokamax/_src/ops/attention/xla_chunked.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/attention/xla_chunked.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.attention.xla_chunked`/
symbols:
  XlaChunkedDotProductAttention._fwd: XlaChunkedDotProductAttention#_fwd().
  _attend_chunked: _attend_chunked().
  _attend_paged: _attend_paged().
  XlaChunkedDotProductAttention: XlaChunkedDotProductAttention#
  PagingInfo: PagingInfo.
  _attend_paged.bq_loop_fn: _attend_paged().bq_loop_fn().
  XlaChunkedDotProductAttention.chunk_size: XlaChunkedDotProductAttention#chunk_size.
  _attend_chunked.q_loop_fn.kv_loop_fn.get_kv_chunk: _attend_chunked().q_loop_fn().kv_loop_fn().get_kv_chunk().
  _attend_chunked.q_loop_fn: _attend_chunked().q_loop_fn().
  _attend_chunked.q_loop_fn.kv_loop_fn: _attend_chunked().q_loop_fn().kv_loop_fn().
  Mask: Mask.
  _attend_chunked.q_loop_fn.get_q_chunk: _attend_chunked().q_loop_fn().get_q_chunk().
  QArray: QArray.
  _attend_chunk: _attend_chunk().
  _attend_paged.bq_loop_fn.sq_loop_fn: _attend_paged().bq_loop_fn().sq_loop_fn().
  _attend_paged.bq_loop_fn.sq_loop_fn.kv_loop_fn: _attend_paged().bq_loop_fn().sq_loop_fn().kv_loop_fn().
  _attend_chunked.get_chunk: _attend_chunked().get_chunk().
  _attend_chunk.chunk: _attend_chunk().chunk().
---
# Module: [`tokamax/_src/ops/attention/xla_chunked.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/xla_chunked.py)

## Classes
### `XlaChunkedDotProductAttention`  ·  implements/extends DotProductAttention
- def: [`tokamax/_src/ops/attention/xla_chunked.py:313`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/xla_chunked.py#L313)
- doc: XLA chunked dot product attention function.
- signature: `class XlaChunkedDotProductAttention(base.DotProductAttention[op.NullConfig, None]):`
- members:
  - `chunk_size` — [`L319`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/xla_chunked.py#L319)
- protocol/private: `_fwd`[`L323`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/xla_chunked.py#L323)
- uses (calls/refs, reference-scoped): [`DotProductAttention`](base.md#DotProductAttention), [`jaxtyped`](../../jaxtyping.md#jaxtyped), [`CanonicalPrecision`](base.md#CanonicalPrecision), [`_attend_chunked`](xla_chunked.md#_attend_chunked), [`_attend_paged`](xla_chunked.md#_attend_paged), [`NullConfig`](../op.md#NullConfig), [`PagingInfo`](xla_chunked.md#PagingInfo), [`Mask`](xla_chunked.md#Mask), [`QArray`](xla_chunked.md#QArray)
- used by: [`DotProductAttention`](base.md#DotProductAttention), [`_fwd`](base.md#DotProductAttention._fwd), [`_IMPLEMENTATIONS`](api.md#_IMPLEMENTATIONS), [`__init__`](xla_chunked_test.md#XlaChunkedAttentionTest.__init__), [`__init__`](xla_chunked_test.md#XlaPagedAttentionTest.__init__), [`test_chunk_sizes`](xla_chunked_test.md#XlaChunkedAttentionTest.test_chunk_sizes)  (1 test-only)

## Functions
- `_attend_chunk(q: Float[Array, "*B T H D"], k: Float[Array, "*B t #H D"], v: Float[Array, "*B t #H d"], accum: Float[Array, "*B T H d"], x_max: Float[Array, "*B H T"], denom: Float[Array, "*B H T"], *, precision: tuple[jax.lax.DotAlgorithmPreset, jax.lax.DotAlgorithmPreset], logits_dtype: jnp.dtype, logits_scale: float, bias: Float[Array, "*#B #H #T #t"] | None, logits_soft_cap: float | None, mask: Bool[Array, "*#B #H #T #t"] | None, dropout_mask: Bool[Array, "*#B #H #T #t"] | None, dropout_rate: float)` — [`L41`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/xla_chunked.py#L41) — Computes a chunk of attention.
- `_attend_chunked(q: Float[Array, "*B T H D"], k: Float[Array, "*B t #H D"], v: Float[Array, "*B t #H d"], *, precision: tuple[jax.lax.DotAlgorithmPreset, jax.lax.DotAlgorithmPreset], logits_dtype: jnp.dtype, logits_scale: float, bias: Float[Array, "*#B #H #T #t"] | None, logits_soft_cap: float | None, mask: Mask, dropout_mask: Bool[Array, "*#B #H #T #t"] | None, dropout_rate: float, paging_info: PagingInfo | None, q_indices: Int[Array, "*#B #H T"] | None, k_indices: Int[Array, "*#B #h t"] | None, normalize_output: bool, chunk_size: tuple[int, int])` — [`L125`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/xla_chunked.py#L125) — Computes chunked attention.
- `_attend_paged(q: Float[Array, "*B T H D"], k: Float[Array, "*b t #H D"], v: Float[Array, "*b t #H d"], *, precision: tuple[jax.lax.DotAlgorithmPreset, jax.lax.DotAlgorithmPreset], logits_dtype: jnp.dtype, logits_scale: float, bias: Float[Array, "*#B #H #T #t"] | None, logits_soft_cap: float | None, mask: Mask, dropout_mask: Bool[Array, "*#B #H #T #t"] | None, dropout_rate: float, paging_info: PagingInfo | None, q_indices: Int[Array, "*#B #H T"] | None, k_indices: Int[Array, "*#B #h t"] | None, normalize_output: bool, chunk_size: int)` — [`L237`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/xla_chunked.py#L237) — Computes paged attention.
- `bq_loop_fn(bidx, q_batch)` — [`L265`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/xla_chunked.py#L265)
- `chunk(q, k, v, bias, mask, accum, x_max, denom)` — [`L65`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/xla_chunked.py#L65)
- `get_chunk(x, idx, size, axis)` — [`L157`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/xla_chunked.py#L157)
- `get_kv_chunk(x, axis)` — [`L181`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/xla_chunked.py#L181)
- `get_q_chunk(x, axis)` — [`L165`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/xla_chunked.py#L165)
- `kv_loop_fn(_, carry, *, kv_chunk_size)` — [`L178`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/xla_chunked.py#L178)
- `kv_loop_fn(i, carry)` — [`L275`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/xla_chunked.py#L275)
- `q_loop_fn(q_chunk_idx, _, *, q_chunk_size)` — [`L164`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/xla_chunked.py#L164)
- `sq_loop_fn(_, q_chunk)` — [`L269`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/xla_chunked.py#L269)

## Module values
- `Mask` — [`L35`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/xla_chunked.py#L35)
- `PagingInfo` — [`L37`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/xla_chunked.py#L37)
- `QArray` — [`L36`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/xla_chunked.py#L36)


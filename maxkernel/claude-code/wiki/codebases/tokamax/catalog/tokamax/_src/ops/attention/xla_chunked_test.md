---
title: 'Module: tokamax/_src/ops/attention/xla_chunked_test.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/attention/xla_chunked_test.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.attention.xla_chunked_test`/
symbols:
  XlaPagedAttentionTest._run_test_with_inputs: XlaPagedAttentionTest#_run_test_with_inputs().
  XlaChunkedAttentionTest.__init__: XlaChunkedAttentionTest#__init__().
  XlaChunkedAttentionTest.test_chunk_sizes: XlaChunkedAttentionTest#test_chunk_sizes().
  XlaPagedAttentionTest.__init__: XlaPagedAttentionTest#__init__().
  _concatenate_pages: _concatenate_pages().
  XlaChunkedAttentionTest.test_normalize_output: XlaChunkedAttentionTest#test_normalize_output().
  XlaChunkedAttentionTest._run_test: XlaChunkedAttentionTest#_run_test().
  XlaChunkedAttentionTest: XlaChunkedAttentionTest#
  XlaChunkedAttentionTest._run_test_with_inputs: XlaChunkedAttentionTest#_run_test_with_inputs().
  XlaPagedAttentionTest: XlaPagedAttentionTest#
  _concatenate_pages.pad_cache: _concatenate_pages().pad_cache().
  _split_into_pages: _split_into_pages().
  _concatenate_pages.pad_mask: _concatenate_pages().pad_mask().
  XlaPagedAttentionTest.ref_impl: XlaPagedAttentionTest#ref_impl().
  XlaPagedAttentionTest.test_normalize_output: XlaPagedAttentionTest#test_normalize_output().
  XlaPagedAttentionTest._test_invalid_shapes: XlaPagedAttentionTest#_test_invalid_shapes().
  XlaPagedAttentionTest._test_quantized_int8: XlaPagedAttentionTest#_test_quantized_int8().
  XlaPagedAttentionTest._test_quantized_int4: XlaPagedAttentionTest#_test_quantized_int4().
---
# Module: [`tokamax/_src/ops/attention/xla_chunked_test.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/xla_chunked_test.py)

## Classes
### `XlaChunkedAttentionTest`  ·  implements/extends AttentionTestBase
- def: [`tokamax/_src/ops/attention/xla_chunked_test.py:34`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/xla_chunked_test.py#L34)
- signature: `class XlaChunkedAttentionTest(test_base.AttentionTestBase):`
- members:
  - `test_chunk_sizes(self, *chunk_size)` — [`L58`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/xla_chunked_test.py#L58)
  - `test_normalize_output(self)` — [`L65`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/xla_chunked_test.py#L65)
- protocol/private: `__init__`[`L36`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/xla_chunked_test.py#L36), `_run_test`[`L40`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/xla_chunked_test.py#L40), `_run_test_with_inputs`[`L52`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/xla_chunked_test.py#L52)
- uses (calls/refs, reference-scoped): [`XlaChunkedDotProductAttention`](xla_chunked.md#XlaChunkedDotProductAttention), [`chunk_size`](xla_chunked.md#XlaChunkedDotProductAttention.chunk_size)  (7 test-only)
- used by: (4 test-only callers)

### `XlaPagedAttentionTest`  ·  implements/extends AttentionTestBase
- def: [`tokamax/_src/ops/attention/xla_chunked_test.py:160`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/xla_chunked_test.py#L160)
- signature: `class XlaPagedAttentionTest(test_base.AttentionTestBase):`
- members:
  - `_run_test_with_inputs(self, *args, **kwargs)` — [`L173`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/xla_chunked_test.py#L173) — Initialize a random KV cache and split it into pages of variable lens.
  - `ref_impl(q, k_pages, v_pages, *args, **kwargs)` — [`L220`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/xla_chunked_test.py#L220)
  - `test_normalize_output(self)` — [`L245`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/xla_chunked_test.py#L245)
- protocol/private: `__init__`[`L162`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/xla_chunked_test.py#L162), `_test_invalid_shapes`[`L249`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/xla_chunked_test.py#L249), `_test_quantized_int4`[`L257`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/xla_chunked_test.py#L257), `_test_quantized_int8`[`L253`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/xla_chunked_test.py#L253)
- uses (calls/refs, reference-scoped): [`is_causal`](base.md#Mask.is_causal), [`XlaChunkedDotProductAttention`](xla_chunked.md#XlaChunkedDotProductAttention), [`PagingInfo`](xla_chunked.md#PagingInfo), [`chunk_size`](xla_chunked.md#XlaChunkedDotProductAttention.chunk_size), [`Mask`](xla_chunked.md#Mask), [`_concatenate_pages`](xla_chunked_test.md#_concatenate_pages), [`_split_into_pages`](xla_chunked_test.md#_split_into_pages)  (3 test-only)
- used by: (6 test-only callers)

## Functions
- `_concatenate_pages(k_pages, v_pages, max_num_active_pages, num_active_pages, active_page_indices, max_page_size)` — [`L121`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/xla_chunked_test.py#L121)
- `_split_into_pages(unpadded_k, unpadded_v, max_num_pages, max_page_size)` — [`L70`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/xla_chunked_test.py#L70)
- `pad_cache(x)` — [`L140`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/xla_chunked_test.py#L140)
- `pad_mask(x)` — [`L144`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/xla_chunked_test.py#L144)


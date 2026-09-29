---
title: 'Module: ejkernel/kernels/_pallas/tpu/prefill_page_attention/_pallas_impl_fwd.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/prefill_page_attention/_pallas_impl_fwd.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.prefill_page_attention._pallas_impl_fwd`/
symbols:
  MultiPageAsyncCopyDescriptor._make_async_copy: MultiPageAsyncCopyDescriptor#_make_async_copy().
  chunked_prefill_attention_kernel.per_kv_chunk_body.body: chunked_prefill_attention_kernel().per_kv_chunk_body().body().
  MultiPageAsyncCopyDescriptor._async_copies: MultiPageAsyncCopyDescriptor#_async_copies.
  MultiPageAsyncCopyDescriptor.wait_and_get_loaded: MultiPageAsyncCopyDescriptor#wait_and_get_loaded().
  MultiPageAsyncCopyDescriptor.start: MultiPageAsyncCopyDescriptor#start().
  chunked_prefill_attention_kernel.create_kv_async_copy_descriptors: chunked_prefill_attention_kernel().create_kv_async_copy_descriptors().
  chunked_prefill_attention_kernel.per_kv_chunk_body.body.prefetch_first_kv: chunked_prefill_attention_kernel().per_kv_chunk_body().body().prefetch_first_kv().
  chunked_prefill_attention_kernel.per_kv_chunk_body.body.prefetch_next_block: chunked_prefill_attention_kernel().per_kv_chunk_body().body().prefetch_next_block().
  MultiPageAsyncCopyDescriptor._pages_hbm_ref: MultiPageAsyncCopyDescriptor#_pages_hbm_ref.
  chunked_prefill_attention_kernel: chunked_prefill_attention_kernel().
  DEFAULT_MASK_VALUE: DEFAULT_MASK_VALUE.
  MultiPageAsyncCopyDescriptor._vmem_buffer: MultiPageAsyncCopyDescriptor#_vmem_buffer.
  ref_prefill_page_attention: ref_prefill_page_attention().
  MultiPageAsyncCopyDescriptor: MultiPageAsyncCopyDescriptor#
  MultiPageAsyncCopyDescriptor._num_pages_to_load: MultiPageAsyncCopyDescriptor#_num_pages_to_load.
  MultiPageAsyncCopyDescriptor._sem: MultiPageAsyncCopyDescriptor#_sem.
  MultiPageAsyncCopyDescriptor._page_indices: MultiPageAsyncCopyDescriptor#_page_indices.
  MultiPageAsyncCopyDescriptor._page_offset: MultiPageAsyncCopyDescriptor#_page_offset.
  chunked_prefill_attention_kernel.per_kv_chunk_body: chunked_prefill_attention_kernel().per_kv_chunk_body().
  chunked_prefill_attention_kernel.per_kv_chunk_body.body.per_group_body: chunked_prefill_attention_kernel().per_kv_chunk_body().body().per_group_body().
  MultiPageAsyncCopyDescriptor.__init__: MultiPageAsyncCopyDescriptor#__init__().
---
# Module: [`ejkernel/kernels/_pallas/tpu/prefill_page_attention/_pallas_impl_fwd.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/prefill_page_attention/_pallas_impl_fwd.py)

## Classes
### `MultiPageAsyncCopyDescriptor`
- def: [`ejkernel/kernels/_pallas/tpu/prefill_page_attention/_pallas_impl_fwd.py:151`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/prefill_page_attention/_pallas_impl_fwd.py#L151)
- doc: Descriptor for asynchronous DMA copy of multiple KV cache pages from HBM to VMEM.
- signature: `class MultiPageAsyncCopyDescriptor:`
- members:
  - `_make_async_copy(self, i)` — [`L192`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/prefill_page_attention/_pallas_impl_fwd.py#L192) — Create an async DMA copy operation for a single page.
  - `start(self)` — [`L204`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/prefill_page_attention/_pallas_impl_fwd.py#L204) — Initiate all asynchronous DMA copy operations.
  - `wait_and_get_loaded(self)` — [`L214`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/prefill_page_attention/_pallas_impl_fwd.py#L214) — Wait for all async copies to complete and return the loaded data.
- protocol/private: `__init__`[`L171`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/prefill_page_attention/_pallas_impl_fwd.py#L171), `_async_copies`[`L190`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/prefill_page_attention/_pallas_impl_fwd.py#L190), `_num_pages_to_load`[`L182`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/prefill_page_attention/_pallas_impl_fwd.py#L182), `_page_indices`[`L188`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/prefill_page_attention/_pallas_impl_fwd.py#L188), `_page_offset`[`L189`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/prefill_page_attention/_pallas_impl_fwd.py#L189), `_pages_hbm_ref`[`L184`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/prefill_page_attention/_pallas_impl_fwd.py#L184), `_sem`[`L187`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/prefill_page_attention/_pallas_impl_fwd.py#L187), `_vmem_buffer`[`L181`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/prefill_page_attention/_pallas_impl_fwd.py#L181)
- used by: [`body`](_pallas_impl_fwd.md#chunked_prefill_attention_kernel.per_kv_chunk_body.body), [`create_kv_async_copy_descriptors`](_pallas_impl_fwd.md#chunked_prefill_attention_kernel.create_kv_async_copy_descriptors), [`prefetch_first_kv`](_pallas_impl_fwd.md#chunked_prefill_attention_kernel.per_kv_chunk_body.body.prefetch_first_kv), [`prefetch_next_block`](_pallas_impl_fwd.md#chunked_prefill_attention_kernel.per_kv_chunk_body.body.prefetch_next_block)

## Functions
- `body()` — [`L328`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/prefill_page_attention/_pallas_impl_fwd.py#L328)
- `chunked_prefill_attention_kernel(length_ref, page_indices_ref, buffer_index_ref, q_ref, k_pages_hbm_ref, v_pages_hbm_ref, out_ref, l_ref, m_ref, k_vmem_buffer, v_vmem_buffer, sem, *, chunk_size: int, page_size: int, num_kv_chunks: int, mask_value: float, attn_logits_soft_cap: float | None, sliding_window: int | None)` — [`L231`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/prefill_page_attention/_pallas_impl_fwd.py#L231) — Pallas kernel for chunked prefill attention with paged KV cache.
- `create_kv_async_copy_descriptors(i, buffer_index)` — [`L301`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/prefill_page_attention/_pallas_impl_fwd.py#L301) — Create async copy descriptors for KV chunk i.
- `per_group_body(group_idx, _)` — [`L366`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/prefill_page_attention/_pallas_impl_fwd.py#L366)
- `per_kv_chunk_body(i, _)` — [`L324`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/prefill_page_attention/_pallas_impl_fwd.py#L324) — Process KV chunk i.
- `prefetch_first_kv()` — [`L332`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/prefill_page_attention/_pallas_impl_fwd.py#L332)
- `prefetch_next_block()` — [`L340`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/prefill_page_attention/_pallas_impl_fwd.py#L340)
- `ref_prefill_page_attention(query: jax.Array, key_cache: jax.Array, value_cache: jax.Array, context_len: jax.Array, page_indices: jax.Array, *, softmax_scale: float | None = None, mask_value: float = DEFAULT_MASK_VALUE, attn_logits_soft_cap: float | None = None, sliding_window: int | None = None)` — [`L75`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/prefill_page_attention/_pallas_impl_fwd.py#L75) — Reference implementation of chunked prefill paged attention for testing.

## Module values
- `DEFAULT_MASK_VALUE` — [`L72`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/prefill_page_attention/_pallas_impl_fwd.py#L72)


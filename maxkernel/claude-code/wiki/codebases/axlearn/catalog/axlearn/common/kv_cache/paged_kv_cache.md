---
title: 'Module: axlearn/common/kv_cache/paged_kv_cache.py'
type: catalog
provenance: extracted
module: axlearn/common/kv_cache/paged_kv_cache.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.kv_cache.paged_kv_cache`/PagedKVCache#
symbols:
  PagedKVCache.as_dense_kv: as_dense_kv().
  PagedKVCache: ''
  PagedKVCache.init_states: init_states().
  PagedKVCache.extend_step: extend_step().
  PagedKVCache.update_kv_pages: update_kv_pages().
  PagedKVCache.PADDING_PAGE_ID: PADDING_PAGE_ID.
---
# Module: [`axlearn/common/kv_cache/paged_kv_cache.py`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_cache.py)

## Classes
### `PagedKVCache`  ·  implements/extends KVCache
- def: [`axlearn/common/kv_cache/paged_kv_cache.py:25`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_cache.py#L25)
- doc: Paged KV cache.
- signature: `class PagedKVCache(KVCache):`
- members:
  - `as_dense_kv(cls, kv_state: Union[KVState, PagedKVStorage])` — [`L42`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_cache.py#L42) — Materialise dense `(k, v)` for any paged attention input.
  - `extend_step(self, cached_states: Nested[Tensor], *, k_proj: Tensor, v_proj: Tensor, key_positions: Tensor, segment_ids: Optional[Tensor] = None, page_pool: Optional[Nested[Tensor]] = None)` — [`L71`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_cache.py#L71) — Extend the cache with the new key and value.
  - `init_states(self, shape: KVCache.Shape, *, dtype: jnp.dtype)` — [`L61`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_cache.py#L61) — Initialize the KV States.
  - `update_kv_pages(kv_pages, page_indices, key_positions, kv_proj)` — [`L154`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_cache.py#L154)
  - `PADDING_PAGE_ID` — [`L39`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_cache.py#L39)
- uses (calls/refs, reference-scoped): [`Tensor`](../utils.md#Tensor), [`Nested`](../utils.md#Nested), [`KVState`](base_kv_cache.md#KVState), [`KVCache`](kv_cache.md#KVCache), [`nowrap`](../module.md#nowrap), [`k_proj`](base_kv_cache.md#KVState.k_proj), [`v_proj`](base_kv_cache.md#KVState.v_proj), [`as_dense_kv`](base_kv_cache.md#BaseKVCache.as_dense_kv), [`Output`](base_kv_cache.md#BaseKVCache.Output), [`reconstruct_kv`](paged_kv_storage.md#reconstruct_kv), [`tpu_scatter_update_pages_shmap_fn`](paged_kv_cache_tpu_kernel.md#tpu_scatter_update_pages_shmap_fn), [`scatter_update_pages_kernel`](paged_kv_storage.md#scatter_update_pages_kernel), [`gpu_scatter_update_pages_shmap_fn`](paged_kv_cache_gpu_kernel.md#gpu_scatter_update_pages_shmap_fn), [`page_indices`](base_kv_cache.md#KVState.page_indices), [`Shape`](base_kv_cache.md#BaseKVCache.Shape), [`scatter_update_pages`](paged_kv_storage.md#scatter_update_pages), [`as_dense`](paged_kv_storage.md#PagedKVStorage.as_dense), [`PagedKVStorage`](paged_kv_storage.md#PagedKVStorage)
- used by: `_test_extend_step`, `test_extend_step`, [`_prepare_layers`](../flash_attention/layer_test.md#_prepare_layers), `test_gqa_extend_step`, [`_test_extend_step`](../flash_attention/layer_test.md#TestFlashAttention._test_extend_step), [`KVCache`](kv_cache.md#KVCache), `_test_prefill_states`, [`flash_attention_implementation`](../flash_attention/utils.md#flash_attention_implementation), [`is_supported`](../flash_attention/common.md#BaseSingleStepDecoding.is_supported), [`init_states`](kv_cache.md#KVCache.init_states), [`is_supported`](../flash_attention/common.md#BasePagedAttention.is_supported), [`is_supported`](../flash_attention/common.md#ReferenceMHA.is_supported), [`extend_step`](kv_cache.md#KVCache.extend_step), [`as_dense_kv`](base_kv_cache.md#BaseKVCache.as_dense_kv)


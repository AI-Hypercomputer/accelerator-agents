---
title: 'Module: axlearn/common/kv_cache/paged_kv_cache_test.py'
type: catalog
provenance: extracted
module: axlearn/common/kv_cache/paged_kv_cache_test.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.kv_cache.paged_kv_cache_test`/
symbols:
  test_fns: test_fns.
  ScatterUpdatePagesTest: ScatterUpdatePagesTest#
  ScatterUpdatePagesTest.test_scatter_update: ScatterUpdatePagesTest#test_scatter_update().
  PagedKVCacheTest: PagedKVCacheTest#
  PagedKVCacheTest.test_paged_kv_cache: PagedKVCacheTest#test_paged_kv_cache().
  PagedKVCacheAsDenseKvTest: PagedKVCacheAsDenseKvTest#
  PagedKVCacheAsDenseKvTest._make_bf16_storage: PagedKVCacheAsDenseKvTest#_make_bf16_storage().
  PagedKVCacheAsDenseKvTest.test_dense_kv_state_falls_through_to_base: PagedKVCacheAsDenseKvTest#test_dense_kv_state_falls_through_to_base().
  PagedKVCacheAsDenseKvTest.test_kv_state_with_page_indices_is_reconstructed: PagedKVCacheAsDenseKvTest#test_kv_state_with_page_indices_is_reconstructed().
  PagedKVCacheAsDenseKvTest.test_paged_storage_returns_dense_reconstruction: PagedKVCacheAsDenseKvTest#test_paged_storage_returns_dense_reconstruction().
---
# Module: [`axlearn/common/kv_cache/paged_kv_cache_test.py`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_cache_test.py)

## Classes
### `PagedKVCacheAsDenseKvTest`
- def: [`axlearn/common/kv_cache/paged_kv_cache_test.py:163`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_cache_test.py#L163)
- doc: Freeze `PagedKVCache.as_dense_kv` dispatch.
- signature: `class PagedKVCacheAsDenseKvTest(TestCase):`
- members:
  - `test_dense_kv_state_falls_through_to_base(self)` — [`L203`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_cache_test.py#L203)
  - `test_kv_state_with_page_indices_is_reconstructed(self)` — [`L215`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_cache_test.py#L215)
  - `test_paged_storage_returns_dense_reconstruction(self)` — [`L252`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_cache_test.py#L252)
- protocol/private: `_make_bf16_storage`[`L171`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_cache_test.py#L171)

### `PagedKVCacheTest`
- def: [`axlearn/common/kv_cache/paged_kv_cache_test.py:70`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_cache_test.py#L70)
- signature: `class PagedKVCacheTest(TestCase):`
- members:
  - `test_paged_kv_cache(self, time_step_value, cache_dtype, max_pages_each_request, page_size)` — [`L77`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_cache_test.py#L77)

### `ScatterUpdatePagesTest`
- def: [`axlearn/common/kv_cache/paged_kv_cache_test.py:31`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_cache_test.py#L31)
- signature: `class ScatterUpdatePagesTest(TestCase):`
- members:
  - `test_scatter_update(self, batch_size, page_size, num_heads, fn)` — [`L38`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_cache_test.py#L38)

## Module values
- `test_fns` — [`L24`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_cache_test.py#L24)


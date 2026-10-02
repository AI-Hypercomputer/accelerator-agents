---
title: 'Module: easydel/inference/esurge/core/utils.py'
type: catalog
provenance: extracted
module: easydel/inference/esurge/core/utils.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.esurge.core.utils`/
symbols:
  CachePage: CachePage#
  CachePage.next_free_page: CachePage#next_free_page.
  FreeCachePageQueue.fake_free_list_tail: FreeCachePageQueue#fake_free_list_tail.
  CachePage.prev_free_page: CachePage#prev_free_page.
  PageHash: PageHash#
  FreeCachePageQueue.num_free_pages: FreeCachePageQueue#num_free_pages.
  FreeCachePageQueue.fake_free_list_head: FreeCachePageQueue#fake_free_list_head.
  FreeCachePageQueue.popleft: FreeCachePageQueue#popleft().
  hash_request_tokens: hash_request_tokens().
  CachePage.page_id: CachePage#page_id.
  CachePage.page_hash: CachePage#page_hash().
  FreeCachePageQueue.popleft_n: FreeCachePageQueue#popleft_n().
  FreeCachePageQueue.append_n: FreeCachePageQueue#append_n().
  CachePage.__repr__: CachePage#__repr__().
  FreeCachePageQueue.append: FreeCachePageQueue#append().
  CachePage.ref_cnt: CachePage#ref_cnt.
  FreeCachePageQueue.remove: FreeCachePageQueue#remove().
  FreeCachePageQueue.get_all_free_pages: FreeCachePageQueue#get_all_free_pages().
  hash_page_tokens: hash_page_tokens().
  CachePage._page_hash: CachePage#_page_hash.
  PageHashWithGroupId: PageHashWithGroupId#
  PageHashWithGroupId.get_hash_value: PageHashWithGroupId#get_hash_value().
  CachePage.is_null: CachePage#is_null.
  CachePage.decr_ref: CachePage#decr_ref().
  CachePage.reset_hash: CachePage#reset_hash().
  init_none_hash: init_none_hash().
  PageHashWithGroupId.page_hash: PageHashWithGroupId#page_hash.
  CachePage.incr_ref: CachePage#incr_ref().
  PageHash.hash_value: PageHash#hash_value.
  none_hash.none_hash: none_hash.none_hash.
  FreeCachePageQueue.__init__: FreeCachePageQueue#__init__().
  FreeCachePageQueue: FreeCachePageQueue#
  PageHash.token_ids: PageHash#token_ids.
  PageHash.extra_keys: PageHash#extra_keys.
  PageHashWithGroupId.group_id: PageHashWithGroupId#group_id.
---
# Module: [`easydel/inference/esurge/core/utils.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/utils.py)

## Classes
### `CachePage`
- def: [`easydel/inference/esurge/core/utils.py:96`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/utils.py#L96)
- doc: Metadata container for a single KV-cache page.
- signature: `class CachePage:`
- members:
  - `__repr__(self)` — [`L177`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/utils.py#L177) — Return a detailed string representation of the page.
  - `decr_ref(self)` — [`L136`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/utils.py#L136) — Decrement the reference count by 1.
  - `incr_ref(self)` — [`L128`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/utils.py#L128) — Increment the reference count by 1.
  - `page_hash(self)` — [`L147`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/utils.py#L147) — Get the page's hash for prefix caching.
  - `reset_hash(self)` — [`L169`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/utils.py#L169) — Clear the page hash when the page is evicted.
  - `is_null` — [`L126`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/utils.py#L126)
  - `next_free_page` — [`L125`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/utils.py#L125)
  - `page_id` — [`L121`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/utils.py#L121)
  - `prev_free_page` — [`L124`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/utils.py#L124)
  - `ref_cnt` — [`L122`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/utils.py#L122)
- protocol/private: `_page_hash`[`L123`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/utils.py#L123)
- uses (calls/refs, reference-scoped): [`PageHashWithGroupId`](utils.md#PageHashWithGroupId)
- used by: [`find_longest_cache_hit`](coordinator.md#HybridCacheCoordinator.find_longest_cache_hit), [`get_new_pages`](page_pool.md#PagePool.get_new_pages), [`cache_full_pages`](page_pool.md#PagePool.cache_full_pages), [`fake_free_list_tail`](utils.md#FreeCachePageQueue.fake_free_list_tail), [`find_longest_cache_hit`](single_type_cache_manager.md#SlidingWindowManager.find_longest_cache_hit), [`get_cached_page`](page_pool.md#PagePool.get_cached_page), [`find_longest_cache_hit`](single_type_cache_manager.md#ChunkedLocalAttentionManager.find_longest_cache_hit), [`find_longest_cache_hit`](single_type_cache_manager.md#SingleTypeCacheManager.find_longest_cache_hit), [`find_longest_cache_hit`](single_type_cache_manager.md#FullAttentionManager.find_longest_cache_hit), [`find_longest_cache_hit`](coordinator.md#UnitaryCacheCoordinator.find_longest_cache_hit), [`remove_skipped_pages`](single_type_cache_manager.md#ChunkedLocalAttentionManager.remove_skipped_pages), [`allocate_new_pages`](single_type_cache_manager.md#SingleTypeCacheManager.allocate_new_pages), [`num_free_pages`](utils.md#FreeCachePageQueue.num_free_pages), [`fake_free_list_head`](utils.md#FreeCachePageQueue.fake_free_list_head), [`remove_skipped_pages`](single_type_cache_manager.md#SlidingWindowManager.remove_skipped_pages), [`free_pages`](page_pool.md#PagePool.free_pages), [`_infer_dp_shard_from_pages`](manager.md#CacheManager._infer_dp_shard_from_pages), [`get_num_pages_to_allocate`](single_type_cache_manager.md#SingleTypeCacheManager.get_num_pages_to_allocate), [`null_page`](page_pool.md#PagePool.null_page), [`popleft`](utils.md#FreeCachePageQueue.popleft), [`reset_prefix_cache`](page_pool.md#PagePool.reset_prefix_cache), [`touch`](page_pool.md#PagePool.touch), [`req_to_pages`](single_type_cache_manager.md#SingleTypeCacheManager.req_to_pages), [`_maybe_evict_cached_page`](page_pool.md#PagePool._maybe_evict_cached_page), [`append_n`](utils.md#FreeCachePageQueue.append_n), [`find_longest_cache_hit`](coordinator.md#CacheCoordinator.find_longest_cache_hit), [`popleft_n`](utils.md#FreeCachePageQueue.popleft_n), [`append`](utils.md#FreeCachePageQueue.append), [`find_longest_cache_hit`](single_type_cache_manager.md#MambaManager.find_longest_cache_hit), [`remove`](utils.md#FreeCachePageQueue.remove), [`cached_page_hash_to_page`](page_pool.md#PagePool.cached_page_hash_to_page), [`get_pages`](coordinator.md#CacheCoordinator.get_pages), [`pages`](manager.md#CachePages.pages), [`allocate_new_pages`](coordinator.md#CacheCoordinator.allocate_new_pages), [`get_all_free_pages`](utils.md#FreeCachePageQueue.get_all_free_pages), [`get_num_pages_to_allocate`](coordinator.md#CacheCoordinator.get_num_pages_to_allocate), [`rollback_new_computed_pages`](coordinator.md#CacheCoordinator.rollback_new_computed_pages), [`rollback_new_computed_pages`](single_type_cache_manager.md#SingleTypeCacheManager.rollback_new_computed_pages), [`save_new_computed_pages`](coordinator.md#CacheCoordinator.save_new_computed_pages), [`save_new_computed_pages`](single_type_cache_manager.md#SingleTypeCacheManager.save_new_computed_pages)  (+7 more)

### `FreeCachePageQueue`
- def: [`easydel/inference/esurge/core/utils.py:194`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/utils.py#L194)
- doc: Doubly-linked list for O(1) free page queue operations.
- signature: `class FreeCachePageQueue:`
- members:
  - `__init__(self, pages: list[CachePage])` — [`L224`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/utils.py#L224) — Initialize the free page queue with a list of pages.
  - `append(self, page: CachePage)` — [`L326`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/utils.py#L326) — Put a page back into the free list and increase
  - `append_n(self, pages: list[CachePage])` — [`L345`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/utils.py#L345) — Put a list of pages back into the free list
  - `get_all_free_pages(self)` — [`L366`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/utils.py#L366) — Get all free pages in the free list. Mainly used for testing.
  - `popleft(self)` — [`L253`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/utils.py#L253) — Pop the first free page and reduce num_free_pages by 1.
  - `popleft_n(self, n: int)` — [`L279`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/utils.py#L279) — Pop the first n free pages and reduce num_free_pages by n.
  - `remove(self, page: CachePage)` — [`L310`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/utils.py#L310) — Remove a page in the free list and reduce num_free_pages by 1.
  - `fake_free_list_head` — [`L242`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/utils.py#L242)
  - `fake_free_list_tail` — [`L243`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/utils.py#L243)
  - `num_free_pages` — [`L234`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/utils.py#L234)
- uses (calls/refs, reference-scoped): [`CachePage`](utils.md#CachePage), [`next_free_page`](utils.md#CachePage.next_free_page), [`prev_free_page`](utils.md#CachePage.prev_free_page), [`page_id`](utils.md#CachePage.page_id)
- used by: [`get_new_pages`](page_pool.md#PagePool.get_new_pages), [`free_pages`](page_pool.md#PagePool.free_pages), [`null_page`](page_pool.md#PagePool.null_page), [`touch`](page_pool.md#PagePool.touch), [`free_page_queue`](page_pool.md#PagePool.free_page_queue), [`get_num_free_pages`](page_pool.md#PagePool.get_num_free_pages)

### `PageHash`  ·  implements/extends NamedTuple
- def: [`easydel/inference/esurge/core/utils.py:47`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/utils.py#L47)
- doc: Hash representation of a cache page for prefix caching.
- signature: `class PageHash(NamedTuple):`
- members:
  - `extra_keys` — [`L69`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/utils.py#L69)
  - `hash_value` — [`L67`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/utils.py#L67)
  - `token_ids` — [`L68`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/utils.py#L68)
- used by: [`find_longest_cache_hit`](coordinator.md#HybridCacheCoordinator.find_longest_cache_hit), [`cache_full_pages`](page_pool.md#PagePool.cache_full_pages), [`find_longest_cache_hit`](single_type_cache_manager.md#SlidingWindowManager.find_longest_cache_hit), [`cache_pages`](single_type_cache_manager.md#SingleTypeCacheManager.cache_pages), [`get_cached_page`](page_pool.md#PagePool.get_cached_page), [`find_longest_cache_hit`](single_type_cache_manager.md#ChunkedLocalAttentionManager.find_longest_cache_hit), [`find_longest_cache_hit`](single_type_cache_manager.md#SingleTypeCacheManager.find_longest_cache_hit), [`find_longest_cache_hit`](single_type_cache_manager.md#FullAttentionManager.find_longest_cache_hit), [`find_longest_cache_hit`](coordinator.md#UnitaryCacheCoordinator.find_longest_cache_hit), [`hash_request_tokens`](utils.md#hash_request_tokens), [`find_longest_cache_hit`](coordinator.md#CacheCoordinator.find_longest_cache_hit), [`find_longest_cache_hit`](single_type_cache_manager.md#MambaManager.find_longest_cache_hit), [`cache_pages`](coordinator.md#CacheCoordinator.cache_pages), [`hash_page_tokens`](utils.md#hash_page_tokens), [`find_longest_cache_hit`](coordinator.md#CacheCoordinatorNoPrefixCache.find_longest_cache_hit), [`req_to_page_hashes`](manager.md#CacheManager.req_to_page_hashes), [`get_hash_value`](utils.md#PageHashWithGroupId.get_hash_value), [`page_hash`](utils.md#PageHashWithGroupId.page_hash)

### `PageHashWithGroupId`  ·  implements/extends NamedTuple
- def: [`easydel/inference/esurge/core/utils.py:72`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/utils.py#L72)
- doc: Page hash combined with its KV cache group identifier.
- signature: `class PageHashWithGroupId(NamedTuple):`
- members:
  - `get_hash_value(self)` — [`L86`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/utils.py#L86) — Get the underlying hash value.
  - `group_id` — [`L84`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/utils.py#L84)
  - `page_hash` — [`L83`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/utils.py#L83)
- uses (calls/refs, reference-scoped): [`PageHash`](utils.md#PageHash), [`hash_value`](utils.md#PageHash.hash_value)
- used by: [`CachePage`](utils.md#CachePage), [`cache_full_pages`](page_pool.md#PagePool.cache_full_pages), [`get_cached_page`](page_pool.md#PagePool.get_cached_page), [`page_hash`](utils.md#CachePage.page_hash), [`cached_page_hash_to_page`](page_pool.md#PagePool.cached_page_hash_to_page), [`_page_hash`](utils.md#CachePage._page_hash)

## Functions
- `hash_page_tokens(hash_function: Callable, parent_page_hash: int | None, curr_page_token_ids: Sequence[int], extra_keys: tuple[Any, ...] | None = None)` — [`L416`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/utils.py#L416) — Computes a hash value corresponding to the contents of a page and
- `hash_request_tokens(hash_function: Any, page_size: int, request: EngineRequest)` — [`L449`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/utils.py#L449) — Compute hash values for all complete pages in a request's token sequence.
- `init_none_hash()` — [`L388`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/utils.py#L388) — Initialize the global none_hash used for prefix caching.

## Module values
- `none_hash` — [`L384`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/utils.py#L384)


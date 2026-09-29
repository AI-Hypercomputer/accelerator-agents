---
title: 'Module: easydel/inference/esurge/core/page_pool.py'
type: catalog
provenance: extracted
module: easydel/inference/esurge/core/page_pool.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.esurge.core.page_pool`/PagePool#
symbols:
  PagePool.get_new_pages: get_new_pages().
  PagePool.cache_full_pages: cache_full_pages().
  PagePool.get_cached_page: get_cached_page().
  PagePool.free_pages: free_pages().
  PagePool.null_page: null_page.
  PagePool.touch: touch().
  PagePool.reset_prefix_cache: reset_prefix_cache().
  PagePool._maybe_evict_cached_page: _maybe_evict_cached_page().
  PagePool: ''
  PagePool.free_page_queue: free_page_queue.
  PagePool.get_num_free_pages: get_num_free_pages().
  PagePool.cached_page_hash_to_page: cached_page_hash_to_page.
  PagePool.num_pages: num_pages.
  PagePool.get_usage: get_usage().
  PagePool.pages: pages.
  PagePool.enable_caching: enable_caching.
  PagePool.__init__: __init__().
---
# Module: [`easydel/inference/esurge/core/page_pool.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/page_pool.py)

## Classes
### `PagePool`
- def: [`easydel/inference/esurge/core/page_pool.py:38`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/page_pool.py#L38)
- doc: Pool manager for KV-cache pages with prefix caching support.
- signature: `class PagePool:`
- members:
  - `__init__(self, num_pages: int, enable_caching: bool)` — [`L67`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/page_pool.py#L67) — Initialize the page pool.
  - `_maybe_evict_cached_page(self, page: CachePage)` — [`L285`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/page_pool.py#L285) — Evict a page from the prefix cache if it is cached.
  - `cache_full_pages(self, request: EngineRequest, pages: list[CachePage], page_hashes: list[PageHash], num_cached_pages: int, num_full_pages: int, page_size: int, kv_cache_group_id: int)` — [`L146`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/page_pool.py#L146) — Cache a list of full pages for prefix caching.
  - `free_pages(self, ordered_pages: Iterable[CachePage])` — [`L329`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/page_pool.py#L329) — Free a list of pages. The pages should be ordered by their
  - `get_cached_page(self, page_hash: PageHash, kv_cache_group_ids: list[int], *, dp_shard_hint: int | None = None, data_parallel_size: int | None = None)` — [`L90`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/page_pool.py#L90) — Get the cached page by the page hash for each group in
  - `get_new_pages(self, num_pages: int, *, dp_shard_hint: int | None = None, data_parallel_size: int | None = None)` — [`L221`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/page_pool.py#L221) — Get new pages from the free page pool.
  - `get_num_free_pages(self)` — [`L364`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/page_pool.py#L364) — Get the number of free pages in the pool.
  - `get_usage(self)` — [`L372`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/page_pool.py#L372) — Get the KV cache usage.
  - `reset_prefix_cache(self)` — [`L343`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/page_pool.py#L343) — Reset prefix cache. This function may be used in RLHF
  - `touch(self, pages: tuple[list[CachePage], ...])` — [`L315`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/page_pool.py#L315) — Touch a page increases its reference count by 1, and may remove
  - `cached_page_hash_to_page` — [`L86`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/page_pool.py#L86)
  - `enable_caching` — [`L80`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/page_pool.py#L80)
  - `free_page_queue` — [`L84`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/page_pool.py#L84)
  - `null_page` — [`L87`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/page_pool.py#L87)
  - `num_pages` — [`L79`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/page_pool.py#L79)
  - `pages` — [`L82`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/page_pool.py#L82)
- uses (calls/refs, reference-scoped): [`logger`](../logger.md#logger), [`EngineRequest`](../request.md#EngineRequest), [`CachePage`](utils.md#CachePage), [`request_id`](../request.md#EngineRequest.request_id), [`PageHash`](utils.md#PageHash), [`num_free_pages`](utils.md#FreeCachePageQueue.num_free_pages), [`pages_per_dp_shard`](dp_sharding.md#pages_per_dp_shard), [`popleft`](utils.md#FreeCachePageQueue.popleft), [`page_hash`](utils.md#CachePage.page_hash), [`page_id`](utils.md#CachePage.page_id), [`append_n`](utils.md#FreeCachePageQueue.append_n), [`popleft_n`](utils.md#FreeCachePageQueue.popleft_n), [`ref_cnt`](utils.md#CachePage.ref_cnt), [`remove`](utils.md#FreeCachePageQueue.remove), [`dp_shard_page_bounds`](dp_sharding.md#dp_shard_page_bounds), [`get_all_free_pages`](utils.md#FreeCachePageQueue.get_all_free_pages), [`all_token_ids`](../request.md#EngineRequest.all_token_ids), [`dp_shard_for_page_id`](dp_sharding.md#dp_shard_for_page_id), [`hash_page_tokens`](utils.md#hash_page_tokens), [`PageHashWithGroupId`](utils.md#PageHashWithGroupId), [`get_hash_value`](utils.md#PageHashWithGroupId.get_hash_value), [`is_null`](utils.md#CachePage.is_null), [`reset_hash`](utils.md#CachePage.reset_hash), [`incr_ref`](utils.md#CachePage.incr_ref), [`hash_value`](utils.md#PageHash.hash_value), [`FreeCachePageQueue`](utils.md#FreeCachePageQueue)
- used by: [`schedule`](../scheduler/scheduler.md#Scheduler.schedule), [`allocate_slots`](manager.md#CacheManager.allocate_slots), [`find_longest_cache_hit`](single_type_cache_manager.md#SlidingWindowManager.find_longest_cache_hit), [`cache_pages`](single_type_cache_manager.md#SingleTypeCacheManager.cache_pages), [`find_longest_cache_hit`](single_type_cache_manager.md#ChunkedLocalAttentionManager.find_longest_cache_hit), [`find_longest_cache_hit`](single_type_cache_manager.md#SingleTypeCacheManager.find_longest_cache_hit), [`find_longest_cache_hit`](single_type_cache_manager.md#FullAttentionManager.find_longest_cache_hit), [`remove_skipped_pages`](single_type_cache_manager.md#ChunkedLocalAttentionManager.remove_skipped_pages), [`allocate_new_pages`](single_type_cache_manager.md#SingleTypeCacheManager.allocate_new_pages), [`begin_cycle`](../scheduler/token_budget.md#TokenBudgetManager.begin_cycle), [`remove_skipped_pages`](single_type_cache_manager.md#SlidingWindowManager.remove_skipped_pages), [`find_longest_cache_hit`](single_type_cache_manager.md#MambaManager.find_longest_cache_hit), [`free`](single_type_cache_manager.md#SingleTypeCacheManager.free), [`page_pool`](coordinator.md#CacheCoordinator.page_pool), [`__init__`](single_type_cache_manager.md#ChunkedLocalAttentionManager.__init__), [`__init__`](single_type_cache_manager.md#SlidingWindowManager.__init__), [`__init__`](single_type_cache_manager.md#SingleTypeCacheManager.__init__), [`reset_prefix_cache`](manager.md#CacheManager.reset_prefix_cache), [`usage`](manager.md#CacheManager.usage), [`_null_page`](single_type_cache_manager.md#ChunkedLocalAttentionManager._null_page), [`_null_page`](single_type_cache_manager.md#SlidingWindowManager._null_page), [`_null_page`](single_type_cache_manager.md#SingleTypeCacheManager._null_page)


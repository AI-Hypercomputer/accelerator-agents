---
title: 'Module: easydel/inference/esurge/core/single_type_cache_manager.py'
type: catalog
provenance: extracted
module: easydel/inference/esurge/core/single_type_cache_manager.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.esurge.core.single_type_cache_manager`/
symbols:
  spec_manager_map.spec_manager_map: spec_manager_map.spec_manager_map.
  SlidingWindowManager.find_longest_cache_hit: SlidingWindowManager#find_longest_cache_hit().
  SingleTypeCacheManager.cache_pages: SingleTypeCacheManager#cache_pages().
  ChunkedLocalAttentionManager.find_longest_cache_hit: ChunkedLocalAttentionManager#find_longest_cache_hit().
  SingleTypeCacheManager.find_longest_cache_hit: SingleTypeCacheManager#find_longest_cache_hit().
  FullAttentionManager.find_longest_cache_hit: FullAttentionManager#find_longest_cache_hit().
  ChunkedLocalAttentionManager.remove_skipped_pages: ChunkedLocalAttentionManager#remove_skipped_pages().
  SingleTypeCacheManager.allocate_new_pages: SingleTypeCacheManager#allocate_new_pages().
  SingleTypeCacheManager: SingleTypeCacheManager#
  SlidingWindowManager.remove_skipped_pages: SlidingWindowManager#remove_skipped_pages().
  SingleTypeCacheManager.get_num_pages_to_allocate: SingleTypeCacheManager#get_num_pages_to_allocate().
  SingleTypeCacheManager.req_to_pages: SingleTypeCacheManager#req_to_pages.
  MambaManager.find_longest_cache_hit: MambaManager#find_longest_cache_hit().
  SingleTypeCacheManager.free: SingleTypeCacheManager#free().
  SingleTypeCacheManager.get_num_common_prefix_pages: SingleTypeCacheManager#get_num_common_prefix_pages().
  SingleTypeCacheManager.remove_skipped_pages: SingleTypeCacheManager#remove_skipped_pages().
  get_manager_for_kv_cache_spec: get_manager_for_kv_cache_spec().
  SingleTypeCacheManager.num_cached_page: SingleTypeCacheManager#num_cached_page.
  SingleTypeCacheManager.page_size: SingleTypeCacheManager#page_size.
  SingleTypeCacheManager.save_new_computed_pages: SingleTypeCacheManager#save_new_computed_pages().
  SingleTypeCacheManager.rollback_new_computed_pages: SingleTypeCacheManager#rollback_new_computed_pages().
  FullAttentionManager: FullAttentionManager#
  SlidingWindowManager.__init__: SlidingWindowManager#__init__().
  ChunkedLocalAttentionManager.__init__: ChunkedLocalAttentionManager#__init__().
  MambaManager.allocate_new_pages: MambaManager#allocate_new_pages().
  SingleTypeCacheManager.__init__: SingleTypeCacheManager#__init__().
  SlidingWindowManager: SlidingWindowManager#
  ChunkedLocalAttentionManager: ChunkedLocalAttentionManager#
  MambaManager: MambaManager#
  SingleTypeCacheManager.page_pool: SingleTypeCacheManager#page_pool.
  FullAttentionManager.get_num_common_prefix_pages: FullAttentionManager#get_num_common_prefix_pages().
  SlidingWindowManager._null_page: SlidingWindowManager#_null_page.
  ChunkedLocalAttentionManager.attention_chunk_size: ChunkedLocalAttentionManager#attention_chunk_size.
  ChunkedLocalAttentionManager._null_page: ChunkedLocalAttentionManager#_null_page.
  SlidingWindowManager.sliding_window: SlidingWindowManager#sliding_window.
  SingleTypeCacheManager._null_page: SingleTypeCacheManager#_null_page.
  SingleTypeCacheManager.kv_cache_group_id: SingleTypeCacheManager#kv_cache_group_id.
  SingleTypeCacheManager.kv_cache_spec: SingleTypeCacheManager#kv_cache_spec.
  FullAttentionManager.remove_skipped_pages: FullAttentionManager#remove_skipped_pages().
  SlidingWindowManager.get_num_common_prefix_pages: SlidingWindowManager#get_num_common_prefix_pages().
  ChunkedLocalAttentionManager.get_num_common_prefix_pages: ChunkedLocalAttentionManager#get_num_common_prefix_pages().
  MambaManager.remove_skipped_pages: MambaManager#remove_skipped_pages().
  MambaManager.get_num_common_prefix_pages: MambaManager#get_num_common_prefix_pages().
---
# Module: [`easydel/inference/esurge/core/single_type_cache_manager.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/single_type_cache_manager.py)

## Classes
### `ChunkedLocalAttentionManager`  ·  implements/extends SingleTypeCacheManager
- def: [`easydel/inference/esurge/core/single_type_cache_manager.py:528`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/single_type_cache_manager.py#L528)
- doc: Cache manager for chunked local attention layers.
- signature: `class ChunkedLocalAttentionManager(SingleTypeCacheManager):`
- members:
  - `__init__(self, kv_cache_spec: ChunkedLocalAttentionSpec, page_pool: PagePool, **kwargs)` — [`L547`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/single_type_cache_manager.py#L547) — Initialize the chunked local attention cache manager.
  - `find_longest_cache_hit(cls, page_hashes: list[PageHash], max_length: int, kv_cache_group_ids: list[int], page_pool: PagePool, kv_cache_spec: CacheSpec, use_eagle: bool, dp_shard_hint: int | None = None, data_parallel_size: int | None = None)` — [`L560`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/single_type_cache_manager.py#L560) — For chunked local attention, we need to find the longest cache hit
  - `get_num_common_prefix_pages(self, request_id: str, num_scheduled_requests: int)` — [`L658`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/single_type_cache_manager.py#L658) — Get number of common prefix pages (always 0 for chunked local).
  - `remove_skipped_pages(self, request_id: str, num_computed_tokens: int)` — [`L632`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/single_type_cache_manager.py#L632) — Remove pages outside the current attention chunk.
  - `attention_chunk_size` — [`L556`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/single_type_cache_manager.py#L556)
- protocol/private: `_null_page`[`L557`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/single_type_cache_manager.py#L557)
- uses (calls/refs, reference-scoped): [`CachePage`](utils.md#CachePage), [`get_cached_page`](page_pool.md#PagePool.get_cached_page), [`PageHash`](utils.md#PageHash), [`page_size`](interface.md#CacheSpec.page_size), [`CacheSpec`](interface.md#CacheSpec), [`SingleTypeCacheManager`](single_type_cache_manager.md#SingleTypeCacheManager), [`free_pages`](page_pool.md#PagePool.free_pages), [`null_page`](page_pool.md#PagePool.null_page), [`req_to_pages`](single_type_cache_manager.md#SingleTypeCacheManager.req_to_pages), [`PagePool`](page_pool.md#PagePool), [`ChunkedLocalAttentionSpec`](interface.md#ChunkedLocalAttentionSpec), [`num_cached_page`](single_type_cache_manager.md#SingleTypeCacheManager.num_cached_page), [`page_size`](single_type_cache_manager.md#SingleTypeCacheManager.page_size), [`__init__`](single_type_cache_manager.md#SingleTypeCacheManager.__init__), [`attention_chunk_size`](interface.md#ChunkedLocalAttentionSpec.attention_chunk_size), [`page_pool`](single_type_cache_manager.md#SingleTypeCacheManager.page_pool)
- used by: [`spec_manager_map`](single_type_cache_manager.md#spec_manager_map.spec_manager_map), [`find_longest_cache_hit`](single_type_cache_manager.md#SingleTypeCacheManager.find_longest_cache_hit), [`SingleTypeCacheManager`](single_type_cache_manager.md#SingleTypeCacheManager), [`get_num_common_prefix_pages`](single_type_cache_manager.md#SingleTypeCacheManager.get_num_common_prefix_pages), [`remove_skipped_pages`](single_type_cache_manager.md#SingleTypeCacheManager.remove_skipped_pages)

### `FullAttentionManager`  ·  implements/extends SingleTypeCacheManager
- def: [`easydel/inference/esurge/core/single_type_cache_manager.py:316`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/single_type_cache_manager.py#L316)
- doc: Cache manager for standard full/causal attention layers.
- signature: `class FullAttentionManager(SingleTypeCacheManager):`
- members:
  - `find_longest_cache_hit(cls, page_hashes: list[PageHash], max_length: int, kv_cache_group_ids: list[int], page_pool: PagePool, kv_cache_spec: CacheSpec, use_eagle: bool, dp_shard_hint: int | None = None, data_parallel_size: int | None = None)` — [`L332`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/single_type_cache_manager.py#L332) — Find the longest prefix cache hit by scanning pages sequentially from the start.
  - `get_num_common_prefix_pages(self, request_id: str, num_scheduled_requests: int)` — [`L377`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/single_type_cache_manager.py#L377) — Count pages shared by all scheduled requests.
  - `remove_skipped_pages(self, request_id: str, num_computed_tokens: int)` — [`L365`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/single_type_cache_manager.py#L365) — Remove pages that are no longer needed (no-op for full attention).
- uses (calls/refs, reference-scoped): [`CachePage`](utils.md#CachePage), [`get_cached_page`](page_pool.md#PagePool.get_cached_page), [`PageHash`](utils.md#PageHash), [`page_size`](interface.md#CacheSpec.page_size), [`CacheSpec`](interface.md#CacheSpec), [`SingleTypeCacheManager`](single_type_cache_manager.md#SingleTypeCacheManager), [`FullAttentionSpec`](interface.md#FullAttentionSpec), [`req_to_pages`](single_type_cache_manager.md#SingleTypeCacheManager.req_to_pages), [`PagePool`](page_pool.md#PagePool), [`ref_cnt`](utils.md#CachePage.ref_cnt), [`ChunkedLocalAttentionSpec`](interface.md#ChunkedLocalAttentionSpec)
- used by: [`find_longest_cache_hit`](coordinator.md#HybridCacheCoordinator.find_longest_cache_hit), [`spec_manager_map`](single_type_cache_manager.md#spec_manager_map.spec_manager_map), [`find_longest_cache_hit`](single_type_cache_manager.md#SingleTypeCacheManager.find_longest_cache_hit), [`SingleTypeCacheManager`](single_type_cache_manager.md#SingleTypeCacheManager), [`get_num_common_prefix_pages`](single_type_cache_manager.md#SingleTypeCacheManager.get_num_common_prefix_pages), [`remove_skipped_pages`](single_type_cache_manager.md#SingleTypeCacheManager.remove_skipped_pages), [`full_attention_manager_cls`](coordinator.md#HybridCacheCoordinator.full_attention_manager_cls)

### `MambaManager`  ·  implements/extends SingleTypeCacheManager
- def: [`easydel/inference/esurge/core/single_type_cache_manager.py:674`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/single_type_cache_manager.py#L674)
- doc: Cache manager for Mamba state-space model layers.
- signature: `class MambaManager(SingleTypeCacheManager):`
- members:
  - `allocate_new_pages(self, request_id: str, num_tokens: int, *, dp_shard_hint: int | None = None, data_parallel_size: int | None = None)` — [`L732`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/single_type_cache_manager.py#L732) — Allocate a single page for Mamba state storage.
  - `find_longest_cache_hit(cls, page_hashes: list[PageHash], max_length: int, kv_cache_group_ids: list[int], page_pool: PagePool, kv_cache_spec: CacheSpec, use_eagle: bool, dp_shard_hint: int | None = None, data_parallel_size: int | None = None)` — [`L690`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/single_type_cache_manager.py#L690) — Return empty pages (Mamba state is never prefix-cached).
  - `get_num_common_prefix_pages(self, request_id: str, num_scheduled_requests: int)` — [`L718`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/single_type_cache_manager.py#L718) — Get number of common prefix pages (always 0 for Mamba).
  - `remove_skipped_pages(self, request_id: str, num_computed_tokens: int)` — [`L707`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/single_type_cache_manager.py#L707) — Remove skipped pages (no-op for Mamba).
- uses (calls/refs, reference-scoped): [`CachePage`](utils.md#CachePage), [`PageHash`](utils.md#PageHash), [`CacheSpec`](interface.md#CacheSpec), [`allocate_new_pages`](single_type_cache_manager.md#SingleTypeCacheManager.allocate_new_pages), [`SingleTypeCacheManager`](single_type_cache_manager.md#SingleTypeCacheManager), [`req_to_pages`](single_type_cache_manager.md#SingleTypeCacheManager.req_to_pages), [`PagePool`](page_pool.md#PagePool), [`MambaSpec`](interface.md#MambaSpec)
- used by: [`spec_manager_map`](single_type_cache_manager.md#spec_manager_map.spec_manager_map), [`find_longest_cache_hit`](single_type_cache_manager.md#SingleTypeCacheManager.find_longest_cache_hit), [`allocate_new_pages`](single_type_cache_manager.md#SingleTypeCacheManager.allocate_new_pages), [`SingleTypeCacheManager`](single_type_cache_manager.md#SingleTypeCacheManager), [`get_num_common_prefix_pages`](single_type_cache_manager.md#SingleTypeCacheManager.get_num_common_prefix_pages), [`remove_skipped_pages`](single_type_cache_manager.md#SingleTypeCacheManager.remove_skipped_pages)

### `SingleTypeCacheManager`  ·  implements/extends ABC
- def: [`easydel/inference/esurge/core/single_type_cache_manager.py:47`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/single_type_cache_manager.py#L47)
- doc: Abstract base class for attention-type-specific cache managers.
- signature: `class SingleTypeCacheManager(ABC):`
- members:
  - `__init__(self, kv_cache_spec: CacheSpec, page_pool: PagePool, kv_cache_group_id: int)` — [`L68`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/single_type_cache_manager.py#L68) — Initialize the cache manager for a specific attention type.
  - `allocate_new_pages(self, request_id: str, num_tokens: int, *, dp_shard_hint: int | None = None, data_parallel_size: int | None = None)` — [`L167`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/single_type_cache_manager.py#L167) — Allocate new pages for the request to give it at least `num_tokens`
  - `cache_pages(self, request: EngineRequest, page_hashes: list[PageHash], num_tokens: int)` — [`L201`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/single_type_cache_manager.py#L201) — Cache the pages for the request.
  - `find_longest_cache_hit(cls, page_hashes: list[PageHash], max_length: int, kv_cache_group_ids: list[int], page_pool: PagePool, kv_cache_spec: CacheSpec, use_eagle: bool, dp_shard_hint: int | None = None, data_parallel_size: int | None = None)` — [`L258`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/single_type_cache_manager.py#L258) — Get the longest cache hit prefix of the pages that is not longer than
  - `free(self, request_id: str)` — [`L226`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/single_type_cache_manager.py#L226) — Free the pages for the request.
  - `get_num_common_prefix_pages(self, request_id: str, num_scheduled_requests: int)` — [`L242`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/single_type_cache_manager.py#L242) — Get the number of common prefix pages for a request.
  - `get_num_pages_to_allocate(self, request_id: str, num_tokens: int, new_computed_pages: list[CachePage])` — [`L94`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/single_type_cache_manager.py#L94) — Get the number of pages needed to be allocated for the request.
  - `remove_skipped_pages(self, request_id: str, num_computed_tokens: int)` — [`L303`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/single_type_cache_manager.py#L303) — Remove the pages that are no longer needed from `pages` and free the
  - `rollback_new_computed_pages(self, request_id: str, new_computed_pages: list[CachePage])` — [`L132`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/single_type_cache_manager.py#L132) — Rollback pages attached by :meth:`save_new_computed_pages`.
  - `save_new_computed_pages(self, request_id: str, new_computed_pages: list[CachePage])` — [`L115`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/single_type_cache_manager.py#L115) — Add the new computed pages to the request.
  - `kv_cache_group_id` — [`L91`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/single_type_cache_manager.py#L91)
  - `kv_cache_spec` — [`L84`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/single_type_cache_manager.py#L84)
  - `num_cached_page` — [`L89`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/single_type_cache_manager.py#L89)
  - `page_pool` — [`L85`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/single_type_cache_manager.py#L85)
  - `page_size` — [`L83`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/single_type_cache_manager.py#L83)
  - `req_to_pages` — [`L87`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/single_type_cache_manager.py#L87)
- protocol/private: `_null_page`[`L92`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/single_type_cache_manager.py#L92)
- uses (calls/refs, reference-scoped): [`EngineRequest`](../request.md#EngineRequest), [`CachePage`](utils.md#CachePage), [`request_id`](../request.md#EngineRequest.request_id), [`get_new_pages`](page_pool.md#PagePool.get_new_pages), [`cache_full_pages`](page_pool.md#PagePool.cache_full_pages), [`find_longest_cache_hit`](single_type_cache_manager.md#SlidingWindowManager.find_longest_cache_hit), [`find_longest_cache_hit`](single_type_cache_manager.md#ChunkedLocalAttentionManager.find_longest_cache_hit), [`PageHash`](utils.md#PageHash), [`page_size`](interface.md#CacheSpec.page_size), [`find_longest_cache_hit`](single_type_cache_manager.md#FullAttentionManager.find_longest_cache_hit), [`CacheSpec`](interface.md#CacheSpec), [`remove_skipped_pages`](single_type_cache_manager.md#ChunkedLocalAttentionManager.remove_skipped_pages), [`remove_skipped_pages`](single_type_cache_manager.md#SlidingWindowManager.remove_skipped_pages), [`free_pages`](page_pool.md#PagePool.free_pages), [`null_page`](page_pool.md#PagePool.null_page), [`PagePool`](page_pool.md#PagePool), [`find_longest_cache_hit`](single_type_cache_manager.md#MambaManager.find_longest_cache_hit), [`ref_cnt`](utils.md#CachePage.ref_cnt), [`cdiv`](../utils.md#cdiv), [`FullAttentionManager`](single_type_cache_manager.md#FullAttentionManager), [`allocate_new_pages`](single_type_cache_manager.md#MambaManager.allocate_new_pages), [`ChunkedLocalAttentionManager`](single_type_cache_manager.md#ChunkedLocalAttentionManager), [`MambaManager`](single_type_cache_manager.md#MambaManager), [`SlidingWindowManager`](single_type_cache_manager.md#SlidingWindowManager), [`get_num_common_prefix_pages`](single_type_cache_manager.md#FullAttentionManager.get_num_common_prefix_pages), [`is_null`](utils.md#CachePage.is_null), [`get_num_common_prefix_pages`](single_type_cache_manager.md#ChunkedLocalAttentionManager.get_num_common_prefix_pages), [`get_num_common_prefix_pages`](single_type_cache_manager.md#MambaManager.get_num_common_prefix_pages), [`get_num_common_prefix_pages`](single_type_cache_manager.md#SlidingWindowManager.get_num_common_prefix_pages), [`remove_skipped_pages`](single_type_cache_manager.md#FullAttentionManager.remove_skipped_pages), [`remove_skipped_pages`](single_type_cache_manager.md#MambaManager.remove_skipped_pages)
- used by: [`find_longest_cache_hit`](coordinator.md#HybridCacheCoordinator.find_longest_cache_hit), [`spec_manager_map`](single_type_cache_manager.md#spec_manager_map.spec_manager_map), [`find_longest_cache_hit`](coordinator.md#UnitaryCacheCoordinator.find_longest_cache_hit), [`remove_skipped_pages`](single_type_cache_manager.md#ChunkedLocalAttentionManager.remove_skipped_pages), [`remove_skipped_pages`](single_type_cache_manager.md#SlidingWindowManager.remove_skipped_pages), [`cache_pages`](coordinator.md#CacheCoordinator.cache_pages), [`get_manager_for_kv_cache_spec`](single_type_cache_manager.md#get_manager_for_kv_cache_spec), [`get_pages`](coordinator.md#CacheCoordinator.get_pages), [`allocate_new_pages`](coordinator.md#CacheCoordinator.allocate_new_pages), [`get_num_common_prefix_pages`](coordinator.md#CacheCoordinator.get_num_common_prefix_pages), [`get_num_pages_to_allocate`](coordinator.md#CacheCoordinator.get_num_pages_to_allocate), [`rollback_new_computed_pages`](coordinator.md#CacheCoordinator.rollback_new_computed_pages), [`save_new_computed_pages`](coordinator.md#CacheCoordinator.save_new_computed_pages), [`FullAttentionManager`](single_type_cache_manager.md#FullAttentionManager), [`__init__`](single_type_cache_manager.md#ChunkedLocalAttentionManager.__init__), [`__init__`](single_type_cache_manager.md#SlidingWindowManager.__init__), [`allocate_new_pages`](single_type_cache_manager.md#MambaManager.allocate_new_pages), [`free`](coordinator.md#CacheCoordinator.free), [`remove_skipped_pages`](coordinator.md#CacheCoordinator.remove_skipped_pages), [`ChunkedLocalAttentionManager`](single_type_cache_manager.md#ChunkedLocalAttentionManager), [`MambaManager`](single_type_cache_manager.md#MambaManager), [`SlidingWindowManager`](single_type_cache_manager.md#SlidingWindowManager), [`get_num_common_prefix_pages`](single_type_cache_manager.md#FullAttentionManager.get_num_common_prefix_pages)

### `SlidingWindowManager`  ·  implements/extends SingleTypeCacheManager
- def: [`easydel/inference/esurge/core/single_type_cache_manager.py:404`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/single_type_cache_manager.py#L404)
- doc: Cache manager for sliding window attention layers.
- signature: `class SlidingWindowManager(SingleTypeCacheManager):`
- members:
  - `__init__(self, kv_cache_spec: SlidingWindowSpec, page_pool: PagePool, **kwargs)` — [`L424`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/single_type_cache_manager.py#L424) — Initialize the sliding window cache manager.
  - `find_longest_cache_hit(cls, page_hashes: list[PageHash], max_length: int, kv_cache_group_ids: list[int], page_pool: PagePool, kv_cache_spec: CacheSpec, use_eagle: bool, dp_shard_hint: int | None = None, data_parallel_size: int | None = None)` — [`L437`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/single_type_cache_manager.py#L437) — Find the longest cache hit within the sliding window by scanning pages backward.
  - `get_num_common_prefix_pages(self, request_id: str, num_scheduled_requests: int)` — [`L508`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/single_type_cache_manager.py#L508) — Get number of common prefix pages (always 0 for sliding window).
  - `remove_skipped_pages(self, request_id: str, num_computed_tokens: int)` — [`L487`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/single_type_cache_manager.py#L487) — Remove pages outside the sliding window and replace with null pages.
  - `sliding_window` — [`L433`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/single_type_cache_manager.py#L433)
- protocol/private: `_null_page`[`L434`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/single_type_cache_manager.py#L434)
- uses (calls/refs, reference-scoped): [`CachePage`](utils.md#CachePage), [`get_cached_page`](page_pool.md#PagePool.get_cached_page), [`PageHash`](utils.md#PageHash), [`page_size`](interface.md#CacheSpec.page_size), [`CacheSpec`](interface.md#CacheSpec), [`SingleTypeCacheManager`](single_type_cache_manager.md#SingleTypeCacheManager), [`free_pages`](page_pool.md#PagePool.free_pages), [`null_page`](page_pool.md#PagePool.null_page), [`req_to_pages`](single_type_cache_manager.md#SingleTypeCacheManager.req_to_pages), [`PagePool`](page_pool.md#PagePool), [`SlidingWindowSpec`](interface.md#SlidingWindowSpec), [`cdiv`](../utils.md#cdiv), [`page_size`](single_type_cache_manager.md#SingleTypeCacheManager.page_size), [`__init__`](single_type_cache_manager.md#SingleTypeCacheManager.__init__), [`page_pool`](single_type_cache_manager.md#SingleTypeCacheManager.page_pool), [`sliding_window`](interface.md#SlidingWindowSpec.sliding_window)
- used by: [`spec_manager_map`](single_type_cache_manager.md#spec_manager_map.spec_manager_map), [`find_longest_cache_hit`](single_type_cache_manager.md#SingleTypeCacheManager.find_longest_cache_hit), [`SingleTypeCacheManager`](single_type_cache_manager.md#SingleTypeCacheManager), [`get_num_common_prefix_pages`](single_type_cache_manager.md#SingleTypeCacheManager.get_num_common_prefix_pages), [`remove_skipped_pages`](single_type_cache_manager.md#SingleTypeCacheManager.remove_skipped_pages)

## Functions
- `get_manager_for_kv_cache_spec(kv_cache_spec: CacheSpec, **kwargs)` — [`L774`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/single_type_cache_manager.py#L774) — Factory function to create the appropriate cache manager for a specification.

## Module values
- `spec_manager_map` — [`L765`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/single_type_cache_manager.py#L765)


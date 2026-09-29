---
title: 'Module: easydel/inference/esurge/core/coordinator.py'
type: catalog
provenance: extracted
module: easydel/inference/esurge/core/coordinator.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.esurge.core.coordinator`/
symbols:
  HybridCacheCoordinator.find_longest_cache_hit: HybridCacheCoordinator#find_longest_cache_hit().
  HybridCacheCoordinator.other_group_ids: HybridCacheCoordinator#other_group_ids.
  CacheCoordinator.single_type_managers: CacheCoordinator#single_type_managers.
  UnitaryCacheCoordinator.find_longest_cache_hit: UnitaryCacheCoordinator#find_longest_cache_hit().
  HybridCacheCoordinator.other_page_size: HybridCacheCoordinator#other_page_size.
  get_kv_cache_coordinator: get_kv_cache_coordinator().
  CacheCoordinator: CacheCoordinator#
  CacheCoordinator.find_longest_cache_hit: CacheCoordinator#find_longest_cache_hit().
  CacheCoordinator.cache_pages: CacheCoordinator#cache_pages().
  CacheCoordinator.page_pool: CacheCoordinator#page_pool.
  HybridCacheCoordinator.full_attention_page_size: HybridCacheCoordinator#full_attention_page_size.
  CacheCoordinator.get_pages: CacheCoordinator#get_pages().
  HybridCacheCoordinator.full_attention_spec: HybridCacheCoordinator#full_attention_spec.
  HybridCacheCoordinator.other_spec: HybridCacheCoordinator#other_spec.
  HybridCacheCoordinator.full_attn_first: HybridCacheCoordinator#full_attn_first.
  CacheCoordinator.get_num_pages_to_allocate: CacheCoordinator#get_num_pages_to_allocate().
  CacheCoordinator.save_new_computed_pages: CacheCoordinator#save_new_computed_pages().
  CacheCoordinator.rollback_new_computed_pages: CacheCoordinator#rollback_new_computed_pages().
  CacheCoordinator.allocate_new_pages: CacheCoordinator#allocate_new_pages().
  CacheCoordinator.get_num_common_prefix_pages: CacheCoordinator#get_num_common_prefix_pages().
  UnitaryCacheCoordinator.page_size: UnitaryCacheCoordinator#page_size.
  CacheCoordinatorNoPrefixCache.find_longest_cache_hit: CacheCoordinatorNoPrefixCache#find_longest_cache_hit().
  HybridCacheCoordinator.__init__: HybridCacheCoordinator#__init__().
  UnitaryCacheCoordinator.kv_cache_spec: UnitaryCacheCoordinator#kv_cache_spec.
  CacheCoordinator.kv_cache_groups: CacheCoordinator#kv_cache_groups.
  CacheCoordinator.free: CacheCoordinator#free().
  CacheCoordinator.remove_skipped_pages: CacheCoordinator#remove_skipped_pages().
  HybridCacheCoordinator.other_attention_cls: HybridCacheCoordinator#other_attention_cls.
  CacheCoordinator.__init__: CacheCoordinator#__init__().
  CacheCoordinatorNoPrefixCache: CacheCoordinatorNoPrefixCache#
  UnitaryCacheCoordinator: UnitaryCacheCoordinator#
  HybridCacheCoordinator: HybridCacheCoordinator#
  CacheCoordinatorNoPrefixCache.__init__: CacheCoordinatorNoPrefixCache#__init__().
  UnitaryCacheCoordinator.__init__: UnitaryCacheCoordinator#__init__().
  HybridCacheCoordinator.full_attention_group_ids: HybridCacheCoordinator#full_attention_group_ids.
  CacheCoordinatorNoPrefixCache.num_single_type_manager: CacheCoordinatorNoPrefixCache#num_single_type_manager.
  HybridCacheCoordinator.full_attention_manager_cls: HybridCacheCoordinator#full_attention_manager_cls.
  CacheCoordinator.use_eagle: CacheCoordinator#use_eagle.
  CacheCoordinatorNoPrefixCache.get_num_common_prefix_pages: CacheCoordinatorNoPrefixCache#get_num_common_prefix_pages().
  CacheCoordinator.num_pages: CacheCoordinator#num_pages.
  CacheCoordinator.enable_caching: CacheCoordinator#enable_caching.
  HybridCacheCoordinator.verify_and_split_kv_cache_groups: HybridCacheCoordinator#verify_and_split_kv_cache_groups().
  CacheCoordinator.max_model_len: CacheCoordinator#max_model_len.
---
# Module: [`easydel/inference/esurge/core/coordinator.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/coordinator.py)

## Classes
### `CacheCoordinator`  ·  implements/extends ABC
- def: [`easydel/inference/esurge/core/coordinator.py:50`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/coordinator.py#L50)
- doc: Abstract base class for coordinating KV-cache across multiple groups.
- signature: `class CacheCoordinator(ABC):`
- members:
  - `__init__(self, num_pages: int, kv_cache_groups: list[CacheGroupSpec], max_model_len: int, use_eagle: bool, enable_caching: bool)` — [`L72`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/coordinator.py#L72) — Initialize the cache coordinator.
  - `allocate_new_pages(self, request_id: str, num_tokens: int, *, dp_shard_hint: int | None = None, data_parallel_size: int | None = None)` — [`L155`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/coordinator.py#L155) — Allocate new pages for the request to give it at least `num_tokens`
  - `cache_pages(self, request: EngineRequest, page_hashes: list[PageHash], num_computed_tokens: int)` — [`L185`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/coordinator.py#L185) — Cache the pages for the request.
  - `find_longest_cache_hit(self, page_hashes: list[PageHash], max_cache_hit_length: int, dp_shard_hint: int | None = None, data_parallel_size: int | None = None)` — [`L249`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/coordinator.py#L249) — Find the longest prefix cache hit for a sequence of page hashes.
  - `free(self, request_id: str)` — [`L198`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/coordinator.py#L198) — Free the pages for the request.
  - `get_num_common_prefix_pages(self, request_id: str, num_scheduled_requests: int)` — [`L208`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/coordinator.py#L208) — Get the number of common prefix pages for a request.
  - `get_num_pages_to_allocate(self, request_id: str, num_tokens: int, new_computed_pages: tuple[list[CachePage], ...])` — [`L106`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/coordinator.py#L106) — Get the number of pages needed to be allocated for the request.
  - `get_pages(self, request_id: str)` — [`L237`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/coordinator.py#L237) — Get all pages allocated to a request across all cache groups.
  - `remove_skipped_pages(self, request_id: str, num_computed_tokens: int)` — [`L225`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/coordinator.py#L225) — Remove the pages that are no longer needed from `pages` and replace
  - `rollback_new_computed_pages(self, request_id: str, new_computed_pages: tuple[list[CachePage], ...])` — [`L142`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/coordinator.py#L142) — Rollback a previous :meth:`save_new_computed_pages` call.
  - `save_new_computed_pages(self, request_id: str, new_computed_pages: tuple[list[CachePage], ...])` — [`L130`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/coordinator.py#L130) — Add the new computed pages to the request.
  - `enable_caching` — [`L92`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/coordinator.py#L92)
  - `kv_cache_groups` — [`L90`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/coordinator.py#L90)
  - `max_model_len` — [`L91`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/coordinator.py#L91)
  - `num_pages` — [`L89`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/coordinator.py#L89)
  - `page_pool` — [`L94`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/coordinator.py#L94)
  - `single_type_managers` — [`L97`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/coordinator.py#L97)
  - `use_eagle` — [`L96`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/coordinator.py#L96)
- uses (calls/refs, reference-scoped): [`EngineRequest`](../request.md#EngineRequest), [`CachePage`](utils.md#CachePage), [`find_longest_cache_hit`](coordinator.md#HybridCacheCoordinator.find_longest_cache_hit), [`cache_pages`](single_type_cache_manager.md#SingleTypeCacheManager.cache_pages), [`PageHash`](utils.md#PageHash), [`find_longest_cache_hit`](coordinator.md#UnitaryCacheCoordinator.find_longest_cache_hit), [`allocate_new_pages`](single_type_cache_manager.md#SingleTypeCacheManager.allocate_new_pages), [`kv_cache_spec`](interface.md#CacheGroupSpec.kv_cache_spec), [`CacheGroupSpec`](interface.md#CacheGroupSpec), [`get_num_pages_to_allocate`](single_type_cache_manager.md#SingleTypeCacheManager.get_num_pages_to_allocate), [`req_to_pages`](single_type_cache_manager.md#SingleTypeCacheManager.req_to_pages), [`PagePool`](page_pool.md#PagePool), [`free`](single_type_cache_manager.md#SingleTypeCacheManager.free), [`get_num_common_prefix_pages`](single_type_cache_manager.md#SingleTypeCacheManager.get_num_common_prefix_pages), [`remove_skipped_pages`](single_type_cache_manager.md#SingleTypeCacheManager.remove_skipped_pages), [`get_manager_for_kv_cache_spec`](single_type_cache_manager.md#get_manager_for_kv_cache_spec), [`rollback_new_computed_pages`](single_type_cache_manager.md#SingleTypeCacheManager.rollback_new_computed_pages), [`save_new_computed_pages`](single_type_cache_manager.md#SingleTypeCacheManager.save_new_computed_pages), [`find_longest_cache_hit`](coordinator.md#CacheCoordinatorNoPrefixCache.find_longest_cache_hit), [`CacheCoordinatorNoPrefixCache`](coordinator.md#CacheCoordinatorNoPrefixCache), [`HybridCacheCoordinator`](coordinator.md#HybridCacheCoordinator), [`UnitaryCacheCoordinator`](coordinator.md#UnitaryCacheCoordinator), [`get_num_common_prefix_pages`](coordinator.md#CacheCoordinatorNoPrefixCache.get_num_common_prefix_pages)
- used by: [`allocate_slots`](manager.md#CacheManager.allocate_slots), [`find_longest_cache_hit`](coordinator.md#HybridCacheCoordinator.find_longest_cache_hit), [`get_computed_pages`](manager.md#CacheManager.get_computed_pages), [`other_group_ids`](coordinator.md#HybridCacheCoordinator.other_group_ids), [`find_longest_cache_hit`](coordinator.md#UnitaryCacheCoordinator.find_longest_cache_hit), [`other_page_size`](coordinator.md#HybridCacheCoordinator.other_page_size), [`get_num_common_prefix_pages`](manager.md#CacheManager.get_num_common_prefix_pages), [`cache_pages`](manager.md#CacheManager.cache_pages), [`get_kv_cache_coordinator`](coordinator.md#get_kv_cache_coordinator), [`page_pool`](manager.md#CacheManager.page_pool), [`free`](manager.md#CacheManager.free), [`get_page_ids`](manager.md#CacheManager.get_page_ids), [`full_attention_spec`](coordinator.md#HybridCacheCoordinator.full_attention_spec), [`other_spec`](coordinator.md#HybridCacheCoordinator.other_spec), [`page_size`](coordinator.md#UnitaryCacheCoordinator.page_size), [`__init__`](coordinator.md#HybridCacheCoordinator.__init__), [`kv_cache_spec`](coordinator.md#UnitaryCacheCoordinator.kv_cache_spec), [`other_attention_cls`](coordinator.md#HybridCacheCoordinator.other_attention_cls), [`CacheCoordinatorNoPrefixCache`](coordinator.md#CacheCoordinatorNoPrefixCache), [`HybridCacheCoordinator`](coordinator.md#HybridCacheCoordinator), [`UnitaryCacheCoordinator`](coordinator.md#UnitaryCacheCoordinator), [`__init__`](coordinator.md#CacheCoordinatorNoPrefixCache.__init__), [`__init__`](coordinator.md#UnitaryCacheCoordinator.__init__), [`num_single_type_manager`](coordinator.md#CacheCoordinatorNoPrefixCache.num_single_type_manager)

### `CacheCoordinatorNoPrefixCache`  ·  implements/extends CacheCoordinator
- def: [`easydel/inference/esurge/core/coordinator.py:273`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/coordinator.py#L273)
- doc: Cache coordinator for configurations without prefix caching.
- signature: `class CacheCoordinatorNoPrefixCache(CacheCoordinator):`
- members:
  - `__init__(self, num_pages: int, kv_cache_groups: list[CacheGroupSpec], max_model_len: int, use_eagle: bool)` — [`L290`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/coordinator.py#L290) — Initialize the no-prefix-cache coordinator.
  - `find_longest_cache_hit(self, page_hashes: list[PageHash], max_cache_hit_length: int, dp_shard_hint: int | None = None, data_parallel_size: int | None = None)` — [`L320`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/coordinator.py#L320) — Find cache hit (always returns empty without caching).
  - `get_num_common_prefix_pages(self, request_id: str, num_scheduled_requests: int)` — [`L308`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/coordinator.py#L308) — Get common prefix page counts (always 0 without caching).
  - `num_single_type_manager` — [`L306`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/coordinator.py#L306)
- uses (calls/refs, reference-scoped): [`CachePage`](utils.md#CachePage), [`PageHash`](utils.md#PageHash), [`single_type_managers`](coordinator.md#CacheCoordinator.single_type_managers), [`CacheGroupSpec`](interface.md#CacheGroupSpec), [`CacheCoordinator`](coordinator.md#CacheCoordinator), [`__init__`](coordinator.md#CacheCoordinator.__init__)
- used by: [`get_kv_cache_coordinator`](coordinator.md#get_kv_cache_coordinator), [`CacheCoordinator`](coordinator.md#CacheCoordinator), [`find_longest_cache_hit`](coordinator.md#CacheCoordinator.find_longest_cache_hit), [`get_num_common_prefix_pages`](coordinator.md#CacheCoordinator.get_num_common_prefix_pages)

### `HybridCacheCoordinator`  ·  implements/extends CacheCoordinator
- def: [`easydel/inference/esurge/core/coordinator.py:419`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/coordinator.py#L419)
- doc: Cache coordinator for hybrid models with multiple attention types.
- signature: `class HybridCacheCoordinator(CacheCoordinator):`
- members:
  - `__init__(self, num_pages: int, kv_cache_groups: list[CacheGroupSpec], max_model_len: int, use_eagle: bool, enable_caching: bool)` — [`L447`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/coordinator.py#L447) — Initialize the hybrid cache coordinator.
  - `find_longest_cache_hit(self, page_hashes: list[PageHash], max_cache_hit_length: int, dp_shard_hint: int | None = None, data_parallel_size: int | None = None)` — [`L538`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/coordinator.py#L538) — Find the longest consistent cache hit across all cache groups.
  - `verify_and_split_kv_cache_groups(self)` — [`L470`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/coordinator.py#L470) — Verify cache group configuration and split by attention type.
  - `full_attention_group_ids` — [`L485`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/coordinator.py#L485)
  - `full_attention_manager_cls` — [`L508`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/coordinator.py#L508) — ---
  - `full_attention_page_size` — [`L514`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/coordinator.py#L514)
  - `full_attention_spec` — [`L511`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/coordinator.py#L511)
  - `full_attn_first` — [`L525`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/coordinator.py#L525)
  - `other_attention_cls` — [`L509`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/coordinator.py#L509) — ---
  - `other_group_ids` — [`L486`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/coordinator.py#L486)
  - `other_page_size` — [`L515`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/coordinator.py#L515)
  - `other_spec` — [`L512`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/coordinator.py#L512)
- uses (calls/refs, reference-scoped): [`CachePage`](utils.md#CachePage), [`PageHash`](utils.md#PageHash), [`find_longest_cache_hit`](single_type_cache_manager.md#SingleTypeCacheManager.find_longest_cache_hit), [`page_size`](interface.md#CacheSpec.page_size), [`single_type_managers`](coordinator.md#CacheCoordinator.single_type_managers), [`find_longest_cache_hit`](single_type_cache_manager.md#FullAttentionManager.find_longest_cache_hit), [`kv_cache_spec`](interface.md#CacheGroupSpec.kv_cache_spec), [`CacheGroupSpec`](interface.md#CacheGroupSpec), [`FullAttentionSpec`](interface.md#FullAttentionSpec), [`type_id`](interface.md#CacheSpec.type_id), [`CacheCoordinator`](coordinator.md#CacheCoordinator), [`page_pool`](coordinator.md#CacheCoordinator.page_pool), [`FullAttentionManager`](single_type_cache_manager.md#FullAttentionManager), [`type_id`](interface.md#FullAttentionSpec.type_id), [`kv_cache_groups`](coordinator.md#CacheCoordinator.kv_cache_groups), [`__init__`](coordinator.md#CacheCoordinator.__init__), [`use_eagle`](coordinator.md#CacheCoordinator.use_eagle), [`enable_caching`](coordinator.md#CacheCoordinator.enable_caching)
- used by: [`get_kv_cache_coordinator`](coordinator.md#get_kv_cache_coordinator), [`CacheCoordinator`](coordinator.md#CacheCoordinator), [`find_longest_cache_hit`](coordinator.md#CacheCoordinator.find_longest_cache_hit)

### `UnitaryCacheCoordinator`  ·  implements/extends CacheCoordinator
- def: [`easydel/inference/esurge/core/coordinator.py:340`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/coordinator.py#L340)
- doc: Cache coordinator for models with a single KV cache group.
- signature: `class UnitaryCacheCoordinator(CacheCoordinator):`
- members:
  - `__init__(self, num_pages: int, kv_cache_groups: list[CacheGroupSpec], max_model_len: int, use_eagle: bool, enable_caching: bool)` — [`L362`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/coordinator.py#L362) — Initialize the unitary cache coordinator.
  - `find_longest_cache_hit(self, page_hashes: list[PageHash], max_cache_hit_length: int, dp_shard_hint: int | None = None, data_parallel_size: int | None = None)` — [`L388`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/coordinator.py#L388) — Find the longest cache hit for the single cache group.
  - `kv_cache_spec` — [`L383`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/coordinator.py#L383)
  - `page_size` — [`L384`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/coordinator.py#L384)
- uses (calls/refs, reference-scoped): [`CachePage`](utils.md#CachePage), [`PageHash`](utils.md#PageHash), [`find_longest_cache_hit`](single_type_cache_manager.md#SingleTypeCacheManager.find_longest_cache_hit), [`page_size`](interface.md#CacheSpec.page_size), [`single_type_managers`](coordinator.md#CacheCoordinator.single_type_managers), [`kv_cache_spec`](interface.md#CacheGroupSpec.kv_cache_spec), [`CacheGroupSpec`](interface.md#CacheGroupSpec), [`CacheCoordinator`](coordinator.md#CacheCoordinator), [`page_pool`](coordinator.md#CacheCoordinator.page_pool), [`kv_cache_groups`](coordinator.md#CacheCoordinator.kv_cache_groups), [`__init__`](coordinator.md#CacheCoordinator.__init__), [`use_eagle`](coordinator.md#CacheCoordinator.use_eagle)
- used by: [`get_kv_cache_coordinator`](coordinator.md#get_kv_cache_coordinator), [`CacheCoordinator`](coordinator.md#CacheCoordinator), [`find_longest_cache_hit`](coordinator.md#CacheCoordinator.find_longest_cache_hit)

## Functions
- `get_kv_cache_coordinator(num_pages: int, kv_cache_groups: list[CacheGroupSpec], max_model_len: int, use_eagle: bool, enable_caching: bool)` — [`L604`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/coordinator.py#L604) — Factory function to create the appropriate cache coordinator.


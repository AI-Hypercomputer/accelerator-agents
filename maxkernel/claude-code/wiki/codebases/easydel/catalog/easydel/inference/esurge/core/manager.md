---
title: 'Module: easydel/inference/esurge/core/manager.py'
type: catalog
provenance: extracted
module: easydel/inference/esurge/core/manager.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.esurge.core.manager`/Cache
symbols:
  CacheManager.allocate_slots: Manager#allocate_slots().
  CacheManager.get_computed_pages: Manager#get_computed_pages().
  CacheManager.coordinator: Manager#coordinator.
  CacheManager.get_num_common_prefix_pages: Manager#get_num_common_prefix_pages().
  CachePages: Pages#
  CacheManager._infer_dp_shard_from_pages: Manager#_infer_dp_shard_from_pages().
  CacheManager.cache_pages: Manager#cache_pages().
  CacheManager.page_pool: Manager#page_pool.
  CacheManager.page_size: Manager#page_size.
  CacheManager.free: Manager#free().
  CacheManager.get_page_ids: Manager#get_page_ids().
  CachePages.pages: Pages#pages.
  CacheManager.enable_caching: Manager#enable_caching.
  CacheManager.free_page_hashes: Manager#free_page_hashes().
  CachePages.get_unhashed_page_ids: Pages#get_unhashed_page_ids().
  CacheManager.req_to_page_hashes: Manager#req_to_page_hashes.
  CachePages.get_page_ids: Pages#get_page_ids().
  CacheManager.create_empty_page_list: Manager#create_empty_page_list().
  CacheManager: Manager#
  CacheManager.reset_prefix_cache: Manager#reset_prefix_cache().
  CachePages.__add__: Pages#__add__().
  CachePages.new_empty: Pages#new_empty().
  CacheManager.usage: Manager#usage().
  CacheManager.num_pages: Manager#num_pages.
  CacheManager.__init__: Manager#__init__().
  CacheManager.kv_cache_groups: Manager#kv_cache_groups.
  CacheManager.max_model_len: Manager#max_model_len.
  CacheManager.use_eagle: Manager#use_eagle.
  CacheManager.num_kv_cache_groups: Manager#num_kv_cache_groups.
---
# Module: [`easydel/inference/esurge/core/manager.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/manager.py)

## Classes
### `CacheManager`
- def: [`easydel/inference/esurge/core/manager.py:132`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/manager.py#L132)
- doc: High-level KV-cache manager for inference requests.
- signature: `class CacheManager:`
- members:
  - `__init__(self, num_pages: int, kv_cache_groups: list[CacheGroupSpec], max_model_len: int, enable_caching: bool = True, use_eagle: bool = False)` — [`L173`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/manager.py#L173) — Initialize the CacheManager.
  - `_infer_dp_shard_from_pages(self, pages: tuple[list[CachePage], ...], *, data_parallel_size: int | None)` — [`L271`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/manager.py#L271) — Infer a consistent DP shard from already-attached pages.
  - `allocate_slots(self, request: EngineRequest, num_new_tokens: int, num_new_computed_tokens: int = 0, new_computed_pages: CachePages | None = None, num_lookahead_tokens: int = 0, delay_cache_pages: bool = False, dp_shard_hint: int | None = None, data_parallel_size: int | None = None)` — [`L306`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/manager.py#L306) — Add slots for a request with new tokens to append.
  - `cache_pages(self, request: EngineRequest, num_computed_tokens: int)` — [`L515`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/manager.py#L515) — Cache the pages for a request to enable prefix reuse.
  - `create_empty_page_list(self)` — [`L534`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/manager.py#L534) — Create an empty CachePages instance for the current configuration.
  - `free(self, request: EngineRequest)` — [`L422`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/manager.py#L422) — Free the pages allocated for the request.
  - `free_page_hashes(self, request: EngineRequest)` — [`L496`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/manager.py#L496) — Discard the page hashes for the request.
  - `get_computed_pages(self, request: EngineRequest, *, dp_shard_hint: int | None = None, data_parallel_size: int | None = None)` — [`L228`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/manager.py#L228) — Get the computed (cached) pages for the request.
  - `get_num_common_prefix_pages(self, request: EngineRequest, num_scheduled_requests: int)` — [`L454`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/manager.py#L454) — Calculate the number of common prefix pages shared by all requests
  - `get_page_ids(self, request_id: str)` — [`L504`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/manager.py#L504) — Get the page IDs allocated to a request.
  - `reset_prefix_cache(self)` — [`L432`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/manager.py#L432) — Reset the prefix cache to invalidate all cached prefixes.
  - `usage(self)` — [`L220`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/manager.py#L220) — Get the KV cache usage.
  - `coordinator` — [`L207`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/manager.py#L207)
  - `enable_caching` — [`L199`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/manager.py#L199)
  - `kv_cache_groups` — [`L194`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/manager.py#L194)
  - `max_model_len` — [`L195`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/manager.py#L195)
  - `num_kv_cache_groups` — [`L214`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/manager.py#L214)
  - `num_pages` — [`L193`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/manager.py#L193)
  - `page_pool` — [`L215`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/manager.py#L215)
  - `page_size` — [`L203`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/manager.py#L203)
  - `req_to_page_hashes` — [`L217`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/manager.py#L217)
  - `use_eagle` — [`L201`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/manager.py#L201)
- uses (calls/refs, reference-scoped): [`EngineRequest`](../request.md#EngineRequest), [`CachePage`](utils.md#CachePage), [`request_id`](../request.md#EngineRequest.request_id), [`EngineRequestStatus`](../request.md#EngineRequestStatus), [`status`](../request.md#EngineRequest.status), [`PageHash`](utils.md#PageHash), [`page_size`](interface.md#CacheSpec.page_size), [`kv_cache_spec`](interface.md#CacheGroupSpec.kv_cache_spec), [`CacheGroupSpec`](interface.md#CacheGroupSpec), [`num_computed_tokens`](../request.md#EngineRequest.num_computed_tokens), [`free_pages`](page_pool.md#PagePool.free_pages), [`CachePages`](manager.md#CachePages), [`pages_per_dp_shard`](dp_sharding.md#pages_per_dp_shard), [`reset_prefix_cache`](page_pool.md#PagePool.reset_prefix_cache), [`touch`](page_pool.md#PagePool.touch), [`get_kv_cache_coordinator`](coordinator.md#get_kv_cache_coordinator), [`hash_request_tokens`](utils.md#hash_request_tokens), [`page_id`](utils.md#CachePage.page_id), [`find_longest_cache_hit`](coordinator.md#CacheCoordinator.find_longest_cache_hit), [`cache_pages`](coordinator.md#CacheCoordinator.cache_pages), [`get_num_free_pages`](page_pool.md#PagePool.get_num_free_pages), [`num_tokens`](../request.md#EngineRequest.num_tokens), [`page_pool`](coordinator.md#CacheCoordinator.page_pool), [`get_pages`](coordinator.md#CacheCoordinator.get_pages), [`pages`](manager.md#CachePages.pages), [`allocate_new_pages`](coordinator.md#CacheCoordinator.allocate_new_pages), [`get_num_common_prefix_pages`](coordinator.md#CacheCoordinator.get_num_common_prefix_pages), [`get_num_pages_to_allocate`](coordinator.md#CacheCoordinator.get_num_pages_to_allocate), [`rollback_new_computed_pages`](coordinator.md#CacheCoordinator.rollback_new_computed_pages), [`save_new_computed_pages`](coordinator.md#CacheCoordinator.save_new_computed_pages), [`dp_shard_for_page_id`](dp_sharding.md#dp_shard_for_page_id), [`get_page_ids`](manager.md#CachePages.get_page_ids), [`free`](coordinator.md#CacheCoordinator.free), [`get_usage`](page_pool.md#PagePool.get_usage), [`remove_skipped_pages`](coordinator.md#CacheCoordinator.remove_skipped_pages), [`RUNNING`](../request.md#EngineRequestStatus.RUNNING), [`is_null`](utils.md#CachePage.is_null), [`prompt_logprobs`](../../sampling_params.md#SamplingParams.prompt_logprobs), [`sampling_params`](../request.md#EngineRequest.sampling_params), [`init_none_hash`](utils.md#init_none_hash)
- used by: [`schedule`](../scheduler/scheduler.md#Scheduler.schedule), [`kv_cache_manager`](../scheduler/scheduler.md#Scheduler.kv_cache_manager), [`_update_request_with_output`](../scheduler/async_scheduler.md#AsyncScheduler._update_request_with_output), [`_free_pages`](../scheduler/scheduler.md#Scheduler._free_pages), [`begin_cycle`](../scheduler/token_budget.md#TokenBudgetManager.begin_cycle), [`_infer_dp_shard_from_pages`](../scheduler/scheduler.md#Scheduler._infer_dp_shard_from_pages), [`reset_prefix_cache`](../scheduler/scheduler.md#Scheduler.reset_prefix_cache)

### `CachePages`
- def: [`easydel/inference/esurge/core/manager.py:48`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/manager.py#L48)
- doc: Allocation result container for cache pages.
- signature: `class CachePages:`
- members:
  - `__add__(self, other: CachePages)` — [`L68`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/manager.py#L68) — Concatenate two CachePages instances.
  - `get_page_ids(self)` — [`L85`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/manager.py#L85) — Convert CachePages to page IDs.
  - `get_unhashed_page_ids(self)` — [`L102`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/manager.py#L102) — Get page IDs of pages without computed hashes.
  - `new_empty(self)` — [`L120`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/manager.py#L120) — Create an empty CachePages instance with the same structure.
  - `pages` — [`L66`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/manager.py#L66)
- uses (calls/refs, reference-scoped): [`CachePage`](utils.md#CachePage), [`page_hash`](utils.md#CachePage.page_hash), [`page_id`](utils.md#CachePage.page_id)
- used by: [`schedule`](../scheduler/scheduler.md#Scheduler.schedule), [`allocate_slots`](manager.md#CacheManager.allocate_slots), [`get_computed_pages`](manager.md#CacheManager.get_computed_pages), [`get_page_ids`](manager.md#CacheManager.get_page_ids), [`create_empty_page_list`](manager.md#CacheManager.create_empty_page_list)


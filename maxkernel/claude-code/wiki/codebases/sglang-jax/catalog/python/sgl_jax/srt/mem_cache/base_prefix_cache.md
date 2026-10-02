---
title: 'Module: python/sgl_jax/srt/mem_cache/base_prefix_cache.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/mem_cache/base_prefix_cache.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.mem_cache.base_prefix_cache`/
symbols:
  BasePrefixCache: BasePrefixCache#
  MatchPrefixParams: MatchPrefixParams#
  InsertParams: InsertParams#
  EvictParams: EvictParams#
  IncLockRefResult: IncLockRefResult#
  MatchResult: MatchResult#
  BasePrefixCache.evict: BasePrefixCache#evict().
  DecLockRefParams: DecLockRefParams#
  BasePrefixCache.match_prefix: BasePrefixCache#match_prefix().
  EvictResult: EvictResult#
  MatchResult.last_device_node: MatchResult#last_device_node.
  BasePrefixCache.inc_lock_ref: BasePrefixCache#inc_lock_ref().
  MatchResult.device_indices: MatchResult#device_indices.
  MatchPrefixParams.key: MatchPrefixParams#key.
  IncLockRefResult.to_dec_params: IncLockRefResult#to_dec_params().
  BasePrefixCache.dec_lock_ref: BasePrefixCache#dec_lock_ref().
  BasePrefixCache.evictable_size: BasePrefixCache#evictable_size().
  BasePrefixCache.full_evictable_size: BasePrefixCache#full_evictable_size().
  InsertParams.key: InsertParams#key.
  MatchResult.last_host_node: MatchResult#last_host_node.
  BasePrefixCache.swa_evictable_size: BasePrefixCache#swa_evictable_size().
  EvictParams.dp_rank: EvictParams#dp_rank.
  BasePrefixCache.cache_unfinished_req: BasePrefixCache#cache_unfinished_req().
  BasePrefixCache.reset: BasePrefixCache#reset().
  BasePrefixCache.cache_finished_req: BasePrefixCache#cache_finished_req().
  BasePrefixCache.pretty_print: BasePrefixCache#pretty_print().
  MatchResult.best_match_node: MatchResult#best_match_node.
  BasePrefixCache.supports_recurrent: BasePrefixCache#supports_recurrent().
  InsertParams.value: InsertParams#value.
  EvictParams.num_tokens: EvictParams#num_tokens.
  BasePrefixCache.protected_size: BasePrefixCache#protected_size().
  BasePrefixCache.full_protected_size: BasePrefixCache#full_protected_size().
  BasePrefixCache.swa_protected_size: BasePrefixCache#swa_protected_size().
  BasePrefixCache.total_size: BasePrefixCache#total_size().
  MatchResult.host_hit_length: MatchResult#host_hit_length.
  IncLockRefResult.delta: IncLockRefResult#delta.
  IncLockRefResult.swa_uuid_for_lock: IncLockRefResult#swa_uuid_for_lock.
  BasePrefixCache.take_events: BasePrefixCache#take_events().
  EvictParams.swa_num_tokens: EvictParams#swa_num_tokens.
  DecLockRefParams.swa_uuid_for_lock: DecLockRefParams#swa_uuid_for_lock.
  MatchPrefixParams.cow_recurrent: MatchPrefixParams#cow_recurrent.
  MatchPrefixParams.req: MatchPrefixParams#req.
  InsertParams.prev_prefix_len: InsertParams#prev_prefix_len.
  InsertParams.swa_evicted_seqlen: InsertParams#swa_evicted_seqlen.
  InsertParams.recurrent_value: InsertParams#recurrent_value.
  EvictParams.recurrent_num: EvictParams#recurrent_num.
  EvictResult.num_tokens_evicted: EvictResult#num_tokens_evicted.
  MatchPrefixParams.full_only: MatchPrefixParams#full_only.
  EvictResult.swa_num_tokens_evicted: EvictResult#swa_num_tokens_evicted.
  DecLockRefParams.skip_lock_node_ids: DecLockRefParams#skip_lock_node_ids.
  IncLockRefResult.skip_lock_node_ids: IncLockRefResult#skip_lock_node_ids.
  EvictResult.recurrent_num_evicted: EvictResult#recurrent_num_evicted.
  MatchResult.recurrent_branching_seqlen: MatchResult#recurrent_branching_seqlen.
  BasePrefixCache.init_load_back: BasePrefixCache#init_load_back().
  BasePrefixCache.ready_to_load_host_cache: BasePrefixCache#ready_to_load_host_cache().
  BasePrefixCache.check_hicache_events: BasePrefixCache#check_hicache_events().
---
# Module: [`python/sgl_jax/srt/mem_cache/base_prefix_cache.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py)

## Classes
### `BasePrefixCache`  ·  implements/extends ABC
- def: [`python/sgl_jax/srt/mem_cache/base_prefix_cache.py:111`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L111) — documented in [python-sgl_jax-srt-mem_cache-base_prefix_cache](../../../../../concepts/python-sgl_jax-srt-mem_cache-base_prefix_cache.md)
- doc: Cache can be indexed by either rid or key.
- signature: `class BasePrefixCache(abc.ABC):`
- members:
  - `cache_finished_req(self, req: Any, **kwargs)` — [`L123`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L123)
  - `cache_unfinished_req(self, req: Any, **kwargs)` — [`L127`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L127)
  - `check_hicache_events(self)` — [`L185`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L185)
  - `dec_lock_ref(self, node: Any, params: DecLockRefParams | None = None)` — [`L139`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L139)
  - `evict(self, params: EvictParams)` — [`L131`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L131)
  - `evictable_size(self, dp_rank: int = 0)` — [`L142`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L142)
  - `full_evictable_size(self, dp_rank: int = 0)` — [`L148`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L148)
  - `full_protected_size(self)` — [`L157`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L157)
  - `inc_lock_ref(self, node: Any)` — [`L135`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L135)
  - `init_load_back(self, last_host_node: Any, host_hit_length: int)` — [`L169`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L169) — Preparing KV cache loading from host to device.
  - `match_prefix(self, params: MatchPrefixParams)` — [`L119`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L119)
  - `pretty_print(self)` — [`L166`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L166)
  - `protected_size(self, dp_rank: int = 0)` — [`L154`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L154)
  - `ready_to_load_host_cache(self)` — [`L179`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L179) — Notify the cache controller to start the KV cache loading
  - `reset(self)` — [`L115`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L115)
  - `supports_recurrent(self)` — [`L145`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L145)
  - `swa_evictable_size(self, dp_rank: int = 0)` — [`L151`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L151)
  - `swa_protected_size(self)` — [`L160`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L160)
  - `take_events(self)` — [`L188`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L188)
  - `total_size(self)` — [`L163`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L163)
- uses (calls/refs, reference-scoped): [`cache_unfinished_req`](unified_radix_cache.md#UnifiedRadixCache.cache_unfinished_req), [`cache_unfinished_req`](swa_radix_cache.md#SWARadixCache.cache_unfinished_req), [`evict`](swa_radix_cache.md#SWARadixCache.evict), [`cache_unfinished_req`](radix_cache.md#RadixCache.cache_unfinished_req), [`cache_finished_req`](swa_radix_cache.md#SWARadixCache.cache_finished_req), [`cache_finished_req`](unified_radix_cache.md#UnifiedRadixCache.cache_finished_req), [`cache_finished_req`](radix_cache.md#RadixCache.cache_finished_req), [`inc_lock_ref`](swa_radix_cache.md#SWARadixCache.inc_lock_ref), [`dec_lock_ref`](swa_radix_cache.md#SWARadixCache.dec_lock_ref), [`match_prefix`](radix_cache.md#RadixCache.match_prefix), [`match_prefix`](unified_radix_cache.md#UnifiedRadixCache.match_prefix), [`evict`](unified_radix_cache.md#UnifiedRadixCache.evict), [`inc_lock_ref`](radix_cache.md#RadixCache.inc_lock_ref), [`dec_lock_ref`](radix_cache.md#RadixCache.dec_lock_ref), [`match_prefix`](swa_radix_cache.md#SWARadixCache.match_prefix), [`evict`](radix_cache.md#RadixCache.evict), [`EvictParams`](base_prefix_cache.md#EvictParams), [`MatchPrefixParams`](base_prefix_cache.md#MatchPrefixParams), [`IncLockRefResult`](base_prefix_cache.md#IncLockRefResult), [`MatchResult`](base_prefix_cache.md#MatchResult), [`ChunkCache`](chunk_cache.md#ChunkCache), [`cache_finished_req`](chunk_cache.md#ChunkCache.cache_finished_req), [`DecLockRefParams`](base_prefix_cache.md#DecLockRefParams), [`inc_lock_ref`](unified_radix_cache.md#UnifiedRadixCache.inc_lock_ref), [`pretty_print`](unified_radix_cache.md#UnifiedRadixCache.pretty_print), [`dec_lock_ref`](unified_radix_cache.md#UnifiedRadixCache.dec_lock_ref), [`EvictResult`](base_prefix_cache.md#EvictResult), [`cache_unfinished_req`](chunk_cache.md#ChunkCache.cache_unfinished_req), [`match_prefix`](chunk_cache.md#ChunkCache.match_prefix), [`pretty_print`](radix_cache.md#RadixCache.pretty_print), [`total_size`](unified_radix_cache.md#UnifiedRadixCache.total_size), [`UnifiedRadixCache`](unified_radix_cache.md#UnifiedRadixCache), [`pretty_print`](swa_radix_cache.md#SWARadixCache.pretty_print), [`supports_recurrent`](unified_radix_cache.md#UnifiedRadixCache.supports_recurrent), [`RadixCache`](radix_cache.md#RadixCache), [`evictable_size`](unified_radix_cache.md#UnifiedRadixCache.evictable_size), [`protected_size`](unified_radix_cache.md#UnifiedRadixCache.protected_size), [`evict`](chunk_cache.md#ChunkCache.evict), [`evictable_size`](swa_radix_cache.md#SWARadixCache.evictable_size), [`inc_lock_ref`](chunk_cache.md#ChunkCache.inc_lock_ref)  (+25 more)
- used by: [`process_batch_result_prefill`](../managers/scheduler_output_processor_mixin.md#SchedulerOutputProcessorMixin.process_batch_result_prefill), `prepare_for_extend`, [`add_one_req`](../managers/schedule_policy.md#PrefillAdder.add_one_req), [`get_internal_state`](../managers/scheduler.md#Scheduler.get_internal_state), [`_compute_prefix_matches`](../managers/schedule_policy.md#SchedulePolicy._compute_prefix_matches), [`default_radix_cache_factory`](registry.md#default_radix_cache_factory), [`flush_cache`](../managers/scheduler.md#Scheduler.flush_cache), `mix_with_running`, [`build_kv_cache`](kv_cache_builder.md#build_kv_cache), `host_hit_length`, [`get_next_batch_to_run`](../managers/scheduler.md#Scheduler.get_next_batch_to_run), `tree_cache`, `init_new`, [`check_memory`](../managers/scheduler.md#Scheduler.check_memory), `alloc_req_slots`, `_evict_tree_cache_if_needed`, [`evict_from_tree_cache`](common.md#evict_from_tree_cache), `_available_and_evictable_str`, [`rem_total_tokens_for_dp`](../managers/schedule_policy.md#PrefillAdder.rem_total_tokens_for_dp), [`ChunkCache`](chunk_cache.md#ChunkCache), [`alloc_token_slots`](common.md#alloc_token_slots), [`cur_rem_tokens_for_dp`](../managers/schedule_policy.md#PrefillAdder.cur_rem_tokens_for_dp), [`_get_swa_token_info`](../managers/scheduler.md#Scheduler._get_swa_token_info), [`release_kv_cache`](common.md#release_kv_cache), [`_get_token_info`](../managers/scheduler.md#Scheduler._get_token_info), [`create_tree_cache`](registry.md#create_tree_cache), [`_cached_prefix_len`](../managers/scheduler.md#Scheduler._cached_prefix_len), [`_sort_by_dfs_weight`](../managers/schedule_policy.md#SchedulePolicy._sort_by_dfs_weight), [`alloc_paged_token_slots_extend`](common.md#alloc_paged_token_slots_extend), [`rem_swa_tokens_for_dp`](../managers/schedule_policy.md#PrefillAdder.rem_swa_tokens_for_dp), [`SWARadixCache`](swa_radix_cache.md#SWARadixCache), [`_lock_node`](../managers/schedule_policy.md#PrefillAdder._lock_node), [`_validate_and_adjust_policy`](../managers/schedule_policy.md#SchedulePolicy._validate_and_adjust_policy), `init_next_round_input`, [`UnifiedRadixCache`](unified_radix_cache.md#UnifiedRadixCache), [`__init__`](../managers/schedule_policy.md#PrefillAdder.__init__), [`RadixCache`](radix_cache.md#RadixCache), `tree_cache`, [`__init__`](../managers/schedule_policy.md#SchedulePolicy.__init__)

### `DecLockRefParams`
- def: [`python/sgl_jax/srt/mem_cache/base_prefix_cache.py:63`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L63)
- doc: Parameters for dec_lock_ref.
- signature: `class DecLockRefParams:`
- members:
  - `skip_lock_node_ids` — [`L68`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L68)
  - `swa_uuid_for_lock` — [`L66`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L66)
- used by: [`cache_unfinished_req`](swa_radix_cache.md#SWARadixCache.cache_unfinished_req), [`cache_finished_req`](swa_radix_cache.md#SWARadixCache.cache_finished_req), [`dec_lock_ref`](swa_radix_cache.md#SWARadixCache.dec_lock_ref), [`release_component_lock`](unified_cache_components/recurrent_component.md#RecurrentComponent.release_component_lock), [`release_component_lock`](unified_cache_components/full_component.md#FullComponent.release_component_lock), [`dec_lock_ref`](radix_cache.md#RadixCache.dec_lock_ref), [`dec_lock_ref`](unified_radix_cache.md#UnifiedRadixCache.dec_lock_ref), [`dec_lock_ref`](base_prefix_cache.md#BasePrefixCache.dec_lock_ref), [`to_dec_params`](base_prefix_cache.md#IncLockRefResult.to_dec_params), [`release_component_lock`](unified_cache_components/tree_component.md#TreeComponent.release_component_lock), [`dec_lock_ref`](chunk_cache.md#ChunkCache.dec_lock_ref)

### `EvictParams`
- def: [`python/sgl_jax/srt/mem_cache/base_prefix_cache.py:44`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L44) — documented in [python-sgl_jax-srt-mem_cache-base_prefix_cache](../../../../../concepts/python-sgl_jax-srt-mem_cache-base_prefix_cache.md)
- doc: Unified parameters for evict across cache types.
- signature: `class EvictParams:`
- members:
  - `dp_rank` — [`L49`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L49)
  - `num_tokens` — [`L47`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L47)
  - `recurrent_num` — [`L50`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L50)
  - `swa_num_tokens` — [`L48`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L48)
- used by: [`evict`](swa_radix_cache.md#SWARadixCache.evict), `alloc_req_slots`, [`evict`](unified_radix_cache.md#UnifiedRadixCache.evict), `_evict_tree_cache_if_needed`, [`evict_from_tree_cache`](common.md#evict_from_tree_cache), [`drive_eviction`](unified_cache_components/recurrent_component.md#RecurrentComponent.drive_eviction), [`drive_eviction`](unified_cache_components/full_component.md#FullComponent.drive_eviction), [`evict`](radix_cache.md#RadixCache.evict), [`evict`](base_prefix_cache.md#BasePrefixCache.evict), [`drive_eviction`](unified_cache_components/tree_component.md#TreeComponent.drive_eviction), [`evict`](chunk_cache.md#ChunkCache.evict)

### `EvictResult`
- def: [`python/sgl_jax/srt/mem_cache/base_prefix_cache.py:54`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L54)
- doc: Result of an evict operation.
- signature: `class EvictResult:`
- members:
  - `num_tokens_evicted` — [`L57`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L57)
  - `recurrent_num_evicted` — [`L59`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L59)
  - `swa_num_tokens_evicted` — [`L58`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L58)
- used by: [`evict`](swa_radix_cache.md#SWARadixCache.evict), [`evict`](unified_radix_cache.md#UnifiedRadixCache.evict), [`evict`](radix_cache.md#RadixCache.evict), [`evict`](base_prefix_cache.md#BasePrefixCache.evict), [`evict`](chunk_cache.md#ChunkCache.evict)

### `IncLockRefResult`
- def: [`python/sgl_jax/srt/mem_cache/base_prefix_cache.py:72`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L72)
- doc: Result of inc_lock_ref.
- signature: `class IncLockRefResult:`
- members:
  - `to_dec_params(self)` — [`L79`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L79)
  - `delta` — [`L75`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L75)
  - `skip_lock_node_ids` — [`L77`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L77)
  - `swa_uuid_for_lock` — [`L76`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L76)
- uses (calls/refs, reference-scoped): [`DecLockRefParams`](base_prefix_cache.md#DecLockRefParams), [`swa_uuid_for_lock`](base_prefix_cache.md#DecLockRefParams.swa_uuid_for_lock), [`skip_lock_node_ids`](base_prefix_cache.md#DecLockRefParams.skip_lock_node_ids)
- used by: [`add_one_req`](../managers/schedule_policy.md#PrefillAdder.add_one_req), [`cache_unfinished_req`](swa_radix_cache.md#SWARadixCache.cache_unfinished_req), [`inc_lock_ref`](swa_radix_cache.md#SWARadixCache.inc_lock_ref), [`acquire_component_lock`](unified_cache_components/full_component.md#FullComponent.acquire_component_lock), [`acquire_component_lock`](unified_cache_components/recurrent_component.md#RecurrentComponent.acquire_component_lock), [`inc_lock_ref`](radix_cache.md#RadixCache.inc_lock_ref), [`inc_lock_ref`](unified_radix_cache.md#UnifiedRadixCache.inc_lock_ref), [`inc_lock_ref`](base_prefix_cache.md#BasePrefixCache.inc_lock_ref), [`_lock_node`](../managers/schedule_policy.md#PrefillAdder._lock_node), [`acquire_component_lock`](unified_cache_components/tree_component.md#TreeComponent.acquire_component_lock), [`inc_lock_ref`](chunk_cache.md#ChunkCache.inc_lock_ref)

### `InsertParams`
- def: [`python/sgl_jax/srt/mem_cache/base_prefix_cache.py:30`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L30) — documented in [python-sgl_jax-srt-mem_cache-base_prefix_cache](../../../../../concepts/python-sgl_jax-srt-mem_cache-base_prefix_cache.md)
- doc: Unified parameters for insert across cache types.
- signature: `class InsertParams:`
- members:
  - `key` — [`L33`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L33)
  - `prev_prefix_len` — [`L36`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L36)
  - `recurrent_value` — [`L40`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L40)
  - `swa_evicted_seqlen` — [`L37`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L37)
  - `value` — [`L34`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L34)
- uses (calls/refs, reference-scoped): [`RadixKey`](radix_cache.md#RadixKey)
- used by: [`cache_unfinished_req`](unified_radix_cache.md#UnifiedRadixCache.cache_unfinished_req), [`cache_unfinished_req`](swa_radix_cache.md#SWARadixCache.cache_unfinished_req), [`cache_unfinished_req`](radix_cache.md#RadixCache.cache_unfinished_req), [`_compute_prefix_matches`](../managers/schedule_policy.md#SchedulePolicy._compute_prefix_matches), [`cache_finished_req`](swa_radix_cache.md#SWARadixCache.cache_finished_req), [`cache_finished_req`](unified_radix_cache.md#UnifiedRadixCache.cache_finished_req), [`cache_finished_req`](radix_cache.md#RadixCache.cache_finished_req), [`_insert_helper`](unified_radix_cache.md#UnifiedRadixCache._insert_helper), [`insert`](unified_radix_cache.md#UnifiedRadixCache.insert), [`insert`](radix_cache.md#RadixCache.insert), [`commit_insert_component_data`](unified_cache_components/recurrent_component.md#RecurrentComponent.commit_insert_component_data), [`insert`](swa_radix_cache.md#SWARadixCache.insert), [`cleanup_after_caching_req`](unified_cache_components/recurrent_component.md#RecurrentComponent.cleanup_after_caching_req), [`cleanup_after_caching_req`](unified_cache_components/tree_component.md#TreeComponent.cleanup_after_caching_req), [`prepare_for_caching_req`](unified_cache_components/recurrent_component.md#RecurrentComponent.prepare_for_caching_req), [`commit_insert_component_data`](unified_cache_components/tree_component.md#TreeComponent.commit_insert_component_data), [`prepare_for_caching_req`](unified_cache_components/tree_component.md#TreeComponent.prepare_for_caching_req), [`update_component_on_insert_overlap`](unified_cache_components/tree_component.md#TreeComponent.update_component_on_insert_overlap), [`recover_after_unevict`](unified_cache_components/tree_component.md#TreeComponent.recover_after_unevict), [`should_skip_leaf_creation`](unified_cache_components/tree_component.md#TreeComponent.should_skip_leaf_creation)

### `MatchPrefixParams`
- def: [`python/sgl_jax/srt/mem_cache/base_prefix_cache.py:16`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L16) — documented in [python-sgl_jax-srt-mem_cache-base_prefix_cache](../../../../../concepts/python-sgl_jax-srt-mem_cache-base_prefix_cache.md)
- doc: Unified parameters for match_prefix across cache types.
- signature: `class MatchPrefixParams:`
- members:
  - `cow_recurrent` — [`L21`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L21)
  - `full_only` — [`L26`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L26)
  - `key` — [`L19`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L19)
  - `req` — [`L22`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L22)
- uses (calls/refs, reference-scoped): [`RadixKey`](radix_cache.md#RadixKey)
- used by: [`cache_unfinished_req`](unified_radix_cache.md#UnifiedRadixCache.cache_unfinished_req), [`cache_unfinished_req`](swa_radix_cache.md#SWARadixCache.cache_unfinished_req), [`cache_unfinished_req`](radix_cache.md#RadixCache.cache_unfinished_req), [`_compute_prefix_matches`](../managers/schedule_policy.md#SchedulePolicy._compute_prefix_matches), `host_hit_length`, [`match_prefix`](radix_cache.md#RadixCache.match_prefix), [`match_prefix`](unified_radix_cache.md#UnifiedRadixCache.match_prefix), [`_match_post_processor`](unified_radix_cache.md#UnifiedRadixCache._match_post_processor), [`match_prefix`](swa_radix_cache.md#SWARadixCache.match_prefix), [`match_prefix`](base_prefix_cache.md#BasePrefixCache.match_prefix), [`_cached_prefix_len`](../managers/scheduler.md#Scheduler._cached_prefix_len), [`finalize_match_result`](unified_cache_components/recurrent_component.md#RecurrentComponent.finalize_match_result), [`match_prefix`](chunk_cache.md#ChunkCache.match_prefix), [`finalize_match_result`](unified_cache_components/tree_component.md#TreeComponent.finalize_match_result)

### `MatchResult`  ·  implements/extends NamedTuple
- def: [`python/sgl_jax/srt/mem_cache/base_prefix_cache.py:86`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L86)
- doc: Result of a prefix match operation.
- signature: `class MatchResult(NamedTuple):`
- members:
  - `best_match_node` — [`L104`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L104)
  - `device_indices` — [`L101`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L101)
  - `host_hit_length` — [`L105`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L105)
  - `last_device_node` — [`L102`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L102)
  - `last_host_node` — [`L103`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L103)
  - `recurrent_branching_seqlen` — [`L108`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/base_prefix_cache.py#L108)
- uses (calls/refs, reference-scoped): [`TreeNode`](radix_cache.md#TreeNode)
- used by: [`cache_unfinished_req`](unified_radix_cache.md#UnifiedRadixCache.cache_unfinished_req), [`cache_unfinished_req`](swa_radix_cache.md#SWARadixCache.cache_unfinished_req), [`cache_unfinished_req`](radix_cache.md#RadixCache.cache_unfinished_req), [`_compute_prefix_matches`](../managers/schedule_policy.md#SchedulePolicy._compute_prefix_matches), `host_hit_length`, [`match_prefix`](radix_cache.md#RadixCache.match_prefix), [`match_prefix`](unified_radix_cache.md#UnifiedRadixCache.match_prefix), [`_match_post_processor`](unified_radix_cache.md#UnifiedRadixCache._match_post_processor), [`match_prefix`](swa_radix_cache.md#SWARadixCache.match_prefix), [`_empty_match_result`](unified_radix_cache.md#UnifiedRadixCache._empty_match_result), [`match_prefix`](base_prefix_cache.md#BasePrefixCache.match_prefix), [`_cached_prefix_len`](../managers/scheduler.md#Scheduler._cached_prefix_len), [`finalize_match_result`](unified_cache_components/recurrent_component.md#RecurrentComponent.finalize_match_result), [`match_prefix`](chunk_cache.md#ChunkCache.match_prefix), [`finalize_match_result`](unified_cache_components/tree_component.md#TreeComponent.finalize_match_result)


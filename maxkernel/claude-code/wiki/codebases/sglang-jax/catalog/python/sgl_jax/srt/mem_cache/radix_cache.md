---
title: 'Module: python/sgl_jax/srt/mem_cache/radix_cache.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/mem_cache/radix_cache.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.mem_cache.radix_cache`/
symbols:
  RadixCache.cache_unfinished_req: RadixCache#cache_unfinished_req().
  RadixKey.dp_rank: RadixKey#dp_rank.
  RadixKey: RadixKey#
  RadixCache.cache_finished_req: RadixCache#cache_finished_req().
  RadixCache.match_prefix: RadixCache#match_prefix().
  RadixCache.insert: RadixCache#insert().
  RadixCache.inc_lock_ref: RadixCache#inc_lock_ref().
  RadixCache._insert_helper: RadixCache#_insert_helper().
  RadixCache.dec_lock_ref: RadixCache#dec_lock_ref().
  RadixCache.root_node: RadixCache#root_node.
  RadixCache.evict: RadixCache#evict().
  TreeNode.key: TreeNode#key.
  TreeNode: TreeNode#
  RadixCache._match_prefix_helper: RadixCache#_match_prefix_helper().
  RadixCache._split_node: RadixCache#_split_node().
  RadixCache._print_helper: RadixCache#_print_helper().
  RadixKey.token_ids: RadixKey#token_ids.
  TreeNode.value: TreeNode#value.
  RadixCache.get_child_key_fn: RadixCache#get_child_key_fn.
  get_child_key: get_child_key().
  RadixKey.extra_key: RadixKey#extra_key.
  RadixCache.pretty_print: RadixCache#pretty_print().
  TreeNode.children: TreeNode#children.
  _key_match_page_size1: _key_match_page_size1().
  _key_match_paged: _key_match_paged().
  RadixCache.page_size: RadixCache#page_size.
  RadixCache._total_size_helper: RadixCache#_total_size_helper().
  RadixCache.key_convert_fn: RadixCache#key_convert_fn.
  RadixKey.__getitem__: RadixKey#__getitem__().
  _check_composite_key: _check_composite_key().
  TreeNode.lock_ref: TreeNode#lock_ref.
  RadixCache.disable: RadixCache#disable.
  RadixCache.is_eagle: RadixCache#is_eagle.
  RadixKey.__repr__: RadixKey#__repr__().
  TreeNode.parent: TreeNode#parent.
  RadixCache: RadixCache#
  TreeNode.last_access_time: TreeNode#last_access_time.
  RadixCache.token_to_kv_pool_allocator: RadixCache#token_to_kv_pool_allocator.
  RadixCache.evictable_size_: RadixCache#evictable_size_.
  RadixCache._collect_leaves: RadixCache#_collect_leaves().
  RadixCache.key_match_fn: RadixCache#key_match_fn.
  TreeNode.id: TreeNode#id.
  TreeNode.__lt__: TreeNode#__lt__().
  RadixCache.__init__: RadixCache#__init__().
  RadixCache.take_events: RadixCache#take_events().
  RadixCache._create_tokens_data: RadixCache#_create_tokens_data().
  RadixCache.req_to_token_pool: RadixCache#req_to_token_pool.
  RadixCache.protected_size_: RadixCache#protected_size_.
  TreeNode.evicted: TreeNode#evicted().
  RadixCache.total_size: RadixCache#total_size().
  RadixCache._delete_leaf: RadixCache#_delete_leaf().
  _convert_to_bigram_key: _convert_to_bigram_key().
  RadixKey.__len__: RadixKey#__len__().
  RadixKey.__iter__: RadixKey#__iter__().
  TreeNode.backuped: TreeNode#backuped().
  RadixCache.evictable_size: RadixCache#evictable_size().
  RadixCache.protected_size: RadixCache#protected_size().
  TreeNode.counter: TreeNode#counter.
  RadixCache.kv_event_queue: RadixCache#kv_event_queue.
  RadixCache.reset: RadixCache#reset().
  TreeNode.host_value: TreeNode#host_value.
  RadixCache.enable_kv_cache_events: RadixCache#enable_kv_cache_events.
  RadixCache.process_id: RadixCache#process_id.
  RadixKey.__init__: RadixKey#__init__().
  TreeNode.__init__: TreeNode#__init__().
  TreeNode.hit_count: TreeNode#hit_count.
  TreeNode.loading: TreeNode#loading.
  RadixCache.kv_head_num: RadixCache#kv_head_num.
  RadixCache.head_dim: RadixCache#head_dim.
  RadixCache.layer_num: RadixCache#layer_num.
  RadixCache.max_seq_len: RadixCache#max_seq_len.
  RadixCache.dtype: RadixCache#dtype.
  RadixCache.num_processes: RadixCache#num_processes.
  RadixCache.local_devices: RadixCache#local_devices.
  RadixCache._delete_leaf_no_size_update: RadixCache#_delete_leaf_no_size_update().
---
# Module: [`python/sgl_jax/srt/mem_cache/radix_cache.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py)

## Classes
### `RadixCache`  ·  implements/extends BasePrefixCache
- def: [`python/sgl_jax/srt/mem_cache/radix_cache.py:154`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L154)
- signature: `class RadixCache(BasePrefixCache):`
- members:
  - `cache_finished_req(self, req: Req, is_insert: bool = True)` — [`L287`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L287) — Cache completed requests. ``is_insert=False`` skips the radix — documented in [python-sgl_jax-srt-mem_cache-base_prefix_cache](../../../../../concepts/python-sgl_jax-srt-mem_cache-base_prefix_cache.md)
  - `cache_unfinished_req(self, req: Req)` — [`L343`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L343) — Cache incomplete requests — documented in [python-sgl_jax-srt-mem_cache-radix_cache](../../../../../concepts/python-sgl_jax-srt-mem_cache-radix_cache.md)
  - `dec_lock_ref(self, node: TreeNode, params: DecLockRefParams | None = None)` — [`L469`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L469) — documented in [python-sgl_jax-srt-mem_cache-radix_cache](../../../../../concepts/python-sgl_jax-srt-mem_cache-radix_cache.md)
  - `evict(self, params: EvictParams)` — [`L418`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L418)
  - `evictable_size(self, dp_rank: int = 0)` — [`L485`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L485)
  - `inc_lock_ref(self, node: TreeNode)` — [`L453`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L453) — documented in [python-sgl_jax-srt-mem_cache-radix_cache](../../../../../concepts/python-sgl_jax-srt-mem_cache-radix_cache.md)
  - `insert(self, params: InsertParams)` — [`L262`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L262) — documented in [python-sgl_jax-srt-mem_cache-radix_cache](../../../../../concepts/python-sgl_jax-srt-mem_cache-radix_cache.md)
  - `match_prefix(self, params: MatchPrefixParams)` — [`L215`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L215) — documented in [python-sgl_jax-srt-mem_cache-base_prefix_cache](../../../../../concepts/python-sgl_jax-srt-mem_cache-base_prefix_cache.md)
  - `pretty_print(self)` — [`L408`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L408)
  - `protected_size(self, dp_rank: int = 0)` — [`L488`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L488)
  - `reset(self)` — [`L206`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L206)
  - `take_events(self)` — [`L491`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L491) — Atomically takes all events and clears the queue.
  - `total_size(self)` — [`L415`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L415)
  - `disable` — [`L172`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L172)
  - `dtype` — [`L177`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L177)
  - `enable_kv_cache_events` — [`L178`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L178)
  - `evictable_size_` — [`L212`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L212)
  - `get_child_key_fn` — [`L194`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L194) — ---
  - `head_dim` — [`L174`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L174)
  - `is_eagle` — [`L185`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L185)
  - `key_convert_fn` — [`L188`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L188)
  - `key_match_fn` — [`L193`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L193)
  - `kv_event_queue` — [`L179`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L179)
  - `kv_head_num` — [`L173`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L173)
  - `layer_num` — [`L175`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L175)
  - `local_devices` — [`L183`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L183)
  - `max_seq_len` — [`L176`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L176)
  - `num_processes` — [`L182`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L182)
  - `page_size` — [`L171`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L171)
  - `process_id` — [`L181`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L181)
  - `protected_size_` — [`L213`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L213)
  - `req_to_token_pool` — [`L169`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L169)
  - `root_node` — [`L207`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L207) — documented in [python-sgl_jax-srt-mem_cache-radix_cache](../../../../../concepts/python-sgl_jax-srt-mem_cache-radix_cache.md)
  - `token_to_kv_pool_allocator` — [`L170`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L170)
- protocol/private: `__init__`[`L155`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L155), `_collect_leaves`[`L630`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L630), `_create_tokens_data`[`L200`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L200), `_delete_leaf`[`L610`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L610), `_delete_leaf_no_size_update`[`L604`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L604), `_insert_helper`[`L547`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L547), `_match_prefix_helper`[`L501`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L501), `_print_helper`[`L586`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L586), `_split_node`[`L527`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L527), `_total_size_helper`[`L619`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L619)
- uses (calls/refs, reference-scoped): `Req`, [`dp_rank`](radix_cache.md#RadixKey.dp_rank), `output_ids`, [`RadixKey`](radix_cache.md#RadixKey), `origin_input_ids`, `prefix_indices`, [`BasePrefixCache`](base_prefix_cache.md#BasePrefixCache), `dp_rank`, `fill_ids`, `req_pool_idx`, [`ReqToTokenPool`](memory_pool.md#ReqToTokenPool), [`BaseTokenToKVPoolAllocator`](allocator.md#BaseTokenToKVPoolAllocator), [`key`](radix_cache.md#TreeNode.key), [`EvictParams`](base_prefix_cache.md#EvictParams), [`InsertParams`](base_prefix_cache.md#InsertParams), [`MatchPrefixParams`](base_prefix_cache.md#MatchPrefixParams), [`IncLockRefResult`](base_prefix_cache.md#IncLockRefResult), `last_node`, [`free`](allocator.md#BaseTokenToKVPoolAllocator.free), [`MatchResult`](base_prefix_cache.md#MatchResult), [`TreeNode`](radix_cache.md#TreeNode), [`token_ids`](radix_cache.md#RadixKey.token_ids), [`value`](radix_cache.md#TreeNode.value), [`DecLockRefParams`](base_prefix_cache.md#DecLockRefParams), [`get_child_key`](radix_cache.md#get_child_key), `last_matched_prefix_len`, [`EvictResult`](base_prefix_cache.md#EvictResult), [`extra_key`](radix_cache.md#RadixKey.extra_key), [`last_device_node`](base_prefix_cache.md#MatchResult.last_device_node), [`device_indices`](base_prefix_cache.md#MatchResult.device_indices), [`children`](radix_cache.md#TreeNode.children), `cache_protected_len`, [`key`](base_prefix_cache.md#MatchPrefixParams.key), [`_key_match_page_size1`](radix_cache.md#_key_match_page_size1), [`_key_match_paged`](radix_cache.md#_key_match_paged), [`key`](base_prefix_cache.md#InsertParams.key), [`last_host_node`](base_prefix_cache.md#MatchResult.last_host_node), [`dp_rank`](base_prefix_cache.md#EvictParams.dp_rank), [`write`](memory_pool.md#ReqToTokenPool.write), [`best_match_node`](base_prefix_cache.md#MatchResult.best_match_node)  (+13 more)
- used by: [`_compute_prefix_matches`](../managers/schedule_policy.md#SchedulePolicy._compute_prefix_matches), [`default_radix_cache_factory`](registry.md#default_radix_cache_factory), [`BasePrefixCache`](base_prefix_cache.md#BasePrefixCache), [`evict`](base_prefix_cache.md#BasePrefixCache.evict), [`match_prefix`](base_prefix_cache.md#BasePrefixCache.match_prefix), [`inc_lock_ref`](base_prefix_cache.md#BasePrefixCache.inc_lock_ref), [`dec_lock_ref`](base_prefix_cache.md#BasePrefixCache.dec_lock_ref), [`evictable_size`](base_prefix_cache.md#BasePrefixCache.evictable_size), [`cache_unfinished_req`](base_prefix_cache.md#BasePrefixCache.cache_unfinished_req), [`cache_finished_req`](base_prefix_cache.md#BasePrefixCache.cache_finished_req), [`pretty_print`](base_prefix_cache.md#BasePrefixCache.pretty_print), [`reset`](base_prefix_cache.md#BasePrefixCache.reset), [`protected_size`](base_prefix_cache.md#BasePrefixCache.protected_size), [`total_size`](base_prefix_cache.md#BasePrefixCache.total_size), [`waiting_queue_radix_tree`](../managers/schedule_policy.md#SchedulePolicy.waiting_queue_radix_tree), [`take_events`](base_prefix_cache.md#BasePrefixCache.take_events)

### `RadixKey`
- def: [`python/sgl_jax/srt/mem_cache/radix_cache.py:31`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L31) — documented in [python-sgl_jax-srt-mem_cache-radix_cache](../../../../../concepts/python-sgl_jax-srt-mem_cache-radix_cache.md)
- doc: Composite key for radix cache that combines token IDs and an optional extra key.
- signature: `class RadixKey:`
- members:
  - `dp_rank` — [`L42`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L42) — documented in [python-sgl_jax-srt-mem_cache-radix_cache](../../../../../concepts/python-sgl_jax-srt-mem_cache-radix_cache.md)
  - `extra_key` — [`L41`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L41)
  - `token_ids` — [`L40`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L40)
- protocol/private: `__getitem__`[`L50`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L50), `__init__`[`L37`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L37), `__iter__`[`L47`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L47), `__len__`[`L44`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L44), `__repr__`[`L55`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L55)
- used by: [`cache_unfinished_req`](unified_radix_cache.md#UnifiedRadixCache.cache_unfinished_req), [`cache_unfinished_req`](swa_radix_cache.md#SWARadixCache.cache_unfinished_req), [`evict`](swa_radix_cache.md#SWARadixCache.evict), [`cache_unfinished_req`](radix_cache.md#RadixCache.cache_unfinished_req), [`_compute_prefix_matches`](../managers/schedule_policy.md#SchedulePolicy._compute_prefix_matches), [`cache_finished_req`](swa_radix_cache.md#SWARadixCache.cache_finished_req), [`cache_finished_req`](unified_radix_cache.md#UnifiedRadixCache.cache_finished_req), `host_hit_length`, [`_insert_helper`](swa_radix_cache.md#SWARadixCache._insert_helper), [`cache_finished_req`](radix_cache.md#RadixCache.cache_finished_req), [`inc_lock_ref`](swa_radix_cache.md#SWARadixCache.inc_lock_ref), [`_insert_helper`](unified_radix_cache.md#UnifiedRadixCache._insert_helper), [`key`](swa_radix_cache.md#TreeNode.key), [`dec_lock_ref`](swa_radix_cache.md#SWARadixCache.dec_lock_ref), [`match_prefix`](radix_cache.md#RadixCache.match_prefix), [`_match_prefix_helper`](unified_radix_cache.md#UnifiedRadixCache._match_prefix_helper), [`root_node`](swa_radix_cache.md#SWARadixCache.root_node), [`_iteratively_delete_tombstone_leaf`](swa_radix_cache.md#SWARadixCache._iteratively_delete_tombstone_leaf), [`_match_prefix_helper`](swa_radix_cache.md#SWARadixCache._match_prefix_helper), [`acquire_component_lock`](unified_cache_components/full_component.md#FullComponent.acquire_component_lock), [`insert`](unified_radix_cache.md#UnifiedRadixCache.insert), [`_add_new_node`](unified_radix_cache.md#UnifiedRadixCache._add_new_node), [`root_node`](unified_radix_cache.md#UnifiedRadixCache.root_node), [`_add_new_node`](swa_radix_cache.md#SWARadixCache._add_new_node), [`match_prefix`](unified_radix_cache.md#UnifiedRadixCache.match_prefix), [`insert`](radix_cache.md#RadixCache.insert), [`release_component_lock`](unified_cache_components/full_component.md#FullComponent.release_component_lock), [`key`](unified_radix_cache.md#UnifiedTreeNode.key), [`_split_node`](unified_radix_cache.md#UnifiedRadixCache._split_node), [`_insert_helper`](radix_cache.md#RadixCache._insert_helper), [`inc_lock_ref`](radix_cache.md#RadixCache.inc_lock_ref), [`_delete_leaf`](swa_radix_cache.md#SWARadixCache._delete_leaf), [`dec_lock_ref`](radix_cache.md#RadixCache.dec_lock_ref), [`root_node`](radix_cache.md#RadixCache.root_node), [`drive_eviction`](unified_cache_components/full_component.md#FullComponent.drive_eviction), [`evict_component`](unified_cache_components/full_component.md#FullComponent.evict_component), [`_print_helper`](unified_radix_cache.md#UnifiedRadixCache._print_helper), [`insert`](swa_radix_cache.md#SWARadixCache.insert), [`key`](radix_cache.md#TreeNode.key), [`_delete_tombstone_leaf`](swa_radix_cache.md#SWARadixCache._delete_tombstone_leaf)  (+13 more)

### `TreeNode`
- def: [`python/sgl_jax/srt/mem_cache/radix_cache.py:60`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L60)
- signature: `class TreeNode:`
- members:
  - `backuped(self)` — [`L85`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L85)
  - `evicted(self)` — [`L81`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L81)
  - `children` — [`L64`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L64)
  - `counter` — [`L61`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L61)
  - `hit_count` — [`L71`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L71)
  - `host_value` — [`L75`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L75)
  - `id` — [`L77`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L77)
  - `key` — [`L66`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L66)
  - `last_access_time` — [`L69`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L69)
  - `loading` — [`L73`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L73)
  - `lock_ref` — [`L68`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L68)
  - `parent` — [`L65`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L65)
  - `value` — [`L67`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L67)
- protocol/private: `__init__`[`L63`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L63), `__lt__`[`L88`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L88)
- uses (calls/refs, reference-scoped): [`RadixKey`](radix_cache.md#RadixKey)
- used by: [`_insert_helper`](radix_cache.md#RadixCache._insert_helper), [`inc_lock_ref`](radix_cache.md#RadixCache.inc_lock_ref), [`dec_lock_ref`](radix_cache.md#RadixCache.dec_lock_ref), [`root_node`](radix_cache.md#RadixCache.root_node), [`_match_prefix_helper`](radix_cache.md#RadixCache._match_prefix_helper), [`_print_helper`](radix_cache.md#RadixCache._print_helper), [`_split_node`](radix_cache.md#RadixCache._split_node), [`last_device_node`](base_prefix_cache.md#MatchResult.last_device_node), [`_get_dfs_priority`](../managers/schedule_policy.md#SchedulePolicy._get_dfs_priority), [`_lock_node`](../managers/schedule_policy.md#PrefillAdder._lock_node), [`last_host_node`](base_prefix_cache.md#MatchResult.last_host_node), [`_calc_weight`](../managers/schedule_policy.md#SchedulePolicy._calc_weight), [`_total_size_helper`](radix_cache.md#RadixCache._total_size_helper), [`best_match_node`](base_prefix_cache.md#MatchResult.best_match_node), [`_collect_leaves`](radix_cache.md#RadixCache._collect_leaves)

## Functions
- `_check_composite_key(key0: RadixKey, key1: RadixKey)` — [`L92`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L92) — Check that two RadixKeys have matching extra_key and dp_rank.
- `_convert_to_bigram_key(tokens: list[int])` — [`L144`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L144)
- `_key_match_page_size1(key0: RadixKey, key1: RadixKey)` — [`L104`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L104)
- `_key_match_paged(key0: RadixKey, key1: RadixKey, page_size: int)` — [`L114`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L114)
- `get_child_key(key: RadixKey, page_size: int = 1)` — [`L127`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/radix_cache.py#L127) — Get child key for tree traversal with namespace isolation via extra_key and dp_rank.


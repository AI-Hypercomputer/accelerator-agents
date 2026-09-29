---
title: 'Module: python/sgl_jax/srt/mem_cache/unified_radix_cache.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/mem_cache/unified_radix_cache.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.mem_cache.unified_radix_cache`/
symbols:
  UnifiedRadixCache.cache_unfinished_req: UnifiedRadixCache#cache_unfinished_req().
  UnifiedRadixCache.cache_finished_req: UnifiedRadixCache#cache_finished_req().
  UnifiedTreeNode: UnifiedTreeNode#
  UnifiedRadixCache._insert_helper: UnifiedRadixCache#_insert_helper().
  UnifiedRadixCache._match_prefix_helper: UnifiedRadixCache#_match_prefix_helper().
  UnifiedRadixCache.insert: UnifiedRadixCache#insert().
  UnifiedRadixCache._evict_device_leaf: UnifiedRadixCache#_evict_device_leaf().
  UnifiedRadixCache._add_new_node: UnifiedRadixCache#_add_new_node().
  UnifiedRadixCache.root_node: UnifiedRadixCache#root_node.
  UnifiedRadixCache.match_prefix: UnifiedRadixCache#match_prefix().
  UnifiedTreeNode.key: UnifiedTreeNode#key.
  UnifiedRadixCache._match_post_processor: UnifiedRadixCache#_match_post_processor().
  UnifiedTreeNode.component_data: UnifiedTreeNode#component_data.
  UnifiedRadixCache.evict: UnifiedRadixCache#evict().
  UnifiedRadixCache._split_node: UnifiedRadixCache#_split_node().
  UnifiedRadixCache._print_helper: UnifiedRadixCache#_print_helper().
  UnifiedRadixCache._components_tuple: UnifiedRadixCache#_components_tuple.
  UnifiedRadixCache._empty_match_result: UnifiedRadixCache#_empty_match_result().
  UnifiedRadixCache._is_device_leaf: UnifiedRadixCache#_is_device_leaf().
  UnifiedRadixCache.get_child_key_fn: UnifiedRadixCache#get_child_key_fn.
  UnifiedRadixCache.inc_lock_ref: UnifiedRadixCache#inc_lock_ref().
  UnifiedRadixCache.pretty_print: UnifiedRadixCache#pretty_print().
  UnifiedRadixCache._update_evictable_leaf_sets: UnifiedRadixCache#_update_evictable_leaf_sets().
  UnifiedRadixCache.component_evictable_size_: UnifiedRadixCache#component_evictable_size_.
  UnifiedRadixCache.page_size: UnifiedRadixCache#page_size.
  UnifiedRadixCache.dec_lock_ref: UnifiedRadixCache#dec_lock_ref().
  UnifiedTreeNode.parent: UnifiedTreeNode#parent.
  COMPONENT_REGISTRY.COMPONENT_REGISTRY: COMPONENT_REGISTRY.COMPONENT_REGISTRY.
  UnifiedTreeNode.children: UnifiedTreeNode#children.
  UnifiedRadixCache.total_size: UnifiedRadixCache#total_size().
  UnifiedRadixCache._cascade_evict: UnifiedRadixCache#_cascade_evict().
  UnifiedRadixCache._remove_leaf_from_parent: UnifiedRadixCache#_remove_leaf_from_parent().
  UnifiedRadixCache.component_protected_size_: UnifiedRadixCache#component_protected_size_.
  UnifiedRadixCache: UnifiedRadixCache#
  UnifiedRadixCache.assert_recurrent_slot_ledger: UnifiedRadixCache#assert_recurrent_slot_ledger().
  UnifiedTreeNode.evicted: UnifiedTreeNode#evicted().
  UnifiedRadixCache.evictable_device_leaves: UnifiedRadixCache#evictable_device_leaves.
  UnifiedRadixCache.key_convert_fn: UnifiedRadixCache#key_convert_fn.
  UnifiedRadixCache.components: UnifiedRadixCache#components.
  UnifiedRadixCache.__init__: UnifiedRadixCache#__init__().
  UnifiedTreeNode.last_access_time: UnifiedTreeNode#last_access_time.
  UnifiedTreeNode.id: UnifiedTreeNode#id.
  UnifiedRadixCache.req_to_token_pool: UnifiedRadixCache#req_to_token_pool.
  UnifiedRadixCache.token_to_kv_pool_allocator: UnifiedRadixCache#token_to_kv_pool_allocator.
  UnifiedRadixCache.is_eagle: UnifiedRadixCache#is_eagle.
  UnifiedTreeNode.backuped: UnifiedTreeNode#backuped().
  UnifiedRadixCache.supports_recurrent: UnifiedRadixCache#supports_recurrent().
  UnifiedRadixCache.disable: UnifiedRadixCache#disable.
  UnifiedRadixCache.tree_components: UnifiedRadixCache#tree_components.
  UnifiedRadixCache.evictable_size: UnifiedRadixCache#evictable_size().
  UnifiedRadixCache.protected_size: UnifiedRadixCache#protected_size().
  UnifiedRadixCache.key_match_fn: UnifiedRadixCache#key_match_fn.
  UnifiedTreeNode.__lt__: UnifiedTreeNode#__lt__().
  UnifiedRadixCache.take_events: UnifiedRadixCache#take_events().
  UnifiedRadixCache._update_best_if_valid: UnifiedRadixCache#_update_best_if_valid().
  UnifiedTreeNode.__init__: UnifiedTreeNode#__init__().
  UnifiedRadixCache.full_evictable_size: UnifiedRadixCache#full_evictable_size().
  UnifiedRadixCache.full_protected_size: UnifiedRadixCache#full_protected_size().
  UnifiedTreeNode.counter: UnifiedTreeNode#counter.
  UnifiedRadixCache.kv_event_queue: UnifiedRadixCache#kv_event_queue.
  UnifiedRadixCache.enable_kv_cache_events: UnifiedRadixCache#enable_kv_cache_events.
  UnifiedRadixCache.process_id: UnifiedRadixCache#process_id.
  UnifiedRadixCache.reset: UnifiedRadixCache#reset().
  UnifiedTreeNode.tree_components: UnifiedTreeNode#tree_components.
  UnifiedRadixCache.kv_head_num: UnifiedRadixCache#kv_head_num.
  UnifiedRadixCache.head_dim: UnifiedRadixCache#head_dim.
  UnifiedRadixCache.layer_num: UnifiedRadixCache#layer_num.
  UnifiedRadixCache.max_seq_len: UnifiedRadixCache#max_seq_len.
  UnifiedRadixCache.dtype: UnifiedRadixCache#dtype.
  UnifiedRadixCache.swa_evictable_size: UnifiedRadixCache#swa_evictable_size().
  UnifiedRadixCache.swa_protected_size: UnifiedRadixCache#swa_protected_size().
---
# Module: [`python/sgl_jax/srt/mem_cache/unified_radix_cache.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py)

## Classes
### `UnifiedRadixCache`  ·  implements/extends BasePrefixCache
- def: [`python/sgl_jax/srt/mem_cache/unified_radix_cache.py:91`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L91)
- signature: `class UnifiedRadixCache(BasePrefixCache):`
- members:
  - `_cascade_evict(self, node: UnifiedTreeNode)` — [`L696`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L696) — Tombstone the base value after all components have been driven.
  - `_evict_device_leaf(self, node: UnifiedTreeNode, tracker: dict[ComponentType, int])` — [`L710`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L710) — Evict a device leaf: free all component device data, tombstone the — documented in [python-sgl_jax-srt-mem_cache-unified_cache_components-tree_component](../../../../../concepts/python-sgl_jax-srt-mem_cache-unified_cache_components-tree_component.md)
  - `_is_device_leaf(self, node: UnifiedTreeNode)` — [`L726`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L726) — D-leaf: FULL device value present, no child with FULL KV on device,
  - `assert_recurrent_slot_ledger(self, dp_rank: int = 0)` — [`L223`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L223) — Assert active + tree_owned + free == slots_per_rank; return active.
  - `cache_finished_req(self, req: Req, is_insert: bool = True, **kwargs)` — [`L261`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L261) — Cache completed requests. ``is_insert=False`` skips the radix — documented in [python-sgl_jax-srt-mem_cache-unified_radix_cache](../../../../../concepts/python-sgl_jax-srt-mem_cache-unified_radix_cache.md)
  - `cache_unfinished_req(self, req: Req, **kwargs)` — [`L344`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L344) — Cache incomplete requests. — documented in [python-sgl_jax-srt-mem_cache-base_prefix_cache](../../../../../concepts/python-sgl_jax-srt-mem_cache-base_prefix_cache.md)
  - `dec_lock_ref(self, node: UnifiedTreeNode, params: DecLockRefParams | None = None)` — [`L251`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L251)
  - `evict(self, params: EvictParams)` — [`L206`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L206) — documented in [python-sgl_jax-srt-mem_cache-unified_cache_components-tree_component](../../../../../concepts/python-sgl_jax-srt-mem_cache-unified_cache_components-tree_component.md)
  - `evictable_size(self, dp_rank: int = 0)` — [`L441`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L441)
  - `full_evictable_size(self, dp_rank: int = 0)` — [`L444`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L444)
  - `full_protected_size(self, dp_rank: int = 0)` — [`L450`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L450)
  - `inc_lock_ref(self, node: UnifiedTreeNode)` — [`L240`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L240)
  - `insert(self, params: InsertParams)` — [`L175`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L175) — documented in [python-sgl_jax-srt-mem_cache-unified_radix_cache](../../../../../concepts/python-sgl_jax-srt-mem_cache-unified_radix_cache.md)
  - `match_prefix(self, params: MatchPrefixParams)` — [`L157`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L157) — documented in [python-sgl_jax-srt-mem_cache-unified_radix_cache](../../../../../concepts/python-sgl_jax-srt-mem_cache-unified_radix_cache.md)
  - `pretty_print(self)` — [`L470`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L470)
  - `protected_size(self, dp_rank: int = 0)` — [`L447`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L447)
  - `reset(self)` — [`L142`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L142)
  - `supports_recurrent(self)` — [`L220`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L220)
  - `swa_evictable_size(self, dp_rank: int = 0)` — [`L453`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L453)
  - `swa_protected_size(self)` — [`L456`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L456)
  - `take_events(self)` — [`L479`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L479) — Atomically takes all events and clears the queue.
  - `total_size(self)` — [`L459`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L459)
  - `component_evictable_size_` — [`L149`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L149)
  - `component_protected_size_` — [`L152`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L152)
  - `components` — [`L135`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L135)
  - `disable` — [`L110`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L110)
  - `dtype` — [`L115`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L115)
  - `enable_kv_cache_events` — [`L116`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L116)
  - `evictable_device_leaves` — [`L155`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L155)
  - `get_child_key_fn` — [`L129`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L129) — ---
  - `head_dim` — [`L112`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L112)
  - `is_eagle` — [`L121`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L121)
  - `key_convert_fn` — [`L123`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L123)
  - `key_match_fn` — [`L128`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L128)
  - `kv_event_queue` — [`L117`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L117)
  - `kv_head_num` — [`L111`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L111)
  - `layer_num` — [`L113`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L113)
  - `max_seq_len` — [`L114`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L114)
  - `page_size` — [`L109`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L109)
  - `process_id` — [`L119`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L119)
  - `req_to_token_pool` — [`L107`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L107)
  - `root_node` — [`L143`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L143)
  - `token_to_kv_pool_allocator` — [`L108`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L108)
  - `tree_components` — [`L134`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L134)
- protocol/private: `__init__`[`L92`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L92), `_add_new_node`[`L608`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L608), `_components_tuple`[`L138`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L138), `_empty_match_result`[`L489`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L489), `_insert_helper`[`L626`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L626), `_match_post_processor`[`L555`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L555), `_match_prefix_helper`[`L498`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L498), `_print_helper`[`L744`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L744), `_remove_leaf_from_parent`[`L705`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L705), `_split_node`[`L591`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L591), `_update_best_if_valid`[`L522`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L522), `_update_evictable_leaf_sets`[`L738`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L738)
- uses (calls/refs, reference-scoped): `Req`, [`dp_rank`](radix_cache.md#RadixKey.dp_rank), `output_ids`, [`RadixKey`](radix_cache.md#RadixKey), [`UnifiedTreeNode`](unified_radix_cache.md#UnifiedTreeNode), `origin_input_ids`, `prefix_indices`, [`BasePrefixCache`](base_prefix_cache.md#BasePrefixCache), `dp_rank`, [`value`](unified_cache_components/tree_component.md#ComponentData.value), `fill_ids`, [`ComponentType`](unified_cache_components/tree_component.md#ComponentType), [`key`](unified_radix_cache.md#UnifiedTreeNode.key), `req_pool_idx`, [`ReqToTokenPool`](memory_pool.md#ReqToTokenPool), [`component_data`](unified_radix_cache.md#UnifiedTreeNode.component_data), [`BaseTokenToKVPoolAllocator`](allocator.md#BaseTokenToKVPoolAllocator), [`EvictParams`](base_prefix_cache.md#EvictParams), [`InsertParams`](base_prefix_cache.md#InsertParams), [`MatchPrefixParams`](base_prefix_cache.md#MatchPrefixParams), [`IncLockRefResult`](base_prefix_cache.md#IncLockRefResult), `last_node`, [`free`](allocator.md#BaseTokenToKVPoolAllocator.free), [`MatchResult`](base_prefix_cache.md#MatchResult), [`BASE_COMPONENT_TYPE`](unified_cache_components/tree_component.md#BASE_COMPONENT_TYPE), [`node_has_component_data`](unified_cache_components/tree_component.md#TreeComponent.node_has_component_data), [`component_type`](unified_cache_components/tree_component.md#TreeComponent.component_type), [`token_ids`](radix_cache.md#RadixKey.token_ids), [`EvictLayer`](unified_cache_components/tree_component.md#EvictLayer), [`DecLockRefParams`](base_prefix_cache.md#DecLockRefParams), [`lock_ref`](unified_cache_components/tree_component.md#ComponentData.lock_ref), [`get_child_key`](radix_cache.md#get_child_key), `last_matched_prefix_len`, [`parent`](unified_radix_cache.md#UnifiedTreeNode.parent), [`EvictResult`](base_prefix_cache.md#EvictResult), [`COMPONENT_REGISTRY`](unified_radix_cache.md#COMPONENT_REGISTRY.COMPONENT_REGISTRY), [`extra_key`](radix_cache.md#RadixKey.extra_key), [`children`](unified_radix_cache.md#UnifiedTreeNode.children), [`last_device_node`](base_prefix_cache.md#MatchResult.last_device_node), [`cleanup_after_caching_req`](unified_cache_components/tree_component.md#TreeComponent.cleanup_after_caching_req)  (+43 more)
- used by: [`default_radix_cache_factory`](registry.md#default_radix_cache_factory), [`BasePrefixCache`](base_prefix_cache.md#BasePrefixCache), [`acquire_component_lock`](unified_cache_components/full_component.md#FullComponent.acquire_component_lock), [`release_component_lock`](unified_cache_components/recurrent_component.md#RecurrentComponent.release_component_lock), [`commit_insert_component_data`](unified_cache_components/recurrent_component.md#RecurrentComponent.commit_insert_component_data), [`release_component_lock`](unified_cache_components/full_component.md#FullComponent.release_component_lock), [`acquire_component_lock`](unified_cache_components/recurrent_component.md#RecurrentComponent.acquire_component_lock), [`drive_eviction`](unified_cache_components/recurrent_component.md#RecurrentComponent.drive_eviction), [`drive_eviction`](unified_cache_components/full_component.md#FullComponent.drive_eviction), [`evict_component`](unified_cache_components/full_component.md#FullComponent.evict_component), [`evict_component`](unified_cache_components/recurrent_component.md#RecurrentComponent.evict_component), [`on_parent_gains_child`](unified_cache_components/recurrent_component.md#RecurrentComponent.on_parent_gains_child), [`evict`](base_prefix_cache.md#BasePrefixCache.evict), [`match_prefix`](base_prefix_cache.md#BasePrefixCache.match_prefix), [`inc_lock_ref`](base_prefix_cache.md#BasePrefixCache.inc_lock_ref), [`__init__`](unified_cache_components/recurrent_component.md#RecurrentComponent.__init__), [`dec_lock_ref`](base_prefix_cache.md#BasePrefixCache.dec_lock_ref), [`evictable_size`](base_prefix_cache.md#BasePrefixCache.evictable_size), [`full_evictable_size`](base_prefix_cache.md#BasePrefixCache.full_evictable_size), [`swa_evictable_size`](base_prefix_cache.md#BasePrefixCache.swa_evictable_size), [`cache_unfinished_req`](base_prefix_cache.md#BasePrefixCache.cache_unfinished_req), [`cache_finished_req`](base_prefix_cache.md#BasePrefixCache.cache_finished_req), [`pretty_print`](base_prefix_cache.md#BasePrefixCache.pretty_print), [`reset`](base_prefix_cache.md#BasePrefixCache.reset), [`supports_recurrent`](base_prefix_cache.md#BasePrefixCache.supports_recurrent), [`full_protected_size`](base_prefix_cache.md#BasePrefixCache.full_protected_size), [`protected_size`](base_prefix_cache.md#BasePrefixCache.protected_size), [`swa_protected_size`](base_prefix_cache.md#BasePrefixCache.swa_protected_size), [`total_size`](base_prefix_cache.md#BasePrefixCache.total_size), [`__init__`](unified_cache_components/full_component.md#FullComponent.__init__), [`req_to_token_pool`](unified_cache_components/recurrent_component.md#RecurrentComponent.req_to_token_pool), [`__init__`](unified_cache_components/tree_component.md#TreeComponent.__init__), [`_free_full`](unified_cache_components/full_component.md#FullComponent._free_full), [`recurrent_state_pool`](unified_cache_components/recurrent_component.md#RecurrentComponent.recurrent_state_pool), [`take_events`](base_prefix_cache.md#BasePrefixCache.take_events)

### `UnifiedTreeNode`
- def: [`python/sgl_jax/srt/mem_cache/unified_radix_cache.py:55`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L55) — documented in [python-sgl_jax-srt-mem_cache-unified_radix_cache](../../../../../concepts/python-sgl_jax-srt-mem_cache-unified_radix_cache.md)
- signature: `class UnifiedTreeNode:`
- members:
  - `backuped(self)` — [`L77`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L77) — Tree-level: FULL KV present on host.
  - `evicted(self)` — [`L72`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L72) — Tree-level: FULL KV not on device (non-root with value=None).
  - `children` — [`L59`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L59)
  - `component_data` — [`L64`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L64)
  - `counter` — [`L56`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L56)
  - `id` — [`L68`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L68)
  - `key` — [`L61`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L61) — documented in [python-sgl_jax-srt-mem_cache-unified_radix_cache](../../../../../concepts/python-sgl_jax-srt-mem_cache-unified_radix_cache.md)
  - `last_access_time` — [`L67`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L67)
  - `parent` — [`L60`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L60)
  - `tree_components` — [`L62`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L62)
- protocol/private: `__init__`[`L58`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L58), `__lt__`[`L81`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L81)
- uses (calls/refs, reference-scoped): [`RadixKey`](radix_cache.md#RadixKey), [`value`](unified_cache_components/tree_component.md#ComponentData.value), [`ComponentType`](unified_cache_components/tree_component.md#ComponentType), [`BASE_COMPONENT_TYPE`](unified_cache_components/tree_component.md#BASE_COMPONENT_TYPE), [`get_and_increase_time_counter`](unified_cache_components/tree_component.md#get_and_increase_time_counter), [`_NUM_COMPONENT_TYPES`](unified_cache_components/tree_component.md#_NUM_COMPONENT_TYPES), [`ComponentData`](unified_cache_components/tree_component.md#ComponentData), [`host_value`](unified_cache_components/tree_component.md#ComponentData.host_value)
- used by: [`_insert_helper`](unified_radix_cache.md#UnifiedRadixCache._insert_helper), [`_match_prefix_helper`](unified_radix_cache.md#UnifiedRadixCache._match_prefix_helper), [`acquire_component_lock`](unified_cache_components/full_component.md#FullComponent.acquire_component_lock), [`release_component_lock`](unified_cache_components/recurrent_component.md#RecurrentComponent.release_component_lock), [`_evict_device_leaf`](unified_radix_cache.md#UnifiedRadixCache._evict_device_leaf), [`_add_new_node`](unified_radix_cache.md#UnifiedRadixCache._add_new_node), [`root_node`](unified_radix_cache.md#UnifiedRadixCache.root_node), [`commit_insert_component_data`](unified_cache_components/recurrent_component.md#RecurrentComponent.commit_insert_component_data), [`release_component_lock`](unified_cache_components/full_component.md#FullComponent.release_component_lock), [`_match_post_processor`](unified_radix_cache.md#UnifiedRadixCache._match_post_processor), [`acquire_component_lock`](unified_cache_components/recurrent_component.md#RecurrentComponent.acquire_component_lock), [`_split_node`](unified_radix_cache.md#UnifiedRadixCache._split_node), [`drive_eviction`](unified_cache_components/full_component.md#FullComponent.drive_eviction), [`evict_component`](unified_cache_components/full_component.md#FullComponent.evict_component), [`evict_component`](unified_cache_components/recurrent_component.md#RecurrentComponent.evict_component), [`_print_helper`](unified_radix_cache.md#UnifiedRadixCache._print_helper), [`on_parent_gains_child`](unified_cache_components/recurrent_component.md#RecurrentComponent.on_parent_gains_child), [`node_has_component_data`](unified_cache_components/tree_component.md#TreeComponent.node_has_component_data), [`_is_device_leaf`](unified_radix_cache.md#UnifiedRadixCache._is_device_leaf), [`inc_lock_ref`](unified_radix_cache.md#UnifiedRadixCache.inc_lock_ref), [`_update_evictable_leaf_sets`](unified_radix_cache.md#UnifiedRadixCache._update_evictable_leaf_sets), [`dec_lock_ref`](unified_radix_cache.md#UnifiedRadixCache.dec_lock_ref), [`_node_dp_rank`](unified_cache_components/recurrent_component.md#_node_dp_rank), [`redistribute_on_node_split`](unified_cache_components/full_component.md#FullComponent.redistribute_on_node_split), [`_cascade_evict`](unified_radix_cache.md#UnifiedRadixCache._cascade_evict), [`_remove_leaf_from_parent`](unified_radix_cache.md#UnifiedRadixCache._remove_leaf_from_parent), [`evict_component`](unified_cache_components/tree_component.md#TreeComponent.evict_component), [`total_size`](unified_radix_cache.md#UnifiedRadixCache.total_size), [`redistribute_on_node_split`](unified_cache_components/recurrent_component.md#RecurrentComponent.redistribute_on_node_split), [`evictable_device_leaves`](unified_radix_cache.md#UnifiedRadixCache.evictable_device_leaves), [`acquire_component_lock`](unified_cache_components/tree_component.md#TreeComponent.acquire_component_lock), [`commit_insert_component_data`](unified_cache_components/tree_component.md#TreeComponent.commit_insert_component_data), [`release_component_lock`](unified_cache_components/tree_component.md#TreeComponent.release_component_lock), [`create_match_validator`](unified_cache_components/full_component.md#FullComponent.create_match_validator), [`create_match_validator`](unified_cache_components/recurrent_component.md#RecurrentComponent.create_match_validator), [`value_len`](unified_cache_components/tree_component.md#TreeComponent.value_len), [`create_match_validator`](unified_cache_components/tree_component.md#TreeComponent.create_match_validator), [`redistribute_on_node_split`](unified_cache_components/tree_component.md#TreeComponent.redistribute_on_node_split), [`build_hicache_transfers`](unified_cache_components/tree_component.md#TreeComponent.build_hicache_transfers), [`commit_hicache_transfer`](unified_cache_components/tree_component.md#TreeComponent.commit_hicache_transfer)  (+4 more)

## Module values
- `COMPONENT_REGISTRY` — [`L85`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_radix_cache.py#L85)


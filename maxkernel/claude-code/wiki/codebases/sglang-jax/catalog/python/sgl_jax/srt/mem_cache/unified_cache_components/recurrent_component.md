---
title: 'Module: python/sgl_jax/srt/mem_cache/unified_cache_components/recurrent_component.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/mem_cache/unified_cache_components/recurrent_component.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.mem_cache.unified_cache_components.recurrent_component`/
symbols:
  RecurrentComponent.release_component_lock: RecurrentComponent#release_component_lock().
  RecurrentComponent.commit_insert_component_data: RecurrentComponent#commit_insert_component_data().
  RecurrentComponent.acquire_component_lock: RecurrentComponent#acquire_component_lock().
  RecurrentComponent.drive_eviction: RecurrentComponent#drive_eviction().
  RecurrentComponent.evict_component: RecurrentComponent#evict_component().
  RecurrentComponent.on_parent_gains_child: RecurrentComponent#on_parent_gains_child().
  _node_dp_rank: _node_dp_rank().
  RecurrentComponent.__init__: RecurrentComponent#__init__().
  RecurrentComponent.finalize_match_result: RecurrentComponent#finalize_match_result().
  RecurrentComponent.prepare_for_caching_req: RecurrentComponent#prepare_for_caching_req().
  RecurrentComponent.cleanup_after_caching_req: RecurrentComponent#cleanup_after_caching_req().
  RecurrentComponent.redistribute_on_node_split: RecurrentComponent#redistribute_on_node_split().
  RecurrentComponent.create_match_validator: RecurrentComponent#create_match_validator().
  RecurrentComponent.req_to_token_pool: RecurrentComponent#req_to_token_pool.
  RecurrentComponent: RecurrentComponent#
  RecurrentComponent.component_type: RecurrentComponent#component_type.
  RecurrentComponent.recurrent_state_pool: RecurrentComponent#recurrent_state_pool.
  RecurrentComponent.eviction_priority: RecurrentComponent#eviction_priority().
---
# Module: [`python/sgl_jax/srt/mem_cache/unified_cache_components/recurrent_component.py`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/recurrent_component.py)

## Classes
### `RecurrentComponent`  ·  implements/extends TreeComponent
- def: [`python/sgl_jax/srt/mem_cache/unified_cache_components/recurrent_component.py:40`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/recurrent_component.py#L40)
- signature: `class RecurrentComponent(TreeComponent):`
- members:
  - `acquire_component_lock(self, node: UnifiedTreeNode, result: IncLockRefResult, lock_host: bool = False)` — [`L200`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/recurrent_component.py#L200) — documented in [python-sgl_jax-srt-mem_cache-unified_cache_components-tree_component](../../../../../../concepts/python-sgl_jax-srt-mem_cache-unified_cache_components-tree_component.md)
  - `cleanup_after_caching_req(self, req: Req, is_finished: bool, insert_result: InsertResult | None = None, insert_params: InsertParams | None = None)` — [`L139`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/recurrent_component.py#L139)
  - `commit_insert_component_data(self, node: UnifiedTreeNode, is_new_leaf: bool, params: InsertParams, result: InsertResult)` — [`L100`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/recurrent_component.py#L100) — documented in [python-sgl_jax-srt-mem_cache-unified_cache_components-tree_component](../../../../../../concepts/python-sgl_jax-srt-mem_cache-unified_cache_components-tree_component.md)
  - `create_match_validator(self, match_device_only: bool = False)` — [`L59`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/recurrent_component.py#L59)
  - `drive_eviction(self, params: EvictParams, tracker: dict[ComponentType, int])` — [`L174`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/recurrent_component.py#L174) — documented in [python-sgl_jax-srt-mem_cache-unified_cache_components-tree_component](../../../../../../concepts/python-sgl_jax-srt-mem_cache-unified_cache_components-tree_component.md)
  - `evict_component(self, node: UnifiedTreeNode, target: EvictLayer = EvictLayer.DEVICE)` — [`L156`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/recurrent_component.py#L156)
  - `eviction_priority(self, is_leaf: bool)` — [`L171`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/recurrent_component.py#L171)
  - `finalize_match_result(self, result: MatchResult, params: MatchPrefixParams, value_chunks: list, best_value_len: int)` — [`L65`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/recurrent_component.py#L65)
  - `on_parent_gains_child(self, node: UnifiedTreeNode)` — [`L90`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/recurrent_component.py#L90)
  - `prepare_for_caching_req(self, req: Req, insert_params: InsertParams, token_ids_len: int, is_finished: bool)` — [`L123`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/recurrent_component.py#L123)
  - `redistribute_on_node_split(self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode)` — [`L85`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/recurrent_component.py#L85)
  - `release_component_lock(self, node: UnifiedTreeNode, params: DecLockRefParams | None, lock_host: bool = False)` — [`L220`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/recurrent_component.py#L220)
  - `component_type` — [`L41`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/recurrent_component.py#L41)
  - `recurrent_state_pool` — [`L55`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/recurrent_component.py#L55)
  - `req_to_token_pool` — [`L54`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/recurrent_component.py#L54)
- protocol/private: `__init__`[`L43`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/recurrent_component.py#L43)
- uses (calls/refs, reference-scoped): `Req`, [`UnifiedTreeNode`](../unified_radix_cache.md#UnifiedTreeNode), [`_evict_device_leaf`](../unified_radix_cache.md#UnifiedRadixCache._evict_device_leaf), [`root_node`](../unified_radix_cache.md#UnifiedRadixCache.root_node), [`value`](tree_component.md#ComponentData.value), [`ComponentType`](tree_component.md#ComponentType), [`component_data`](../unified_radix_cache.md#UnifiedTreeNode.component_data), [`cache`](tree_component.md#TreeComponent.cache), [`EvictParams`](../base_prefix_cache.md#EvictParams), [`InsertParams`](../base_prefix_cache.md#InsertParams), [`MatchPrefixParams`](../base_prefix_cache.md#MatchPrefixParams), [`IncLockRefResult`](../base_prefix_cache.md#IncLockRefResult), [`MatchResult`](../base_prefix_cache.md#MatchResult), [`node_has_component_data`](tree_component.md#TreeComponent.node_has_component_data), [`component_type`](tree_component.md#TreeComponent.component_type), [`EvictLayer`](tree_component.md#EvictLayer), [`DecLockRefParams`](../base_prefix_cache.md#DecLockRefParams), [`lock_ref`](tree_component.md#ComponentData.lock_ref), [`_update_evictable_leaf_sets`](../unified_radix_cache.md#UnifiedRadixCache._update_evictable_leaf_sets), [`component_evictable_size_`](../unified_radix_cache.md#UnifiedRadixCache.component_evictable_size_), [`page_size`](../unified_radix_cache.md#UnifiedRadixCache.page_size), [`_node_dp_rank`](recurrent_component.md#_node_dp_rank), [`children`](../unified_radix_cache.md#UnifiedTreeNode.children), [`InsertResult`](tree_component.md#InsertResult), [`TreeComponent`](tree_component.md#TreeComponent), [`component_protected_size_`](../unified_radix_cache.md#UnifiedRadixCache.component_protected_size_), `recurrent_pool_idx`, [`HybridReqToTokenPool`](../memory_pool.md#HybridReqToTokenPool), [`UnifiedRadixCache`](../unified_radix_cache.md#UnifiedRadixCache), [`DEVICE`](tree_component.md#EvictLayer.DEVICE), [`dp_rank`](../base_prefix_cache.md#EvictParams.dp_rank), [`evictable_device_leaves`](../unified_radix_cache.md#UnifiedRadixCache.evictable_device_leaves), [`commit_to_tree`](../memory_pool.md#HybridReqToTokenPool.commit_to_tree), [`best_match_node`](../base_prefix_cache.md#MatchResult.best_match_node), [`CacheInitParams`](../cache_init_params.md#CacheInitParams), [`id`](../unified_radix_cache.md#UnifiedTreeNode.id), [`req_to_token_pool`](../unified_radix_cache.md#UnifiedRadixCache.req_to_token_pool), [`RECURRENT`](tree_component.md#ComponentType.RECURRENT), [`__init__`](tree_component.md#TreeComponent.__init__), [`free_recurrent_slot`](../memory_pool.md#HybridReqToTokenPool.free_recurrent_slot)  (+10 more)
- used by: [`COMPONENT_REGISTRY`](../unified_radix_cache.md#COMPONENT_REGISTRY.COMPONENT_REGISTRY), [`cleanup_after_caching_req`](tree_component.md#TreeComponent.cleanup_after_caching_req), [`TreeComponent`](tree_component.md#TreeComponent), [`evict_component`](tree_component.md#TreeComponent.evict_component), [`acquire_component_lock`](tree_component.md#TreeComponent.acquire_component_lock), [`commit_insert_component_data`](tree_component.md#TreeComponent.commit_insert_component_data), [`drive_eviction`](tree_component.md#TreeComponent.drive_eviction), [`release_component_lock`](tree_component.md#TreeComponent.release_component_lock), [`create_match_validator`](tree_component.md#TreeComponent.create_match_validator), [`prepare_for_caching_req`](tree_component.md#TreeComponent.prepare_for_caching_req), [`finalize_match_result`](tree_component.md#TreeComponent.finalize_match_result), [`redistribute_on_node_split`](tree_component.md#TreeComponent.redistribute_on_node_split), [`on_parent_gains_child`](tree_component.md#TreeComponent.on_parent_gains_child), [`eviction_priority`](tree_component.md#TreeComponent.eviction_priority)

## Functions
- `_node_dp_rank(node: UnifiedTreeNode)` — [`L36`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/recurrent_component.py#L36)


---
title: 'Module: python/sgl_jax/srt/mem_cache/unified_cache_components/full_component.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/mem_cache/unified_cache_components/full_component.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.mem_cache.unified_cache_components.full_component`/FullComponent#
symbols:
  FullComponent.acquire_component_lock: acquire_component_lock().
  FullComponent.release_component_lock: release_component_lock().
  FullComponent.evict_component: evict_component().
  FullComponent.drive_eviction: drive_eviction().
  FullComponent.redistribute_on_node_split: redistribute_on_node_split().
  FullComponent.create_match_validator: create_match_validator().
  FullComponent.__init__: __init__().
  FullComponent._free_full: _free_full.
  FullComponent: ''
  FullComponent.component_type: component_type.
  FullComponent._full_kv_pool_host: _full_kv_pool_host.
  FullComponent.eviction_priority: eviction_priority().
---
# Module: [`python/sgl_jax/srt/mem_cache/unified_cache_components/full_component.py`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/full_component.py)

## Classes
### `FullComponent`  ·  implements/extends TreeComponent
- def: [`python/sgl_jax/srt/mem_cache/unified_cache_components/full_component.py:26`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/full_component.py#L26)
- signature: `class FullComponent(TreeComponent):`
- members:
  - `acquire_component_lock(self, node: UnifiedTreeNode, result: IncLockRefResult, lock_host: bool = False)` — [`L89`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/full_component.py#L89) — documented in [python-sgl_jax-srt-mem_cache-unified_cache_components-tree_component](../../../../../../concepts/python-sgl_jax-srt-mem_cache-unified_cache_components-tree_component.md)
  - `create_match_validator(self, match_device_only: bool = False)` — [`L35`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/full_component.py#L35)
  - `drive_eviction(self, params: EvictParams, tracker: dict[ComponentType, int])` — [`L70`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/full_component.py#L70) — documented in [python-sgl_jax-srt-mem_cache-unified_cache_components-tree_component](../../../../../../concepts/python-sgl_jax-srt-mem_cache-unified_cache_components-tree_component.md)
  - `evict_component(self, node: UnifiedTreeNode, target: EvictLayer = EvictLayer.DEVICE)` — [`L50`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/full_component.py#L50)
  - `eviction_priority(self, is_leaf: bool)` — [`L67`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/full_component.py#L67)
  - `redistribute_on_node_split(self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode)` — [`L41`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/full_component.py#L41)
  - `release_component_lock(self, node: UnifiedTreeNode, params: DecLockRefParams | None, lock_host: bool = False)` — [`L125`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/full_component.py#L125)
  - `component_type` — [`L27`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/full_component.py#L27)
- protocol/private: `__init__`[`L29`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/full_component.py#L29), `_free_full`[`L31`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/full_component.py#L31), `_full_kv_pool_host`[`L33`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/full_component.py#L33)
- uses (calls/refs, reference-scoped): [`dp_rank`](../radix_cache.md#RadixKey.dp_rank), [`UnifiedTreeNode`](../unified_radix_cache.md#UnifiedTreeNode), [`_evict_device_leaf`](../unified_radix_cache.md#UnifiedRadixCache._evict_device_leaf), [`root_node`](../unified_radix_cache.md#UnifiedRadixCache.root_node), [`value`](tree_component.md#ComponentData.value), [`ComponentType`](tree_component.md#ComponentType), [`key`](../unified_radix_cache.md#UnifiedTreeNode.key), [`component_data`](../unified_radix_cache.md#UnifiedTreeNode.component_data), [`cache`](tree_component.md#TreeComponent.cache), [`EvictParams`](../base_prefix_cache.md#EvictParams), [`IncLockRefResult`](../base_prefix_cache.md#IncLockRefResult), [`free`](../allocator.md#BaseTokenToKVPoolAllocator.free), [`component_type`](tree_component.md#TreeComponent.component_type), [`EvictLayer`](tree_component.md#EvictLayer), [`DecLockRefParams`](../base_prefix_cache.md#DecLockRefParams), [`lock_ref`](tree_component.md#ComponentData.lock_ref), [`_update_evictable_leaf_sets`](../unified_radix_cache.md#UnifiedRadixCache._update_evictable_leaf_sets), [`component_evictable_size_`](../unified_radix_cache.md#UnifiedRadixCache.component_evictable_size_), [`parent`](../unified_radix_cache.md#UnifiedTreeNode.parent), [`TreeComponent`](tree_component.md#TreeComponent), [`component_protected_size_`](../unified_radix_cache.md#UnifiedRadixCache.component_protected_size_), [`UnifiedRadixCache`](../unified_radix_cache.md#UnifiedRadixCache), [`DEVICE`](tree_component.md#EvictLayer.DEVICE), [`dp_rank`](../base_prefix_cache.md#EvictParams.dp_rank), [`evictable_device_leaves`](../unified_radix_cache.md#UnifiedRadixCache.evictable_device_leaves), [`CacheInitParams`](../cache_init_params.md#CacheInitParams), [`id`](../unified_radix_cache.md#UnifiedTreeNode.id), [`num_tokens`](../base_prefix_cache.md#EvictParams.num_tokens), [`token_to_kv_pool_allocator`](../unified_radix_cache.md#UnifiedRadixCache.token_to_kv_pool_allocator), [`FULL`](tree_component.md#ComponentType.FULL), [`__init__`](tree_component.md#TreeComponent.__init__), [`delta`](../base_prefix_cache.md#IncLockRefResult.delta)
- used by: [`COMPONENT_REGISTRY`](../unified_radix_cache.md#COMPONENT_REGISTRY.COMPONENT_REGISTRY), [`TreeComponent`](tree_component.md#TreeComponent), [`evict_component`](tree_component.md#TreeComponent.evict_component), [`acquire_component_lock`](tree_component.md#TreeComponent.acquire_component_lock), [`drive_eviction`](tree_component.md#TreeComponent.drive_eviction), [`release_component_lock`](tree_component.md#TreeComponent.release_component_lock), [`create_match_validator`](tree_component.md#TreeComponent.create_match_validator), [`redistribute_on_node_split`](tree_component.md#TreeComponent.redistribute_on_node_split), [`eviction_priority`](tree_component.md#TreeComponent.eviction_priority)


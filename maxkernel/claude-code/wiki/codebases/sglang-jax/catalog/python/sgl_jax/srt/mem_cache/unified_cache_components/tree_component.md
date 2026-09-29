---
title: 'Module: python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.mem_cache.unified_cache_components.tree_component`/
symbols:
  ComponentData.value: ComponentData#value.
  ComponentType: ComponentType#
  TreeComponent.cache: TreeComponent#cache.
  BASE_COMPONENT_TYPE: BASE_COMPONENT_TYPE.
  TreeComponent.node_has_component_data: TreeComponent#node_has_component_data().
  TreeComponent.component_type: TreeComponent#component_type.
  EvictLayer: EvictLayer#
  ComponentData.lock_ref: ComponentData#lock_ref.
  TreeComponent.cleanup_after_caching_req: TreeComponent#cleanup_after_caching_req().
  InsertResult: InsertResult#
  TreeComponent: TreeComponent#
  TreeComponent.evict_component: TreeComponent#evict_component().
  EvictLayer.DEVICE: EvictLayer#DEVICE.
  get_and_increase_time_counter: get_and_increase_time_counter().
  TreeComponent.commit_insert_component_data: TreeComponent#commit_insert_component_data().
  TreeComponent.drive_eviction: TreeComponent#drive_eviction().
  TreeComponent.acquire_component_lock: TreeComponent#acquire_component_lock().
  TreeComponent.release_component_lock: TreeComponent#release_component_lock().
  TreeComponent.value_len: TreeComponent#value_len().
  TreeComponent.create_match_validator: TreeComponent#create_match_validator().
  TreeComponent.prepare_for_caching_req: TreeComponent#prepare_for_caching_req().
  TreeComponent.finalize_match_result: TreeComponent#finalize_match_result().
  TreeComponent.redistribute_on_node_split: TreeComponent#redistribute_on_node_split().
  TreeComponent.build_hicache_transfers: TreeComponent#build_hicache_transfers().
  TreeComponent.commit_hicache_transfer: TreeComponent#commit_hicache_transfer().
  ComponentType.FULL: ComponentType#FULL.
  ComponentType.RECURRENT: ComponentType#RECURRENT.
  TreeComponent.__init__: TreeComponent#__init__().
  InsertResult.prefix_len: InsertResult#prefix_len.
  TreeComponent.update_component_on_insert_overlap: TreeComponent#update_component_on_insert_overlap().
  TreeComponent.on_parent_gains_child: TreeComponent#on_parent_gains_child().
  _NUM_COMPONENT_TYPES: _NUM_COMPONENT_TYPES.
  ComponentType.is_full: ComponentType#is_full().
  ComponentType.is_swa: ComponentType#is_swa().
  ComponentType.is_recurrent: ComponentType#is_recurrent().
  EvictLayer.ALL: EvictLayer#ALL.
  TreeComponent.recover_after_unevict: TreeComponent#recover_after_unevict().
  TreeComponent.eviction_priority: TreeComponent#eviction_priority().
  ComponentData: ComponentData#
  InsertResult.recurrent_committed: InsertResult#recurrent_committed.
  next_component_uuid: next_component_uuid().
  TreeComponent.should_skip_leaf_creation: TreeComponent#should_skip_leaf_creation().
  _LAST_ACCESS_TIME_COUNTER_FLOAT: _LAST_ACCESS_TIME_COUNTER_FLOAT.
  _COMPONENT_UUID_COUNTER: _COMPONENT_UUID_COUNTER.
  CacheTransferPhase: CacheTransferPhase#
  TreeComponent.drive_host_eviction: TreeComponent#drive_host_eviction().
  ComponentType.SWA: ComponentType#SWA.
  ComponentData.host_value: ComponentData#host_value.
  EvictLayer.HOST: EvictLayer#HOST.
  InsertResult.recurrent_exist: InsertResult#recurrent_exist.
  LRURefreshPhase: LRURefreshPhase#
  ComponentType.__str__: ComponentType#__str__().
  ComponentData.host_lock_ref: ComponentData#host_lock_ref.
  CacheTransferPhase.BACKUP_HOST: CacheTransferPhase#BACKUP_HOST.
  CacheTransferPhase.LOAD_BACK: CacheTransferPhase#LOAD_BACK.
  CacheTransferPhase.BACKUP_STORAGE: CacheTransferPhase#BACKUP_STORAGE.
  CacheTransferPhase.PREFETCH: CacheTransferPhase#PREFETCH.
  LRURefreshPhase.WALKDOWN: LRURefreshPhase#WALKDOWN.
  LRURefreshPhase.MATCH_END: LRURefreshPhase#MATCH_END.
  LRURefreshPhase.INSERT_END: LRURefreshPhase#INSERT_END.
---
# Module: [`python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py)

## Classes
### `CacheTransferPhase`  ·  implements/extends StrEnum
- def: [`python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py:99`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L99)
- signature: `class CacheTransferPhase(StrEnum):`
- members:
  - `BACKUP_HOST` — [`L101`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L101)
  - `BACKUP_STORAGE` — [`L103`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L103)
  - `LOAD_BACK` — [`L102`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L102)
  - `PREFETCH` — [`L104`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L104)
- used by: [`build_hicache_transfers`](tree_component.md#TreeComponent.build_hicache_transfers), [`commit_hicache_transfer`](tree_component.md#TreeComponent.commit_hicache_transfer)

### `ComponentData`
- def: [`python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py:70`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L70)
- signature: `class ComponentData:`
- members:
  - `host_lock_ref` — [`L74`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L74)
  - `host_value` — [`L73`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L73)
  - `lock_ref` — [`L72`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L72)
  - `value` — [`L71`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L71) — documented in [python-sgl_jax-srt-mem_cache-unified_cache_components-tree_component](../../../../../../concepts/python-sgl_jax-srt-mem_cache-unified_cache_components-tree_component.md)
- used by: [`_match_prefix_helper`](../unified_radix_cache.md#UnifiedRadixCache._match_prefix_helper), [`acquire_component_lock`](full_component.md#FullComponent.acquire_component_lock), [`release_component_lock`](recurrent_component.md#RecurrentComponent.release_component_lock), [`_add_new_node`](../unified_radix_cache.md#UnifiedRadixCache._add_new_node), [`root_node`](../unified_radix_cache.md#UnifiedRadixCache.root_node), [`commit_insert_component_data`](recurrent_component.md#RecurrentComponent.commit_insert_component_data), [`release_component_lock`](full_component.md#FullComponent.release_component_lock), [`component_data`](../unified_radix_cache.md#UnifiedTreeNode.component_data), [`acquire_component_lock`](recurrent_component.md#RecurrentComponent.acquire_component_lock), [`evict_component`](full_component.md#FullComponent.evict_component), [`evict_component`](recurrent_component.md#RecurrentComponent.evict_component), [`_print_helper`](../unified_radix_cache.md#UnifiedRadixCache._print_helper), [`on_parent_gains_child`](recurrent_component.md#RecurrentComponent.on_parent_gains_child), [`node_has_component_data`](tree_component.md#TreeComponent.node_has_component_data), [`_is_device_leaf`](../unified_radix_cache.md#UnifiedRadixCache._is_device_leaf), [`redistribute_on_node_split`](full_component.md#FullComponent.redistribute_on_node_split), [`_cascade_evict`](../unified_radix_cache.md#UnifiedRadixCache._cascade_evict), [`total_size`](../unified_radix_cache.md#UnifiedRadixCache.total_size), [`redistribute_on_node_split`](recurrent_component.md#RecurrentComponent.redistribute_on_node_split), [`evicted`](../unified_radix_cache.md#UnifiedTreeNode.evicted), [`create_match_validator`](full_component.md#FullComponent.create_match_validator), [`create_match_validator`](recurrent_component.md#RecurrentComponent.create_match_validator), [`value_len`](tree_component.md#TreeComponent.value_len), [`backuped`](../unified_radix_cache.md#UnifiedTreeNode.backuped)

### `ComponentType`  ·  implements/extends IntEnum
- def: [`python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py:39`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L39) — documented in [python-sgl_jax-srt-mem_cache-unified_cache_components-tree_component](../../../../../../concepts/python-sgl_jax-srt-mem_cache-unified_cache_components-tree_component.md)
- doc: Integer enum so that per-node list/tuple storage can be indexed directly.
- signature: `class ComponentType(IntEnum):`
- members:
  - `is_full(self)` — [`L50`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L50)
  - `is_recurrent(self)` — [`L58`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L58)
  - `is_swa(self)` — [`L54`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L54)
  - `FULL` — [`L42`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L42)
  - `RECURRENT` — [`L44`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L44)
  - `SWA` — [`L43`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L43)
- protocol/private: `__str__`[`L46`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L46)
- used by: [`default_radix_cache_factory`](../registry.md#default_radix_cache_factory), [`_evict_device_leaf`](../unified_radix_cache.md#UnifiedRadixCache._evict_device_leaf), [`evict`](../unified_radix_cache.md#UnifiedRadixCache.evict), [`drive_eviction`](recurrent_component.md#RecurrentComponent.drive_eviction), [`drive_eviction`](full_component.md#FullComponent.drive_eviction), [`BASE_COMPONENT_TYPE`](tree_component.md#BASE_COMPONENT_TYPE), [`component_type`](tree_component.md#TreeComponent.component_type), [`component_evictable_size_`](../unified_radix_cache.md#UnifiedRadixCache.component_evictable_size_), [`COMPONENT_REGISTRY`](../unified_radix_cache.md#COMPONENT_REGISTRY.COMPONENT_REGISTRY), [`component_protected_size_`](../unified_radix_cache.md#UnifiedRadixCache.component_protected_size_), [`assert_recurrent_slot_ledger`](../unified_radix_cache.md#UnifiedRadixCache.assert_recurrent_slot_ledger), [`drive_eviction`](tree_component.md#TreeComponent.drive_eviction), [`components`](../unified_radix_cache.md#UnifiedRadixCache.components), [`__init__`](../unified_radix_cache.md#UnifiedRadixCache.__init__), [`supports_recurrent`](../unified_radix_cache.md#UnifiedRadixCache.supports_recurrent), [`_NUM_COMPONENT_TYPES`](tree_component.md#_NUM_COMPONENT_TYPES), [`component_type`](full_component.md#FullComponent.component_type), [`component_type`](recurrent_component.md#RecurrentComponent.component_type), [`__init__`](../unified_radix_cache.md#UnifiedTreeNode.__init__), [`drive_host_eviction`](tree_component.md#TreeComponent.drive_host_eviction)

### `EvictLayer`  ·  implements/extends IntFlag
- def: [`python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py:91`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L91)
- doc: Which storage layer(s) to evict.  Combinable via bitwise OR.
- signature: `class EvictLayer(IntFlag):`
- members:
  - `ALL` — [`L96`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L96)
  - `DEVICE` — [`L94`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L94)
  - `HOST` — [`L95`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L95)
- used by: [`_evict_device_leaf`](../unified_radix_cache.md#UnifiedRadixCache._evict_device_leaf), [`drive_eviction`](recurrent_component.md#RecurrentComponent.drive_eviction), [`evict_component`](full_component.md#FullComponent.evict_component), [`evict_component`](recurrent_component.md#RecurrentComponent.evict_component), [`node_has_component_data`](tree_component.md#TreeComponent.node_has_component_data), [`evict_component`](tree_component.md#TreeComponent.evict_component)

### `InsertResult`
- def: [`python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py:78`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L78)
- doc: Result of an insert operation.
- signature: `class InsertResult:`
- members:
  - `prefix_len` — [`L84`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L84)
  - `recurrent_committed` — [`L88`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L88)
  - `recurrent_exist` — [`L87`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L87)
- used by: [`cache_unfinished_req`](../unified_radix_cache.md#UnifiedRadixCache.cache_unfinished_req), [`cache_finished_req`](../unified_radix_cache.md#UnifiedRadixCache.cache_finished_req), [`_insert_helper`](../unified_radix_cache.md#UnifiedRadixCache._insert_helper), [`insert`](../unified_radix_cache.md#UnifiedRadixCache.insert), [`commit_insert_component_data`](recurrent_component.md#RecurrentComponent.commit_insert_component_data), [`cleanup_after_caching_req`](recurrent_component.md#RecurrentComponent.cleanup_after_caching_req), [`cleanup_after_caching_req`](tree_component.md#TreeComponent.cleanup_after_caching_req), [`commit_insert_component_data`](tree_component.md#TreeComponent.commit_insert_component_data), [`commit_hicache_transfer`](tree_component.md#TreeComponent.commit_hicache_transfer)

### `LRURefreshPhase`  ·  implements/extends StrEnum
- def: [`python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py:107`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L107)
- signature: `class LRURefreshPhase(StrEnum):`
- members:
  - `INSERT_END` — [`L111`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L111)
  - `MATCH_END` — [`L110`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L110)
  - `WALKDOWN` — [`L109`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L109)

### `TreeComponent`  ·  implements/extends ABC
- def: [`python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py:127`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L127)
- signature: `class TreeComponent(ABC):`
- members:
  - `acquire_component_lock(self, node: UnifiedTreeNode, result: IncLockRefResult, lock_host: bool = False)` — [`L305`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L305) — Increment component lock refs, protecting nodes from
  - `build_hicache_transfers(self, node: UnifiedTreeNode, phase: CacheTransferPhase, *, req: Req | None = None, token_ids: Sequence[int] | None = None, prefetch_tokens: int = 0, last_hash: str | None = None)` — [`L383`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L383) — Build transfer descriptors for this component in the given phase.
  - `cleanup_after_caching_req(self, req: Req, is_finished: bool, insert_result: InsertResult | None = None, insert_params: InsertParams | None = None)` — [`L363`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L363) — Post-cache cleanup for component-specific resources.
  - `commit_hicache_transfer(self, node: UnifiedTreeNode, phase: CacheTransferPhase, transfers: Sequence[Any] = (), *, insert_result: InsertResult | None = None, pool_storage_result: Any = None)` — [`L397`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L397) — Post-transfer bookkeeping: store host indices, update LRU, etc.
  - `commit_insert_component_data(self, node: UnifiedTreeNode, is_new_leaf: bool, params: InsertParams, result: InsertResult)` — [`L221`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L221) — Finalize component data on the target (leaf) node after the insert
  - `create_match_validator(self, match_device_only: bool = False)` — [`L148`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L148) — Return a per-match stateful predicate that decides whether a node
  - `drive_eviction(self, params: EvictParams, tracker: dict[ComponentType, int])` — [`L292`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L292) — Drive eviction for this component.
  - `drive_host_eviction(self, num_tokens: int, tracker: dict[ComponentType, int])` — [`L409`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L409) — Evict from this component's host-side resources.
  - `evict_component(self, node: UnifiedTreeNode, target: EvictLayer = EvictLayer.DEVICE)` — [`L254`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L254) — Free this component's KV resources on a node being evicted.
  - `eviction_priority(self, is_leaf: bool)` — [`L275`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L275) — Eviction priority on this node type. Higher = evicted later; evicting
  - `finalize_match_result(self, result: MatchResult, params: MatchPrefixParams, value_chunks: list[np.ndarray], best_value_len: int)` — [`L162`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L162) — Post-process the match result after prefix matching completes.
  - `node_has_component_data(self, node: UnifiedTreeNode, target: EvictLayer = EvictLayer.DEVICE)` — [`L134`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L134) — documented in [python-sgl_jax-srt-mem_cache-unified_cache_components-tree_component](../../../../../../concepts/python-sgl_jax-srt-mem_cache-unified_cache_components-tree_component.md)
  - `on_parent_gains_child(self, node: UnifiedTreeNode)` — [`L216`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L216) — ``node`` just gained its first child (leaf->internal); leaf-only
  - `prepare_for_caching_req(self, req: Req, insert_params: InsertParams, token_ids_len: int, is_finished: bool)` — [`L346`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L346) — Prepare component-specific data before insert, fill component
  - `recover_after_unevict(self, node: UnifiedTreeNode, prefix_len: int, total_prefix_len: int, params: InsertParams)` — [`L201`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L201) — Hook for a future tombstoning component (no-op today: device
  - `redistribute_on_node_split(self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode)` — [`L242`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L242) — Redistribute component data between new_parent and child when a
  - `release_component_lock(self, node: UnifiedTreeNode, params: DecLockRefParams | None, lock_host: bool = False)` — [`L328`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L328) — Decrement component lock refs, un-protecting nodes.
  - `should_skip_leaf_creation(self, total_prefix_len: int, key_len: int, params: InsertParams)` — [`L194`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L194) — Return True to veto leaf creation when the entire new leaf would
  - `update_component_on_insert_overlap(self, node: UnifiedTreeNode, prefix_len: int, total_prefix_len: int, value_slice: np.ndarray, params: InsertParams)` — [`L176`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L176) — Called per-node when an insert's key overlaps an existing node.
  - `value_len(self, node: UnifiedTreeNode)` — [`L143`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L143)
  - `cache` — [`L129`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L129) — documented in [python-sgl_jax-srt-mem_cache-unified_cache_components-tree_component](../../../../../../concepts/python-sgl_jax-srt-mem_cache-unified_cache_components-tree_component.md)
  - `component_type` — [`L132`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L132)
- protocol/private: `__init__`[`L128`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L128)
- uses (calls/refs, reference-scoped): `Req`, [`UnifiedTreeNode`](../unified_radix_cache.md#UnifiedTreeNode), [`acquire_component_lock`](full_component.md#FullComponent.acquire_component_lock), [`release_component_lock`](recurrent_component.md#RecurrentComponent.release_component_lock), [`value`](tree_component.md#ComponentData.value), [`ComponentType`](tree_component.md#ComponentType), [`commit_insert_component_data`](recurrent_component.md#RecurrentComponent.commit_insert_component_data), [`release_component_lock`](full_component.md#FullComponent.release_component_lock), [`component_data`](../unified_radix_cache.md#UnifiedTreeNode.component_data), [`acquire_component_lock`](recurrent_component.md#RecurrentComponent.acquire_component_lock), [`drive_eviction`](recurrent_component.md#RecurrentComponent.drive_eviction), [`drive_eviction`](full_component.md#FullComponent.drive_eviction), [`evict_component`](full_component.md#FullComponent.evict_component), [`evict_component`](recurrent_component.md#RecurrentComponent.evict_component), [`EvictParams`](../base_prefix_cache.md#EvictParams), [`InsertParams`](../base_prefix_cache.md#InsertParams), [`MatchPrefixParams`](../base_prefix_cache.md#MatchPrefixParams), [`IncLockRefResult`](../base_prefix_cache.md#IncLockRefResult), [`on_parent_gains_child`](recurrent_component.md#RecurrentComponent.on_parent_gains_child), [`MatchResult`](../base_prefix_cache.md#MatchResult), [`EvictLayer`](tree_component.md#EvictLayer), [`DecLockRefParams`](../base_prefix_cache.md#DecLockRefParams), [`cleanup_after_caching_req`](recurrent_component.md#RecurrentComponent.cleanup_after_caching_req), [`finalize_match_result`](recurrent_component.md#RecurrentComponent.finalize_match_result), [`prepare_for_caching_req`](recurrent_component.md#RecurrentComponent.prepare_for_caching_req), [`redistribute_on_node_split`](full_component.md#FullComponent.redistribute_on_node_split), [`InsertResult`](tree_component.md#InsertResult), [`UnifiedRadixCache`](../unified_radix_cache.md#UnifiedRadixCache), [`redistribute_on_node_split`](recurrent_component.md#RecurrentComponent.redistribute_on_node_split), [`DEVICE`](tree_component.md#EvictLayer.DEVICE), [`CacheInitParams`](../cache_init_params.md#CacheInitParams), [`create_match_validator`](full_component.md#FullComponent.create_match_validator), [`create_match_validator`](recurrent_component.md#RecurrentComponent.create_match_validator), [`FullComponent`](full_component.md#FullComponent), [`RecurrentComponent`](recurrent_component.md#RecurrentComponent), [`CacheTransferPhase`](tree_component.md#CacheTransferPhase), [`HOST`](tree_component.md#EvictLayer.HOST), [`host_value`](tree_component.md#ComponentData.host_value), [`eviction_priority`](full_component.md#FullComponent.eviction_priority), [`eviction_priority`](recurrent_component.md#RecurrentComponent.eviction_priority)
- used by: [`cache_unfinished_req`](../unified_radix_cache.md#UnifiedRadixCache.cache_unfinished_req), [`cache_finished_req`](../unified_radix_cache.md#UnifiedRadixCache.cache_finished_req), [`_insert_helper`](../unified_radix_cache.md#UnifiedRadixCache._insert_helper), [`_match_prefix_helper`](../unified_radix_cache.md#UnifiedRadixCache._match_prefix_helper), [`acquire_component_lock`](full_component.md#FullComponent.acquire_component_lock), [`release_component_lock`](recurrent_component.md#RecurrentComponent.release_component_lock), [`_evict_device_leaf`](../unified_radix_cache.md#UnifiedRadixCache._evict_device_leaf), [`_add_new_node`](../unified_radix_cache.md#UnifiedRadixCache._add_new_node), [`commit_insert_component_data`](recurrent_component.md#RecurrentComponent.commit_insert_component_data), [`release_component_lock`](full_component.md#FullComponent.release_component_lock), [`_match_post_processor`](../unified_radix_cache.md#UnifiedRadixCache._match_post_processor), [`acquire_component_lock`](recurrent_component.md#RecurrentComponent.acquire_component_lock), [`evict`](../unified_radix_cache.md#UnifiedRadixCache.evict), [`_split_node`](../unified_radix_cache.md#UnifiedRadixCache._split_node), [`drive_eviction`](recurrent_component.md#RecurrentComponent.drive_eviction), [`drive_eviction`](full_component.md#FullComponent.drive_eviction), [`evict_component`](full_component.md#FullComponent.evict_component), [`evict_component`](recurrent_component.md#RecurrentComponent.evict_component), [`_components_tuple`](../unified_radix_cache.md#UnifiedRadixCache._components_tuple), [`on_parent_gains_child`](recurrent_component.md#RecurrentComponent.on_parent_gains_child), [`inc_lock_ref`](../unified_radix_cache.md#UnifiedRadixCache.inc_lock_ref), [`dec_lock_ref`](../unified_radix_cache.md#UnifiedRadixCache.dec_lock_ref), [`COMPONENT_REGISTRY`](../unified_radix_cache.md#COMPONENT_REGISTRY.COMPONENT_REGISTRY), [`__init__`](recurrent_component.md#RecurrentComponent.__init__), [`finalize_match_result`](recurrent_component.md#RecurrentComponent.finalize_match_result), [`redistribute_on_node_split`](full_component.md#FullComponent.redistribute_on_node_split), [`redistribute_on_node_split`](recurrent_component.md#RecurrentComponent.redistribute_on_node_split), [`components`](../unified_radix_cache.md#UnifiedRadixCache.components), [`create_match_validator`](full_component.md#FullComponent.create_match_validator), [`create_match_validator`](recurrent_component.md#RecurrentComponent.create_match_validator), [`__init__`](full_component.md#FullComponent.__init__), [`FullComponent`](full_component.md#FullComponent), [`RecurrentComponent`](recurrent_component.md#RecurrentComponent)

## Functions
- `get_and_increase_time_counter()` — [`L114`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L114)
- `next_component_uuid()` — [`L121`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L121)

## Module values
- `BASE_COMPONENT_TYPE` — [`L62`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L62) — documented in [python-sgl_jax-srt-mem_cache-unified_cache_components-tree_component](../../../../../../concepts/python-sgl_jax-srt-mem_cache-unified_cache_components-tree_component.md)
- `_COMPONENT_UUID_COUNTER` — [`L66`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L66)
- `_LAST_ACCESS_TIME_COUNTER_FLOAT` — [`L65`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L65)
- `_NUM_COMPONENT_TYPES` — [`L63`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/unified_cache_components/tree_component.py#L63)


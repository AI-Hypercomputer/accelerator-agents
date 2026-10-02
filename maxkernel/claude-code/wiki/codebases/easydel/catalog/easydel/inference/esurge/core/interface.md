---
title: 'Module: easydel/inference/esurge/core/interface.py'
type: catalog
provenance: extracted
module: easydel/inference/esurge/core/interface.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.esurge.core.interface`/
symbols:
  create_kv_cache_specs_from_config: create_kv_cache_specs_from_config().
  estimate_runtime_page_budget: estimate_runtime_page_budget().
  CacheSpec.page_size: CacheSpec#page_size.
  AttentionSpec.page_size_bytes: AttentionSpec#page_size_bytes().
  CacheSpec: CacheSpec#
  CacheGroupSpec.kv_cache_spec: CacheGroupSpec#kv_cache_spec.
  CacheGroupSpec: CacheGroupSpec#
  AttentionSpec: AttentionSpec#
  FullAttentionSpec: FullAttentionSpec#
  CacheSpec.type_id: CacheSpec#type_id().
  CacheSpec.max_memory_usage_bytes: CacheSpec#max_memory_usage_bytes().
  MambaSpec.page_size_bytes: MambaSpec#page_size_bytes().
  ChunkedLocalAttentionSpec: ChunkedLocalAttentionSpec#
  FullAttentionSpec.merge: FullAttentionSpec#merge().
  ChunkedLocalAttentionSpec.max_memory_usage_bytes: ChunkedLocalAttentionSpec#max_memory_usage_bytes().
  SlidingWindowSpec.max_memory_usage_bytes: SlidingWindowSpec#max_memory_usage_bytes().
  SlidingWindowSpec: SlidingWindowSpec#
  _estimate_group_pages_per_request: _estimate_group_pages_per_request().
  FullAttentionSpec.max_memory_usage_bytes: FullAttentionSpec#max_memory_usage_bytes().
  ChunkedLocalAttentionSpec.type_id: ChunkedLocalAttentionSpec#type_id().
  SlidingWindowSpec.type_id: SlidingWindowSpec#type_id().
  MambaSpec: MambaSpec#
  AttentionSpec.use_mla: AttentionSpec#use_mla.
  FullAttentionSpec.type_id: FullAttentionSpec#type_id().
  CacheSpec.merge: CacheSpec#merge().
  AttentionSpec.num_kv_heads: AttentionSpec#num_kv_heads.
  AttentionSpec.head_size: AttentionSpec#head_size.
  AttentionSpec.dtype: AttentionSpec#dtype.
  ChunkedLocalAttentionSpec.attention_chunk_size: ChunkedLocalAttentionSpec#attention_chunk_size.
  _resolve_layer_attention_geometry: _resolve_layer_attention_geometry().
  CacheGroupsConfig.kv_cache_groups: CacheGroupsConfig#kv_cache_groups.
  CacheSpec.page_size_bytes: CacheSpec#page_size_bytes().
  SlidingWindowSpec.sliding_window: SlidingWindowSpec#sliding_window.
  MambaSpec.type_id: MambaSpec#type_id().
  CacheGroupSpec.layer_names: CacheGroupSpec#layer_names.
  CacheGroupsConfig: CacheGroupsConfig#
  FullAttentionSpec.sliding_window: FullAttentionSpec#sliding_window.
  FullAttentionSpec.attention_chunk_size: FullAttentionSpec#attention_chunk_size.
  _resolve_positive_layer_config_value: _resolve_positive_layer_config_value().
  MambaSpec.page_size_padded: MambaSpec#page_size_padded.
  RuntimePageBudgetEstimate: RuntimePageBudgetEstimate#
  RuntimePageBudgetEstimate.pages_per_request: RuntimePageBudgetEstimate#pages_per_request.
  RuntimePageBudgetEstimate.max_num_seqs: RuntimePageBudgetEstimate#max_num_seqs.
  SlidingWindowSpec.__post_init__: SlidingWindowSpec#__post_init__().
  MambaSpec.__post_init__: MambaSpec#__post_init__().
  MambaSpec.max_memory_usage_bytes: MambaSpec#max_memory_usage_bytes().
  FullAttentionSpec.merge_window_sizes: FullAttentionSpec#merge_window_sizes().
  MambaSpec.shapes: MambaSpec#shapes.
  MambaSpec.dtype: MambaSpec#dtype.
  RuntimePageBudgetEstimate.usable_pages: RuntimePageBudgetEstimate#usable_pages.
  RuntimePageBudgetEstimate.per_group_pages: RuntimePageBudgetEstimate#per_group_pages.
  MambaSpec.num_elements: MambaSpec#num_elements.
  CacheGroupsConfig.num_pages: CacheGroupsConfig#num_pages.
---
# Module: [`easydel/inference/esurge/core/interface.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py)

## Classes
### `AttentionSpec`  ·  implements/extends CacheSpec
- def: [`easydel/inference/esurge/core/interface.py:148`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L148)
- doc: Base specification for attention-based KV-cache.
- signature: `class AttentionSpec(CacheSpec):`
- members:
  - `page_size_bytes(self)` — [`L168`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L168) — Calculate page size in bytes for attention KV-cache.
  - `dtype` — [`L164`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L164)
  - `head_size` — [`L163`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L163)
  - `num_kv_heads` — [`L162`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L162)
  - `use_mla` — [`L165`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L165)
- uses (calls/refs, reference-scoped): [`page_size`](interface.md#CacheSpec.page_size), [`CacheSpec`](interface.md#CacheSpec), [`FullAttentionSpec`](interface.md#FullAttentionSpec), [`ChunkedLocalAttentionSpec`](interface.md#ChunkedLocalAttentionSpec), [`SlidingWindowSpec`](interface.md#SlidingWindowSpec), [`get_dtype_size`](../utils.md#get_dtype_size)
- used by: [`create_kv_cache_specs_from_config`](interface.md#create_kv_cache_specs_from_config), [`CacheSpec`](interface.md#CacheSpec), [`FullAttentionSpec`](interface.md#FullAttentionSpec), [`ChunkedLocalAttentionSpec`](interface.md#ChunkedLocalAttentionSpec), [`max_memory_usage_bytes`](interface.md#ChunkedLocalAttentionSpec.max_memory_usage_bytes), [`max_memory_usage_bytes`](interface.md#SlidingWindowSpec.max_memory_usage_bytes), [`SlidingWindowSpec`](interface.md#SlidingWindowSpec), [`max_memory_usage_bytes`](interface.md#FullAttentionSpec.max_memory_usage_bytes), [`type_id`](interface.md#ChunkedLocalAttentionSpec.type_id), [`type_id`](interface.md#SlidingWindowSpec.type_id), [`type_id`](interface.md#FullAttentionSpec.type_id), [`page_size_bytes`](interface.md#CacheSpec.page_size_bytes), [`__post_init__`](interface.md#SlidingWindowSpec.__post_init__)

### `CacheGroupSpec`
- def: [`easydel/inference/esurge/core/interface.py:473`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L473)
- doc: Specification for a group of layers sharing a KV-cache page table.
- signature: `class CacheGroupSpec:`
- members:
  - `kv_cache_spec` — [`L491`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L491)
  - `layer_names` — [`L493`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L493)
- uses (calls/refs, reference-scoped): [`CacheSpec`](interface.md#CacheSpec)
- used by: [`create_kv_cache_specs_from_config`](interface.md#create_kv_cache_specs_from_config), [`estimate_runtime_page_budget`](interface.md#estimate_runtime_page_budget), [`other_group_ids`](coordinator.md#HybridCacheCoordinator.other_group_ids), [`single_type_managers`](coordinator.md#CacheCoordinator.single_type_managers), [`get_kv_cache_coordinator`](coordinator.md#get_kv_cache_coordinator), [`page_size`](manager.md#CacheManager.page_size), [`full_attention_spec`](coordinator.md#HybridCacheCoordinator.full_attention_spec), [`other_spec`](coordinator.md#HybridCacheCoordinator.other_spec), [`_get_full_attention_page_table_index`](../runners/model_runner.md#eSurgeRunner._get_full_attention_page_table_index), [`__init__`](coordinator.md#HybridCacheCoordinator.__init__), [`kv_cache_spec`](coordinator.md#UnitaryCacheCoordinator.kv_cache_spec), [`__init__`](coordinator.md#CacheCoordinator.__init__), [`kv_cache_groups`](interface.md#CacheGroupsConfig.kv_cache_groups), [`__init__`](coordinator.md#CacheCoordinatorNoPrefixCache.__init__), [`__init__`](coordinator.md#UnitaryCacheCoordinator.__init__), [`__init__`](manager.md#CacheManager.__init__)

### `CacheGroupsConfig`
- def: [`easydel/inference/esurge/core/interface.py:497`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L497)
- doc: Complete KV-cache configuration for a model.
- signature: `class CacheGroupsConfig:`
- members:
  - `kv_cache_groups` — [`L515`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L515)
  - `num_pages` — [`L514`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L514)
- uses (calls/refs, reference-scoped): [`CacheGroupSpec`](interface.md#CacheGroupSpec)
- used by: [`schedule`](../scheduler/scheduler.md#Scheduler.schedule), [`from_runner`](../scheduler/scheduler.md#Scheduler.from_runner), [`kv_cache_manager`](../scheduler/scheduler.md#Scheduler.kv_cache_manager), [`__init__`](../scheduler/scheduler.md#Scheduler.__init__)

### `CacheSpec`
- def: [`easydel/inference/esurge/core/interface.py:62`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L62)
- doc: Base class for specifying KV-cache format for a layer type.
- signature: `class CacheSpec:`
- members:
  - `max_memory_usage_bytes(self, *args, **kwargs)` — [`L114`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L114) — Calculate maximum possible memory usage for this cache.
  - `merge(cls, specs: list[Self])` — [`L126`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L126) — Merge multiple cache specifications into one.
  - `page_size_bytes(self)` — [`L103`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L103) — Calculate the memory size of one cache page in bytes.
  - `type_id(self)` — [`L81`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L81) — Get the unique type identifier for this cache specification.
  - `page_size` — [`L78`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L78)
- uses (calls/refs, reference-scoped): [`page_size_bytes`](interface.md#AttentionSpec.page_size_bytes), [`AttentionSpec`](interface.md#AttentionSpec), [`page_size_bytes`](interface.md#MambaSpec.page_size_bytes), [`max_memory_usage_bytes`](interface.md#ChunkedLocalAttentionSpec.max_memory_usage_bytes), [`max_memory_usage_bytes`](interface.md#SlidingWindowSpec.max_memory_usage_bytes), [`merge`](interface.md#FullAttentionSpec.merge), [`MambaSpec`](interface.md#MambaSpec), [`max_memory_usage_bytes`](interface.md#FullAttentionSpec.max_memory_usage_bytes), [`type_id`](interface.md#ChunkedLocalAttentionSpec.type_id), [`type_id`](interface.md#SlidingWindowSpec.type_id), [`type_id`](interface.md#FullAttentionSpec.type_id), [`type_id`](interface.md#MambaSpec.type_id), [`max_memory_usage_bytes`](interface.md#MambaSpec.max_memory_usage_bytes)
- used by: [`create_kv_cache_specs_from_config`](interface.md#create_kv_cache_specs_from_config), [`spec_manager_map`](single_type_cache_manager.md#spec_manager_map.spec_manager_map), [`find_longest_cache_hit`](single_type_cache_manager.md#SlidingWindowManager.find_longest_cache_hit), [`other_group_ids`](coordinator.md#HybridCacheCoordinator.other_group_ids), [`find_longest_cache_hit`](single_type_cache_manager.md#ChunkedLocalAttentionManager.find_longest_cache_hit), [`find_longest_cache_hit`](single_type_cache_manager.md#SingleTypeCacheManager.find_longest_cache_hit), [`find_longest_cache_hit`](single_type_cache_manager.md#FullAttentionManager.find_longest_cache_hit), [`page_size_bytes`](interface.md#AttentionSpec.page_size_bytes), [`kv_cache_spec`](interface.md#CacheGroupSpec.kv_cache_spec), [`other_page_size`](coordinator.md#HybridCacheCoordinator.other_page_size), [`AttentionSpec`](interface.md#AttentionSpec), [`page_size`](manager.md#CacheManager.page_size), [`find_longest_cache_hit`](single_type_cache_manager.md#MambaManager.find_longest_cache_hit), [`full_attention_page_size`](coordinator.md#HybridCacheCoordinator.full_attention_page_size), [`max_memory_usage_bytes`](interface.md#ChunkedLocalAttentionSpec.max_memory_usage_bytes), [`max_memory_usage_bytes`](interface.md#SlidingWindowSpec.max_memory_usage_bytes), [`merge`](interface.md#FullAttentionSpec.merge), [`get_manager_for_kv_cache_spec`](single_type_cache_manager.md#get_manager_for_kv_cache_spec), [`_estimate_group_pages_per_request`](interface.md#_estimate_group_pages_per_request), [`page_size`](coordinator.md#UnitaryCacheCoordinator.page_size), [`page_size`](single_type_cache_manager.md#SingleTypeCacheManager.page_size), [`MambaSpec`](interface.md#MambaSpec), [`max_memory_usage_bytes`](interface.md#FullAttentionSpec.max_memory_usage_bytes), [`type_id`](interface.md#ChunkedLocalAttentionSpec.type_id), [`type_id`](interface.md#SlidingWindowSpec.type_id), [`__init__`](single_type_cache_manager.md#SingleTypeCacheManager.__init__), [`type_id`](interface.md#FullAttentionSpec.type_id)

### `ChunkedLocalAttentionSpec`  ·  implements/extends AttentionSpec
- def: [`easydel/inference/esurge/core/interface.py:290`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L290)
- doc: Cache specification for chunked local attention layers.
- signature: `class ChunkedLocalAttentionSpec(AttentionSpec):`
- members:
  - `max_memory_usage_bytes(self, max_model_len: int, max_num_batched_tokens: int, **kwargs)` — [`L322`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L322) — Calculate maximum memory for chunked local attention.
  - `type_id(self)` — [`L314`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L314) — Get the type identifier for chunked local attention.
  - `attention_chunk_size` — [`L311`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L311)
- uses (calls/refs, reference-scoped): [`page_size`](interface.md#CacheSpec.page_size), [`page_size_bytes`](interface.md#AttentionSpec.page_size_bytes), [`AttentionSpec`](interface.md#AttentionSpec), [`cdiv`](../utils.md#cdiv)
- used by: [`create_kv_cache_specs_from_config`](interface.md#create_kv_cache_specs_from_config), [`spec_manager_map`](single_type_cache_manager.md#spec_manager_map.spec_manager_map), [`find_longest_cache_hit`](single_type_cache_manager.md#ChunkedLocalAttentionManager.find_longest_cache_hit), [`find_longest_cache_hit`](single_type_cache_manager.md#FullAttentionManager.find_longest_cache_hit), [`AttentionSpec`](interface.md#AttentionSpec), [`type_id`](interface.md#CacheSpec.type_id), [`max_memory_usage_bytes`](interface.md#CacheSpec.max_memory_usage_bytes), [`__init__`](single_type_cache_manager.md#ChunkedLocalAttentionManager.__init__), [`attention_chunk_size`](single_type_cache_manager.md#ChunkedLocalAttentionManager.attention_chunk_size)

### `FullAttentionSpec`  ·  implements/extends AttentionSpec
- def: [`easydel/inference/esurge/core/interface.py:184`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L184)
- doc: Cache specification for full (causal) attention layers.
- signature: `class FullAttentionSpec(AttentionSpec):`
- members:
  - `max_memory_usage_bytes(self, max_model_len: int, **kwargs)` — [`L226`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L226) — Calculate maximum memory for full attention KV-cache.
  - `merge(cls, specs: list[Self])` — [`L262`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L262) — Merge multiple FullAttentionSpec objects into one.
  - `merge_window_sizes(cls, window_sizes: set[int])` — [`L242`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L242) — Merge window sizes from multiple specifications.
  - `type_id(self)` — [`L218`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L218) — Get the type identifier for full attention.
  - `attention_chunk_size` — [`L215`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L215)
  - `sliding_window` — [`L214`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L214)
- uses (calls/refs, reference-scoped): [`page_size`](interface.md#CacheSpec.page_size), [`page_size_bytes`](interface.md#AttentionSpec.page_size_bytes), [`AttentionSpec`](interface.md#AttentionSpec), [`cdiv`](../utils.md#cdiv), [`merge`](interface.md#CacheSpec.merge)
- used by: [`create_kv_cache_specs_from_config`](interface.md#create_kv_cache_specs_from_config), [`spec_manager_map`](single_type_cache_manager.md#spec_manager_map.spec_manager_map), [`other_group_ids`](coordinator.md#HybridCacheCoordinator.other_group_ids), [`find_longest_cache_hit`](single_type_cache_manager.md#FullAttentionManager.find_longest_cache_hit), [`AttentionSpec`](interface.md#AttentionSpec), [`type_id`](interface.md#CacheSpec.type_id), [`max_memory_usage_bytes`](interface.md#CacheSpec.max_memory_usage_bytes), [`_get_full_attention_page_table_index`](../runners/model_runner.md#eSurgeRunner._get_full_attention_page_table_index), [`merge`](interface.md#CacheSpec.merge)

### `MambaSpec`  ·  implements/extends CacheSpec
- def: [`easydel/inference/esurge/core/interface.py:405`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L405)
- doc: Cache specification for Mamba state-space model layers.
- signature: `class MambaSpec(CacheSpec):`
- members:
  - `__post_init__(self)` — [`L429`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L429) — Calculate total number of elements across all state shapes.
  - `max_memory_usage_bytes(self, **kwargs)` — [`L458`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L458) — Calculate maximum memory for Mamba state (constant).
  - `page_size_bytes(self)` — [`L443`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L443) — Calculate page size in bytes for Mamba state.
  - `type_id(self)` — [`L434`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L434) — Get the type identifier for Mamba state caching.
  - `dtype` — [`L426`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L426)
  - `num_elements` — [`L431`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L431)
  - `page_size_padded` — [`L427`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L427)
  - `shapes` — [`L425`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L425)
- uses (calls/refs, reference-scoped): [`CacheSpec`](interface.md#CacheSpec), [`get_dtype_size`](../utils.md#get_dtype_size)
- used by: [`spec_manager_map`](single_type_cache_manager.md#spec_manager_map.spec_manager_map), [`CacheSpec`](interface.md#CacheSpec), [`type_id`](interface.md#CacheSpec.type_id), [`find_longest_cache_hit`](single_type_cache_manager.md#MambaManager.find_longest_cache_hit), [`max_memory_usage_bytes`](interface.md#CacheSpec.max_memory_usage_bytes), [`page_size_bytes`](interface.md#CacheSpec.page_size_bytes)

### `RuntimePageBudgetEstimate`
- def: [`easydel/inference/esurge/core/interface.py:519`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L519)
- doc: Window-aware runtime page-budget estimate for eSurge.
- signature: `class RuntimePageBudgetEstimate:`
- members:
  - `max_num_seqs` — [`L532`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L532)
  - `pages_per_request` — [`L531`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L531)
  - `per_group_pages` — [`L534`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L534)
  - `usable_pages` — [`L533`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L533)
- used by: [`_apply_window_aware_runtime_cap`](../runners/model_runner.md#eSurgeRunner._apply_window_aware_runtime_cap), [`estimate_runtime_page_budget`](interface.md#estimate_runtime_page_budget)

### `SlidingWindowSpec`  ·  implements/extends AttentionSpec
- def: [`easydel/inference/esurge/core/interface.py:342`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L342)
- doc: Cache specification for sliding window attention layers.
- signature: `class SlidingWindowSpec(AttentionSpec):`
- members:
  - `__post_init__(self)` — [`L368`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L368) — Validate sliding window configuration.
  - `max_memory_usage_bytes(self, max_model_len: int, max_num_batched_tokens: int, **kwargs)` — [`L385`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L385) — Calculate maximum memory for sliding window attention.
  - `type_id(self)` — [`L377`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L377) — Get the type identifier for sliding window attention.
  - `sliding_window` — [`L366`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L366)
- uses (calls/refs, reference-scoped): [`page_size`](interface.md#CacheSpec.page_size), [`page_size_bytes`](interface.md#AttentionSpec.page_size_bytes), [`AttentionSpec`](interface.md#AttentionSpec), [`cdiv`](../utils.md#cdiv), [`use_mla`](interface.md#AttentionSpec.use_mla)
- used by: [`create_kv_cache_specs_from_config`](interface.md#create_kv_cache_specs_from_config), [`spec_manager_map`](single_type_cache_manager.md#spec_manager_map.spec_manager_map), [`find_longest_cache_hit`](single_type_cache_manager.md#SlidingWindowManager.find_longest_cache_hit), [`AttentionSpec`](interface.md#AttentionSpec), [`type_id`](interface.md#CacheSpec.type_id), [`max_memory_usage_bytes`](interface.md#CacheSpec.max_memory_usage_bytes), [`__init__`](single_type_cache_manager.md#SlidingWindowManager.__init__), [`sliding_window`](single_type_cache_manager.md#SlidingWindowManager.sliding_window)

## Functions
- `_estimate_group_pages_per_request(kv_cache_spec: CacheSpec, *, max_model_len: int, max_num_batched_tokens: int)` — [`L537`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L537) — Estimate logical page IDs needed for one request in a cache group.
- `_resolve_layer_attention_geometry(config: EasyDeLBaseConfig, layer_idx: int, default_num_kv_heads: int, default_head_size: int)` — [`L691`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L691) — Resolve per-layer KV geometry for hybrid attention models like Gemma4.
- `_resolve_positive_layer_config_value(config: EasyDeLBaseConfig, attribute_names: tuple[str, ...], *, layer_idx: int | None = None)` — [`L646`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L646) — Resolve a positive scalar or per-layer integer config value.
- `create_kv_cache_specs_from_config(config: EasyDeLBaseConfig, page_size: int, num_kv_heads: int, head_size: int, dtype: jnp.dtype, use_mla: bool = False)` — [`L742`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L742) — Convert model config's get_mask_details() to CacheGroupSpec list.
- `estimate_runtime_page_budget(*, num_pages: int, kv_cache_groups: list[CacheGroupSpec], max_model_len: int, max_num_batched_tokens: int, data_parallel_size: int = 1)` — [`L571`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/interface.py#L571) — Estimate window-aware max-length concurrency from KV-cache groups.


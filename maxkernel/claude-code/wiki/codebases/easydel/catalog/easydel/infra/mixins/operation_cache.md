---
title: 'Module: easydel/infra/mixins/operation_cache.py'
type: catalog
provenance: extracted
module: easydel/infra/mixins/operation_cache.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.infra.mixins.operation_cache`/
symbols:
  OperationCacheMixin._get_operations_cache_info_from_config: OperationCacheMixin#_get_operations_cache_info_from_config().
  OperationCacheMixin.get_operations_cache_info_dynamic: OperationCacheMixin#get_operations_cache_info_dynamic().
  OperationCacheMixin._build_cache_info_from_layers: OperationCacheMixin#_build_cache_info_from_layers().
  OperationCacheMixin.get_operations_cache_view: OperationCacheMixin#get_operations_cache_view().
  OperationCacheMixin.get_required_cache_class: OperationCacheMixin#get_required_cache_class().
  OperationCacheMixin.get_operations_cache_info_by_slot: OperationCacheMixin#get_operations_cache_info_by_slot().
  OperationCacheMixin.get_operations_cache_info: OperationCacheMixin#get_operations_cache_info().
  OperationsCacheInfo.layers: OperationsCacheInfo#layers.
  OperationCacheMixin._get_operation_requirements: OperationCacheMixin#_get_operation_requirements().
  LayerOperationInfo.is_recurrent_layer: LayerOperationInfo#is_recurrent_layer().
  LayerOperationInfo: LayerOperationInfo#
  LayerOperationInfo.requirements: LayerOperationInfo#requirements.
  LayerOperationInfo.supported_cache_types: LayerOperationInfo#supported_cache_types.
  OperationCacheMixin.get_unique_cache_view_classes: OperationCacheMixin#get_unique_cache_view_classes().
  LayerOperationInfo.layer_index: LayerOperationInfo#layer_index.
  LayerOperationInfo.is_attention_layer: LayerOperationInfo#is_attention_layer().
  OperationsCacheInfo.get_recommended_cache_type: OperationsCacheInfo#get_recommended_cache_type().
  OperationCacheMixin.get_operations_cache_view_by_slot: OperationCacheMixin#get_operations_cache_view_by_slot().
  LayerOperationInfo.operation_name: LayerOperationInfo#operation_name.
  OperationsCacheInfo: OperationsCacheInfo#
  OperationsCacheInfo.get_layer_by_index: OperationsCacheInfo#get_layer_by_index().
  OperationsCacheInfo.get_layers_by_index: OperationsCacheInfo#get_layers_by_index().
  OperationCacheMixin.get_layer_cache_requirements: OperationCacheMixin#get_layer_cache_requirements().
  LayerOperationInfo.slot: LayerOperationInfo#slot.
  LayerOperationInfo.decode_operation_name: LayerOperationInfo#decode_operation_name.
  OperationsCacheInfo.combined_cache_types: OperationsCacheInfo#combined_cache_types.
  OperationsCacheInfo.combined_metadata: OperationsCacheInfo#combined_metadata.
  OperationsCacheInfo.num_attention_layers: OperationsCacheInfo#num_attention_layers().
  OperationCacheMixin: OperationCacheMixin#
  LayerOperationInfo.has_separate_decode: LayerOperationInfo#has_separate_decode().
  OperationsCacheInfo.num_recurrent_layers: OperationsCacheInfo#num_recurrent_layers().
  OperationsCacheInfo.attention_ratio: OperationsCacheInfo#attention_ratio().
  LayerOperationInfo.layer_type: LayerOperationInfo#layer_type.
  LayerOperationInfo.requires_kv_cache: LayerOperationInfo#requires_kv_cache.
  LayerOperationInfo.requires_state_cache: LayerOperationInfo#requires_state_cache.
  OperationsCacheInfo.supports_ragged_pages: OperationsCacheInfo#supports_ragged_pages.
  OperationsCacheInfo.supports_transformer_cache: OperationsCacheInfo#supports_transformer_cache.
  OperationsCacheInfo.requires_hybrid_cache: OperationsCacheInfo#requires_hybrid_cache.
  OperationsCacheInfo.requires_state_management: OperationsCacheInfo#requires_state_management.
  OperationsCacheInfo.requires_ragged_pages: OperationsCacheInfo#requires_ragged_pages().
  OperationsCacheInfo.requires_transformer_cache: OperationsCacheInfo#requires_transformer_cache().
  OperationsCacheInfo.prefill_operation: OperationsCacheInfo#prefill_operation.
  OperationsCacheInfo.decode_operation: OperationsCacheInfo#decode_operation.
  OperationsCacheInfo.is_hybrid_model: OperationsCacheInfo#is_hybrid_model.
  OperationsCacheInfo.has_separate_decode_ops: OperationsCacheInfo#has_separate_decode_ops.
  OperationCacheMixin._extract_layer_index_from_path: OperationCacheMixin#_extract_layer_index_from_path().
  OperationCacheMixin._infer_layer_type_from_path: OperationCacheMixin#_infer_layer_type_from_path().
  __all__: __all__.
---
# Module: [`easydel/infra/mixins/operation_cache.py`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/operation_cache.py)

## Classes
### `LayerOperationInfo`
- def: [`easydel/infra/mixins/operation_cache.py:58`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/operation_cache.py#L58)
- doc: Information about a single layer's operation and cache requirements.
- signature: `class LayerOperationInfo:`
- members:
  - `has_separate_decode(self)` — [`L151`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/operation_cache.py#L151) — Whether this layer uses different operations for prefill and decode phases.
  - `is_attention_layer(self)` — [`L105`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/operation_cache.py#L105) — Check if this is an attention-based layer.
  - `is_recurrent_layer(self)` — [`L126`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/operation_cache.py#L126) — Check if this is a recurrent/linear attention layer.
  - `decode_operation_name` — [`L100`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/operation_cache.py#L100)
  - `layer_index` — [`L94`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/operation_cache.py#L94)
  - `layer_type` — [`L96`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/operation_cache.py#L96)
  - `operation_name` — [`L97`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/operation_cache.py#L97)
  - `requirements` — [`L98`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/operation_cache.py#L98)
  - `requires_kv_cache` — [`L101`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/operation_cache.py#L101)
  - `requires_state_cache` — [`L102`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/operation_cache.py#L102)
  - `slot` — [`L95`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/operation_cache.py#L95)
  - `supported_cache_types` — [`L99`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/operation_cache.py#L99)
- uses (calls/refs, reference-scoped): [`CacheType`](../../operations/requirements/types.md#CacheType), [`OperationRequirements`](../../operations/requirements/requirements.md#OperationRequirements), [`TRANSFORMER`](../../operations/requirements/types.md#CacheType.TRANSFORMER), [`RAGGED_PAGES`](../../operations/requirements/types.md#CacheType.RAGGED_PAGES), [`RECURRENT`](../../operations/requirements/types.md#CacheType.RECURRENT)
- used by: [`_get_operations_cache_info_from_config`](operation_cache.md#OperationCacheMixin._get_operations_cache_info_from_config), [`get_operations_cache_info_dynamic`](operation_cache.md#OperationCacheMixin.get_operations_cache_info_dynamic), [`_build_cache_info_from_layers`](operation_cache.md#OperationCacheMixin._build_cache_info_from_layers), [`_log_startup_summary`](../../inference/esurge/runners/model_runner.md#eSurgeRunner._log_startup_summary), [`get_operations_cache_view`](operation_cache.md#OperationCacheMixin.get_operations_cache_view), [`get_required_cache_class`](operation_cache.md#OperationCacheMixin.get_required_cache_class), [`get_operations_cache_info_by_slot`](operation_cache.md#OperationCacheMixin.get_operations_cache_info_by_slot), [`layers`](operation_cache.md#OperationsCacheInfo.layers), [`get_unique_cache_view_classes`](operation_cache.md#OperationCacheMixin.get_unique_cache_view_classes), [`get_operations_cache_view_by_slot`](operation_cache.md#OperationCacheMixin.get_operations_cache_view_by_slot), [`get_layer_by_index`](operation_cache.md#OperationsCacheInfo.get_layer_by_index), [`get_layer_cache_requirements`](operation_cache.md#OperationCacheMixin.get_layer_cache_requirements), [`get_layers_by_index`](operation_cache.md#OperationsCacheInfo.get_layers_by_index), [`num_attention_layers`](operation_cache.md#OperationsCacheInfo.num_attention_layers), [`num_recurrent_layers`](operation_cache.md#OperationsCacheInfo.num_recurrent_layers)

### `OperationCacheMixin`
- def: [`easydel/infra/mixins/operation_cache.py:367`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/operation_cache.py#L367)
- doc: Mixin that provides operation cache requirements discovery.
- signature: `class OperationCacheMixin:`
- members:
  - `_build_cache_info_from_layers(self, layers: list[LayerOperationInfo], has_separate_decode: bool = False)` — [`L892`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/operation_cache.py#L892) — Build OperationsCacheInfo from discovered layers.
  - `_extract_layer_index_from_path(self, path: tuple)` — [`L818`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/operation_cache.py#L818) — Extract layer index from module path.
  - `_get_operation_requirements(self, name: str | None, mode: ExecutionMode = ExecutionMode.MIXED)` — [`L399`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/operation_cache.py#L399) — Get requirements for an operation by name from the registry.
  - `_get_operations_cache_info_from_config(self, mode: ExecutionMode = ExecutionMode.MIXED)` — [`L474`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/operation_cache.py#L474) — Get cache info by reading from model configuration (static fallback).
  - `_infer_layer_type_from_path(self, path: tuple)` — [`L861`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/operation_cache.py#L861) — Infer layer type from the module path.
  - `get_layer_cache_requirements(self, layer_index: int)` — [`L635`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/operation_cache.py#L635) — Get cache requirements for a specific layer.
  - `get_operations_cache_info(self, mode: ExecutionMode = ExecutionMode.MIXED, dynamic: bool = True)` — [`L435`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/operation_cache.py#L435) — Get complete information about operations and their cache requirements.
  - `get_operations_cache_info_by_slot(self)` — [`L1082`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/operation_cache.py#L1082) — Get per-layer cache requirements grouped by slot.
  - `get_operations_cache_info_dynamic(self, mode: ExecutionMode = ExecutionMode.MIXED)` — [`L656`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/operation_cache.py#L656) — Discover operations dynamically from the module graph.
  - `get_operations_cache_view(self)` — [`L1006`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/operation_cache.py#L1006) — Get the cache view class required for each layer.
  - `get_operations_cache_view_by_slot(self)` — [`L1115`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/operation_cache.py#L1115) — Get the cache view classes per layer, grouped by slot.
  - `get_required_cache_class(self)` — [`L955`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/operation_cache.py#L955) — Get the required cache class based on operation requirements.
  - `get_unique_cache_view_classes(self)` — [`L1150`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/operation_cache.py#L1150) — Get all unique cache view classes used by this model.
- uses (calls/refs, reference-scoped): [`EasyDeLBaseModule`](../base_module.md#EasyDeLBaseModule), [`TransformerCache`](../../caching/transformer/cache.md#TransformerCache), [`HybridCache`](../../caching/hybrid/cache.md#HybridCache), [`RaggedPagesCacheView`](../../caching/ragged_page/cache.md#RaggedPagesCacheView), [`TransformerCacheView`](../../caching/transformer/cache.md#TransformerCacheView), [`MetadataField`](../../operations/requirements/types.md#MetadataField), [`CacheType`](../../operations/requirements/types.md#CacheType), [`ExecutionMode`](../../operations/requirements/types.md#ExecutionMode), [`OperationRequirements`](../../operations/requirements/requirements.md#OperationRequirements), [`FlexibleAttentionModule`](../../layers/attention/_flexible.md#FlexibleAttentionModule), [`get_requirements`](../../operations/_base_operation.md#BaseOperation.get_requirements), [`RecurrentCacheView`](../../caching/recurrent/cache.md#RecurrentCacheView), [`RecurrentCache`](../../caching/recurrent/cache.md#RecurrentCache), [`UnifiedAttentionCacheView`](../../caching/unified_attention/cache.md#UnifiedAttentionCacheView), [`cache`](../../operations/requirements/requirements.md#OperationRequirements.cache), [`get_impl_name`](../../operations/_base_operation.md#BaseOperation.get_impl_name), [`iter_module_search`](../../utils/traversals.md#iter_module_search), [`supported`](../../operations/requirements/requirements.md#CacheRequirements.supported), [`MLARaggedPagesCacheView`](../../caching/mla_ragged_page/cache.md#MLARaggedPagesCacheView), [`ParallelHybridCacheView`](../../caching/hybrid/cache.md#ParallelHybridCacheView), [`MIXED`](../../operations/requirements/types.md#ExecutionMode.MIXED), [`default`](../../operations/requirements/requirements.md#OperationRequirements.default), [`BaseOperation`](../../operations/_base_operation.md#BaseOperation), [`get_combined_requirements`](../../operations/executor.md#OperationExecutor.get_combined_requirements), [`any`](../../operations/requirements/types.md#CacheType.any), [`required`](../../operations/requirements/requirements.md#MetadataRequirements.required), [`operation_executor`](../../layers/attention/_flexible.md#FlexibleAttentionModule.operation_executor), [`TRANSFORMER`](../../operations/requirements/types.md#CacheType.TRANSFORMER), [`RAGGED_PAGES`](../../operations/requirements/types.md#CacheType.RAGGED_PAGES), [`cache_view_class`](../../operations/requirements/requirements.md#CacheRequirements.cache_view_class), [`get_operation`](../../operations/executor.md#OperationExecutor.get_operation), [`layers`](operation_cache.md#OperationsCacheInfo.layers), [`is_recurrent_layer`](operation_cache.md#LayerOperationInfo.is_recurrent_layer), [`metadata`](../../operations/requirements/requirements.md#OperationRequirements.metadata), [`LayerOperationInfo`](operation_cache.md#LayerOperationInfo), [`RECURRENT`](../../operations/requirements/types.md#CacheType.RECURRENT), [`requirements`](operation_cache.md#LayerOperationInfo.requirements), [`decode_operation`](../../operations/executor.md#OperationExecutor.decode_operation), [`requires_cache`](../../operations/requirements/requirements.md#CacheRequirements.requires_cache), [`supported_cache_types`](operation_cache.md#LayerOperationInfo.supported_cache_types)  (+26 more)
- used by: [`EasyDeLBaseModule`](../base_module.md#EasyDeLBaseModule), [`_log_startup_summary`](../../inference/esurge/runners/model_runner.md#eSurgeRunner._log_startup_summary)

### `OperationsCacheInfo`
- def: [`easydel/infra/mixins/operation_cache.py:175`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/operation_cache.py#L175)
- doc: Complete information about all operations' cache requirements.
- signature: `class OperationsCacheInfo:`
- members:
  - `attention_ratio(self)` — [`L269`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/operation_cache.py#L269) — Ratio of attention layers to total layers.
  - `get_layer_by_index(self, index: int)` — [`L320`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/operation_cache.py#L320) — Get layer info by index.
  - `get_layers_by_index(self, index: int)` — [`L344`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/operation_cache.py#L344) — Get all layer infos for an index (slot-aware).
  - `get_recommended_cache_type(self)` — [`L287`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/operation_cache.py#L287) — Get recommended cache type based on model requirements.
  - `num_attention_layers(self)` — [`L241`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/operation_cache.py#L241) — Count of attention-based layers in the model.
  - `num_recurrent_layers(self)` — [`L255`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/operation_cache.py#L255) — Count of recurrent/linear attention layers in the model.
  - `requires_ragged_pages(self)` — [`L217`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/operation_cache.py#L217) — Deprecated: Use supports_ragged_pages instead.
  - `requires_transformer_cache(self)` — [`L229`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/operation_cache.py#L229) — Deprecated: Use supports_transformer_cache instead.
  - `combined_cache_types` — [`L207`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/operation_cache.py#L207)
  - `combined_metadata` — [`L208`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/operation_cache.py#L208)
  - `decode_operation` — [`L206`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/operation_cache.py#L206)
  - `has_separate_decode_ops` — [`L214`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/operation_cache.py#L214)
  - `is_hybrid_model` — [`L209`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/operation_cache.py#L209)
  - `layers` — [`L204`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/operation_cache.py#L204)
  - `prefill_operation` — [`L205`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/operation_cache.py#L205)
  - `requires_hybrid_cache` — [`L212`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/operation_cache.py#L212)
  - `requires_state_management` — [`L213`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/operation_cache.py#L213)
  - `supports_ragged_pages` — [`L210`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/operation_cache.py#L210)
  - `supports_transformer_cache` — [`L211`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/operation_cache.py#L211)
- uses (calls/refs, reference-scoped): [`MetadataField`](../../operations/requirements/types.md#MetadataField), [`CacheType`](../../operations/requirements/types.md#CacheType), [`any`](../../operations/requirements/types.md#CacheType.any), [`basic`](../../operations/requirements/types.md#MetadataField.basic), [`is_recurrent_layer`](operation_cache.md#LayerOperationInfo.is_recurrent_layer), [`LayerOperationInfo`](operation_cache.md#LayerOperationInfo), [`is_attention_layer`](operation_cache.md#LayerOperationInfo.is_attention_layer), [`layer_index`](operation_cache.md#LayerOperationInfo.layer_index)
- used by: [`_get_operations_cache_info_from_config`](operation_cache.md#OperationCacheMixin._get_operations_cache_info_from_config), [`get_operations_cache_info_dynamic`](operation_cache.md#OperationCacheMixin.get_operations_cache_info_dynamic), [`_build_cache_info_from_layers`](operation_cache.md#OperationCacheMixin._build_cache_info_from_layers), [`_log_startup_summary`](../../inference/esurge/runners/model_runner.md#eSurgeRunner._log_startup_summary), [`get_operations_cache_view`](operation_cache.md#OperationCacheMixin.get_operations_cache_view), [`get_required_cache_class`](operation_cache.md#OperationCacheMixin.get_required_cache_class), [`get_operations_cache_info_by_slot`](operation_cache.md#OperationCacheMixin.get_operations_cache_info_by_slot), [`get_operations_cache_info`](operation_cache.md#OperationCacheMixin.get_operations_cache_info), [`get_unique_cache_view_classes`](operation_cache.md#OperationCacheMixin.get_unique_cache_view_classes), [`get_layer_cache_requirements`](operation_cache.md#OperationCacheMixin.get_layer_cache_requirements)

## Module values
- `__all__` — [`L50`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/operation_cache.py#L50)


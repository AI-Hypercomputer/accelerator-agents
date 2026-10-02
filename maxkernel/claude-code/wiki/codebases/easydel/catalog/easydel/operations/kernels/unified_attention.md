---
title: 'Module: easydel/operations/kernels/unified_attention.py'
type: catalog
provenance: extracted
module: easydel/operations/kernels/unified_attention.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.operations.kernels.unified_attention`/
symbols:
  UnifiedAttn.forward_native: UnifiedAttn#forward_native().
  UnifiedAttn.get_requirements: UnifiedAttn#get_requirements().
  UnifiedAttn._mapped: UnifiedAttn#_mapped().
  _dp_page_axis: _dp_page_axis().
  UnifiedAttn: UnifiedAttn#
  ENABLE_DP_LOCAL_PAGE_PATH: ENABLE_DP_LOCAL_PAGE_PATH.
  _normalize_axis_names: _normalize_axis_names().
  _mesh_axis_size: _mesh_axis_size().
  _axis_index: _axis_index().
  UnifiedAttn.get_impl_name: UnifiedAttn#get_impl_name().
---
# Module: [`easydel/operations/kernels/unified_attention.py`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/unified_attention.py)

## Classes
### `UnifiedAttn`  ·  implements/extends OperationImpl
- def: [`easydel/operations/kernels/unified_attention.py:85`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/unified_attention.py#L85)
- doc: Attention implementation using vLLM-style Unified (Paged) Attention mechanism with Triton kernels.
- signature: `class UnifiedAttn(OperationImpl):`
- members:
  - `forward_native(self, query: Float[Array, "... num_q_heads head_dim"], cache_view: UnifiedAttentionCacheView | None, cache_metadata: RaggedPagesMetadata | OperationsMetadata, softmax_scale: float | None = None, causal: bool = True, sliding_window: int | None = None, logits_soft_cap: float | None = None, softmax_aux: Float[Array, "num_kv_heads num_sinks"] | Float[Array, num_sinks] | None = None, **ignore)` — [`L150`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/unified_attention.py#L150) — Native forward pass for unified paged attention using Triton kernels.
  - `get_impl_name(cls)` — [`L106`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/unified_attention.py#L106) — Returns the registered name for this attention implementation.
  - `get_requirements(cls, mode: ExecutionMode = ExecutionMode.MIXED)` — [`L116`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/unified_attention.py#L116) — Returns the operation requirements for UnifiedAttention.
- protocol/private: `_mapped`[`L286`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/unified_attention.py#L286)
- uses (calls/refs, reference-scoped): [`OperationsMetadata`](../../caching/_abstracts.md#OperationsMetadata), [`RaggedPagesMetadata`](../../caching/ragged_page/cache.md#RaggedPagesMetadata), [`AttentionOutput`](../_attention_outputs.md#AttentionOutput), [`MetadataField`](../requirements/types.md#MetadataField), [`CacheType`](../requirements/types.md#CacheType), [`metadata`](../_base_operation.md#BaseOperation.metadata), [`ExecutionMode`](../requirements/types.md#ExecutionMode), [`OperationRequirements`](../requirements/requirements.md#OperationRequirements), [`OperationImpl`](../_operation_impl.md#OperationImpl), [`build`](../requirements/builder.md#RequirementsBuilder.build), [`UnifiedAttentionCacheView`](../../caching/unified_attention/cache.md#UnifiedAttentionCacheView), [`mesh`](../_operation_meta.md#OperationMetadata.mesh), [`RequirementsBuilder`](../requirements/builder.md#RequirementsBuilder), [`unwrap_metadata`](../../caching/_abstracts.md#unwrap_metadata), [`register`](../_base_operation.md#OperationRegistry.register), [`MIXED`](../requirements/types.md#ExecutionMode.MIXED), [`ATTN_DP`](../../axis.md#ATTN_DP), [`require_metadata`](../requirements/builder.md#RequirementsBuilder.require_metadata), [`support_cache`](../requirements/builder.md#RequirementsBuilder.support_cache), [`context_lens`](../../caching/ragged_page/cache.md#RaggedPagesMetadata.context_lens), [`optional_metadata`](../requirements/builder.md#RequirementsBuilder.optional_metadata), [`RAGGED_PAGES`](../requirements/types.md#CacheType.RAGGED_PAGES), [`get_operation_config`](../_operation_meta.md#OperationMetadata.get_operation_config), [`pages_tables`](../../caching/ragged_page/cache.md#RaggedPagesMetadata.pages_tables), [`use_cache_view`](../requirements/builder.md#RequirementsBuilder.use_cache_view), [`runtime_dtype`](../_operation_meta.md#OperationMetadata.runtime_dtype), [`POSITIONS`](../requirements/types.md#MetadataField.POSITIONS), [`SEQ_LENS`](../requirements/types.md#MetadataField.SEQ_LENS), [`query_start_loc`](../../caching/ragged_page/cache.md#RaggedPagesMetadata.query_start_loc), [`partition_manager`](../_operation_meta.md#OperationMetadata.partition_manager), [`LOGITS_INDICES`](../requirements/types.md#MetadataField.LOGITS_INDICES), [`key_cache`](../../caching/unified_attention/cache.md#UnifiedAttentionCacheView.key_cache), [`_dp_page_axis`](unified_attention.md#_dp_page_axis), [`CONTEXT_LENS`](../requirements/types.md#MetadataField.CONTEXT_LENS), [`PAGES_TABLES`](../requirements/types.md#MetadataField.PAGES_TABLES), [`value_cache`](../../caching/unified_attention/cache.md#UnifiedAttentionCacheView.value_cache), [`QUERY_START_LOC`](../requirements/types.md#MetadataField.QUERY_START_LOC), [`ENABLE_DP_LOCAL_PAGE_PATH`](unified_attention.md#ENABLE_DP_LOCAL_PAGE_PATH), [`SLOT_MAPPING`](../requirements/types.md#MetadataField.SLOT_MAPPING), [`_axis_index`](unified_attention.md#_axis_index)  (+2 more)
- used by: [`OperationImpl`](../_operation_impl.md#OperationImpl), [`get_requirements`](../_base_operation.md#BaseOperation.get_requirements), [`forward_native`](paged_flash_attention.md#PagedFlashAttn.forward_native), [`forward_native`](../_base_operation.md#BaseOperation.forward_native), [`get_impl_name`](../_base_operation.md#BaseOperation.get_impl_name)

## Functions
- `_axis_index(axis_names: tuple[str, ...])` — [`L67`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/unified_attention.py#L67) — Return a linearized axis index over one or more mesh axes.
- `_dp_page_axis(cache_view: UnifiedAttentionCacheView)` — [`L78`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/unified_attention.py#L78) — Resolve the logical page axis for the active cache view.
- `_mesh_axis_size(mesh, axis_names: tuple[str, ...])` — [`L57`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/unified_attention.py#L57) — Compute the product of mesh axis sizes for the provided axis names.
- `_normalize_axis_names(axis: str | tuple[str, ...] | list[str] | None)` — [`L48`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/unified_attention.py#L48) — Normalize a partition axis spec into concrete mesh axis names.

## Module values
- `ENABLE_DP_LOCAL_PAGE_PATH` — [`L45`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/unified_attention.py#L45)


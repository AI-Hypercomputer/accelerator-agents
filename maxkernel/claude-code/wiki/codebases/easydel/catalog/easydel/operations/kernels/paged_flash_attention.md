---
title: 'Module: easydel/operations/kernels/paged_flash_attention.py'
type: catalog
provenance: extracted
module: easydel/operations/kernels/paged_flash_attention.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.operations.kernels.paged_flash_attention`/
symbols:
  PagedFlashAttn.forward_native: PagedFlashAttn#forward_native().
  PagedFlashAttn.get_requirements: PagedFlashAttn#get_requirements().
  _dp_page_axis: _dp_page_axis().
  PagedFlashAttn: PagedFlashAttn#
  PagedFlashAttn.forward_gpu: PagedFlashAttn#forward_gpu().
  PagedFlashAttn.forward_cuda: PagedFlashAttn#forward_cuda().
  PagedFlashAttn.forward_tpu: PagedFlashAttn#forward_tpu().
  PagedFlashAttn.forward_cpu: PagedFlashAttn#forward_cpu().
  PagedFlashAttn.forward_rocm: PagedFlashAttn#forward_rocm().
  ENABLE_DP_LOCAL_PAGE_PATH: ENABLE_DP_LOCAL_PAGE_PATH.
  _localize_block_tables_for_dp_pages: _localize_block_tables_for_dp_pages().
  PagedFlashAttn.get_impl_name: PagedFlashAttn#get_impl_name().
---
# Module: [`easydel/operations/kernels/paged_flash_attention.py`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/paged_flash_attention.py)

## Classes
### `PagedFlashAttn`  ·  implements/extends OperationImpl
- def: [`easydel/operations/kernels/paged_flash_attention.py:78`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/paged_flash_attention.py#L78)
- doc: Paged Flash Attention using CUDA flash_attention with block tables.
- signature: `class PagedFlashAttn(OperationImpl):`
- members:
  - `forward_cpu(self, *args, **kwargs)` — [`L275`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/paged_flash_attention.py#L275) — CPU forward pass. Delegates to ``forward_native``.
  - `forward_cuda(self, *args, **kwargs)` — [`L267`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/paged_flash_attention.py#L267) — CUDA forward pass. Delegates to ``forward_gpu``.
  - `forward_gpu(self, *args, **kwargs)` — [`L279`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/paged_flash_attention.py#L279) — GPU forward pass. Delegates to ``forward_native``.
  - `forward_native(self, query: Float[Array, "batch seq_len_q num_heads head_dim"], key: Float[Array, "batch seq_len_k num_kv_heads head_dim"], value: Float[Array, "batch seq_len_k num_kv_heads head_dim"], cache_view: UnifiedAttentionCacheView | None, cache_metadata: RaggedPagesMetadata | OperationsMetadata | None, mask_info: MaskInfo | None = None, bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None, softmax_scale: float | None = None, dropout_prob: float = 0, causal: bool = False, dropout_seed: int | None = None, sliding_window: int | tuple[int, int] | None = None, logits_soft_cap: float | None = None, softmax_aux: Float[Array, "num_heads num_sinks"] | Float[Array, num_sinks] | None = None, normalize_output: bool = True, precision: lax.PrecisionLike = lax.Precision.DEFAULT, **ignore)` — [`L95`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/paged_flash_attention.py#L95) — Forward pass for paged flash attention on GPU.
  - `forward_rocm(self, *args, **kwargs)` — [`L283`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/paged_flash_attention.py#L283) — ROCm forward pass. Delegates to ``forward_native``.
  - `forward_tpu(self, *args, **kwargs)` — [`L271`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/paged_flash_attention.py#L271) — TPU forward pass. Delegates to ``forward_native``.
  - `get_impl_name(cls)` — [`L82`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/paged_flash_attention.py#L82)
  - `get_requirements(cls, mode: ExecutionMode = ExecutionMode.MIXED)` — [`L86`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/paged_flash_attention.py#L86)
- uses (calls/refs, reference-scoped): [`OperationsMetadata`](../../caching/_abstracts.md#OperationsMetadata), [`RaggedPagesMetadata`](../../caching/ragged_page/cache.md#RaggedPagesMetadata), [`AttentionOutput`](../_attention_outputs.md#AttentionOutput), [`MetadataField`](../requirements/types.md#MetadataField), [`CacheType`](../requirements/types.md#CacheType), [`metadata`](../_base_operation.md#BaseOperation.metadata), [`ExecutionMode`](../requirements/types.md#ExecutionMode), [`OperationRequirements`](../requirements/requirements.md#OperationRequirements), [`OperationImpl`](../_operation_impl.md#OperationImpl), [`get_shardings`](../_operation_meta.md#OperationMetadata.get_shardings), [`build`](../requirements/builder.md#RequirementsBuilder.build), [`forward_native`](unified_attention.md#UnifiedAttn.forward_native), [`UnifiedAttentionCacheView`](../../caching/unified_attention/cache.md#UnifiedAttentionCacheView), [`create_stable_sharding`](../_operation_impl.md#OperationImpl.create_stable_sharding), [`mesh`](../_operation_meta.md#OperationMetadata.mesh), [`RequirementsBuilder`](../requirements/builder.md#RequirementsBuilder), [`unwrap_metadata`](../../caching/_abstracts.md#unwrap_metadata), [`register`](../_base_operation.md#OperationRegistry.register), [`MIXED`](../requirements/types.md#ExecutionMode.MIXED), [`require_metadata`](../requirements/builder.md#RequirementsBuilder.require_metadata), [`support_cache`](../requirements/builder.md#RequirementsBuilder.support_cache), [`query`](../_operation_meta.md#AttnShardingRules.query), [`RAGGED_PAGES`](../requirements/types.md#CacheType.RAGGED_PAGES), [`get_operation_config`](../_operation_meta.md#OperationMetadata.get_operation_config), [`pages_tables`](../../caching/ragged_page/cache.md#RaggedPagesMetadata.pages_tables), [`use_cache_view`](../requirements/builder.md#RequirementsBuilder.use_cache_view), [`get_mode`](../_operation_impl.md#OperationImpl.get_mode), [`runtime_dtype`](../_operation_meta.md#OperationMetadata.runtime_dtype), [`output`](../_operation_meta.md#AttnShardingRules.output), [`partition_manager`](../_operation_meta.md#OperationMetadata.partition_manager), [`paged_v2`](../requirements/types.md#MetadataField.paged_v2), [`key_cache`](../../caching/unified_attention/cache.md#UnifiedAttentionCacheView.key_cache), [`_dp_page_axis`](paged_flash_attention.md#_dp_page_axis), [`UnifiedAttn`](unified_attention.md#UnifiedAttn), [`metadata`](../../caching/unified_attention/cache.md#UnifiedAttentionCacheView.metadata), [`value_cache`](../../caching/unified_attention/cache.md#UnifiedAttentionCacheView.value_cache), [`softmax_aux`](../_operation_meta.md#AttnShardingRules.softmax_aux), [`bias`](../_operation_meta.md#AttnShardingRules.bias), [`_localize_block_tables_for_dp_pages`](paged_flash_attention.md#_localize_block_tables_for_dp_pages)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`OperationImpl`](../_operation_impl.md#OperationImpl), [`get_requirements`](../_base_operation.md#BaseOperation.get_requirements), [`forward_native`](../_base_operation.md#BaseOperation.forward_native), [`get_impl_name`](../_base_operation.md#BaseOperation.get_impl_name), [`forward_gpu`](../_base_operation.md#BaseOperation.forward_gpu), [`forward_tpu`](../_base_operation.md#BaseOperation.forward_tpu), [`forward_cpu`](../_base_operation.md#BaseOperation.forward_cpu), [`forward_cuda`](../_base_operation.md#BaseOperation.forward_cuda), [`forward_rocm`](../_base_operation.md#BaseOperation.forward_rocm)

## Functions
- `_dp_page_axis(cache_view: UnifiedAttentionCacheView)` — [`L45`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/paged_flash_attention.py#L45) — Resolve the logical page axis for the active cache view.
- `_localize_block_tables_for_dp_pages(block_tables: Array, *, num_pages: int, dp_size: int)` — [`L51`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/paged_flash_attention.py#L51) — Translate global page IDs into DP-local page IDs per request row.

## Module values
- `ENABLE_DP_LOCAL_PAGE_PATH` — [`L42`](../../../../../../../raw/code/EasyDeL/easydel/operations/kernels/paged_flash_attention.py#L42)


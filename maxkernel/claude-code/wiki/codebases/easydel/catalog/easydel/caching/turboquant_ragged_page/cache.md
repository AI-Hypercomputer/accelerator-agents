---
title: 'Module: easydel/caching/turboquant_ragged_page/cache.py'
type: catalog
provenance: extracted
module: easydel/caching/turboquant_ragged_page/cache.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.caching.turboquant_ragged_page.cache`/
symbols:
  TurboQuantRaggedPagesCacheView.init: TurboQuantRaggedPagesCacheView#init().
  TurboQuantRaggedPagesCacheView.init_all_layers: TurboQuantRaggedPagesCacheView#init_all_layers().
  TurboQuantRaggedPagesCacheConfig: TurboQuantRaggedPagesCacheConfig#
  TurboQuantRaggedPagesCacheView: TurboQuantRaggedPagesCacheView#
  TurboQuantRaggedPagesCacheConfig.create: TurboQuantRaggedPagesCacheConfig#create().
  TurboQuantRaggedPagesCache.init_cache: TurboQuantRaggedPagesCache#init_cache().
  TurboQuantRaggedPagesCache: TurboQuantRaggedPagesCache#
  TurboQuantRaggedPagesCacheView.concatenate_to_cache: TurboQuantRaggedPagesCacheView#concatenate_to_cache().
  TurboQuantRaggedPagesCache.metadata: TurboQuantRaggedPagesCache#metadata().
  TurboQuantRaggedPagesCacheConfig.turboquant_config: TurboQuantRaggedPagesCacheConfig#turboquant_config.
  TurboQuantRaggedPagesCacheView.key_indices_pages: TurboQuantRaggedPagesCacheView#key_indices_pages.
  TurboQuantRaggedPagesCacheView.__repr__: TurboQuantRaggedPagesCacheView#__repr__().
  TurboQuantRaggedPagesCache.views: TurboQuantRaggedPagesCache#views.
  TurboQuantRaggedPagesCacheView.value_norms_pages: TurboQuantRaggedPagesCacheView#value_norms_pages.
  TurboQuantRaggedPagesCacheView._make_pages: TurboQuantRaggedPagesCacheView#_make_pages().
  TurboQuantRaggedPagesCacheView.constants: TurboQuantRaggedPagesCacheView#constants.
  logger: logger.
  TurboQuantRaggedPagesCacheView.key_signs_pages: TurboQuantRaggedPagesCacheView#key_signs_pages.
  TurboQuantRaggedPagesCacheView.key_norms_pages: TurboQuantRaggedPagesCacheView#key_norms_pages.
  TurboQuantRaggedPagesCacheView.value_indices_pages: TurboQuantRaggedPagesCacheView#value_indices_pages.
  TurboQuantRaggedPagesCacheView.__str__: TurboQuantRaggedPagesCacheView#__str__.
---
# Module: [`easydel/caching/turboquant_ragged_page/cache.py`](../../../../../../../raw/code/EasyDeL/easydel/caching/turboquant_ragged_page/cache.py)

## Classes
### `TurboQuantRaggedPagesCache`  ·  implements/extends RaggedPagesCache
- def: [`easydel/caching/turboquant_ragged_page/cache.py:397`](../../../../../../../raw/code/EasyDeL/easydel/caching/turboquant_ragged_page/cache.py#L397)
- doc: Multi-layer container for TurboQuant-compressed ragged page caches.
- signature: `class TurboQuantRaggedPagesCache(RaggedPagesCache):`
- members:
  - `init_cache(cls, mesh: Mesh, config: TurboQuantRaggedPagesCacheConfig, partition_manager: PartitionManager, quantizer: EasyQuantizer | None = None)` — [`L413`](../../../../../../../raw/code/EasyDeL/easydel/caching/turboquant_ragged_page/cache.py#L413) — Allocate TurboQuant cache for all transformer layers.
  - `metadata(self)` — [`L407`](../../../../../../../raw/code/EasyDeL/easydel/caching/turboquant_ragged_page/cache.py#L407)
  - `views` — [`L404`](../../../../../../../raw/code/EasyDeL/easydel/caching/turboquant_ragged_page/cache.py#L404)
- uses (calls/refs, reference-scoped): [`RaggedPagesCache`](../ragged_page/cache.md#RaggedPagesCache), [`EasyQuantizer`](../../layers/quantization/_quants.md#EasyQuantizer), [`init_all_layers`](cache.md#TurboQuantRaggedPagesCacheView.init_all_layers), [`TurboQuantRaggedPagesCacheConfig`](cache.md#TurboQuantRaggedPagesCacheConfig), [`TurboQuantRaggedPagesCacheView`](cache.md#TurboQuantRaggedPagesCacheView), [`metadata`](../ragged_page/cache.md#RaggedPagesCacheView.metadata), [`num_hidden_layers`](../ragged_page/cache.md#RaggedPagesCacheConfig.num_hidden_layers)
- used by: [`RaggedPagesCache`](../ragged_page/cache.md#RaggedPagesCache), [`init_ragged_pages`](../../infra/mixins/generation.md#EasyGenerationMixin.init_ragged_pages), [`init_cache`](../ragged_page/cache.md#RaggedPagesCache.init_cache), [`metadata`](../ragged_page/cache.md#RaggedPagesCache.metadata), [`_materialize_mixed_turboquant_ragged_cache`](../../infra/mixins/generation.md#_materialize_mixed_turboquant_ragged_cache)

### `TurboQuantRaggedPagesCacheConfig`  ·  implements/extends RaggedPagesCacheConfig
- def: [`easydel/caching/turboquant_ragged_page/cache.py:65`](../../../../../../../raw/code/EasyDeL/easydel/caching/turboquant_ragged_page/cache.py#L65)
- doc: Configuration for TurboQuant-compressed ragged pages cache.
- signature: `class TurboQuantRaggedPagesCacheConfig(RaggedPagesCacheConfig):`
- members:
  - `create(cls, mesh: Mesh, partition_manager: PartitionManager, turboquant_config: TurboQuantConfig, num_hidden_layers: int, num_kv_heads: int, max_model_length: int, kv_head_dim_size: int, hbm_utilization: float = 0.9, page_size: int = 128)` — [`L80`](../../../../../../../raw/code/EasyDeL/easydel/caching/turboquant_ragged_page/cache.py#L80) — Create a TurboQuant cache config from model hyperparameters.
  - `turboquant_config` — [`L77`](../../../../../../../raw/code/EasyDeL/easydel/caching/turboquant_ragged_page/cache.py#L77)
- uses (calls/refs, reference-scoped): [`RaggedPagesCacheConfig`](../ragged_page/cache.md#RaggedPagesCacheConfig), [`_mesh_axis_size`](../ragged_page/cache.md#_mesh_axis_size), [`TurboQuantConfig`](../../layers/quantization/_turboquant.md#TurboQuantConfig), [`resolve_attention_data_parallel_axis`](../../axis.md#resolve_attention_data_parallel_axis), [`cdiv`](../ragged_page/cache.md#cdiv), [`qjl_dim`](../../layers/quantization/_turboquant.md#TurboQuantConfig.qjl_dim), [`per_device_hbm_budget_bytes`](../ragged_page/cache.md#per_device_hbm_budget_bytes), [`logger`](cache.md#logger)
- used by: [`init_operations_cache_config`](../../infra/mixins/generation.md#EasyGenerationMixin.init_operations_cache_config), [`RaggedPagesCacheConfig`](../ragged_page/cache.md#RaggedPagesCacheConfig), [`init_operations_cache`](../../infra/mixins/generation.md#EasyGenerationMixin.init_operations_cache), [`init_ragged_pages`](../../infra/mixins/generation.md#EasyGenerationMixin.init_ragged_pages), [`create`](../ragged_page/cache.md#RaggedPagesCacheConfig.create), [`init`](cache.md#TurboQuantRaggedPagesCacheView.init), [`init_all_layers`](cache.md#TurboQuantRaggedPagesCacheView.init_all_layers), [`_create_mixed_turboquant_ragged_page_cache_configs`](../../infra/mixins/generation.md#_create_mixed_turboquant_ragged_page_cache_configs), [`_tq_signature`](../../infra/mixins/generation.md#EasyGenerationMixin._tq_signature), [`init_cache`](cache.md#TurboQuantRaggedPagesCache.init_cache), [`metadata`](cache.md#TurboQuantRaggedPagesCache.metadata), [`_build_config`](../../infra/mixins/generation.md#_create_mixed_turboquant_ragged_page_cache_configs._build_config)

### `TurboQuantRaggedPagesCacheView`  ·  implements/extends RaggedPagesCacheView
- def: [`easydel/caching/turboquant_ragged_page/cache.py:171`](../../../../../../../raw/code/EasyDeL/easydel/caching/turboquant_ragged_page/cache.py#L171)
- doc: Per-layer view into the TurboQuant-compressed page cache.
- signature: `class TurboQuantRaggedPagesCacheView(RaggedPagesCacheView):`
- members:
  - `concatenate_to_cache(self, key: Float[Array, "batch seq_len num_key_heads head_dim"], value: Float[Array, "batch seq_len num_value_heads head_dim"], cache_metadata: RaggedPagesMetadata | OperationsMetadata)` — [`L366`](../../../../../../../raw/code/EasyDeL/easydel/caching/turboquant_ragged_page/cache.py#L366) — Update cache with new KV pairs.
  - `init(cls, config: TurboQuantRaggedPagesCacheConfig, layer_index: int | None = None, *, mesh: Mesh | None = None, partition_manager: PartitionManager | None = None, quantizer: EasyQuantizer | None = None, constants: TurboQuantConstants | None = None)` — [`L190`](../../../../../../../raw/code/EasyDeL/easydel/caching/turboquant_ragged_page/cache.py#L190) — Initialize a TurboQuant cache view with zeroed page buffers.
  - `init_all_layers(cls, config: TurboQuantRaggedPagesCacheConfig, num_layers: int, *, mesh: Mesh | None = None, partition_manager: PartitionManager | None = None, layer_indices: list[int] | None = None)` — [`L266`](../../../../../../../raw/code/EasyDeL/easydel/caching/turboquant_ragged_page/cache.py#L266) — Batch-allocate cache views for all layers at once.
  - `constants` — [`L187`](../../../../../../../raw/code/EasyDeL/easydel/caching/turboquant_ragged_page/cache.py#L187)
  - `key_indices_pages` — [`L182`](../../../../../../../raw/code/EasyDeL/easydel/caching/turboquant_ragged_page/cache.py#L182)
  - `key_norms_pages` — [`L184`](../../../../../../../raw/code/EasyDeL/easydel/caching/turboquant_ragged_page/cache.py#L184)
  - `key_signs_pages` — [`L183`](../../../../../../../raw/code/EasyDeL/easydel/caching/turboquant_ragged_page/cache.py#L183)
  - `value_indices_pages` — [`L185`](../../../../../../../raw/code/EasyDeL/easydel/caching/turboquant_ragged_page/cache.py#L185)
  - `value_norms_pages` — [`L186`](../../../../../../../raw/code/EasyDeL/easydel/caching/turboquant_ragged_page/cache.py#L186)
- protocol/private: `__repr__`[`L389`](../../../../../../../raw/code/EasyDeL/easydel/caching/turboquant_ragged_page/cache.py#L389), `__str__`[`L393`](../../../../../../../raw/code/EasyDeL/easydel/caching/turboquant_ragged_page/cache.py#L393), `_make_pages`[`L230`](../../../../../../../raw/code/EasyDeL/easydel/caching/turboquant_ragged_page/cache.py#L230)
- uses (calls/refs, reference-scoped): [`OperationsMetadata`](../_abstracts.md#OperationsMetadata), [`RaggedPagesMetadata`](../ragged_page/cache.md#RaggedPagesMetadata), [`RaggedPagesCacheView`](../ragged_page/cache.md#RaggedPagesCacheView), [`EasyQuantizer`](../../layers/quantization/_quants.md#EasyQuantizer), [`generate`](../../layers/quantization/_turboquant.md#TurboQuantConstants.generate), [`ATTN_DP`](../../axis.md#ATTN_DP), [`TurboQuantRaggedPagesCacheConfig`](cache.md#TurboQuantRaggedPagesCacheConfig), [`page_size`](../ragged_page/cache.md#RaggedPagesCacheConfig.page_size), [`TurboQuantConstants`](../../layers/quantization/_turboquant.md#TurboQuantConstants), [`k_headdim`](../ragged_page/cache.md#RaggedPagesCacheConfig.k_headdim), [`qjl_dim`](../../layers/quantization/_turboquant.md#TurboQuantConfig.qjl_dim), [`turboquant_config`](cache.md#TurboQuantRaggedPagesCacheConfig.turboquant_config), [`num_kv_heads`](../ragged_page/cache.md#RaggedPagesCacheConfig.num_kv_heads), [`num_pages`](../ragged_page/cache.md#RaggedPagesCacheConfig.num_pages), [`data_parallel_size`](../ragged_page/cache.md#RaggedPagesCacheConfig.data_parallel_size), [`layer_index`](../ragged_page/cache.md#RaggedPagesCacheView.layer_index), [`logger`](cache.md#logger)
- used by: [`RaggedPagesCacheView`](../ragged_page/cache.md#RaggedPagesCacheView), [`init_operations_cache`](../../infra/mixins/generation.md#EasyGenerationMixin.init_operations_cache), [`_forward_v2_turboquant`](../../operations/kernels/ragged_page_attention.md#_RaggedPageAttn._forward_v2_turboquant), [`_forward_v3_turboquant`](../../operations/kernels/ragged_page_attention.md#_RaggedPageAttn._forward_v3_turboquant), [`concatenate_to_cache`](../ragged_page/cache.md#RaggedPagesCacheView.concatenate_to_cache), [`forward_v2`](../../operations/kernels/ragged_page_attention.md#_RaggedPageAttn.forward_v2), [`init`](../ragged_page/cache.md#RaggedPagesCacheView.init), [`forward_v3`](../../operations/kernels/ragged_page_attention.md#_RaggedPageAttn.forward_v3), [`init_cache`](cache.md#TurboQuantRaggedPagesCache.init_cache), [`_materialize_mixed_turboquant_ragged_cache`](../../infra/mixins/generation.md#_materialize_mixed_turboquant_ragged_cache), [`views`](cache.md#TurboQuantRaggedPagesCache.views)

## Module values
- `logger` — [`L61`](../../../../../../../raw/code/EasyDeL/easydel/caching/turboquant_ragged_page/cache.py#L61)


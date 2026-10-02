---
title: 'Module: easydel/caching/unified_attention/cache.py'
type: catalog
provenance: extracted
module: easydel/caching/unified_attention/cache.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.caching.unified_attention.cache`/
symbols:
  UnifiedAttentionCacheView: UnifiedAttentionCacheView#
  UnifiedAttentionCacheConfig: UnifiedAttentionCacheConfig#
  UnifiedAttentionCacheView.init: UnifiedAttentionCacheView#init().
  UnifiedAttentionCacheView.concatenate_to_cache: UnifiedAttentionCacheView#concatenate_to_cache().
  UnifiedAttentionCache: UnifiedAttentionCache#
  UnifiedAttentionCacheConfig.create: UnifiedAttentionCacheConfig#create().
  UnifiedAttentionCache.init_cache: UnifiedAttentionCache#init_cache().
  EMPTY: EMPTY.
  UnifiedAttentionCacheConfig.get_max_num_seqs: UnifiedAttentionCacheConfig#get_max_num_seqs().
  UnifiedAttentionCacheConfig._compute_free_hbm: UnifiedAttentionCacheConfig#_compute_free_hbm().
  UnifiedAttentionCacheConfig.get_padded_num_slices: UnifiedAttentionCacheConfig#get_padded_num_slices().
  UnifiedAttentionCacheView.key_cache: UnifiedAttentionCacheView#key_cache.
  UnifiedAttentionCacheConfig.page_size: UnifiedAttentionCacheConfig#page_size.
  UnifiedAttentionCacheView._update_pages: UnifiedAttentionCacheView#_update_pages().
  UnifiedAttentionCacheView.__repr__: UnifiedAttentionCacheView#__repr__().
  _attention_dp_axis: _attention_dp_axis().
  _mesh_axis_size: _mesh_axis_size().
  UnifiedAttentionCacheView.metadata: UnifiedAttentionCacheView#metadata.
  UnifiedAttentionCache.metadata: UnifiedAttentionCache#metadata().
  MODE_PREFILL: MODE_PREFILL.
  UnifiedAttentionCacheView.value_cache: UnifiedAttentionCacheView#value_cache.
  UnifiedAttentionCacheConfig.kvdtype: UnifiedAttentionCacheConfig#kvdtype().
  UnifiedAttentionCache.views: UnifiedAttentionCache#views.
  KV_HEAD: KV_HEAD.
  cdiv: cdiv().
  UnifiedAttentionCacheConfig.num_pages: UnifiedAttentionCacheConfig#num_pages.
  logger: logger.
  _previous_power_of_2: _previous_power_of_2().
  UnifiedAttentionCacheConfig.window_aware_max_num_seqs: UnifiedAttentionCacheConfig#window_aware_max_num_seqs.
  UnifiedAttentionCacheView.__str__: UnifiedAttentionCacheView#__str__.
  UnifiedAttentionCache.init_empty: UnifiedAttentionCache#init_empty().
  UnifiedAttentionCacheConfig.max_model_length: UnifiedAttentionCacheConfig#max_model_length.
  UnifiedAttentionCacheConfig.num_kv_heads: UnifiedAttentionCacheConfig#num_kv_heads.
  UnifiedAttentionCacheView.partition_manager: UnifiedAttentionCacheView#partition_manager.
  per_device_hbm_budget_bytes: per_device_hbm_budget_bytes().
  _axis_index: _axis_index().
  UnifiedAttentionCacheConfig.num_hidden_layers: UnifiedAttentionCacheConfig#num_hidden_layers.
  UnifiedAttentionCacheConfig.head_dim: UnifiedAttentionCacheConfig#head_dim.
  UnifiedAttentionCacheConfig.data_parallel_size: UnifiedAttentionCacheConfig#data_parallel_size.
  UnifiedAttentionCacheConfig.max_num_pages_per_req: UnifiedAttentionCacheConfig#max_num_pages_per_req.
  UnifiedAttentionCacheConfig.window_aware_pages_per_request: UnifiedAttentionCacheConfig#window_aware_pages_per_request.
  UnifiedAttentionCacheConfig.window_aware_max_num_batched_tokens: UnifiedAttentionCacheConfig#window_aware_max_num_batched_tokens.
  UnifiedAttentionCacheConfig.num_slices_per_kv_cache_update_page: UnifiedAttentionCacheConfig#num_slices_per_kv_cache_update_page.
  UnifiedAttentionCacheConfig._kvdtype_str: UnifiedAttentionCacheConfig#_kvdtype_str.
  UnifiedAttentionCacheView.layer_index: UnifiedAttentionCacheView#layer_index.
  UnifiedAttentionCacheConfig.hbm_utilization: UnifiedAttentionCacheConfig#hbm_utilization.
  UnifiedAttentionCacheConfig.version: UnifiedAttentionCacheConfig#version.
---
# Module: [`easydel/caching/unified_attention/cache.py`](../../../../../../../raw/code/EasyDeL/easydel/caching/unified_attention/cache.py)

## Classes
### `UnifiedAttentionCache`  ·  implements/extends BaseCache
- def: [`easydel/caching/unified_attention/cache.py:623`](../../../../../../../raw/code/EasyDeL/easydel/caching/unified_attention/cache.py#L623)
- doc: Complete unified attention KV cache for all model layers.
- signature: `class UnifiedAttentionCache(BaseCache):`
- members:
  - `init_cache(cls, *, mesh: Mesh, config: UnifiedAttentionCacheConfig, partition_manager: es.PartitionManager, quantizer: EasyQuantizer | None = None)` — [`L656`](../../../../../../../raw/code/EasyDeL/easydel/caching/unified_attention/cache.py#L656) — Initialize the complete cache for all layers.
  - `init_empty(cls, num_hidden_layers: int, *args, **kwargs)` — [`L691`](../../../../../../../raw/code/EasyDeL/easydel/caching/unified_attention/cache.py#L691) — Create an empty cache with None views for each layer.
  - `metadata(self)` — [`L644`](../../../../../../../raw/code/EasyDeL/easydel/caching/unified_attention/cache.py#L644) — Get the shared cache configuration.
  - `views` — [`L641`](../../../../../../../raw/code/EasyDeL/easydel/caching/unified_attention/cache.py#L641)
- uses (calls/refs, reference-scoped): [`UnifiedAttentionCacheView`](cache.md#UnifiedAttentionCacheView), [`EasyQuantizer`](../../layers/quantization/_quants.md#EasyQuantizer), [`UnifiedAttentionCacheConfig`](cache.md#UnifiedAttentionCacheConfig), [`init`](cache.md#UnifiedAttentionCacheView.init), [`BaseCache`](../_abstracts.md#BaseCache), [`metadata`](cache.md#UnifiedAttentionCacheView.metadata), [`num_hidden_layers`](cache.md#UnifiedAttentionCacheConfig.num_hidden_layers)
- used by: [`_backbone_step`](../../inference/esurge/runners/executors/model_executor.md#ModelStepExecutor._backbone_step), [`get_compile_configurations`](../../inference/esurge/runners/execution_manager.md#ExecutionManager.get_compile_configurations), [`init_ragged_pages`](../../infra/mixins/generation.md#EasyGenerationMixin.init_ragged_pages), [`BaseCache`](../_abstracts.md#BaseCache), [`_build_backbone_fn`](../../inference/esurge/runners/executors/model_executor.md#ModelStepExecutor._build_backbone_fn), [`init_unified_attention_cache`](../../infra/mixins/generation.md#EasyGenerationMixin.init_unified_attention_cache), [`init_cache`](../_abstracts.md#BaseCache.init_cache), [`init_empty`](../_abstracts.md#BaseCache.init_empty), [`kv_pages`](../../inference/esurge/runners/execution_types.md#StepFunctionInputs.kv_pages), [`__init__`](../../inference/esurge/runners/executors/model_executor.md#ModelStepExecutor.__init__), [`_materialize_mixed_unified_attention_cache`](../../infra/mixins/generation.md#_materialize_mixed_unified_attention_cache), [`kv_pages`](../../inference/esurge/runners/execution_types.md#BackboneOutputs.kv_pages), [`kv_pages`](../../inference/esurge/runners/execution_types.md#ModelStepOutputs.kv_pages), [`kv_pages`](../../inference/esurge/runners/execution_types.md#StepFunctionOutputs.kv_pages)

### `UnifiedAttentionCacheConfig`  ·  implements/extends BaseCacheConfig
- def: [`easydel/caching/unified_attention/cache.py:209`](../../../../../../../raw/code/EasyDeL/easydel/caching/unified_attention/cache.py#L209)
- doc: Configuration for vLLM-style unified attention paged KV-cache.
- signature: `class UnifiedAttentionCacheConfig(BaseCacheConfig):`
- members:
  - `_compute_free_hbm(mesh: Mesh, partition_manager: PartitionManager, hbm_utilization: float)` — [`L274`](../../../../../../../raw/code/EasyDeL/easydel/caching/unified_attention/cache.py#L274) — Compute available HBM for cache allocation across mesh.
  - `create(cls, mesh: Mesh, partition_manager: PartitionManager, kvdtype: jnp.dtype, num_hidden_layers: int, num_kv_heads: int, max_model_length: int, head_dim: int, *, hbm_utilization: float = 0.9, page_size: int = 128)` — [`L299`](../../../../../../../raw/code/EasyDeL/easydel/caching/unified_attention/cache.py#L299) — Create a UnifiedAttentionCacheConfig with automatic capacity calculation.
  - `get_max_num_seqs(self)` — [`L428`](../../../../../../../raw/code/EasyDeL/easydel/caching/unified_attention/cache.py#L428) — Estimate maximum concurrent sequences based on page budget.
  - `get_padded_num_slices(self, num_tokens: int | None = None, max_num_reqs: int | None = None)` — [`L399`](../../../../../../../raw/code/EasyDeL/easydel/caching/unified_attention/cache.py#L399) — Calculate padded slice count for slot mapping.
  - `kvdtype(self)` — [`L389`](../../../../../../../raw/code/EasyDeL/easydel/caching/unified_attention/cache.py#L389) — Get the JAX dtype for KV cache storage.
  - `data_parallel_size` — [`L257`](../../../../../../../raw/code/EasyDeL/easydel/caching/unified_attention/cache.py#L257)
  - `hbm_utilization` — [`L256`](../../../../../../../raw/code/EasyDeL/easydel/caching/unified_attention/cache.py#L256)
  - `head_dim` — [`L254`](../../../../../../../raw/code/EasyDeL/easydel/caching/unified_attention/cache.py#L254)
  - `max_model_length` — [`L252`](../../../../../../../raw/code/EasyDeL/easydel/caching/unified_attention/cache.py#L252)
  - `max_num_pages_per_req` — [`L260`](../../../../../../../raw/code/EasyDeL/easydel/caching/unified_attention/cache.py#L260)
  - `num_hidden_layers` — [`L251`](../../../../../../../raw/code/EasyDeL/easydel/caching/unified_attention/cache.py#L251)
  - `num_kv_heads` — [`L253`](../../../../../../../raw/code/EasyDeL/easydel/caching/unified_attention/cache.py#L253)
  - `num_pages` — [`L259`](../../../../../../../raw/code/EasyDeL/easydel/caching/unified_attention/cache.py#L259)
  - `num_slices_per_kv_cache_update_page` — [`L266`](../../../../../../../raw/code/EasyDeL/easydel/caching/unified_attention/cache.py#L266)
  - `page_size` — [`L258`](../../../../../../../raw/code/EasyDeL/easydel/caching/unified_attention/cache.py#L258)
  - `version` — [`L269`](../../../../../../../raw/code/EasyDeL/easydel/caching/unified_attention/cache.py#L269)
  - `window_aware_max_num_batched_tokens` — [`L263`](../../../../../../../raw/code/EasyDeL/easydel/caching/unified_attention/cache.py#L263)
  - `window_aware_max_num_seqs` — [`L261`](../../../../../../../raw/code/EasyDeL/easydel/caching/unified_attention/cache.py#L261)
  - `window_aware_pages_per_request` — [`L262`](../../../../../../../raw/code/EasyDeL/easydel/caching/unified_attention/cache.py#L262)
- protocol/private: `_kvdtype_str`[`L271`](../../../../../../../raw/code/EasyDeL/easydel/caching/unified_attention/cache.py#L271)
- uses (calls/refs, reference-scoped): [`BaseCacheConfig`](../_abstracts.md#BaseCacheConfig), [`_attention_dp_axis`](cache.md#_attention_dp_axis), [`_mesh_axis_size`](cache.md#_mesh_axis_size), [`cdiv`](cache.md#cdiv), [`_previous_power_of_2`](cache.md#_previous_power_of_2), [`logger`](cache.md#logger), [`per_device_hbm_budget_bytes`](cache.md#per_device_hbm_budget_bytes)
- used by: [`sequence_buffer`](../../inference/esurge/runners/model_runner.md#eSurgeRunner.sequence_buffer), [`from_runner`](../../inference/esurge/scheduler/scheduler.md#Scheduler.from_runner), [`get_compile_configurations`](../../inference/esurge/runners/execution_manager.md#ExecutionManager.get_compile_configurations), [`init_operations_cache_config`](../../infra/mixins/generation.md#EasyGenerationMixin.init_operations_cache_config), [`init_operations_cache`](../../infra/mixins/generation.md#EasyGenerationMixin.init_operations_cache), [`init_ragged_pages`](../../infra/mixins/generation.md#EasyGenerationMixin.init_ragged_pages), [`_compile_sampler_variant`](../../inference/esurge/runners/execution_manager.md#ExecutionManager._compile_sampler_variant), [`BaseCacheConfig`](../_abstracts.md#BaseCacheConfig), [`init`](cache.md#UnifiedAttentionCacheView.init), [`_log_startup_summary`](../../inference/esurge/runners/model_runner.md#eSurgeRunner._log_startup_summary), [`initialize_kv_cache`](../../inference/esurge/runners/model_runner.md#eSurgeRunner.initialize_kv_cache), [`_apply_window_aware_runtime_cap`](../../inference/esurge/runners/model_runner.md#eSurgeRunner._apply_window_aware_runtime_cap), [`_compile_backbone_variant`](../../inference/esurge/runners/execution_manager.md#ExecutionManager._compile_backbone_variant), [`compile`](../../inference/esurge/runners/execution_manager.md#ExecutionManager.compile), [`_create_mixed_standard_unified_attention_cache_configs`](../../infra/mixins/generation.md#_create_mixed_standard_unified_attention_cache_configs), [`init_unified_attention_cache`](../../infra/mixins/generation.md#EasyGenerationMixin.init_unified_attention_cache), [`kv_pages`](../../inference/esurge/runners/execution_manager.md#ExecutionManager.kv_pages), [`create_unified_attention_cache_config`](../../infra/mixins/generation.md#EasyGenerationMixin.create_unified_attention_cache_config), [`_compile_lm_head_variant`](../../inference/esurge/runners/execution_manager.md#ExecutionManager._compile_lm_head_variant), [`create`](../_abstracts.md#BaseCacheConfig.create), [`init_cache`](cache.md#UnifiedAttentionCache.init_cache), [`_build_kv_cache_groups`](../../inference/esurge/runners/model_runner.md#eSurgeRunner._build_kv_cache_groups), [`__init__`](../../inference/esurge/runners/execution_manager.md#ExecutionManager.__init__), [`__init__`](../../inference/esurge/runners/executors/model_executor.md#ModelStepExecutor.__init__), [`num_reqs_max_model_len`](../../inference/esurge/runners/model_runner.md#eSurgeRunner.num_reqs_max_model_len), [`_materialize_mixed_unified_attention_cache`](../../infra/mixins/generation.md#_materialize_mixed_unified_attention_cache), [`max_pages_per_req`](../../inference/esurge/runners/model_runner.md#eSurgeRunner.max_pages_per_req), [`create_unified_attention_cache_config`](../../modules/glm_moe_dsa/modeling_glm_moe_dsa.md#GlmMoeDsaForCausalLM.create_unified_attention_cache_config), [`_build_config`](../../infra/mixins/generation.md#_create_mixed_standard_unified_attention_cache_configs._build_config), [`metadata`](cache.md#UnifiedAttentionCache.metadata), [`metadata`](cache.md#UnifiedAttentionCacheView.metadata), [`__init__`](../../inference/esurge/runners/executors/batch_preparer.md#BatchMetadataPreparer.__init__), [`page_size`](../../inference/esurge/runners/model_runner.md#eSurgeRunner.page_size)

### `UnifiedAttentionCacheView`  ·  implements/extends BaseCacheView
- def: [`easydel/caching/unified_attention/cache.py:445`](../../../../../../../raw/code/EasyDeL/easydel/caching/unified_attention/cache.py#L445) — documented in [easydel-layers-attention-_unified](../../../../concepts/easydel-layers-attention-_unified.md)
- doc: Per-layer KV-cache view for unified attention.
- signature: `class UnifiedAttentionCacheView(BaseCacheView):`
- members:
  - `concatenate_to_cache(self, key: Float[Array, "batch seq_len num_kv_heads head_dim"], value: Float[Array, "batch seq_len num_kv_heads head_dim"], cache_metadata: tp.Any)` — [`L523`](../../../../../../../raw/code/EasyDeL/easydel/caching/unified_attention/cache.py#L523) — Update cache with new key-value pairs using slot mapping.
  - `init(cls, config: UnifiedAttentionCacheConfig, layer_index: int | None = None, *, mesh: Mesh | None = None, partition_manager: es.PartitionManager | None = None, quantizer: EasyQuantizer | None = None)` — [`L474`](../../../../../../../raw/code/EasyDeL/easydel/caching/unified_attention/cache.py#L474) — Initialize a UnifiedAttentionCacheView from a cache config.
  - `key_cache` — [`L465`](../../../../../../../raw/code/EasyDeL/easydel/caching/unified_attention/cache.py#L465)
  - `layer_index` — [`L463`](../../../../../../../raw/code/EasyDeL/easydel/caching/unified_attention/cache.py#L463)
  - `metadata` — [`L462`](../../../../../../../raw/code/EasyDeL/easydel/caching/unified_attention/cache.py#L462)
  - `partition_manager` — [`L468`](../../../../../../../raw/code/EasyDeL/easydel/caching/unified_attention/cache.py#L468)
  - `value_cache` — [`L466`](../../../../../../../raw/code/EasyDeL/easydel/caching/unified_attention/cache.py#L466)
- protocol/private: `__repr__`[`L612`](../../../../../../../raw/code/EasyDeL/easydel/caching/unified_attention/cache.py#L612), `__str__`[`L619`](../../../../../../../raw/code/EasyDeL/easydel/caching/unified_attention/cache.py#L619), `_update_pages`[`L561`](../../../../../../../raw/code/EasyDeL/easydel/caching/unified_attention/cache.py#L561)
- uses (calls/refs, reference-scoped): [`EasyQuantizer`](../../layers/quantization/_quants.md#EasyQuantizer), [`UnifiedAttentionCacheConfig`](cache.md#UnifiedAttentionCacheConfig), [`BaseCacheView`](../_abstracts.md#BaseCacheView), [`unwrap_metadata`](../_abstracts.md#unwrap_metadata), [`ATTN_DP`](../../axis.md#ATTN_DP), [`EMPTY`](cache.md#EMPTY), [`page_size`](cache.md#UnifiedAttentionCacheConfig.page_size), [`_attention_dp_axis`](cache.md#_attention_dp_axis), [`MODE_PREFILL`](cache.md#MODE_PREFILL), [`kv_cache_update_jax`](../ragged_page/utils.md#kv_cache_update_jax), [`kvdtype`](cache.md#UnifiedAttentionCacheConfig.kvdtype), [`KV_HEAD`](cache.md#KV_HEAD), [`num_pages`](cache.md#UnifiedAttentionCacheConfig.num_pages), [`num_kv_heads`](cache.md#UnifiedAttentionCacheConfig.num_kv_heads), [`_axis_index`](cache.md#_axis_index), [`data_parallel_size`](cache.md#UnifiedAttentionCacheConfig.data_parallel_size), [`head_dim`](cache.md#UnifiedAttentionCacheConfig.head_dim)
- used by: [`AttentionOutput`](../../operations/_attention_outputs.md#AttentionOutput), [`forward`](../../layers/attention/_unified.md#UnifiedAttention.forward), [`forward`](../../layers/attention/_flexible.md#FlexibleAttentionModule.forward), [`forward_mla`](../../layers/attention/_unified.md#UnifiedAttention.forward_mla), [`print_status`](../../inference/esurge/runners/execution_types.md#StepFunctionInputs.print_status), [`concatenate`](../../layers/attention/_flexible.md#AttentionModule.concatenate), [`forward_alibi`](../../layers/attention/_unified.md#UnifiedAttention.forward_alibi), [`init_operations_cache_config`](../../infra/mixins/generation.md#EasyGenerationMixin.init_operations_cache_config), [`init_operations_cache`](../../infra/mixins/generation.md#EasyGenerationMixin.init_operations_cache), [`forward_native`](../../operations/kernels/paged_flash_attention.md#PagedFlashAttn.forward_native), [`forward_native`](../../operations/kernels/unified_attention.md#UnifiedAttn.forward_native), [`get_requirements`](../../operations/kernels/unified_attention.md#UnifiedAttn.get_requirements), [`BaseCacheView`](../_abstracts.md#BaseCacheView), [`standard_decoder_layer_call`](../../layers/attention/_decoder_base.md#BaseDecoderLayer.standard_decoder_layer_call), [`get_operations_cache_view`](../../infra/mixins/operation_cache.md#OperationCacheMixin.get_operations_cache_view), [`get_requirements`](../../operations/kernels/paged_flash_attention.md#PagedFlashAttn.get_requirements), [`__call__`](../../layers/attention/_unified.md#UnifiedAttention.__call__), [`init`](../_abstracts.md#BaseCacheView.init), [`pre_norm_residual_attn`](../../layers/attention/_decoder_base.md#BaseDecoderLayer.pre_norm_residual_attn), [`concatenate_to_cache`](../_abstracts.md#BaseCacheView.concatenate_to_cache), [`transformer`](../hybrid/cache.md#ParallelHybridCacheView.transformer), [`init_cache`](cache.md#UnifiedAttentionCache.init_cache), [`_resolve_parallel_hybrid_attention_view_class`](../../infra/mixins/generation.md#EasyGenerationMixin._resolve_parallel_hybrid_attention_view_class), [`_materialize_mixed_unified_attention_cache`](../../infra/mixins/generation.md#_materialize_mixed_unified_attention_cache), [`is_ragged`](../hybrid/cache.md#ParallelHybridCacheView.is_ragged), [`_dp_page_axis`](../../operations/kernels/paged_flash_attention.md#_dp_page_axis), [`_dp_page_axis`](../../operations/kernels/unified_attention.md#_dp_page_axis), [`metadata`](cache.md#UnifiedAttentionCache.metadata), [`views`](cache.md#UnifiedAttentionCache.views)

## Functions
- `_attention_dp_axis(partition_manager: PartitionManager)` — [`L83`](../../../../../../../raw/code/EasyDeL/easydel/caching/unified_attention/cache.py#L83) — Resolve the concrete mesh axis used for KV-page data parallelism.
- `_axis_index(axis: str | tuple[str, ...] | list[str] | None)` — [`L179`](../../../../../../../raw/code/EasyDeL/easydel/caching/unified_attention/cache.py#L179) — Return a linearized axis index over one or more mesh axes.
- `_mesh_axis_size(mesh: Mesh, axis: str | tuple[str, ...] | list[str] | None)` — [`L157`](../../../../../../../raw/code/EasyDeL/easydel/caching/unified_attention/cache.py#L157) — Return product of mesh sizes for a semantic axis mapping.
- `_previous_power_of_2(n: int)` — [`L143`](../../../../../../../raw/code/EasyDeL/easydel/caching/unified_attention/cache.py#L143) — Find the largest power of 2 less than or equal to n.
- `cdiv(a: int, b: int)` — [`L88`](../../../../../../../raw/code/EasyDeL/easydel/caching/unified_attention/cache.py#L88) — Ceiling division: compute ceil(a / b) using integer arithmetic.
- `per_device_hbm_budget_bytes(util: float = 0.9, mode: str = "free", safety_margin: int = 256 << 20)` — [`L101`](../../../../../../../raw/code/EasyDeL/easydel/caching/unified_attention/cache.py#L101) — Calculate available HBM budget per device for cache allocation.

## Module values
- `EMPTY` — [`L78`](../../../../../../../raw/code/EasyDeL/easydel/caching/unified_attention/cache.py#L78)
- `KV_HEAD` — [`L79`](../../../../../../../raw/code/EasyDeL/easydel/caching/unified_attention/cache.py#L79)
- `MODE_PREFILL` — [`L80`](../../../../../../../raw/code/EasyDeL/easydel/caching/unified_attention/cache.py#L80)
- `logger` — [`L76`](../../../../../../../raw/code/EasyDeL/easydel/caching/unified_attention/cache.py#L76)


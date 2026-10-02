---
title: 'Module: easydel/caching/ragged_page/cache.py'
type: catalog
provenance: extracted
module: easydel/caching/ragged_page/cache.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.caching.ragged_page.cache`/
symbols:
  RaggedPagesMetadata: RaggedPagesMetadata#
  RaggedPagesCache: RaggedPagesCache#
  RaggedPagesCacheView: RaggedPagesCacheView#
  RaggedPagesCacheConfig: RaggedPagesCacheConfig#
  RaggedPagesCacheView.concatenate_to_cache: RaggedPagesCacheView#concatenate_to_cache().
  RaggedPagesCacheConfig.create: RaggedPagesCacheConfig#create().
  RaggedPagesCacheConfig.get_shape_and_axes: RaggedPagesCacheConfig#get_shape_and_axes().
  RaggedPagesCache.init_cache: RaggedPagesCache#init_cache().
  RaggedPagesCacheView.init: RaggedPagesCacheView#init().
  RaggedPagesMetadata.context_lens: RaggedPagesMetadata#context_lens.
  _mesh_axis_size: _mesh_axis_size().
  RaggedPagesCacheConfig.get_padded_num_slices: RaggedPagesCacheConfig#get_padded_num_slices().
  RaggedPagesCache.insert: RaggedPagesCache#insert().
  RaggedPagesMetadata.pages_tables: RaggedPagesMetadata#pages_tables.
  cdiv: cdiv().
  RaggedPagesCacheConfig.get_max_num_seqs: RaggedPagesCacheConfig#get_max_num_seqs().
  RaggedPagesCacheConfig._compute_free_hbm: RaggedPagesCacheConfig#_compute_free_hbm().
  RaggedPagesCacheView.kv_pages: RaggedPagesCacheView#kv_pages.
  RaggedPagesCacheView.metadata: RaggedPagesCacheView#metadata.
  RaggedPagesCacheView._update_fn: RaggedPagesCacheView#_update_fn().
  align_to_multiple: align_to_multiple().
  RaggedPagesCache.metadata: RaggedPagesCache#metadata().
  RaggedPagesCache.to_pure: RaggedPagesCache#to_pure().
  RaggedPagesCacheConfig.page_size: RaggedPagesCacheConfig#page_size.
  RaggedPagesCacheView.flattened_kv_pages: RaggedPagesCacheView#flattened_kv_pages().
  RaggedPagesMetadata.query_start_loc: RaggedPagesMetadata#query_start_loc.
  _resolve_ragged_cache_layout: _resolve_ragged_cache_layout().
  EMPTY: EMPTY.
  _select_compatible_v3_kv_cache_dtype: _select_compatible_v3_kv_cache_dtype().
  RaggedPagesCacheConfig.storage_num_combined_kv_heads: RaggedPagesCacheConfig#storage_num_combined_kv_heads().
  _dtype_to_string: _dtype_to_string().
  RaggedPagesCacheConfig.version: RaggedPagesCacheConfig#version.
  RaggedPagesCache.views: RaggedPagesCache#views.
  RaggedPagesCacheConfig.kv_head_packing: RaggedPagesCacheConfig#kv_head_packing().
  _attention_dp_axis: _attention_dp_axis().
  get_page_size_bytes: get_page_size_bytes().
  RaggedPagesCacheConfig.k_headdim: RaggedPagesCacheConfig#k_headdim.
  get_num_slices_per_kv_cache_update_page: get_num_slices_per_kv_cache_update_page().
  RaggedPagesCache.from_pure: RaggedPagesCache#from_pure().
  get_dtype_packing: get_dtype_packing().
  _canonicalize_dtype: _canonicalize_dtype().
  _select_compatible_v3_kv_cache_dtype._storage_groups: _select_compatible_v3_kv_cache_dtype()._storage_groups().
  per_device_hbm_budget_bytes: per_device_hbm_budget_bytes().
  RaggedPagesCacheConfig.kvdtype: RaggedPagesCacheConfig#kvdtype().
  MODE_PREFILL: MODE_PREFILL.
  logger: logger.
  _storage_num_combined_kv_heads_for_dtype: _storage_num_combined_kv_heads_for_dtype().
  RaggedPagesCacheConfig.num_kv_heads: RaggedPagesCacheConfig#num_kv_heads.
  RaggedPagesCacheConfig.num_pages: RaggedPagesCacheConfig#num_pages.
  RaggedPagesCacheConfig.num_slices_per_kv_cache_update_page: RaggedPagesCacheConfig#num_slices_per_kv_cache_update_page.
  RaggedPagesCacheConfig.storage_num_kv_groups: RaggedPagesCacheConfig#storage_num_kv_groups().
  RaggedPagesCacheConfig.storage_head_dim: RaggedPagesCacheConfig#storage_head_dim().
  RaggedPagesCacheView.key_pages: RaggedPagesCacheView#key_pages().
  RaggedPagesCacheView.__repr__: RaggedPagesCacheView#__repr__().
  RaggedPagesCache.__repr__: RaggedPagesCache#__repr__().
  RaggedPagesCacheConfig.data_parallel_size: RaggedPagesCacheConfig#data_parallel_size.
  RaggedPagesCacheView.layer_index: RaggedPagesCacheView#layer_index.
  RaggedPagesCacheView.value_pages: RaggedPagesCacheView#value_pages().
  RaggedPagesCacheConfig.is_v2: RaggedPagesCacheConfig#is_v2().
  RaggedPagesMetadata.request_distribution: RaggedPagesMetadata#request_distribution.
  PERMITTED_KV_KERNELS: PERMITTED_KV_KERNELS.
  RaggedPagesCacheConfig.num_hidden_layers: RaggedPagesCacheConfig#num_hidden_layers.
  RaggedPagesCacheConfig.max_model_length: RaggedPagesCacheConfig#max_model_length.
  RaggedPagesCacheConfig.max_num_pages_per_req: RaggedPagesCacheConfig#max_num_pages_per_req.
  RaggedPagesCacheView.partition_manager: RaggedPagesCacheView#partition_manager.
  RaggedPagesCacheConfig.is_v3: RaggedPagesCacheConfig#is_v3().
  RaggedPagesCacheView.__str__: RaggedPagesCacheView#__str__.
  RaggedPagesCache.__str__: RaggedPagesCache#__str__.
  RaggedPagesMetadata.create_empty: RaggedPagesMetadata#create_empty().
  RaggedPagesCacheConfig.v_headdim: RaggedPagesCacheConfig#v_headdim.
  RaggedPagesCacheConfig.max_num_tokens: RaggedPagesCacheConfig#max_num_tokens.
  RaggedPagesCacheConfig.max_num_reqs: RaggedPagesCacheConfig#max_num_reqs.
  RaggedPagesCacheConfig.window_aware_max_num_seqs: RaggedPagesCacheConfig#window_aware_max_num_seqs.
  RaggedPagesMetadata.num_seqs: RaggedPagesMetadata#num_seqs.
  RaggedPagesMetadata.page_size: RaggedPagesMetadata#page_size.
  previous_power_of_2: previous_power_of_2().
  RaggedPagesCacheConfig.kv_head_shards: RaggedPagesCacheConfig#kv_head_shards.
  RaggedPagesCacheConfig._kvdtype_str: RaggedPagesCacheConfig#_kvdtype_str.
  RaggedPagesCacheConfig._mixed_layer_configs: RaggedPagesCacheConfig#_mixed_layer_configs.
  RaggedPagesCache.init_empty: RaggedPagesCache#init_empty().
  RaggedPagesMetadata.slot_mapping: RaggedPagesMetadata#slot_mapping.
  RaggedPagesMetadata.num_kv_update_slices: RaggedPagesMetadata#num_kv_update_slices.
  RaggedPagesMetadata.num_slices_per_kv_cache_update_page: RaggedPagesMetadata#num_slices_per_kv_cache_update_page.
  KV_HEAD: KV_HEAD.
  RaggedPagesCacheConfig.hbm_utilization: RaggedPagesCacheConfig#hbm_utilization.
  RaggedPagesCacheConfig.window_aware_pages_per_request: RaggedPagesCacheConfig#window_aware_pages_per_request.
  RaggedPagesCacheConfig.window_aware_max_num_batched_tokens: RaggedPagesCacheConfig#window_aware_max_num_batched_tokens.
  RaggedPagesMetadata.position_ids: RaggedPagesMetadata#position_ids.
  RaggedPagesMetadata.version: RaggedPagesMetadata#version.
  RaggedPagesMetadata.prefill_chunk_size: RaggedPagesMetadata#prefill_chunk_size.
---
# Module: [`easydel/caching/ragged_page/cache.py`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py)

## Classes
### `RaggedPagesCache`  ·  implements/extends BaseCache
- def: [`easydel/caching/ragged_page/cache.py:982`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L982) — documented in [easydel-caching-ragged_page-cache](../../../../concepts/easydel-caching-ragged_page-cache.md)
- doc: Represents the complete Paged Attention KV cache for all layers of a model.
- signature: `class RaggedPagesCache(BaseCache):`
- members:
  - `__repr__(self)` — [`L1041`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L1041) — Provides a string representation of the entire paged cache.
  - `from_pure(cls, cache_data: list[dict[str, tp.Any]], metadata: RaggedPagesCacheConfig | None = None, partition_manager: PartitionManager | None = None)` — [`L1077`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L1077) — Reconstruct cache from pure Python data.
  - `init_cache(cls, mesh: Mesh, config: RaggedPagesCacheConfig, partition_manager: es.PartitionManager, quantizer: EasyQuantizer | None = None)` — [`L1003`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L1003) — Initializes the entire RaggedPagesCache for all layers.
  - `init_empty(self, *args, **kwargs)` — [`L1037`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L1037) — Not typically used for RaggedPagesCache; returns None.
  - `insert(self, other: RaggedPagesCache, slot: int)` — [`L1111`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L1111) — Insert another cache's pages at a specific slot offset.
  - `metadata(self)` — [`L997`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L997)
  - `to_pure(self)` — [`L1050`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L1050) — Convert cache to pure Python data for serialization.
  - `views` — [`L994`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L994)
- protocol/private: `__str__`[`L1153`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L1153)
- uses (calls/refs, reference-scoped): [`RaggedPagesCacheView`](cache.md#RaggedPagesCacheView), [`RaggedPagesCacheConfig`](cache.md#RaggedPagesCacheConfig), [`EasyQuantizer`](../../layers/quantization/_quants.md#EasyQuantizer), [`BaseCache`](../_abstracts.md#BaseCache), [`init`](cache.md#RaggedPagesCacheView.init), [`init_cache`](../mla_ragged_page/cache.md#MLARaggedPagesCache.init_cache), [`init_cache`](../turboquant_ragged_page/cache.md#TurboQuantRaggedPagesCache.init_cache), [`kv_pages`](cache.md#RaggedPagesCacheView.kv_pages), [`metadata`](cache.md#RaggedPagesCacheView.metadata), [`TurboQuantRaggedPagesCache`](../turboquant_ragged_page/cache.md#TurboQuantRaggedPagesCache), [`MLARaggedPagesCache`](../mla_ragged_page/cache.md#MLARaggedPagesCache), [`metadata`](../mla_ragged_page/cache.md#MLARaggedPagesCache.metadata), [`metadata`](../turboquant_ragged_page/cache.md#TurboQuantRaggedPagesCache.metadata), [`layer_index`](cache.md#RaggedPagesCacheView.layer_index), [`num_hidden_layers`](cache.md#RaggedPagesCacheConfig.num_hidden_layers), [`partition_manager`](cache.md#RaggedPagesCacheView.partition_manager)
- used by: [`BaseModuleProtocol`](../../infra/mixins/protocol.md#BaseModuleProtocol), [`_backbone_step`](../../inference/esurge/runners/executors/model_executor.md#ModelStepExecutor._backbone_step), [`__all__`](../../modules/whisper/modeling_whisper.md#__all__), [`get_compile_configurations`](../../inference/esurge/runners/execution_manager.md#ExecutionManager.get_compile_configurations), [`init_ragged_pages`](../../infra/mixins/generation.md#EasyGenerationMixin.init_ragged_pages), [`mesh_call`](../../infra/mixins/protocol.md#BaseModuleProtocol.mesh_call), [`__call__`](../../infra/mixins/protocol.md#BaseModuleProtocol.__call__), [`__call__`](../../modules/minimax/modeling_minimax.md#MiniMaxModel.__call__), [`__call__`](../../modules/gemma4/modeling_gemma4.md#Gemma4TextModel.__call__), [`__call__`](../../modules/qwen3_vl_moe/modeling_qwen3_vl_moe.md#Qwen3VLMoeTextModel.__call__), [`__call__`](../../modules/gpt_oss/modeling_gpt_oss.md#GptOssModel.__call__), [`__call__`](../../modules/kimi_linear/modeling_kimi_linear.md#KimiLinearModel.__call__), [`__call__`](../../modules/mixtral/modeling_mixtral.md#MixtralModel.__call__), [`__call__`](../../modules/opt/modeling_opt.md#OPTDecoder.__call__), [`__call__`](../../modules/qwen2_moe/modeling_qwen2_moe.md#Qwen2MoeModel.__call__), [`__call__`](../../modules/qwen3_next/modeling_qwen3_next.md#Qwen3NextModel.__call__), [`__call__`](../../modules/arctic/modeling_arctic.md#ArcticModel.__call__), [`__call__`](../../modules/dbrx/modeling_dbrx.md#DbrxModel.__call__), [`__call__`](../../modules/glm4_moe/modeling_glm4_moe.md#Glm4MoeModel.__call__), [`__call__`](../../modules/glm4_moe_lite/modeling_glm4_moe_lite.md#Glm4MoeLiteModel.__call__), [`__call__`](../../modules/glm4v_moe/modeling_glm4v_moe.md#Glm4vMoeTextModel.__call__), [`__call__`](../../modules/glm_moe_dsa/modeling_glm_moe_dsa.md#GlmMoeDsaModel.__call__), [`__call__`](../../modules/gpt_neox/modeling_gpt_neox.md#GPTNeoXModel.__call__), [`__call__`](../../modules/llama/modeling_llama.md#LlamaModel.__call__), [`__call__`](../../modules/qwen2/modeling_qwen.md#Qwen2Model.__call__), [`__call__`](../../modules/qwen3_moe/modeling_qwen3_moe.md#Qwen3MoeModel.__call__), [`__call__`](../../modules/qwen3_vl/modeling_qwen3_vl.md#Qwen3VLTextModel.__call__), [`__call__`](../../modules/xerxes2/modeling_xerxes2.md#Xerxes2Model.__call__), [`__call__`](../../modules/cohere/modeling_cohere.md#CohereModel.__call__), [`__call__`](../../modules/cohere2/modeling_cohere2.md#Cohere2Model.__call__), [`__call__`](../../modules/exaone/modeling_exaone.md#ExaoneModel.__call__), [`__call__`](../../modules/exaone4/modeling_exaone4.md#Exaone4Model.__call__), [`__call__`](../../modules/gemma2/modeling_gemma2.md#Gemma2Model.__call__), [`__call__`](../../modules/gidd/modeling_gidd.md#GiddModel.__call__), [`__call__`](../../modules/glm/modeling_glm.md#GlmModel.__call__), [`__call__`](../../modules/glm4/modeling_glm4.md#Glm4Model.__call__), [`__call__`](../../modules/glm4v/modeling_glm4v.md#Glm4vTextModel.__call__), [`__call__`](../../modules/gpt_j/modeling_gpt_j.md#GPTJModel.__call__), [`__call__`](../../modules/grok_1/modeling_grok_1.md#Grok1Model.__call__), [`__call__`](../../modules/internlm2/modeling_internlm2.md#InternLM2Model.__call__)  (+132 more)

### `RaggedPagesCacheConfig`  ·  implements/extends BaseCacheConfig
- def: [`easydel/caching/ragged_page/cache.py:375`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L375) — documented in [easydel-caching-ragged_page-cache](../../../../concepts/easydel-caching-ragged_page-cache.md)
- doc: Configuration for the Paged Attention KV cache.
- signature: `class RaggedPagesCacheConfig(BaseCacheConfig):`
- members:
  - `_compute_free_hbm(mesh: Mesh, partition_manager: PartitionManager, hbm_utilization: float, kv_head_shards: int | None = None)` — [`L444`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L444) — Compute available HBM for cache allocation across mesh.
  - `create(cls, mesh: Mesh, partition_manager: PartitionManager, kvdtype: jnp.dtype, num_hidden_layers: int, num_kv_heads: int, max_model_length: int, kv_head_dim_size: int | None = None, k_headdim: int | None = None, v_headdim: int | None = None, hbm_utilization: float = 0.9, page_size: int = 128, version: tp.Literal["v3", "v2"] = "v3")` — [`L483`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L483) — Create a RaggedPagesCacheConfig with automatic capacity calculation.
  - `get_max_num_seqs(self)` — [`L681`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L681) — Estimate maximum concurrent sequences based on page budget.
  - `get_padded_num_slices(self, num_tokens: int | None = None, max_num_reqs: int | None = None)` — [`L649`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L649) — Calculate padded slice count for v2 slot mapping.
  - `get_shape_and_axes(self)` — [`L695`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L695) — Get KV pages tensor shape and sharding axes for this version.
  - `is_v2(self)` — [`L741`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L741) — Check if using v2 cache format.
  - `is_v3(self)` — [`L732`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L732) — Check if using v3 cache format.
  - `kv_head_packing(self)` — [`L608`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L608) — Get the packing factor for KV heads (elements per 32 bits).
  - `kvdtype(self)` — [`L597`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L597) — Get the JAX dtype for KV cache storage.
  - `storage_head_dim(self)` — [`L639`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L639) — Get aligned head dimension for storage.
  - `storage_num_combined_kv_heads(self)` — [`L617`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L617) — Get aligned combined KV head count for storage.
  - `storage_num_kv_groups(self)` — [`L630`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L630) — Get number of KV groups after packing.
  - `data_parallel_size` — [`L426`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L426)
  - `hbm_utilization` — [`L425`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L425)
  - `k_headdim` — [`L423`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L423)
  - `kv_head_shards` — [`L427`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L427)
  - `max_model_length` — [`L421`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L421)
  - `max_num_pages_per_req` — [`L430`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L430)
  - `max_num_reqs` — [`L433`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L433)
  - `max_num_tokens` — [`L432`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L432)
  - `num_hidden_layers` — [`L420`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L420)
  - `num_kv_heads` — [`L422`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L422)
  - `num_pages` — [`L429`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L429)
  - `num_slices_per_kv_cache_update_page` — [`L431`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L431)
  - `page_size` — [`L428`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L428)
  - `v_headdim` — [`L424`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L424)
  - `version` — [`L438`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L438)
  - `window_aware_max_num_batched_tokens` — [`L436`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L436)
  - `window_aware_max_num_seqs` — [`L434`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L434)
  - `window_aware_pages_per_request` — [`L435`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L435)
- protocol/private: `_kvdtype_str`[`L440`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L440), `_mixed_layer_configs`[`L441`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L441)
- uses (calls/refs, reference-scoped): [`BaseCacheConfig`](../_abstracts.md#BaseCacheConfig), [`create`](../mla_ragged_page/cache.md#MLARaggedPagesCacheConfig.create), [`ATTN_DP`](../../axis.md#ATTN_DP), [`MLARaggedPagesCacheConfig`](../mla_ragged_page/cache.md#MLARaggedPagesCacheConfig), [`TurboQuantRaggedPagesCacheConfig`](../turboquant_ragged_page/cache.md#TurboQuantRaggedPagesCacheConfig), [`create`](../turboquant_ragged_page/cache.md#TurboQuantRaggedPagesCacheConfig.create), [`_mesh_axis_size`](cache.md#_mesh_axis_size), [`get_shape_and_axes`](../mla_ragged_page/cache.md#MLARaggedPagesCacheConfig.get_shape_and_axes), [`cdiv`](cache.md#cdiv), [`align_to_multiple`](cache.md#align_to_multiple), [`_resolve_ragged_cache_layout`](cache.md#_resolve_ragged_cache_layout), [`_compute_free_hbm`](../mla_ragged_page/cache.md#MLARaggedPagesCacheConfig._compute_free_hbm), [`_dtype_to_string`](cache.md#_dtype_to_string), [`_attention_dp_axis`](cache.md#_attention_dp_axis), [`get_page_size_bytes`](cache.md#get_page_size_bytes), [`get_num_slices_per_kv_cache_update_page`](cache.md#get_num_slices_per_kv_cache_update_page), [`_canonicalize_dtype`](cache.md#_canonicalize_dtype), [`get_dtype_packing`](cache.md#get_dtype_packing), [`per_device_hbm_budget_bytes`](cache.md#per_device_hbm_budget_bytes), [`logger`](cache.md#logger)
- used by: [`_backbone_step`](../../inference/esurge/runners/executors/model_executor.md#ModelStepExecutor._backbone_step), [`get_compile_configurations`](../../inference/esurge/runners/execution_manager.md#ExecutionManager.get_compile_configurations), [`init_operations_cache_config`](../../infra/mixins/generation.md#EasyGenerationMixin.init_operations_cache_config), [`init_operations_cache`](../../infra/mixins/generation.md#EasyGenerationMixin.init_operations_cache), [`init_ragged_pages`](../../infra/mixins/generation.md#EasyGenerationMixin.init_ragged_pages), [`_compile_sampler_variant`](../../inference/esurge/runners/execution_manager.md#ExecutionManager._compile_sampler_variant), [`concatenate_to_cache`](cache.md#RaggedPagesCacheView.concatenate_to_cache), [`_create_mixed_standard_ragged_page_cache_configs`](../../infra/mixins/generation.md#_create_mixed_standard_ragged_page_cache_configs), [`init`](../turboquant_ragged_page/cache.md#TurboQuantRaggedPagesCacheView.init), [`BaseCacheConfig`](../_abstracts.md#BaseCacheConfig), [`build`](../_metadatabuilder.md#AttentionMetadataBuilder.build), [`compute_paged_attention_batch_fields_cpu`](../_metadatabuilder.md#AttentionMetadataBuilder.compute_paged_attention_batch_fields_cpu), [`init_all_layers`](../turboquant_ragged_page/cache.md#TurboQuantRaggedPagesCacheView.init_all_layers), [`initialize_kv_cache`](../../inference/esurge/runners/model_runner.md#eSurgeRunner.initialize_kv_cache), [`_build_backbone_fn`](../../inference/esurge/runners/executors/model_executor.md#ModelStepExecutor._build_backbone_fn), [`_compile_backbone_variant`](../../inference/esurge/runners/execution_manager.md#ExecutionManager._compile_backbone_variant), [`compile`](../../inference/esurge/runners/execution_manager.md#ExecutionManager.compile), [`_compute_ragged_from_batch_cpu`](../_metadatabuilder.md#AttentionMetadataBuilder._compute_ragged_from_batch_cpu), [`MLARaggedPagesCacheConfig`](../mla_ragged_page/cache.md#MLARaggedPagesCacheConfig), [`TurboQuantRaggedPagesCacheConfig`](../turboquant_ragged_page/cache.md#TurboQuantRaggedPagesCacheConfig), [`kv_pages`](../../inference/esurge/runners/execution_manager.md#ExecutionManager.kv_pages), [`_tq_signature`](../../infra/mixins/generation.md#EasyGenerationMixin._tq_signature), [`create_ragged_page_cache_config`](../../modules/deepseek_v2/modeling_deepseek.md#DeepseekV2ForCausalLM.create_ragged_page_cache_config), [`create_ragged_page_cache_config`](../../modules/deepseek_v3/modeling_deepseek.md#DeepseekV3ForCausalLM.create_ragged_page_cache_config), [`_compute_slot_mapping_v2`](../../inference/esurge/runners/executors/batch_preparer.md#BatchMetadataPreparer._compute_slot_mapping_v2), [`_compile_lm_head_variant`](../../inference/esurge/runners/execution_manager.md#ExecutionManager._compile_lm_head_variant), [`init_cache`](cache.md#RaggedPagesCache.init_cache), [`init`](cache.md#RaggedPagesCacheView.init), [`create`](../_abstracts.md#BaseCacheConfig.create), [`create_ragged_page_cache_config`](../../modules/glm4_moe_lite/modeling_glm4_moe_lite.md#Glm4MoeLiteForCausalLM.create_ragged_page_cache_config), [`create_ragged_page_cache_config`](../../modules/glm_moe_dsa/modeling_glm_moe_dsa.md#GlmMoeDsaForCausalLM.create_ragged_page_cache_config), [`_create_standard_ragged_page_cache_config`](../../infra/mixins/generation.md#EasyGenerationMixin._create_standard_ragged_page_cache_config), [`build_ragged_page_metadata`](../_metadatabuilder.md#AttentionMetadataBuilder.build_ragged_page_metadata), [`build_page_metadata`](../_metadatabuilder.md#AttentionMetadataBuilder.build_page_metadata), [`build_paged_metadata`](../_metadatabuilder.md#AttentionMetadataBuilder.build_paged_metadata), [`get_shape_and_axes`](../mla_ragged_page/cache.md#MLARaggedPagesCacheConfig.get_shape_and_axes), [`init_cache`](../mla_ragged_page/cache.md#MLARaggedPagesCache.init_cache), [`init_cache`](../turboquant_ragged_page/cache.md#TurboQuantRaggedPagesCache.init_cache), [`__init__`](../../inference/esurge/runners/execution_manager.md#ExecutionManager.__init__), [`__init__`](../../inference/esurge/runners/executors/model_executor.md#ModelStepExecutor.__init__)  (+22 more)

### `RaggedPagesCacheView`  ·  implements/extends BaseCacheView
- def: [`easydel/caching/ragged_page/cache.py:751`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L751) — documented in [easydel-caching-ragged_page-cache](../../../../concepts/easydel-caching-ragged_page-cache.md)
- doc: Represents the view of the Paged Attention KV cache for a single transformer layer.
- signature: `class RaggedPagesCacheView(BaseCacheView):`
- members:
  - `concatenate_to_cache(self, key: Float[Array, "batch seq_len num_key_heads head_dim"], value: Float[Array, "batch seq_len num_value_heads head_dim"], cache_metadata: RaggedPagesMetadata | OperationsMetadata)` — [`L825`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L825) — Update cache pages with new key-value pairs.
  - `flattened_kv_pages(self)` — [`L933`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L933) — Get KV pages in flattened format with interleaved K and V.
  - `init(cls, config: RaggedPagesCacheConfig, layer_index: int | None = None, *, mesh: Mesh | None = None, partition_manager: es.PartitionManager | None = None, quantizer: EasyQuantizer | None = None)` — [`L782`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L782) — Initialize a RaggedPagesCacheView from a cache config.
  - `key_pages(self)` — [`L954`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L954) — Extract key pages from interleaved KV storage.
  - `value_pages(self)` — [`L965`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L965) — Extract value pages from interleaved KV storage.
  - `kv_pages` — [`L771`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L771)
  - `layer_index` — [`L769`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L769)
  - `metadata` — [`L768`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L768)
  - `partition_manager` — [`L776`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L776)
- protocol/private: `__repr__`[`L975`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L975), `__str__`[`L978`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L978), `_update_fn`[`L872`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L872)
- uses (calls/refs, reference-scoped): [`OperationsMetadata`](../_abstracts.md#OperationsMetadata), [`RaggedPagesMetadata`](cache.md#RaggedPagesMetadata), [`RaggedPagesCacheConfig`](cache.md#RaggedPagesCacheConfig), [`EasyQuantizer`](../../layers/quantization/_quants.md#EasyQuantizer), [`MLARaggedPagesCacheView`](../mla_ragged_page/cache.md#MLARaggedPagesCacheView), [`BaseCacheView`](../_abstracts.md#BaseCacheView), [`init`](../turboquant_ragged_page/cache.md#TurboQuantRaggedPagesCacheView.init), [`get_shape_and_axes`](cache.md#RaggedPagesCacheConfig.get_shape_and_axes), [`unwrap_metadata`](../_abstracts.md#unwrap_metadata), [`ATTN_DP`](../../axis.md#ATTN_DP), [`TurboQuantRaggedPagesCacheView`](../turboquant_ragged_page/cache.md#TurboQuantRaggedPagesCacheView), [`init`](../mla_ragged_page/cache.md#MLARaggedPagesCacheView.init), [`EMPTY`](cache.md#EMPTY), [`kv_cache_update`](utils.md#kv_cache_update), [`_attention_dp_axis`](cache.md#_attention_dp_axis), [`concatenate_to_cache`](../turboquant_ragged_page/cache.md#TurboQuantRaggedPagesCacheView.concatenate_to_cache), [`kv_cache_update_jax`](utils.md#kv_cache_update_jax), [`kvdtype`](cache.md#RaggedPagesCacheConfig.kvdtype), [`MODE_PREFILL`](cache.md#MODE_PREFILL), [`flattened_kv_pages`](../mla_ragged_page/cache.md#MLARaggedPagesCacheView.flattened_kv_pages), [`is_v2`](cache.md#RaggedPagesCacheConfig.is_v2), [`PERMITTED_KV_KERNELS`](cache.md#PERMITTED_KV_KERNELS), [`key_pages`](../mla_ragged_page/cache.md#MLARaggedPagesCacheView.key_pages), [`value_pages`](../mla_ragged_page/cache.md#MLARaggedPagesCacheView.value_pages), [`page_size`](cache.md#RaggedPagesMetadata.page_size), [`num_kv_update_slices`](cache.md#RaggedPagesMetadata.num_kv_update_slices), [`num_slices_per_kv_cache_update_page`](cache.md#RaggedPagesMetadata.num_slices_per_kv_cache_update_page), [`slot_mapping`](cache.md#RaggedPagesMetadata.slot_mapping)
- used by: [`AttentionOutput`](../../operations/_attention_outputs.md#AttentionOutput), [`forward_mla`](../../modules/glm_moe_dsa/modeling_glm_moe_dsa.md#GlmMoeDsaAttention.forward_mla), [`forward`](../../layers/attention/_unified.md#UnifiedAttention.forward), [`forward_mla`](../../modules/glm4_moe_lite/modeling_glm4_moe_lite.md#Glm4MoeLiteAttention.forward_mla), [`__call__`](../../modules/gemma4/modeling_gemma4.md#Gemma4DecoderLayer.__call__), [`forward`](../../layers/attention/_flexible.md#FlexibleAttentionModule.forward), [`__all__`](../../modules/whisper/modeling_whisper.md#__all__), [`forward_mla`](../../layers/attention/_unified.md#UnifiedAttention.forward_mla), [`concatenate`](../../layers/attention/_flexible.md#AttentionModule.concatenate), [`forward_alibi`](../../layers/attention/_unified.md#UnifiedAttention.forward_alibi), [`init_operations_cache_config`](../../infra/mixins/generation.md#EasyGenerationMixin.init_operations_cache_config), [`init_operations_cache`](../../infra/mixins/generation.md#EasyGenerationMixin.init_operations_cache), [`__call__`](../../modules/falcon/modeling_falcon.md#FalconBlock.__call__), [`_forward_v2_turboquant`](../../operations/kernels/ragged_page_attention.md#_RaggedPageAttn._forward_v2_turboquant), [`_forward_v3_turboquant`](../../operations/kernels/ragged_page_attention.md#_RaggedPageAttn._forward_v3_turboquant), [`__call__`](../../modules/gidd/modeling_gidd.md#GiddAttention.__call__), [`__call__`](../../modules/stablelm/modeling_stablelm.md#StableLmAttention.__call__), [`forward`](../../modules/dbrx/modeling_dbrx.md#DbrxAttention.forward), [`__call__`](../../modules/grok_1/modeling_grok_1.md#Grok1Attention.__call__), [`forward`](../../modules/openelm/modeling_openelm.md#OpenELMMultiHeadCausalAttention.forward), [`_forward_v3_standard`](../../operations/kernels/ragged_page_attention.md#_RaggedPageAttn._forward_v3_standard), [`__call__`](../../modules/gemma3/modeling_gemma3.md#Gemma3DecoderLayer.__call__), [`__call__`](../../modules/phi3/modeling_phi3.md#Phi3Attention.__call__), [`__call__`](../../modules/xerxes/modeling_xerxes.md#XerxesDecoderLayer.__call__), [`__call__`](../../modules/gemma2/modeling_gemma2.md#Gemma2DecoderLayer.__call__), [`__call__`](../../modules/glm4/modeling_glm4.md#Glm4DecoderLayer.__call__), [`__call__`](../../modules/glm4v/modeling_glm4v.md#Glm4vTextDecoderLayer.__call__), [`__call__`](../../modules/gpt2/modeling_gpt2.md#GPT2Block.__call__), [`__call__`](../../modules/opt/modeling_opt.md#OPTAttention.__call__), [`__call__`](../../modules/phi3/modeling_phi3.md#Phi3DecoderLayer.__call__), [`__call__`](../../modules/roberta/modeling_roberta.md#RobertaSelfAttention.__call__), [`__call__`](../../modules/stablelm/modeling_stablelm.md#StableLmDecoderLayer.__call__), [`forward`](../../modules/qwen3_next/modeling_qwen3_next.md#Qwen3NextFullAttention.forward), [`forward_alibi`](../../modules/mosaic_mpt/modeling_mosaic.md#MptAttention.forward_alibi), [`get_requirements`](../../operations/kernels/ragged_page_attention.md#RaggedPageAttnV2.get_requirements), [`get_requirements`](../../operations/kernels/ragged_page_attention.md#RaggedPageAttnV3.get_requirements), [`__call__`](../../modules/gpt2/modeling_gpt2.md#GPT2Attention.__call__), [`__call__`](../../modules/opt/modeling_opt.md#OPTDecoderLayer.__call__), [`__call__`](../../modules/qwen3_vl_moe/modeling_qwen3_vl_moe.md#Qwen3VLMoeTextDecoderLayer.__call__), [`__call__`](../../modules/whisper/modeling_whisper.md#WhisperAttention.__call__)  (+85 more)

### `RaggedPagesMetadata`
- def: [`easydel/caching/ragged_page/cache.py:1157`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L1157) — documented in [easydel-caching-ragged_page-cache](../../../../concepts/easydel-caching-ragged_page-cache.md)
- members:
  - `create_empty(cls, num_tokens: int, max_num_reqs: int, max_pages: int, page_size: int = 128, version: tp.Literal["v3", "v2"] = "v3")` — [`L1216`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L1216) — Create empty metadata with proper shapes for compilation.
  - `context_lens` — [`L1199`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L1199)
  - `num_kv_update_slices` — [`L1207`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L1207)
  - `num_seqs` — [`L1201`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L1201)
  - `num_slices_per_kv_cache_update_page` — [`L1211`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L1211)
  - `page_size` — [`L1212`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L1212)
  - `pages_tables` — [`L1198`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L1198)
  - `position_ids` — [`L1204`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L1204)
  - `prefill_chunk_size` — [`L1213`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L1213)
  - `query_start_loc` — [`L1200`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L1200)
  - `request_distribution` — [`L1206`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L1206)
  - `slot_mapping` — [`L1203`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L1203)
  - `version` — [`L1209`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L1209)
- used by: [`BaseModuleProtocol`](../../infra/mixins/protocol.md#BaseModuleProtocol), [`forward_mla`](../../modules/glm_moe_dsa/modeling_glm_moe_dsa.md#GlmMoeDsaAttention.forward_mla), [`forward`](../../layers/attention/_unified.md#UnifiedAttention.forward), [`forward_mla`](../../modules/glm4_moe_lite/modeling_glm4_moe_lite.md#Glm4MoeLiteAttention.forward_mla), [`_backbone_step`](../../inference/esurge/runners/executors/model_executor.md#ModelStepExecutor._backbone_step), [`__call__`](../../modules/gemma4/modeling_gemma4.md#Gemma4DecoderLayer.__call__), [`forward`](../../layers/attention/_flexible.md#FlexibleAttentionModule.forward), [`__all__`](../../modules/whisper/modeling_whisper.md#__all__), [`forward_mla`](../../layers/attention/_unified.md#UnifiedAttention.forward_mla), [`concatenate`](../../layers/attention/_flexible.md#AttentionModule.concatenate), [`forward_alibi`](../../layers/attention/_unified.md#UnifiedAttention.forward_alibi), [`forward_native`](../../operations/kernels/paged_flash_attention.md#PagedFlashAttn.forward_native), [`mesh_call`](../../infra/mixins/protocol.md#BaseModuleProtocol.mesh_call), [`__call__`](../../infra/mixins/protocol.md#BaseModuleProtocol.__call__), [`__call__`](../../modules/falcon/modeling_falcon.md#FalconBlock.__call__), [`_forward_v2_turboquant`](../../operations/kernels/ragged_page_attention.md#_RaggedPageAttn._forward_v2_turboquant), [`_forward_v3_turboquant`](../../operations/kernels/ragged_page_attention.md#_RaggedPageAttn._forward_v3_turboquant), [`__call__`](../../modules/gidd/modeling_gidd.md#GiddAttention.__call__), [`__call__`](../../modules/stablelm/modeling_stablelm.md#StableLmAttention.__call__), [`forward`](../../modules/dbrx/modeling_dbrx.md#DbrxAttention.forward), [`__call__`](../../modules/minimax/modeling_minimax.md#MiniMaxModel.__call__), [`forward_core`](../../operations/kernels/multi_latent_ragged_page_attention.md#MultiLatentRaggedPageAttn.forward_core), [`forward_core`](../../operations/kernels/multi_latent_ragged_page_attention.md#MultiLatentRaggedPageAttnV2.forward_core), [`__call__`](../../modules/gemma4/modeling_gemma4.md#Gemma4TextModel.__call__), [`__call__`](../../modules/grok_1/modeling_grok_1.md#Grok1Attention.__call__), [`__call__`](../../modules/minimax/modeling_minimax.md#MiniMaxDecoderLayer.__call__), [`__call__`](../../modules/qwen3_vl_moe/modeling_qwen3_vl_moe.md#Qwen3VLMoeTextModel.__call__), [`forward`](../../modules/openelm/modeling_openelm.md#OpenELMMultiHeadCausalAttention.forward), [`_forward_v3_standard`](../../operations/kernels/ragged_page_attention.md#_RaggedPageAttn._forward_v3_standard), [`forward_native`](../../operations/kernels/unified_attention.md#UnifiedAttn.forward_native), [`__call__`](../../modules/gemma3/modeling_gemma3.md#Gemma3DecoderLayer.__call__), [`__call__`](../../modules/gpt_oss/modeling_gpt_oss.md#GptOssModel.__call__), [`__call__`](../../modules/kimi_linear/modeling_kimi_linear.md#KimiLinearModel.__call__), [`__call__`](../../modules/mixtral/modeling_mixtral.md#MixtralModel.__call__), [`__call__`](../../modules/opt/modeling_opt.md#OPTDecoder.__call__), [`__call__`](../../modules/phi3/modeling_phi3.md#Phi3Attention.__call__), [`__call__`](../../modules/qwen2_moe/modeling_qwen2_moe.md#Qwen2MoeModel.__call__), [`__call__`](../../modules/qwen3_next/modeling_qwen3_next.md#Qwen3NextModel.__call__), [`__call__`](../../modules/arctic/modeling_arctic.md#ArcticModel.__call__), [`__call__`](../../modules/dbrx/modeling_dbrx.md#DbrxModel.__call__)  (+220 more)

## Functions
- `_attention_dp_axis(partition_manager: PartitionManager)` — [`L87`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L87) — Resolve the concrete mesh axis used for KV-page data parallelism.
- `_canonicalize_dtype(dtype: jnp.dtype)` — [`L204`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L204) — Normalize dtype objects/classes to the scalar type form used by eformer maps.
- `_dtype_to_string(dtype: jnp.dtype)` — [`L209`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L209) — Convert a dtype to the stable cache-config string representation.
- `_mesh_axis_size(mesh: Mesh, axis: str | tuple[str, ...] | list[str] | None)` — [`L174`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L174) — Return product of mesh sizes for a semantic axis mapping.
- `_resolve_ragged_cache_layout(kvdtype: jnp.dtype, *, version: str, num_kv_heads: int, k_headdim: int, kv_head_shards: int)` — [`L256`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L256) — Resolve cache dtype and effective KV-head shard count for ragged caches.
- `_select_compatible_v3_kv_cache_dtype(kvdtype: jnp.dtype, *, num_kv_heads: int, k_headdim: int, kv_head_shards: int)` — [`L215`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L215) — Upcast packed v3 cache storage when TP sharding would otherwise be invalid.
- `_storage_groups(dtype: jnp.dtype)` — [`L227`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L227)
- `_storage_num_combined_kv_heads_for_dtype(num_kv_heads: int, k_headdim: int, kvdtype: jnp.dtype)` — [`L196`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L196) — Return packed combined KV heads for a given cache dtype.
- `align_to_multiple(value: int, multiple: int)` — [`L161`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L161) — Align a value up to the nearest multiple.
- `cdiv(a: int, b: int)` — [`L92`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L92) — Ceiling division: compute ceil(a / b) using integer arithmetic.
- `get_dtype_packing(dtype: jnp.dtype)` — [`L143`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L143) — Get the packing factor for a dtype (elements per 32 bits).
- `get_num_slices_per_kv_cache_update_page(page_size_bytes: int)` — [`L119`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L119) — Calculate the number of update slices per processing page.
- `get_page_size_bytes(page_size: int, num_kv_heads: int, head_size: int, kv_cache_dtype: jnp.dtype)` — [`L304`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L304) — Calculate the size in bytes of one page of the KV cache.
- `per_device_hbm_budget_bytes(util: float = 0.9, mode: str = "free", safety_margin: int = 256 << 20)` — [`L332`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L332) — Calculate available HBM budget per device for cache allocation.
- `previous_power_of_2(n: int)` — [`L105`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L105) — Find the largest power of 2 less than or equal to n.

## Module values
- `EMPTY` — [`L78`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L78)
- `KV_HEAD` — [`L79`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L79)
- `MODE_PREFILL` — [`L80`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L80)
- `PERMITTED_KV_KERNELS` — [`L84`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L84)
- `logger` — [`L82`](../../../../../../../raw/code/EasyDeL/easydel/caching/ragged_page/cache.py#L82)


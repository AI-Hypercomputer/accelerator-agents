---
title: 'Module: easydel/caching/mla_ragged_page/cache.py'
type: catalog
provenance: extracted
module: easydel/caching/mla_ragged_page/cache.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.caching.mla_ragged_page.cache`/
symbols:
  MLARaggedPagesCacheView: MLARaggedPagesCacheView#
  MLARaggedPagesCacheConfig.create: MLARaggedPagesCacheConfig#create().
  MLARaggedPagesCacheConfig: MLARaggedPagesCacheConfig#
  MLARaggedPagesCacheConfig.get_shape_and_axes: MLARaggedPagesCacheConfig#get_shape_and_axes().
  MLARaggedPagesCache.init_cache: MLARaggedPagesCache#init_cache().
  MLARaggedPagesCacheView.init: MLARaggedPagesCacheView#init().
  MLARaggedPagesCacheConfig._compute_free_hbm: MLARaggedPagesCacheConfig#_compute_free_hbm().
  MLARaggedPagesCacheConfig.kv_dim_padded: MLARaggedPagesCacheConfig#kv_dim_padded().
  MLARaggedPagesCacheConfig.page_size_per_kv_packing: MLARaggedPagesCacheConfig#page_size_per_kv_packing().
  MLARaggedPagesCache: MLARaggedPagesCache#
  MLARaggedPagesCache.metadata: MLARaggedPagesCache#metadata().
  MLARaggedPagesCacheConfig.kv_packing: MLARaggedPagesCacheConfig#kv_packing().
  MLARaggedPagesCacheView.kv_pages: MLARaggedPagesCacheView#kv_pages.
  MLARaggedPagesCacheView.__repr__: MLARaggedPagesCacheView#__repr__().
  MLARaggedPagesCache.views: MLARaggedPagesCache#views.
  MLARaggedPagesCacheConfig.kv_dim: MLARaggedPagesCacheConfig#kv_dim().
  MLARaggedPagesCacheConfig.kv_lora_rank: MLARaggedPagesCacheConfig#kv_lora_rank().
  MLARaggedPagesCacheConfig.qk_rope_head_dim: MLARaggedPagesCacheConfig#qk_rope_head_dim().
  MLARaggedPagesCacheView.metadata: MLARaggedPagesCacheView#metadata.
  MLARaggedPagesCacheView.flattened_kv_pages: MLARaggedPagesCacheView#flattened_kv_pages().
  logger: logger.
  MLARaggedPagesCacheView.key_pages: MLARaggedPagesCacheView#key_pages().
  MLARaggedPagesCacheView.value_pages: MLARaggedPagesCacheView#value_pages().
  MLARaggedPagesCacheView.__str__: MLARaggedPagesCacheView#__str__.
---
# Module: [`easydel/caching/mla_ragged_page/cache.py`](../../../../../../../raw/code/EasyDeL/easydel/caching/mla_ragged_page/cache.py)

## Classes
### `MLARaggedPagesCache`  ·  implements/extends RaggedPagesCache
- def: [`easydel/caching/mla_ragged_page/cache.py:396`](../../../../../../../raw/code/EasyDeL/easydel/caching/mla_ragged_page/cache.py#L396)
- doc: Top-level container holding one `MLARaggedPagesCacheView` per transformer layer.
- signature: `class MLARaggedPagesCache(RaggedPagesCache):`
- members:
  - `init_cache(cls, mesh: Mesh, config: MLARaggedPagesCacheConfig, partition_manager: es.PartitionManager, quantizer: EasyQuantizer | None = None)` — [`L419`](../../../../../../../raw/code/EasyDeL/easydel/caching/mla_ragged_page/cache.py#L419) — Allocate an MLA ragged page cache for all transformer layers.
  - `metadata(self)` — [`L412`](../../../../../../../raw/code/EasyDeL/easydel/caching/mla_ragged_page/cache.py#L412) — Return the shared config from the last layer view, or ``None`` if empty.
  - `views` — [`L409`](../../../../../../../raw/code/EasyDeL/easydel/caching/mla_ragged_page/cache.py#L409)
- uses (calls/refs, reference-scoped): [`RaggedPagesCache`](../ragged_page/cache.md#RaggedPagesCache), [`EasyQuantizer`](../../layers/quantization/_quants.md#EasyQuantizer), [`MLARaggedPagesCacheView`](cache.md#MLARaggedPagesCacheView), [`MLARaggedPagesCacheConfig`](cache.md#MLARaggedPagesCacheConfig), [`init`](cache.md#MLARaggedPagesCacheView.init), [`metadata`](cache.md#MLARaggedPagesCacheView.metadata), [`num_hidden_layers`](../ragged_page/cache.md#RaggedPagesCacheConfig.num_hidden_layers)
- used by: [`RaggedPagesCache`](../ragged_page/cache.md#RaggedPagesCache), [`init_ragged_pages`](../../infra/mixins/generation.md#EasyGenerationMixin.init_ragged_pages), [`init_cache`](../ragged_page/cache.md#RaggedPagesCache.init_cache), [`metadata`](../ragged_page/cache.md#RaggedPagesCache.metadata)

### `MLARaggedPagesCacheConfig`  ·  implements/extends RaggedPagesCacheConfig
- def: [`easydel/caching/mla_ragged_page/cache.py:62`](../../../../../../../raw/code/EasyDeL/easydel/caching/mla_ragged_page/cache.py#L62)
- doc: Configuration for Multi-Latent Attention (MLA) compressed ragged pages cache.
- signature: `class MLARaggedPagesCacheConfig(RaggedPagesCacheConfig):`
- members:
  - `_compute_free_hbm(mesh: Mesh, partition_manager: PartitionManager, hbm_utilization: float)` — [`L80`](../../../../../../../raw/code/EasyDeL/easydel/caching/mla_ragged_page/cache.py#L80) — Compute free HBM budget available for MLA cache pages.
  - `create(cls, mesh: Mesh, partition_manager: PartitionManager, kvdtype: jnp.dtype, num_hidden_layers: int, num_kv_heads: int, max_model_length: int, *, kv_lora_rank: int | None = None, qk_rope_head_dim: int | None = None, kv_head_dim_size: int | None = None, k_headdim: int | None = None, v_headdim: int | None = None, hbm_utilization: float = 0.9, page_size: int = 128, version: tp.Literal["v1"] = "v1")` — [`L107`](../../../../../../../raw/code/EasyDeL/easydel/caching/mla_ragged_page/cache.py#L107) — Create an MLA ragged-page cache config from model hyperparameters.
  - `get_shape_and_axes(self)` — [`L270`](../../../../../../../raw/code/EasyDeL/easydel/caching/mla_ragged_page/cache.py#L270) — Return the page tensor shape and corresponding sharding axes.
  - `kv_dim(self)` — [`L246`](../../../../../../../raw/code/EasyDeL/easydel/caching/mla_ragged_page/cache.py#L246) — Total unpadded KV dimension (``kv_lora_rank + qk_rope_head_dim``).
  - `kv_dim_padded(self)` — [`L251`](../../../../../../../raw/code/EasyDeL/easydel/caching/mla_ragged_page/cache.py#L251) — Total KV dimension after padding each component to a multiple of 128.
  - `kv_lora_rank(self)` — [`L236`](../../../../../../../raw/code/EasyDeL/easydel/caching/mla_ragged_page/cache.py#L236) — Low-rank KV projection dimension (stored as ``k_headdim``).
  - `kv_packing(self)` — [`L261`](../../../../../../../raw/code/EasyDeL/easydel/caching/mla_ragged_page/cache.py#L261) — Number of elements packed per dtype unit (e.g. 2 for ``bfloat16``).
  - `page_size_per_kv_packing(self)` — [`L266`](../../../../../../../raw/code/EasyDeL/easydel/caching/mla_ragged_page/cache.py#L266) — Page size divided by the dtype packing factor, rounded up.
  - `qk_rope_head_dim(self)` — [`L241`](../../../../../../../raw/code/EasyDeL/easydel/caching/mla_ragged_page/cache.py#L241) — RoPE head dimension for queries/keys (stored as ``v_headdim``).
- uses (calls/refs, reference-scoped): [`RaggedPagesCacheConfig`](../ragged_page/cache.md#RaggedPagesCacheConfig), [`ATTN_DP`](../../axis.md#ATTN_DP), [`_mesh_axis_size`](../ragged_page/cache.md#_mesh_axis_size), [`resolve_attention_data_parallel_axis`](../../axis.md#resolve_attention_data_parallel_axis), [`cdiv`](../ragged_page/cache.md#cdiv), [`align_to_multiple`](../ragged_page/cache.md#align_to_multiple), [`page_size`](../ragged_page/cache.md#RaggedPagesCacheConfig.page_size), [`k_headdim`](../ragged_page/cache.md#RaggedPagesCacheConfig.k_headdim), [`get_num_slices_per_kv_cache_update_page`](../ragged_page/cache.md#get_num_slices_per_kv_cache_update_page), [`get_dtype_packing`](../ragged_page/cache.md#get_dtype_packing), [`per_device_hbm_budget_bytes`](../ragged_page/cache.md#per_device_hbm_budget_bytes), [`kvdtype`](../ragged_page/cache.md#RaggedPagesCacheConfig.kvdtype), [`num_pages`](../ragged_page/cache.md#RaggedPagesCacheConfig.num_pages), [`data_parallel_size`](../ragged_page/cache.md#RaggedPagesCacheConfig.data_parallel_size), [`logger`](cache.md#logger), [`v_headdim`](../ragged_page/cache.md#RaggedPagesCacheConfig.v_headdim)
- used by: [`init_operations_cache_config`](../../infra/mixins/generation.md#EasyGenerationMixin.init_operations_cache_config), [`RaggedPagesCacheConfig`](../ragged_page/cache.md#RaggedPagesCacheConfig), [`init_operations_cache`](../../infra/mixins/generation.md#EasyGenerationMixin.init_operations_cache), [`init_ragged_pages`](../../infra/mixins/generation.md#EasyGenerationMixin.init_ragged_pages), [`create`](../ragged_page/cache.md#RaggedPagesCacheConfig.create), [`get_shape_and_axes`](../ragged_page/cache.md#RaggedPagesCacheConfig.get_shape_and_axes), [`create_ragged_page_cache_config`](../../modules/deepseek_v2/modeling_deepseek.md#DeepseekV2ForCausalLM.create_ragged_page_cache_config), [`create_ragged_page_cache_config`](../../modules/deepseek_v3/modeling_deepseek.md#DeepseekV3ForCausalLM.create_ragged_page_cache_config), [`init_cache`](cache.md#MLARaggedPagesCache.init_cache), [`_create_mla_ragged_page_cache_config`](../../infra/mixins/generation.md#EasyGenerationMixin._create_mla_ragged_page_cache_config), [`_compute_free_hbm`](../ragged_page/cache.md#RaggedPagesCacheConfig._compute_free_hbm), [`init`](cache.md#MLARaggedPagesCacheView.init), [`_create_mla_ragged_page_cache_config`](../../modules/glm4_moe_lite/modeling_glm4_moe_lite.md#Glm4MoeLiteForCausalLM._create_mla_ragged_page_cache_config), [`metadata`](cache.md#MLARaggedPagesCache.metadata), [`metadata`](cache.md#MLARaggedPagesCacheView.metadata)

### `MLARaggedPagesCacheView`  ·  implements/extends RaggedPagesCacheView
- def: [`easydel/caching/mla_ragged_page/cache.py:291`](../../../../../../../raw/code/EasyDeL/easydel/caching/mla_ragged_page/cache.py#L291)
- doc: Per-layer view into the MLA ragged page cache.
- signature: `class MLARaggedPagesCacheView(RaggedPagesCacheView):`
- members:
  - `flattened_kv_pages(self)` — [`L353`](../../../../../../../raw/code/EasyDeL/easydel/caching/mla_ragged_page/cache.py#L353) — Materialize and return the KV page buffer as a concrete JAX array.
  - `init(cls, config: MLARaggedPagesCacheConfig, layer_index: int | None = None, *, mesh: Mesh | None = None, partition_manager: es.PartitionManager | None = None, quantizer: EasyQuantizer | None = None)` — [`L311`](../../../../../../../raw/code/EasyDeL/easydel/caching/mla_ragged_page/cache.py#L311) — Allocate a single-layer MLA ragged page cache view.
  - `key_pages(self)` — [`L369`](../../../../../../../raw/code/EasyDeL/easydel/caching/mla_ragged_page/cache.py#L369) — Return the key page buffer (alias for :meth:`flattened_kv_pages`).
  - `value_pages(self)` — [`L378`](../../../../../../../raw/code/EasyDeL/easydel/caching/mla_ragged_page/cache.py#L378) — Return the value page buffer (alias for :meth:`flattened_kv_pages`).
  - `kv_pages` — [`L308`](../../../../../../../raw/code/EasyDeL/easydel/caching/mla_ragged_page/cache.py#L308)
  - `metadata` — [`L307`](../../../../../../../raw/code/EasyDeL/easydel/caching/mla_ragged_page/cache.py#L307)
- protocol/private: `__repr__`[`L386`](../../../../../../../raw/code/EasyDeL/easydel/caching/mla_ragged_page/cache.py#L386), `__str__`[`L392`](../../../../../../../raw/code/EasyDeL/easydel/caching/mla_ragged_page/cache.py#L392)
- uses (calls/refs, reference-scoped): [`RaggedPagesCacheView`](../ragged_page/cache.md#RaggedPagesCacheView), [`EasyQuantizer`](../../layers/quantization/_quants.md#EasyQuantizer), [`MLARaggedPagesCacheConfig`](cache.md#MLARaggedPagesCacheConfig), [`get_shape_and_axes`](cache.md#MLARaggedPagesCacheConfig.get_shape_and_axes), [`kvdtype`](../ragged_page/cache.md#RaggedPagesCacheConfig.kvdtype), [`layer_index`](../ragged_page/cache.md#RaggedPagesCacheView.layer_index)
- used by: [`RaggedPagesCacheView`](../ragged_page/cache.md#RaggedPagesCacheView), [`forward_mla`](../../modules/glm_moe_dsa/modeling_glm_moe_dsa.md#GlmMoeDsaAttention.forward_mla), [`forward_mla`](../../modules/glm4_moe_lite/modeling_glm4_moe_lite.md#Glm4MoeLiteAttention.forward_mla), [`forward_mla`](../../layers/attention/_unified.md#UnifiedAttention.forward_mla), [`init_operations_cache_config`](../../infra/mixins/generation.md#EasyGenerationMixin.init_operations_cache_config), [`init_operations_cache`](../../infra/mixins/generation.md#EasyGenerationMixin.init_operations_cache), [`forward_core`](../../operations/kernels/multi_latent_ragged_page_attention.md#MultiLatentRaggedPageAttn.forward_core), [`forward_core`](../../operations/kernels/multi_latent_ragged_page_attention.md#MultiLatentRaggedPageAttnV2.forward_core), [`get_requirements`](../../operations/kernels/multi_latent_ragged_page_attention.md#MultiLatentRaggedPageAttn.get_requirements), [`get_requirements`](../../operations/kernels/multi_latent_ragged_page_attention.md#MultiLatentRaggedPageAttnV2.get_requirements), [`get_operations_cache_view`](../../infra/mixins/operation_cache.md#OperationCacheMixin.get_operations_cache_view), [`init`](../ragged_page/cache.md#RaggedPagesCacheView.init), [`init_cache`](cache.md#MLARaggedPagesCache.init_cache), [`_resolve_parallel_hybrid_attention_view_class`](../../infra/mixins/generation.md#EasyGenerationMixin._resolve_parallel_hybrid_attention_view_class), [`forward_native`](../../operations/kernels/multi_latent_ragged_page_attention.md#MultiLatentRaggedPageAttn.forward_native), [`flattened_kv_pages`](../ragged_page/cache.md#RaggedPagesCacheView.flattened_kv_pages), [`__call__`](../../operations/kernels/multi_latent_ragged_page_attention.md#MultiLatentRaggedPageAttn.__call__), [`_dp_page_axis`](../../operations/kernels/multi_latent_ragged_page_attention.md#_dp_page_axis), [`metadata`](cache.md#MLARaggedPagesCache.metadata), [`key_pages`](../ragged_page/cache.md#RaggedPagesCacheView.key_pages), [`views`](cache.md#MLARaggedPagesCache.views), [`value_pages`](../ragged_page/cache.md#RaggedPagesCacheView.value_pages)

## Module values
- `logger` — [`L58`](../../../../../../../raw/code/EasyDeL/easydel/caching/mla_ragged_page/cache.py#L58)


---
title: 'Module: easydel/caching/_abstracts.py'
type: catalog
provenance: extracted
module: easydel/caching/_abstracts.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.caching._abstracts`/
symbols:
  OperationsMetadata: OperationsMetadata#
  BaseCacheView: BaseCacheView#
  BaseCacheConfig: BaseCacheConfig#
  BaseCache: BaseCache#
  unwrap_metadata: unwrap_metadata().
  BaseCacheView.init: BaseCacheView#init().
  BaseRunTimeMetadata: BaseRunTimeMetadata#
  BaseCache.init_cache: BaseCache#init_cache().
  BaseCacheConfig.create: BaseCacheConfig#create().
  BaseCacheView.concatenate_to_cache: BaseCacheView#concatenate_to_cache().
  BaseCache.init_empty: BaseCache#init_empty().
  OperationsMetadata.get_inner: OperationsMetadata#get_inner().
  OperationsMetadata.cache_type: OperationsMetadata#cache_type().
  OperationsMetadata.hybrid: OperationsMetadata#hybrid.
  OperationsMetadata.for_hybrid: OperationsMetadata#for_hybrid().
  OperationsMetadata.for_ragged: OperationsMetadata#for_ragged().
  BaseCache.views: BaseCache#views.
  OperationsMetadata.transformer: OperationsMetadata#transformer.
  OperationsMetadata.for_transformer: OperationsMetadata#for_transformer().
  OperationsMetadata.for_recurrent: OperationsMetadata#for_recurrent().
  OperationsMetadata.ragged: OperationsMetadata#ragged.
  OperationsMetadata.recurrent: OperationsMetadata#recurrent.
  BaseCacheView.metadata: BaseCacheView#metadata.
  BaseCache.__getitem__: BaseCache#__getitem__().
  BaseCache.__setitem__: BaseCache#__setitem__().
  BaseCache.__len__: BaseCache#__len__().
  BaseCacheView.layer_index: BaseCacheView#layer_index.
---
# Module: [`easydel/caching/_abstracts.py`](../../../../../../raw/code/EasyDeL/easydel/caching/_abstracts.py)

## Classes
### `BaseCache`  ·  implements/extends ABC
- def: [`easydel/caching/_abstracts.py:585`](../../../../../../raw/code/EasyDeL/easydel/caching/_abstracts.py#L585) — documented in [easydel-caching-_abstracts](../../../concepts/easydel-caching-_abstracts.md)
- doc: Abstract base class for multi-layer cache orchestration.
- signature: `class BaseCache(ABC):`
- members:
  - `__getitem__(self, index)` — [`L710`](../../../../../../raw/code/EasyDeL/easydel/caching/_abstracts.py#L710) — Access cache views by index using subscript notation.
  - `__len__(self)` — [`L768`](../../../../../../raw/code/EasyDeL/easydel/caching/_abstracts.py#L768) — Return the number of cache views in this container.
  - `__setitem__(self, index, value)` — [`L739`](../../../../../../raw/code/EasyDeL/easydel/caching/_abstracts.py#L739) — Update cache views by index using subscript notation.
  - `init_cache(cls, metadata: BaseCacheConfig, *args, **kwargs)` — [`L626`](../../../../../../raw/code/EasyDeL/easydel/caching/_abstracts.py#L626) — Initialize a complete cache with views for all layers.
  - `init_empty(cls, *args, **kwargs)` — [`L680`](../../../../../../raw/code/EasyDeL/easydel/caching/_abstracts.py#L680) — Initialize an empty cache container without allocated storage.
  - `views` — [`L622`](../../../../../../raw/code/EasyDeL/easydel/caching/_abstracts.py#L622)
- uses (calls/refs, reference-scoped): [`TransformerCache`](transformer/cache.md#TransformerCache), [`HybridCache`](hybrid/cache.md#HybridCache), [`RaggedPagesCache`](ragged_page/cache.md#RaggedPagesCache), [`init_empty`](transformer/cache.md#TransformerCache.init_empty), [`RecurrentCache`](recurrent/cache.md#RecurrentCache), [`BaseCacheView`](_abstracts.md#BaseCacheView), [`BaseCacheConfig`](_abstracts.md#BaseCacheConfig), [`UnifiedAttentionCache`](unified_attention/cache.md#UnifiedAttentionCache), [`init_cache`](ragged_page/cache.md#RaggedPagesCache.init_cache), [`init_cache`](unified_attention/cache.md#UnifiedAttentionCache.init_cache), [`init_cache`](transformer/cache.md#TransformerCache.init_cache), [`init_cache`](recurrent/cache.md#RecurrentCache.init_cache), [`init_cache`](hybrid/cache.md#HybridCache.init_cache), [`KDACache`](kda/cache.md#KDACache), [`init_cache`](kda/cache.md#KDACache.init_cache), [`LightningCache`](lightning/cache.md#LightningCache), [`init_cache`](lightning/cache.md#LightningCache.init_cache), [`init_empty`](hybrid/cache.md#HybridCache.init_empty), [`init_empty`](recurrent/cache.md#RecurrentCache.init_empty), [`init_empty`](kda/cache.md#KDACache.init_empty), [`init_empty`](lightning/cache.md#LightningCache.init_empty), [`init_empty`](unified_attention/cache.md#UnifiedAttentionCache.init_empty), [`init_empty`](ragged_page/cache.md#RaggedPagesCache.init_empty)
- used by: [`TransformerCache`](transformer/cache.md#TransformerCache), [`HybridCache`](hybrid/cache.md#HybridCache), [`RaggedPagesCache`](ragged_page/cache.md#RaggedPagesCache), [`RecurrentCache`](recurrent/cache.md#RecurrentCache), [`UnifiedAttentionCache`](unified_attention/cache.md#UnifiedAttentionCache), [`KDACache`](kda/cache.md#KDACache), [`LightningCache`](lightning/cache.md#LightningCache)

### `BaseCacheConfig`  ·  implements/extends ABC
- def: [`easydel/caching/_abstracts.py:66`](../../../../../../raw/code/EasyDeL/easydel/caching/_abstracts.py#L66) — documented in [easydel-caching-_abstracts](../../../concepts/easydel-caching-_abstracts.md)
- doc: Abstract base class defining the interface for cache metadata.
- signature: `class BaseCacheConfig(ABC):`
- members:
  - `create(cls, *args, **kwargs)` — [`L97`](../../../../../../raw/code/EasyDeL/easydel/caching/_abstracts.py#L97) — Factory method to create and validate a metadata instance.
- uses (calls/refs, reference-scoped): [`RaggedPagesCacheConfig`](ragged_page/cache.md#RaggedPagesCacheConfig), [`UnifiedAttentionCacheConfig`](unified_attention/cache.md#UnifiedAttentionCacheConfig), [`create`](ragged_page/cache.md#RaggedPagesCacheConfig.create), [`RecurrentCacheConfig`](recurrent/cache.md#RecurrentCacheConfig), [`TransformerCacheConfig`](transformer/cache.md#TransformerCacheConfig), [`create`](unified_attention/cache.md#UnifiedAttentionCacheConfig.create), [`create`](hybrid/cache.md#HybridCacheConfig.create), [`KDACacheConfig`](kda/cache.md#KDACacheConfig), [`LightningCacheConfig`](lightning/cache.md#LightningCacheConfig), [`create`](recurrent/cache.md#RecurrentCacheConfig.create), [`HybridCacheConfig`](hybrid/cache.md#HybridCacheConfig), [`create`](transformer/cache.md#TransformerCacheConfig.create), [`create`](kda/cache.md#KDACacheConfig.create), [`create`](lightning/cache.md#LightningCacheConfig.create)
- used by: [`RaggedPagesCacheConfig`](ragged_page/cache.md#RaggedPagesCacheConfig), [`UnifiedAttentionCacheConfig`](unified_attention/cache.md#UnifiedAttentionCacheConfig), [`RecurrentCacheConfig`](recurrent/cache.md#RecurrentCacheConfig), [`init`](_abstracts.md#BaseCacheView.init), [`init_cache`](_abstracts.md#BaseCache.init_cache), [`TransformerCacheConfig`](transformer/cache.md#TransformerCacheConfig), [`KDACacheConfig`](kda/cache.md#KDACacheConfig), [`LightningCacheConfig`](lightning/cache.md#LightningCacheConfig), [`HybridCacheConfig`](hybrid/cache.md#HybridCacheConfig), [`init`](hybrid/cache.md#ParallelHybridCacheView.init), [`metadata`](_abstracts.md#BaseCacheView.metadata)

### `BaseCacheView`  ·  implements/extends ABC
- def: [`easydel/caching/_abstracts.py:438`](../../../../../../raw/code/EasyDeL/easydel/caching/_abstracts.py#L438) — documented in [easydel-caching-_abstracts](../../../concepts/easydel-caching-_abstracts.md)
- doc: Abstract base class for single-layer cache management.
- signature: `class BaseCacheView(ABC):`
- members:
  - `concatenate_to_cache(self, *args, **kwargs)` — [`L528`](../../../../../../raw/code/EasyDeL/easydel/caching/_abstracts.py#L528) — Update the cache with new computed states.
  - `init(cls, metadata: BaseCacheConfig, *args, **kwargs)` — [`L481`](../../../../../../raw/code/EasyDeL/easydel/caching/_abstracts.py#L481) — Initialize a new cache view for a single layer.
  - `layer_index` — [`L477`](../../../../../../raw/code/EasyDeL/easydel/caching/_abstracts.py#L477)
  - `metadata` — [`L476`](../../../../../../raw/code/EasyDeL/easydel/caching/_abstracts.py#L476)
- uses (calls/refs, reference-scoped): [`RaggedPagesCacheView`](ragged_page/cache.md#RaggedPagesCacheView), [`TransformerCacheView`](transformer/cache.md#TransformerCacheView), [`RecurrentCacheView`](recurrent/cache.md#RecurrentCacheView), [`concatenate_to_cache`](transformer/cache.md#TransformerCacheView.concatenate_to_cache), [`UnifiedAttentionCacheView`](unified_attention/cache.md#UnifiedAttentionCacheView), [`init`](transformer/cache.md#TransformerCacheView.init), [`concatenate_to_cache`](ragged_page/cache.md#RaggedPagesCacheView.concatenate_to_cache), [`init`](hybrid/cache.md#HybridCacheView.init), [`concatenate_to_cache`](hybrid/cache.md#HybridCacheView.concatenate_to_cache), [`BaseCacheConfig`](_abstracts.md#BaseCacheConfig), [`init`](unified_attention/cache.md#UnifiedAttentionCacheView.init), [`ParallelHybridCacheView`](hybrid/cache.md#ParallelHybridCacheView), [`concatenate_to_cache`](unified_attention/cache.md#UnifiedAttentionCacheView.concatenate_to_cache), [`KDACacheView`](kda/cache.md#KDACacheView), [`concatenate_to_cache`](recurrent/cache.md#RecurrentCacheView.concatenate_to_cache), [`concatenate_to_cache`](kda/cache.md#KDACacheView.concatenate_to_cache), [`init`](recurrent/cache.md#RecurrentCacheView.init), [`init`](kda/cache.md#KDACacheView.init), [`init`](ragged_page/cache.md#RaggedPagesCacheView.init), [`HybridCacheView`](hybrid/cache.md#HybridCacheView), [`concatenate_to_cache`](hybrid/cache.md#ParallelHybridCacheView.concatenate_to_cache), [`LightningCacheView`](lightning/cache.md#LightningCacheView), [`init`](lightning/cache.md#LightningCacheView.init), [`init`](hybrid/cache.md#ParallelHybridCacheView.init), [`concatenate_to_cache`](lightning/cache.md#LightningCacheView.concatenate_to_cache)
- used by: [`RaggedPagesCacheView`](ragged_page/cache.md#RaggedPagesCacheView), [`TransformerCacheView`](transformer/cache.md#TransformerCacheView), [`RecurrentCacheView`](recurrent/cache.md#RecurrentCacheView), [`UnifiedAttentionCacheView`](unified_attention/cache.md#UnifiedAttentionCacheView), [`ParallelHybridCacheView`](hybrid/cache.md#ParallelHybridCacheView), [`KDACacheView`](kda/cache.md#KDACacheView), [`HybridCacheView`](hybrid/cache.md#HybridCacheView), [`LightningCacheView`](lightning/cache.md#LightningCacheView), [`views`](_abstracts.md#BaseCache.views)

### `BaseRunTimeMetadata`
- def: [`easydel/caching/_abstracts.py:143`](../../../../../../raw/code/EasyDeL/easydel/caching/_abstracts.py#L143)
- doc: Abstract base class for runtime metadata used during cache operations.
- signature: `class BaseRunTimeMetadata:`
- uses (calls/refs, reference-scoped): [`OperationsMetadata`](_abstracts.md#OperationsMetadata), [`RecurrentMetadata`](recurrent/cache.md#RecurrentMetadata), [`HybridMetadata`](hybrid/cache.md#HybridMetadata), [`KDAMetadata`](kda/cache.md#KDAMetadata), [`LightningMetadata`](lightning/cache.md#LightningMetadata)
- used by: [`OperationsMetadata`](_abstracts.md#OperationsMetadata), [`TransformerMetadata`](transformer/cache.md#TransformerMetadata), [`RecurrentMetadata`](recurrent/cache.md#RecurrentMetadata), [`get_inner`](_abstracts.md#OperationsMetadata.get_inner), [`HybridMetadata`](hybrid/cache.md#HybridMetadata), [`KDAMetadata`](kda/cache.md#KDAMetadata), [`LightningMetadata`](lightning/cache.md#LightningMetadata)

### `OperationsMetadata`  ·  implements/extends BaseRunTimeMetadata
- def: [`easydel/caching/_abstracts.py:178`](../../../../../../raw/code/EasyDeL/easydel/caching/_abstracts.py#L178) — documented in [easydel-caching-_abstracts](../../../concepts/easydel-caching-_abstracts.md)
- doc: Unified runtime metadata for all cache types using composition.
- signature: `class OperationsMetadata(BaseRunTimeMetadata):`
- members:
  - `cache_type(self)` — [`L329`](../../../../../../raw/code/EasyDeL/easydel/caching/_abstracts.py#L329) — Determine the cache type based on which field is populated.
  - `for_hybrid(cls, postpadded: bool = False, starts: tp.Any | None = None, indexs: tp.Any | None = None)` — [`L242`](../../../../../../raw/code/EasyDeL/easydel/caching/_abstracts.py#L242) — Create OperationsMetadata for hybrid cache.
  - `for_ragged(cls, pages_tables: tp.Any, context_lens: tp.Any, query_start_loc: tp.Any, num_seqs: tp.Any, slot_mapping: tp.Any | None = None, position_ids: tp.Any | None = None, request_distribution: tp.Any | None = None, num_kv_update_slices: tp.Any | None = None, version: str = "v3", page_size: int = 128, prefill_chunk_size: int = 512)` — [`L267`](../../../../../../raw/code/EasyDeL/easydel/caching/_abstracts.py#L267) — Create OperationsMetadata for ragged pages cache.
  - `for_recurrent(cls)` — [`L318`](../../../../../../raw/code/EasyDeL/easydel/caching/_abstracts.py#L318) — Create OperationsMetadata for recurrent cache.
  - `for_transformer(cls, postpadded: bool = False, starts: tp.Any | None = None, indexs: tp.Any | None = None)` — [`L221`](../../../../../../raw/code/EasyDeL/easydel/caching/_abstracts.py#L221) — Create OperationsMetadata for transformer cache.
  - `get_inner(self)` — [`L345`](../../../../../../raw/code/EasyDeL/easydel/caching/_abstracts.py#L345) — Get the inner type-specific metadata.
  - `hybrid` — [`L216`](../../../../../../raw/code/EasyDeL/easydel/caching/_abstracts.py#L216)
  - `ragged` — [`L217`](../../../../../../raw/code/EasyDeL/easydel/caching/_abstracts.py#L217)
  - `recurrent` — [`L218`](../../../../../../raw/code/EasyDeL/easydel/caching/_abstracts.py#L218)
  - `transformer` — [`L215`](../../../../../../raw/code/EasyDeL/easydel/caching/_abstracts.py#L215)
- uses (calls/refs, reference-scoped): [`RaggedPagesMetadata`](ragged_page/cache.md#RaggedPagesMetadata), [`TransformerMetadata`](transformer/cache.md#TransformerMetadata), [`BaseRunTimeMetadata`](_abstracts.md#BaseRunTimeMetadata), [`RecurrentMetadata`](recurrent/cache.md#RecurrentMetadata), [`HybridMetadata`](hybrid/cache.md#HybridMetadata)
- used by: [`BaseModuleProtocol`](../infra/mixins/protocol.md#BaseModuleProtocol), [`forward_mla`](../modules/glm_moe_dsa/modeling_glm_moe_dsa.md#GlmMoeDsaAttention.forward_mla), [`forward`](../layers/attention/_unified.md#UnifiedAttention.forward), [`__call__`](../modules/falcon_h1/modeling_falcon_h1.md#FalconH1Mixer.__call__), [`forward_mla`](../modules/glm4_moe_lite/modeling_glm4_moe_lite.md#Glm4MoeLiteAttention.forward_mla), [`__call__`](../modules/gemma4/modeling_gemma4.md#Gemma4DecoderLayer.__call__), [`forward`](../layers/attention/_flexible.md#FlexibleAttentionModule.forward), [`__all__`](../modules/whisper/modeling_whisper.md#__all__), [`forward_mla`](../layers/attention/_unified.md#UnifiedAttention.forward_mla), [`__call__`](../modules/falcon_mamba/modeling_falcon_mamba.md#FalconMambaMixer.__call__), [`concatenate`](../layers/attention/_flexible.md#AttentionModule.concatenate), [`forward_alibi`](../layers/attention/_unified.md#UnifiedAttention.forward_alibi), [`forward_native`](../operations/kernels/paged_flash_attention.md#PagedFlashAttn.forward_native), [`mesh_call`](../infra/mixins/protocol.md#BaseModuleProtocol.mesh_call), [`__call__`](../infra/mixins/protocol.md#BaseModuleProtocol.__call__), [`__call__`](../modules/falcon/modeling_falcon.md#FalconBlock.__call__), [`__call__`](../modules/gidd/modeling_gidd.md#GiddAttention.__call__), [`__call__`](../modules/stablelm/modeling_stablelm.md#StableLmAttention.__call__), [`forward`](../modules/dbrx/modeling_dbrx.md#DbrxAttention.forward), [`concatenate_to_cache`](transformer/cache.md#TransformerCacheView.concatenate_to_cache), [`__call__`](../modules/minimax/modeling_minimax.md#MiniMaxModel.__call__), [`__call__`](../modules/gemma4/modeling_gemma4.md#Gemma4TextModel.__call__), [`__call__`](../modules/grok_1/modeling_grok_1.md#Grok1Attention.__call__), [`__call__`](../modules/minimax/modeling_minimax.md#MiniMaxDecoderLayer.__call__), [`__call__`](../modules/qwen3_vl_moe/modeling_qwen3_vl_moe.md#Qwen3VLMoeTextModel.__call__), [`forward`](../modules/openelm/modeling_openelm.md#OpenELMMultiHeadCausalAttention.forward), [`forward_native`](../operations/kernels/unified_attention.md#UnifiedAttn.forward_native), [`__call__`](../modules/gemma3/modeling_gemma3.md#Gemma3DecoderLayer.__call__), [`__call__`](../modules/gpt_oss/modeling_gpt_oss.md#GptOssModel.__call__), [`__call__`](../modules/kimi_linear/modeling_kimi_linear.md#KimiLinearModel.__call__), [`__call__`](../modules/mixtral/modeling_mixtral.md#MixtralModel.__call__), [`__call__`](../modules/opt/modeling_opt.md#OPTDecoder.__call__), [`__call__`](../modules/phi3/modeling_phi3.md#Phi3Attention.__call__), [`__call__`](../modules/qwen2_moe/modeling_qwen2_moe.md#Qwen2MoeModel.__call__), [`__call__`](../modules/qwen3_next/modeling_qwen3_next.md#Qwen3NextModel.__call__), [`__call__`](../modules/arctic/modeling_arctic.md#ArcticModel.__call__), [`__call__`](../modules/dbrx/modeling_dbrx.md#DbrxModel.__call__), [`__call__`](../modules/gemma2/modeling_gemma2.md#Gemma2DecoderLayer.__call__), [`__call__`](../modules/glm4/modeling_glm4.md#Glm4DecoderLayer.__call__), [`__call__`](../modules/glm4_moe/modeling_glm4_moe.md#Glm4MoeModel.__call__)  (+216 more)

## Functions
- `unwrap_metadata(metadata: tp.Any, expected_type: str | None = None)` — [`L362`](../../../../../../raw/code/EasyDeL/easydel/caching/_abstracts.py#L362) — Unwrap OperationsMetadata or HybridMetadata to the inner type-specific metadata.


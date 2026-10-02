---
title: 'Module: easydel/caching/transformer/cache.py'
type: catalog
provenance: extracted
module: easydel/caching/transformer/cache.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.caching.transformer.cache`/
symbols:
  TransformerMetadata: TransformerMetadata#
  TransformerCache: TransformerCache#
  TransformerCacheView: TransformerCacheView#
  TransformerCache.views: TransformerCache#views.
  TransformerCache.init_empty: TransformerCache#init_empty().
  TransformerCacheView.concatenate_to_cache: TransformerCacheView#concatenate_to_cache().
  TransformerCacheView.init: TransformerCacheView#init().
  TransformerCache.insert: TransformerCache#insert().
  TransformerCacheConfig: TransformerCacheConfig#
  TransformerCacheView.key: TransformerCacheView#key.
  TransformerCache.to_pure: TransformerCache#to_pure().
  TransformerCache.init_cache: TransformerCache#init_cache().
  MODE_PREFILL: MODE_PREFILL.
  TransformerCacheView.value: TransformerCacheView#value.
  TransformerCache.insert_starts: TransformerCache#insert_starts().
  TransformerCache.insert_index: TransformerCache#insert_index().
  BATCH: BATCH.
  TransformerCacheView._kv_struct_shard: TransformerCacheView#_kv_struct_shard().
  TransformerCacheView.indexs: TransformerCacheView#indexs.
  TransformerCacheView.kv_sharding_axes: TransformerCacheView#kv_sharding_axes.
  TransformerCacheView.__repr__: TransformerCacheView#__repr__().
  _maybe_materialize: _maybe_materialize().
  TransformerCacheConfig.batch_size: TransformerCacheConfig#batch_size.
  TransformerCacheView.starts: TransformerCacheView#starts.
  TransformerCacheConfig.create: TransformerCacheConfig#create().
  TransformerCache.from_pure: TransformerCache#from_pure().
  TransformerCacheView._update_kv_sliding: TransformerCacheView#_update_kv_sliding().
  _sanitize_sharding_axes_for_shape: _sanitize_sharding_axes_for_shape().
  TransformerCacheView.masking_details: TransformerCacheView#masking_details.
  KV_LENGTH: KV_LENGTH.
  KV_HEAD: KV_HEAD.
  KV_HEAD_DIM: KV_HEAD_DIM.
  TransformerCacheConfig.sequence_length: TransformerCacheConfig#sequence_length.
  TransformerCacheView.metadata: TransformerCacheView#metadata.
  AttnMaskDetail: AttnMaskDetail#
  AttnMaskDetail.size: AttnMaskDetail#size.
  TransformerCache.__repr__: TransformerCache#__repr__().
  AttnMaskDetail.mask_type: AttnMaskDetail#mask_type.
  TransformerMetadata.indexs: TransformerMetadata#indexs.
  TransformerCacheView.batch_sharding_axes: TransformerCacheView#batch_sharding_axes.
  TransformerCacheView.is_empty: TransformerCacheView#is_empty().
  TransformerCacheView.__str__: TransformerCacheView#__str__.
  TransformerCache.__str__: TransformerCache#__str__.
  TransformerCacheConfig.key_heads: TransformerCacheConfig#key_heads.
  TransformerCacheConfig.value_heads: TransformerCacheConfig#value_heads.
  TransformerCacheConfig.key_dim: TransformerCacheConfig#key_dim.
  TransformerCacheConfig.value_dim: TransformerCacheConfig#value_dim.
  TransformerCacheView.layer_index: TransformerCacheView#layer_index.
  TransformerCacheView._update_kv: TransformerCacheView#_update_kv().
  TransformerMetadata.starts: TransformerMetadata#starts.
  _mesh_partition_product: _mesh_partition_product().
  _expand_mask_kv_dim: _expand_mask_kv_dim().
  TransformerCacheConfig.num_hidden_layers: TransformerCacheConfig#num_hidden_layers.
  TransformerCacheView._fits_in_window: TransformerCacheView#_fits_in_window().
  TransformerCacheView._overflow_window: TransformerCacheView#_overflow_window().
  AttnMaskDetail.offset: AttnMaskDetail#offset.
  AttnMaskDetail.chunks: AttnMaskDetail#chunks.
  AttnMaskDetail.bricks: AttnMaskDetail#bricks.
  NOT_GIVEN: NOT_GIVEN.
  RUNTIME_MODE_TYPES: RUNTIME_MODE_TYPES.
  QUERY_LENGTH: QUERY_LENGTH.
  HEAD: HEAD.
  HEAD_DIM: HEAD_DIM.
  BIAS_HEAD_SEQ: BIAS_HEAD_SEQ.
  BIAS_KV_SEQ: BIAS_KV_SEQ.
  TransformerCacheConfig.pad_token_id: TransformerCacheConfig#pad_token_id.
  TransformerCacheConfig.num_heads: TransformerCacheConfig#num_heads.
  TransformerCacheConfig.head_dim: TransformerCacheConfig#head_dim.
  TransformerCacheConfig.sliding_window: TransformerCacheConfig#sliding_window.
  TransformerCacheConfig.update_causal_mask: TransformerCacheConfig#update_causal_mask.
  TransformerCacheConfig.create_attention_bias: TransformerCacheConfig#create_attention_bias.
  TransformerCacheView.maximum_sequence_length: TransformerCacheView#maximum_sequence_length.
  TransformerMetadata.postpadded: TransformerMetadata#postpadded.
---
# Module: [`easydel/caching/transformer/cache.py`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py)

## Classes
### `AttnMaskDetail`
- def: [`easydel/caching/transformer/cache.py:132`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L132)
- doc: Configuration for attention masking patterns.
- signature: `class AttnMaskDetail:`
- members:
  - `bricks` — [`L151`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L151)
  - `chunks` — [`L150`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L150)
  - `mask_type` — [`L147`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L147)
  - `offset` — [`L149`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L149)
  - `size` — [`L148`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L148)
- used by: [`concatenate`](../../layers/attention/_flexible.md#AttentionModule.concatenate), [`concatenate_to_cache`](cache.md#TransformerCacheView.concatenate_to_cache), [`init`](cache.md#TransformerCacheView.init), [`init_cache`](cache.md#TransformerCache.init_cache), [`masking_details`](cache.md#TransformerCacheView.masking_details)

### `TransformerCache`  ·  implements/extends BaseCache
- def: [`easydel/caching/transformer/cache.py:754`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L754) — documented in [easydel-caching-transformer-cache](../../../../concepts/easydel-caching-transformer-cache.md)
- doc: Multi-layer transformer cache container.
- signature: `class TransformerCache(BaseCache):`
- members:
  - `from_pure(cls, pure: list[list[JAXArray | ImplicitArray | None]], metadata: TransformerCacheConfig)` — [`L842`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L842) — Reconstruct cache from pure Python data structure.
  - `init_cache(cls, mesh: Mesh, config: TransformerCacheConfig, partition_manager: PartitionManager, dtype: jnp.dtype | None = None, starts: Int[JAXArray, batch] | None = None, quantizer: EasyQuantizer | None = None, mask_type_details: dict[int, AttnMaskDetail] | None = None)` — [`L775`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L775) — Initialize a complete transformer cache with views for all layers.
  - `init_empty(cls, num_hidden_layers: int)` — [`L1001`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L1001) — Initialize an empty transformer cache without allocated storage. — documented in [easydel-caching-transformer-cache](../../../../concepts/easydel-caching-transformer-cache.md)
  - `insert(self, other: TransformerCache, slot: int, quantizer: EasyQuantizer, partition_manager: PartitionManager)` — [`L929`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L929) — Insert another cache's contents at specified batch slot.
  - `insert_index(self, index: Int[JAXArray, ...], slot: int, partition_manager: PartitionManager)` — [`L900`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L900) — Insert position indices at specified batch slot.
  - `insert_starts(self, starts: Int[JAXArray, ...], slot: int, partition_manager: PartitionManager)` — [`L870`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L870) — Insert starting positions at specified batch slot.
  - `to_pure(self)` — [`L825`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L825) — Convert cache to pure Python data structure for serialization.
  - `views` — [`L772`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L772) — documented in [easydel-caching-transformer-cache](../../../../concepts/easydel-caching-transformer-cache.md)
- protocol/private: `__repr__`[`L1014`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L1014), `__str__`[`L1017`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L1017)
- uses (calls/refs, reference-scoped): [`TransformerCacheView`](cache.md#TransformerCacheView), [`EasyQuantizer`](../../layers/quantization/_quants.md#EasyQuantizer), [`init`](cache.md#TransformerCacheView.init), [`BaseCache`](../_abstracts.md#BaseCache), [`TransformerCacheConfig`](cache.md#TransformerCacheConfig), [`key`](cache.md#TransformerCacheView.key), [`MODE_PREFILL`](cache.md#MODE_PREFILL), [`value`](cache.md#TransformerCacheView.value), [`BATCH`](cache.md#BATCH), [`indexs`](cache.md#TransformerCacheView.indexs), [`_maybe_materialize`](cache.md#_maybe_materialize), [`starts`](cache.md#TransformerCacheView.starts), [`KV_HEAD`](cache.md#KV_HEAD), [`KV_HEAD_DIM`](cache.md#KV_HEAD_DIM), [`KV_LENGTH`](cache.md#KV_LENGTH), [`metadata`](cache.md#TransformerCacheView.metadata), [`AttnMaskDetail`](cache.md#AttnMaskDetail), [`num_hidden_layers`](cache.md#TransformerCacheConfig.num_hidden_layers)
- used by: [`BaseModuleProtocol`](../../infra/mixins/protocol.md#BaseModuleProtocol), [`__all__`](../../modules/whisper/modeling_whisper.md#__all__), [`mesh_call`](../../infra/mixins/protocol.md#BaseModuleProtocol.mesh_call), [`__call__`](../../infra/mixins/protocol.md#BaseModuleProtocol.__call__), [`__call__`](../../modules/minimax/modeling_minimax.md#MiniMaxModel.__call__), [`__call__`](../../modules/gemma4/modeling_gemma4.md#Gemma4TextModel.__call__), [`__call__`](../../modules/qwen3_vl_moe/modeling_qwen3_vl_moe.md#Qwen3VLMoeTextModel.__call__), [`__call__`](../../modules/gpt_oss/modeling_gpt_oss.md#GptOssModel.__call__), [`__call__`](../../modules/kimi_linear/modeling_kimi_linear.md#KimiLinearModel.__call__), [`__call__`](../../modules/mixtral/modeling_mixtral.md#MixtralModel.__call__), [`__call__`](../../modules/opt/modeling_opt.md#OPTDecoder.__call__), [`__call__`](../../modules/qwen2_moe/modeling_qwen2_moe.md#Qwen2MoeModel.__call__), [`__call__`](../../modules/qwen3_next/modeling_qwen3_next.md#Qwen3NextModel.__call__), [`init_cache`](../../infra/mixins/generation.md#EasyGenerationMixin.init_cache), [`__call__`](../../modules/arctic/modeling_arctic.md#ArcticModel.__call__), [`__call__`](../../modules/dbrx/modeling_dbrx.md#DbrxModel.__call__), [`__call__`](../../modules/glm4_moe/modeling_glm4_moe.md#Glm4MoeModel.__call__), [`__call__`](../../modules/glm4_moe_lite/modeling_glm4_moe_lite.md#Glm4MoeLiteModel.__call__), [`__call__`](../../modules/glm4v_moe/modeling_glm4v_moe.md#Glm4vMoeTextModel.__call__), [`__call__`](../../modules/glm_moe_dsa/modeling_glm_moe_dsa.md#GlmMoeDsaModel.__call__), [`__call__`](../../modules/gpt_neox/modeling_gpt_neox.md#GPTNeoXModel.__call__), [`__call__`](../../modules/llama/modeling_llama.md#LlamaModel.__call__), [`__call__`](../../modules/qwen2/modeling_qwen.md#Qwen2Model.__call__), [`__call__`](../../modules/qwen3_moe/modeling_qwen3_moe.md#Qwen3MoeModel.__call__), [`__call__`](../../modules/qwen3_vl/modeling_qwen3_vl.md#Qwen3VLTextModel.__call__), [`__call__`](../../modules/xerxes2/modeling_xerxes2.md#Xerxes2Model.__call__), [`__call__`](../../modules/cohere/modeling_cohere.md#CohereModel.__call__), [`__call__`](../../modules/cohere2/modeling_cohere2.md#Cohere2Model.__call__), [`__call__`](../../modules/exaone/modeling_exaone.md#ExaoneModel.__call__), [`__call__`](../../modules/exaone4/modeling_exaone4.md#Exaone4Model.__call__), [`__call__`](../../modules/gemma2/modeling_gemma2.md#Gemma2Model.__call__), [`__call__`](../../modules/gidd/modeling_gidd.md#GiddModel.__call__), [`__call__`](../../modules/glm/modeling_glm.md#GlmModel.__call__), [`__call__`](../../modules/glm4/modeling_glm4.md#Glm4Model.__call__), [`__call__`](../../modules/glm4v/modeling_glm4v.md#Glm4vTextModel.__call__), [`__call__`](../../modules/gpt_j/modeling_gpt_j.md#GPTJModel.__call__), [`__call__`](../../modules/grok_1/modeling_grok_1.md#Grok1Model.__call__), [`__call__`](../../modules/internlm2/modeling_internlm2.md#InternLM2Model.__call__), [`__call__`](../../modules/llama4/modeling_llama4.md#Llama4TextModel.__call__), [`__call__`](../../modules/mistral/modeling_mistral.md#MistralModel.__call__)  (+146 more)

### `TransformerCacheConfig`  ·  implements/extends BaseCacheConfig
- def: [`easydel/caching/transformer/cache.py:329`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L329)
- doc: Metadata configuration for transformer key-value caching.
- signature: `class TransformerCacheConfig(BaseCacheConfig):`
- members:
  - `create(cls, batch_size: int, sequence_length: int, num_hidden_layers: int, pad_token_id: int = -100, num_heads: int | None = None, head_dim: int | None = None, key_heads: int | None = None, value_heads: int | None = None, key_dim: int | None = None, value_dim: int | None = None, update_causal_mask: bool = True, create_attention_bias: bool = True, sliding_window: int | None = None)` — [`L376`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L376) — Create a TransformerCacheConfig instance with validation.
  - `batch_size` — [`L358`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L358)
  - `create_attention_bias` — [`L373`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L373)
  - `head_dim` — [`L364`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L364)
  - `key_dim` — [`L367`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L367)
  - `key_heads` — [`L365`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L365)
  - `num_heads` — [`L363`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L363)
  - `num_hidden_layers` — [`L360`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L360)
  - `pad_token_id` — [`L361`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L361)
  - `sequence_length` — [`L359`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L359)
  - `sliding_window` — [`L369`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L369)
  - `update_causal_mask` — [`L372`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L372)
  - `value_dim` — [`L368`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L368)
  - `value_heads` — [`L366`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L366)
- uses (calls/refs, reference-scoped): [`BaseCacheConfig`](../_abstracts.md#BaseCacheConfig)
- used by: [`init`](cache.md#TransformerCacheView.init), [`BaseCacheConfig`](../_abstracts.md#BaseCacheConfig), [`create_transformer_cache_config`](../../infra/mixins/generation.md#EasyGenerationMixin.create_transformer_cache_config), [`create`](../_abstracts.md#BaseCacheConfig.create), [`to_pure`](cache.md#TransformerCache.to_pure), [`init_cache`](cache.md#TransformerCache.init_cache), [`create_transformer_cache_config`](../../modules/deepseek_v2/modeling_deepseek.md#DeepseekV2ForCausalLM.create_transformer_cache_config), [`create_transformer_cache_config`](../../modules/deepseek_v3/modeling_deepseek.md#DeepseekV3ForCausalLM.create_transformer_cache_config), [`create_transformer_cache_config`](../../modules/xerxes2/modeling_xerxes2.md#Xerxes2ForCausalLM.create_transformer_cache_config), [`from_pure`](cache.md#TransformerCache.from_pure), [`metadata`](cache.md#TransformerCacheView.metadata)

### `TransformerCacheView`  ·  implements/extends BaseCacheView
- def: [`easydel/caching/transformer/cache.py:457`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L457) — documented in [easydel-caching-hybrid-cache](../../../../concepts/easydel-caching-hybrid-cache.md)
- doc: Single-layer cache view for transformer key-value states.
- signature: `class TransformerCacheView(BaseCacheView):`
- members:
  - `_update_kv_sliding(old_cache: Float[JAXArray, "window_size num_heads head_dim"], new_values: Float[JAXArray, "query_len num_heads head_dim"], current_index: Int[JAXArray, ""])` — [`L676`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L676) — Update sliding window KV cache.
  - `concatenate_to_cache(self, query: Float[JAXArray, "batch query_len num_heads head_dim"], key: Float[JAXArray, "batch query_len num_key_heads key_dim"], value: Float[JAXArray, "batch query_len num_value_heads value_dim"], mode: common_types.RUNTIME_MODE_TYPES, quantizer: EasyQuantizer, cache_metadata: TransformerMetadata | OperationsMetadata | HybridMetadata | None, mask_info: MaskInfo, partition_manager: PartitionManager)` — [`L596`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L596) — Update the KV cache functionally and return updated tensors with attention mask. — documented in [easydel-caching-transformer-cache](../../../../concepts/easydel-caching-transformer-cache.md)
  - `init(cls, config: TransformerCacheConfig, layer_index: int | None = None, *, mesh: Mesh | None = None, dtype: jnp.dtype = jnp.bfloat16, partition_manager: PartitionManager | None = None, quantizer: EasyQuantizer | None = None, masking_details: AttnMaskDetail | None = None, starts: Int[JAXArray, batch] | None = None)` — [`L502`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L502) — Initialize a TransformerCacheView from a cache config.
  - `is_empty(self)` — [`L747`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L747)
  - `batch_sharding_axes` — [`L499`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L499)
  - `indexs` — [`L487`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L487)
  - `key` — [`L485`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L485)
  - `kv_sharding_axes` — [`L496`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L496)
  - `layer_index` — [`L494`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L494)
  - `masking_details` — [`L495`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L495)
  - `maximum_sequence_length` — [`L492`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L492)
  - `metadata` — [`L490`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L490)
  - `starts` — [`L488`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L488)
  - `value` — [`L486`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L486)
- protocol/private: `__repr__`[`L737`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L737), `__str__`[`L750`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L750), `_fits_in_window`[`L689`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L689), `_kv_struct_shard`[`L661`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L661), `_overflow_window`[`L692`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L692), `_update_kv`[`L668`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L668)
- uses (calls/refs, reference-scoped): [`OperationsMetadata`](../_abstracts.md#OperationsMetadata), [`TransformerMetadata`](cache.md#TransformerMetadata), [`AttnMaskType`](../../infra/utils.md#AttnMaskType), [`EasyQuantizer`](../../layers/quantization/_quants.md#EasyQuantizer), [`BaseCacheView`](../_abstracts.md#BaseCacheView), [`unwrap_metadata`](../_abstracts.md#unwrap_metadata), [`TransformerCacheConfig`](cache.md#TransformerCacheConfig), [`MODE_PREFILL`](cache.md#MODE_PREFILL), [`SLIDING`](../../infra/utils.md#AttnMaskType.SLIDING), [`BATCH`](cache.md#BATCH), [`HybridMetadata`](../hybrid/cache.md#HybridMetadata), [`_maybe_materialize`](cache.md#_maybe_materialize), [`batch_size`](cache.md#TransformerCacheConfig.batch_size), [`_sanitize_sharding_axes_for_shape`](cache.md#_sanitize_sharding_axes_for_shape), [`KV_HEAD`](cache.md#KV_HEAD), [`KV_HEAD_DIM`](cache.md#KV_HEAD_DIM), [`KV_LENGTH`](cache.md#KV_LENGTH), [`sequence_length`](cache.md#TransformerCacheConfig.sequence_length), [`AttnMaskDetail`](cache.md#AttnMaskDetail), [`size`](cache.md#AttnMaskDetail.size), [`mask_type`](cache.md#AttnMaskDetail.mask_type), [`key_dim`](cache.md#TransformerCacheConfig.key_dim), [`key_heads`](cache.md#TransformerCacheConfig.key_heads), [`value_dim`](cache.md#TransformerCacheConfig.value_dim), [`value_heads`](cache.md#TransformerCacheConfig.value_heads), [`_expand_mask_kv_dim`](cache.md#_expand_mask_kv_dim)
- used by: [`AttentionOutput`](../../operations/_attention_outputs.md#AttentionOutput), [`forward_mla`](../../modules/glm_moe_dsa/modeling_glm_moe_dsa.md#GlmMoeDsaAttention.forward_mla), [`forward`](../../layers/attention/_unified.md#UnifiedAttention.forward), [`forward_mla`](../../modules/glm4_moe_lite/modeling_glm4_moe_lite.md#Glm4MoeLiteAttention.forward_mla), [`__call__`](../../modules/gemma4/modeling_gemma4.md#Gemma4DecoderLayer.__call__), [`forward`](../../layers/attention/_flexible.md#FlexibleAttentionModule.forward), [`__all__`](../../modules/whisper/modeling_whisper.md#__all__), [`forward_mla`](../../layers/attention/_unified.md#UnifiedAttention.forward_mla), [`views`](cache.md#TransformerCache.views), [`concatenate`](../../layers/attention/_flexible.md#AttentionModule.concatenate), [`forward_alibi`](../../layers/attention/_unified.md#UnifiedAttention.forward_alibi), [`init_operations_cache_config`](../../infra/mixins/generation.md#EasyGenerationMixin.init_operations_cache_config), [`cache_view`](../../infra/modeling_outputs.md#AttentionLayerOutput.cache_view), [`init_operations_cache`](../../infra/mixins/generation.md#EasyGenerationMixin.init_operations_cache), [`__call__`](../../modules/falcon/modeling_falcon.md#FalconBlock.__call__), [`__call__`](../../modules/gidd/modeling_gidd.md#GiddAttention.__call__), [`__call__`](../../modules/stablelm/modeling_stablelm.md#StableLmAttention.__call__), [`forward`](../../modules/dbrx/modeling_dbrx.md#DbrxAttention.forward), [`__call__`](../../modules/grok_1/modeling_grok_1.md#Grok1Attention.__call__), [`__call__`](../../modules/minimax/modeling_minimax.md#MiniMaxDecoderLayer.__call__), [`forward`](../../modules/openelm/modeling_openelm.md#OpenELMMultiHeadCausalAttention.forward), [`cache_view`](../../infra/modeling_outputs.md#DecoderLayerOutput.cache_view), [`__call__`](../../modules/gemma3/modeling_gemma3.md#Gemma3DecoderLayer.__call__), [`__call__`](../../modules/phi3/modeling_phi3.md#Phi3Attention.__call__), [`__call__`](../../modules/xerxes/modeling_xerxes.md#XerxesDecoderLayer.__call__), [`__call__`](../../modules/gemma2/modeling_gemma2.md#Gemma2DecoderLayer.__call__), [`__call__`](../../modules/glm4/modeling_glm4.md#Glm4DecoderLayer.__call__), [`__call__`](../../modules/glm4v/modeling_glm4v.md#Glm4vTextDecoderLayer.__call__), [`__call__`](../../modules/gpt2/modeling_gpt2.md#GPT2Block.__call__), [`__call__`](../../modules/opt/modeling_opt.md#OPTAttention.__call__), [`__call__`](../../modules/phi3/modeling_phi3.md#Phi3DecoderLayer.__call__), [`__call__`](../../modules/roberta/modeling_roberta.md#RobertaSelfAttention.__call__), [`__call__`](../../modules/stablelm/modeling_stablelm.md#StableLmDecoderLayer.__call__), [`forward`](../../modules/qwen3_next/modeling_qwen3_next.md#Qwen3NextFullAttention.forward), [`forward_alibi`](../../modules/mosaic_mpt/modeling_mosaic.md#MptAttention.forward_alibi), [`__call__`](../../modules/gpt2/modeling_gpt2.md#GPT2Attention.__call__), [`__call__`](../../modules/opt/modeling_opt.md#OPTDecoderLayer.__call__), [`__call__`](../../modules/qwen3_vl_moe/modeling_qwen3_vl_moe.md#Qwen3VLMoeTextDecoderLayer.__call__), [`__call__`](../../modules/whisper/modeling_whisper.md#WhisperAttention.__call__), [`__call__`](../../modules/arctic/modeling_arctic.md#ArcticDecoderLayer.__call__)  (+85 more)

### `TransformerMetadata`
- def: [`easydel/caching/transformer/cache.py:1021`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L1021) — documented in [easydel-caching-transformer-cache](../../../../concepts/easydel-caching-transformer-cache.md)
- members:
  - `indexs` — [`L1039`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L1039)
  - `postpadded` — [`L1037`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L1037)
  - `starts` — [`L1038`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L1038)
- uses (calls/refs, reference-scoped): [`BaseRunTimeMetadata`](../_abstracts.md#BaseRunTimeMetadata)
- used by: [`BaseModuleProtocol`](../../infra/mixins/protocol.md#BaseModuleProtocol), [`forward_mla`](../../modules/glm_moe_dsa/modeling_glm_moe_dsa.md#GlmMoeDsaAttention.forward_mla), [`forward`](../../layers/attention/_unified.md#UnifiedAttention.forward), [`forward_mla`](../../modules/glm4_moe_lite/modeling_glm4_moe_lite.md#Glm4MoeLiteAttention.forward_mla), [`__call__`](../../modules/gemma4/modeling_gemma4.md#Gemma4DecoderLayer.__call__), [`forward`](../../layers/attention/_flexible.md#FlexibleAttentionModule.forward), [`__all__`](../../modules/whisper/modeling_whisper.md#__all__), [`forward_mla`](../../layers/attention/_unified.md#UnifiedAttention.forward_mla), [`concatenate`](../../layers/attention/_flexible.md#AttentionModule.concatenate), [`forward_alibi`](../../layers/attention/_unified.md#UnifiedAttention.forward_alibi), [`mesh_call`](../../infra/mixins/protocol.md#BaseModuleProtocol.mesh_call), [`__call__`](../../infra/mixins/protocol.md#BaseModuleProtocol.__call__), [`__call__`](../../modules/falcon/modeling_falcon.md#FalconBlock.__call__), [`__call__`](../../modules/gidd/modeling_gidd.md#GiddAttention.__call__), [`__call__`](../../modules/stablelm/modeling_stablelm.md#StableLmAttention.__call__), [`forward`](../../modules/dbrx/modeling_dbrx.md#DbrxAttention.forward), [`concatenate_to_cache`](cache.md#TransformerCacheView.concatenate_to_cache), [`__call__`](../../modules/minimax/modeling_minimax.md#MiniMaxModel.__call__), [`__call__`](../../modules/gemma4/modeling_gemma4.md#Gemma4TextModel.__call__), [`__call__`](../../modules/grok_1/modeling_grok_1.md#Grok1Attention.__call__), [`__call__`](../../modules/minimax/modeling_minimax.md#MiniMaxDecoderLayer.__call__), [`__call__`](../../modules/qwen3_vl_moe/modeling_qwen3_vl_moe.md#Qwen3VLMoeTextModel.__call__), [`forward`](../../modules/openelm/modeling_openelm.md#OpenELMMultiHeadCausalAttention.forward), [`__call__`](../../modules/gemma3/modeling_gemma3.md#Gemma3DecoderLayer.__call__), [`__call__`](../../modules/gpt_oss/modeling_gpt_oss.md#GptOssModel.__call__), [`__call__`](../../modules/kimi_linear/modeling_kimi_linear.md#KimiLinearModel.__call__), [`__call__`](../../modules/mixtral/modeling_mixtral.md#MixtralModel.__call__), [`__call__`](../../modules/opt/modeling_opt.md#OPTDecoder.__call__), [`__call__`](../../modules/phi3/modeling_phi3.md#Phi3Attention.__call__), [`__call__`](../../modules/qwen2_moe/modeling_qwen2_moe.md#Qwen2MoeModel.__call__), [`__call__`](../../modules/qwen3_next/modeling_qwen3_next.md#Qwen3NextModel.__call__), [`__call__`](../../modules/xerxes/modeling_xerxes.md#XerxesDecoderLayer.__call__), [`__call__`](../../modules/arctic/modeling_arctic.md#ArcticModel.__call__), [`__call__`](../../modules/dbrx/modeling_dbrx.md#DbrxModel.__call__), [`__call__`](../../modules/gemma2/modeling_gemma2.md#Gemma2DecoderLayer.__call__), [`__call__`](../../modules/glm4/modeling_glm4.md#Glm4DecoderLayer.__call__), [`__call__`](../../modules/glm4_moe/modeling_glm4_moe.md#Glm4MoeModel.__call__), [`__call__`](../../modules/glm4_moe_lite/modeling_glm4_moe_lite.md#Glm4MoeLiteModel.__call__), [`__call__`](../../modules/glm4v/modeling_glm4v.md#Glm4vTextDecoderLayer.__call__), [`__call__`](../../modules/glm4v_moe/modeling_glm4v_moe.md#Glm4vMoeTextModel.__call__)  (+215 more)

## Functions
- `_expand_mask_kv_dim(mask_info: MaskInfo, target_kv_len: int, cache_position: jnp.ndarray, query_len: int)` — [`L229`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L229) — Expand mask's KV dimension to match cache size.
- `_maybe_materialize(x: JAXArray | ImplicitArray | None)` — [`L108`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L108) — Materialize an ImplicitArray if needed, or return a regular JAXArray.
- `_mesh_partition_product(mesh: Mesh, axis_spec: object)` — [`L168`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L168) — Return the number of shards implied by a PartitionSpec entry for a given mesh.
- `_sanitize_sharding_axes_for_shape(*, mesh: Mesh, partition_manager: PartitionManager, axes: list[object | None], mode: common_types.RUNTIME_MODE_TYPES, shape: tuple[int, ...])` — [`L190`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L190) — Disable incompatible sharding axes for a given shape.

## Module values
- `BATCH` — [`L156`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L156)
- `BIAS_HEAD_SEQ` — [`L163`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L163)
- `BIAS_KV_SEQ` — [`L164`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L164)
- `HEAD` — [`L159`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L159)
- `HEAD_DIM` — [`L161`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L161)
- `KV_HEAD` — [`L160`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L160)
- `KV_HEAD_DIM` — [`L162`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L162)
- `KV_LENGTH` — [`L158`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L158)
- `MODE_PREFILL` — [`L165`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L165)
- `NOT_GIVEN` — [`L154`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L154)
- `QUERY_LENGTH` — [`L157`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L157)
- `RUNTIME_MODE_TYPES` — [`L155`](../../../../../../../raw/code/EasyDeL/easydel/caching/transformer/cache.py#L155)


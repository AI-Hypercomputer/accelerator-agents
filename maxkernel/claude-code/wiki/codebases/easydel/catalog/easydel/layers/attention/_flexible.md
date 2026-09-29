---
title: 'Module: easydel/layers/attention/_flexible.py'
type: catalog
provenance: extracted
module: easydel/layers/attention/_flexible.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.layers.attention._flexible`/
symbols:
  FlexibleAttentionModule: FlexibleAttentionModule#
  AttentionModule.config: AttentionModule#config.
  FlexibleAttentionModule.forward: FlexibleAttentionModule#forward().
  AttentionModule.concatenate: AttentionModule#concatenate().
  AttentionModule: AttentionModule#
  AttentionMechanisms: AttentionMechanisms#
  FlexibleAttentionModule.__init__: FlexibleAttentionModule#__init__().
  AttentionModule.shard_attention_prod: AttentionModule#shard_attention_prod().
  AttentionModule.apply_qkv_shardings: AttentionModule#apply_qkv_shardings().
  FlexibleAttentionModule.operation_executor: FlexibleAttentionModule#operation_executor().
  get_optimal_config: get_optimal_config().
  FlexibleAttentionModule._maybe_route_varlen_multihost_tpu_attention: FlexibleAttentionModule#_maybe_route_varlen_multihost_tpu_attention().
  AttentionModule._merge_heads: AttentionModule#_merge_heads().
  AttentionModule._handle_cache_concat: AttentionModule#_handle_cache_concat().
  AttentionModule._apply_sliding_window: AttentionModule#_apply_sliding_window().
  FlexibleAttentionModule.impl_decode: FlexibleAttentionModule#impl_decode.
  AttentionModule.__init__: AttentionModule#__init__().
  AttentionModule.quantizer: AttentionModule#quantizer().
  FlexibleAttentionModule.impl: FlexibleAttentionModule#impl.
  AttentionModule.default_key_value_sharding: AttentionModule#default_key_value_sharding().
  AttentionMechanisms.RAGGED_PAGE_ATTENTION_V3: AttentionMechanisms#RAGGED_PAGE_ATTENTION_V3.
  AttentionMechanisms.RAGGED_PAGE_ATTENTION_V2: AttentionMechanisms#RAGGED_PAGE_ATTENTION_V2.
  FlexibleAttentionModule.operation_requirements: FlexibleAttentionModule#operation_requirements().
  FlexibleAttentionModule.requires_cache: FlexibleAttentionModule#requires_cache().
  FlexibleAttentionModule.has_separate_decode: FlexibleAttentionModule#has_separate_decode().
  AttentionModule.apply_qk_shardings: AttentionModule#apply_qk_shardings().
  AttentionModule.build_cache_pos: AttentionModule#build_cache_pos().
  Cfg: Cfg.
  AttentionMechanisms.UNIFIED_ATTENTION: AttentionMechanisms#UNIFIED_ATTENTION.
  FlexibleAttentionModule.config: FlexibleAttentionModule#config.
  FlexibleAttentionModule.metadata: FlexibleAttentionModule#metadata.
  FlexibleAttentionModule.operation: FlexibleAttentionModule#operation().
  logger: logger.
  AttentionMechanisms.VANILLA: AttentionMechanisms#VANILLA.
  AttentionModule.init_attention_bias: AttentionModule#init_attention_bias().
  FlexibleAttentionModule.__call__: FlexibleAttentionModule#__call__.
  FlexibleAttentionModule.decode_operation: FlexibleAttentionModule#decode_operation().
  AttentionModule.get_sharding_safely: AttentionModule#get_sharding_safely().
  AttentionModule._transpose_sequence_head: AttentionModule#_transpose_sequence_head().
  _get_jax_dtype_from_string: _get_jax_dtype_from_string().
  AttentionMechanisms.PAGED_FLASH_ATTENTION: AttentionMechanisms#PAGED_FLASH_ATTENTION.
  tpu_version_check: tpu_version_check().
  FlexibleAttentionModule.cast_to_dtype: FlexibleAttentionModule#cast_to_dtype().
  AttentionMechanisms.AUTO: AttentionMechanisms#AUTO.
  AttentionMechanisms.FLASH_ATTN2: AttentionMechanisms#FLASH_ATTN2.
  AttentionMechanisms.BLOCKSPARSE: AttentionMechanisms#BLOCKSPARSE.
  AttentionMechanisms.SDPA: AttentionMechanisms#SDPA.
  FlexibleAttentionModule.softmax_scale: FlexibleAttentionModule#softmax_scale.
  FlexibleAttentionModule.deterministic: FlexibleAttentionModule#deterministic.
  FlexibleAttentionModule._get_impl_names: FlexibleAttentionModule#_get_impl_names().
  AttentionModule.apply_complex_rotary: AttentionModule#apply_complex_rotary().
  AttentionModule.transpose_array: AttentionModule#transpose_array().
  AttentionModule._select_slices: AttentionModule#_select_slices().
  AttentionMechanisms.RING: AttentionMechanisms#RING.
  AttentionMechanisms.SPLASH: AttentionMechanisms#SPLASH.
  AttentionMechanisms.CUDNN: AttentionMechanisms#CUDNN.
  AttentionMechanisms.BLOCKWISE: AttentionMechanisms#BLOCKWISE.
  AttentionMechanisms.CUDA_FLASH_ATTN2: AttentionMechanisms#CUDA_FLASH_ATTN2.
  AttentionMechanisms.MULTI_LATENT_RAGGED_PAGE_ATTENTION_V1: AttentionMechanisms#MULTI_LATENT_RAGGED_PAGE_ATTENTION_V1.
  AttentionMechanisms.MULTI_LATENT_RAGGED_PAGE_ATTENTION_V2: AttentionMechanisms#MULTI_LATENT_RAGGED_PAGE_ATTENTION_V2.
  AttentionMechanisms.PAGED_ATTENTION: AttentionMechanisms#PAGED_ATTENTION.
  AttentionMechanisms.REGRESSIVE_DECODE: AttentionMechanisms#REGRESSIVE_DECODE.
  DEFAULT_ATTENTION_MECHANISM: DEFAULT_ATTENTION_MECHANISM.
  FlexibleAttentionModule.dropout_prob: FlexibleAttentionModule#dropout_prob.
  FlexibleAttentionModule._requires_cache: FlexibleAttentionModule#_requires_cache.
  AttentionModule.cached_key: AttentionModule#cached_key.
  AttentionModule.cached_value: AttentionModule#cached_value.
  AttentionModule.cache_index: AttentionModule#cache_index.
  AttentionModule.repeat_key_value: AttentionModule#repeat_key_value().
---
# Module: [`easydel/layers/attention/_flexible.py`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py)

## Classes
### `AttentionMechanisms`  ·  implements/extends StrEnum
- def: [`easydel/layers/attention/_flexible.py:116`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L116)
- doc: Available attention mechanism implementations.
- signature: `class AttentionMechanisms(StrEnum):`
- members:
  - `AUTO` — [`L141`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L141)
  - `BLOCKSPARSE` — [`L146`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L146)
  - `BLOCKWISE` — [`L148`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L148)
  - `CUDA_FLASH_ATTN2` — [`L150`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L150)
  - `CUDNN` — [`L147`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L147)
  - `FLASH_ATTN2` — [`L142`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L142)
  - `MULTI_LATENT_RAGGED_PAGE_ATTENTION_V1` — [`L153`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L153)
  - `MULTI_LATENT_RAGGED_PAGE_ATTENTION_V2` — [`L154`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L154)
  - `PAGED_ATTENTION` — [`L155`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L155)
  - `PAGED_FLASH_ATTENTION` — [`L157`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L157)
  - `RAGGED_PAGE_ATTENTION_V2` — [`L152`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L152)
  - `RAGGED_PAGE_ATTENTION_V3` — [`L151`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L151)
  - `REGRESSIVE_DECODE` — [`L158`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L158)
  - `RING` — [`L143`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L143)
  - `SDPA` — [`L149`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L149)
  - `SPLASH` — [`L145`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L145)
  - `UNIFIED_ATTENTION` — [`L156`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L156)
  - `VANILLA` — [`L144`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L144)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`_worker_startup_timeout`](../../inference/esurge/esurge_engine.md#eSurge._worker_startup_timeout), [`__init__`](../../inference/esurge/esurge_engine.md#eSurge.__init__), [`__init__`](_flexible.md#FlexibleAttentionModule.__init__), [`create_ragged_page_cache_config`](../../modules/deepseek_v2/modeling_deepseek.md#DeepseekV2ForCausalLM.create_ragged_page_cache_config), [`create_ragged_page_cache_config`](../../modules/deepseek_v3/modeling_deepseek.md#DeepseekV3ForCausalLM.create_ragged_page_cache_config), [`create_ragged_page_cache_config`](../../modules/glm4_moe_lite/modeling_glm4_moe_lite.md#Glm4MoeLiteForCausalLM.create_ragged_page_cache_config), [`create_ragged_page_cache_config`](../../modules/glm_moe_dsa/modeling_glm_moe_dsa.md#GlmMoeDsaForCausalLM.create_ragged_page_cache_config), [`_maybe_route_varlen_multihost_tpu_attention`](_flexible.md#FlexibleAttentionModule._maybe_route_varlen_multihost_tpu_attention), [`get_optimal_config`](_flexible.md#get_optimal_config)

### `AttentionModule`  ·  implements/extends Generic, Module
- def: [`easydel/layers/attention/_flexible.py:638`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L638) — documented in [easydel-layers-attention-_unified](../../../../concepts/easydel-layers-attention-_unified.md)
- doc: Base class for Flax attention modules in EasyDeL, providing common utilities.
- signature: `class AttentionModule(nn.Module, tp.Generic[Cfg]):`
- members:
  - `__init__(self, config: Cfg)` — [`L653`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L653) — Initializes the AttentionModule.
  - `_apply_sliding_window(self, key: Array, value: Array, mask_info: MaskInfo, mode: common_types.RUNTIME_MODE_TYPES, cache_view: TransformerCacheView | None, sliding_window: int | tuple[int, int], query_length: int, masking_details: AttnMaskDetail | None, cache_metadata: TransformerMetadata | None)` — [`L967`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L967) — Apply sliding window masking and slicing to KV tensors and mask.
  - `_handle_cache_concat(self, query: Float[JArray, "batch seq_q heads dim"], key: Float[JArray, "batch seq_k heads dim"], value: Float[JArray, "batch seq_v heads dim"], mode: common_types.RUNTIME_MODE_TYPES, mask_info: MaskInfo, cache_view: TransformerCacheView | None, cache_metadata: TransformerMetadata | None)` — [`L909`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L909) — Handle concatenation of current KV states to the cache.
  - `_merge_heads(self, hidden_states: Float[JArray, "batch seq heads dim"])` — [`L1280`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L1280) — Merges the attention heads into a single hidden state tensor.
  - `_transpose_sequence_head(*args: Float[JArray, "batch seq heads dim"])` — [`L886`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L886) — Transposes the sequence and head dimensions of input tensors.
  - `apply_complex_rotary(xq: Float[JArray, "... seq heads dim"], xk: Float[JArray, "... seq heads dim"], freqs_cis: Complex[JArray, "batch seq 1 dim_2"])` — [`L669`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L669) — Apply rotary position embeddings using complex number multiplication.
  - `apply_qk_shardings(self, q: Float[JArray, "batch seq heads dim"], k: Float[JArray, "batch seq heads dim"])` — [`L715`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L715) — Apply logical sharding constraints to query and key tensors.
  - `apply_qkv_shardings(self, q: Float[JArray, "batch seq heads dim"], k: Float[JArray, "batch seq heads dim"], v: Float[JArray, "batch seq heads dim"])` — [`L749`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L749) — Apply logical sharding constraints to query, key, and value tensors.
  - `build_cache_pos(attention_mask: Bool[JArray, "batch heads seq_q seq_k"], mode: common_types.RUNTIME_MODE_TYPES, cache_view: TransformerCacheView | None = None)` — [`L794`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L794) — Calculates the position indices within the sequence for cache-aware operations.
  - `concatenate(self, *, query: Array, key: Array, value: Array, mask_info: MaskInfo, mode: common_types.RUNTIME_MODE_TYPES | common_types.EMPTY_VAL = common_types.NOT_GIVEN, cache_view: TransformerCacheView | RaggedPagesCacheView | UnifiedAttentionCacheView | None = None, cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None, sliding_window: int | None = None)` — [`L1090`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L1090) — Prepares inputs for attention calculation, handling KV caching and mask merging. — documented in [easydel-layers-attention-_unified](../../../../concepts/easydel-layers-attention-_unified.md)
  - `default_key_value_sharding(self)` — [`L850`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L850) — Defines the default JAX sharding for key and value tensors.
  - `get_sharding_safely(self, tensor: Float[JArray, ...])` — [`L873`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L873) — Retrieves the PartitionSpec of a tensor, falling back to the default KV sharding.
  - `init_attention_bias()` — [`L1170`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L1170)
  - `quantizer(self)` — [`L828`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L828) — Provides an EasyQuantizer instance based on the module's configuration.
  - `repeat_key_value(key: Float[JArray, "batch seq kv_heads dim"], value: Float[JArray, "batch seq kv_heads dim"], num_reps: int)` — [`L1295`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L1295) — Repeats key and value tensors for Grouped Query Attention (GQA).
  - `shard_attention_prod(self, attn_output: Float[JArray, "batch seq heads dim"])` — [`L1258`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L1258) — Applies sharding constraints to the attention output tensor. — documented in [easydel-layers-attention-_unified](../../../../concepts/easydel-layers-attention-_unified.md)
  - `transpose_array(x: Float[JArray, "batch seq heads dim"])` — [`L902`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L902)
  - `cache_index` — [`L666`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L666)
  - `cached_key` — [`L664`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L664)
  - `cached_value` — [`L665`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L665)
  - `config` — [`L662`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L662)
- protocol/private: `_select_slices`[`L1040`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L1040)
- uses (calls/refs, reference-scoped): [`OperationsMetadata`](../../caching/_abstracts.md#OperationsMetadata), [`RaggedPagesMetadata`](../../caching/ragged_page/cache.md#RaggedPagesMetadata), [`TransformerMetadata`](../../caching/transformer/cache.md#TransformerMetadata), [`partition_manager`](../../infra/base_config.md#EasyDeLBaseConfig.partition_manager), [`UnifiedAttention`](_unified.md#UnifiedAttention), [`RaggedPagesCacheView`](../../caching/ragged_page/cache.md#RaggedPagesCacheView), [`TransformerCacheView`](../../caching/transformer/cache.md#TransformerCacheView), [`AttnMaskDetail`](../../infra/utils.md#AttnMaskDetail), [`AttnMaskType`](../../infra/utils.md#AttnMaskType), [`concatenate_to_cache`](../../caching/transformer/cache.md#TransformerCacheView.concatenate_to_cache), [`UnifiedAttentionCacheView`](../../caching/unified_attention/cache.md#UnifiedAttentionCacheView), [`EasyQuantizer`](../quantization/_quants.md#EasyQuantizer), [`ParallelHybridCacheView`](../../caching/hybrid/cache.md#ParallelHybridCacheView), [`mesh`](../../infra/base_config.md#EasyDeLBaseConfig.mesh), [`_merge_heads`](_unified.md#UnifiedAttention._merge_heads), [`key`](../../caching/transformer/cache.md#TransformerCacheView.key), [`TurboQuantConfig`](../quantization/_turboquant.md#TurboQuantConfig), [`concatenate`](../../modules/gidd/modeling_gidd.md#GiddAttention.concatenate), [`SLIDING`](../../infra/utils.md#AttnMaskType.SLIDING), [`indexs`](../../caching/transformer/cache.md#TransformerCacheView.indexs), [`partition_axis`](../../infra/base_config.md#EasyDeLBaseConfig.partition_axis), [`is_ragged`](../../caching/hybrid/cache.md#ParallelHybridCacheView.is_ragged), [`starts`](../../caching/transformer/cache.md#TransformerCacheView.starts), [`WhisperAttention`](../../modules/whisper/modeling_whisper.md#WhisperAttention), [`masking_details`](../../caching/transformer/cache.md#TransformerCacheView.masking_details), [`GiddAttention`](../../modules/gidd/modeling_gidd.md#GiddAttention), [`Cfg`](_flexible.md#Cfg), [`CLIPAttention`](../../modules/clip/modeling_clip.md#CLIPAttention), [`Grok1Attention`](../../modules/grok_1/modeling_grok_1.md#Grok1Attention), [`Llama4VisionAttention`](../../modules/llama4/modeling_llama4.md#Llama4VisionAttention), [`OPTAttention`](../../modules/opt/modeling_opt.md#OPTAttention), [`PixtralAttention`](../../modules/pixtral/modeling_pixtral.md#PixtralAttention), [`RobertaSelfAttention`](../../modules/roberta/modeling_roberta.md#RobertaSelfAttention), [`SiglipAttention`](../../modules/siglip/modeling_siglip.md#SiglipAttention), [`_merge_heads`](../../modules/clip/modeling_clip.md#CLIPAttention._merge_heads), [`_merge_heads`](../../modules/grok_1/modeling_grok_1.md#Grok1Attention._merge_heads), [`_merge_heads`](../../modules/opt/modeling_opt.md#OPTAttention._merge_heads), [`_merge_heads`](../../modules/siglip/modeling_siglip.md#SiglipAttention._merge_heads), [`_merge_heads`](../../modules/whisper/modeling_whisper.md#WhisperAttention._merge_heads), [`indexs`](../../caching/transformer/cache.md#TransformerMetadata.indexs)  (+4 more)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`UnifiedAttention`](_unified.md#UnifiedAttention), [`forward_mla`](../../modules/glm_moe_dsa/modeling_glm_moe_dsa.md#GlmMoeDsaAttention.forward_mla), [`forward`](_unified.md#UnifiedAttention.forward), [`forward_mla`](../../modules/glm4_moe_lite/modeling_glm4_moe_lite.md#Glm4MoeLiteAttention.forward_mla), [`__all__`](../../modules/whisper/modeling_whisper.md#__all__), [`forward_mla`](_unified.md#UnifiedAttention.forward_mla), [`forward_alibi`](_unified.md#UnifiedAttention.forward_alibi), [`__init__`](_unified.md#UnifiedAttention.__init__), [`__call__`](../../modules/gidd/modeling_gidd.md#GiddAttention.__call__), [`__call__`](../../modules/stablelm/modeling_stablelm.md#StableLmAttention.__call__), [`forward`](../../modules/dbrx/modeling_dbrx.md#DbrxAttention.forward), [`_forward_with_kv_capture`](../../modules/gemma4/modeling_gemma4.md#Gemma4Attention._forward_with_kv_capture), [`__call__`](../../modules/grok_1/modeling_grok_1.md#Grok1Attention.__call__), [`forward`](../../modules/openelm/modeling_openelm.md#OpenELMMultiHeadCausalAttention.forward), [`__call__`](../../modules/phi3/modeling_phi3.md#Phi3Attention.__call__), [`__call__`](../../modules/opt/modeling_opt.md#OPTAttention.__call__), [`__call__`](../../modules/roberta/modeling_roberta.md#RobertaSelfAttention.__call__), [`forward`](../../modules/qwen3_next/modeling_qwen3_next.md#Qwen3NextFullAttention.forward), [`forward_alibi`](../../modules/mosaic_mpt/modeling_mosaic.md#MptAttention.forward_alibi), [`__call__`](../../modules/gemma4/modeling_gemma4.md#Gemma4Attention.__call__), [`__call__`](../../modules/gpt2/modeling_gpt2.md#GPT2Attention.__call__), [`__call__`](../../modules/whisper/modeling_whisper.md#WhisperAttention.__call__), [`__call__`](../../modules/pixtral/modeling_pixtral.md#PixtralAttention.__call__), [`__call__`](../../modules/llama4/modeling_llama4.md#Llama4VisionAttention.__call__), [`qk_scale`](../../modules/gidd/modeling_gidd.md#GiddAttention.qk_scale), [`_create_o_proj`](../../modules/seed_oss/modeling_seed_oss.md#SeedOssAttention._create_o_proj), [`_compute_alibi_bias`](../../modules/mosaic_mpt/modeling_mosaic.md#MptAttention._compute_alibi_bias), [`k_norm`](../../modules/cohere/modeling_cohere.md#CohereAttention.k_norm), [`q_norm`](../../modules/cohere/modeling_cohere.md#CohereAttention.q_norm), [`head_dim`](../../modules/whisper/modeling_whisper.md#WhisperAttention.head_dim), [`num_key_value_groups`](../../modules/pixtral/modeling_pixtral.md#PixtralAttention.num_key_value_groups), [`qk_norm_eps`](../../modules/gidd/modeling_gidd.md#GiddAttention.qk_norm_eps), [`_create_k_norm`](../../modules/olmo2/modeling_olmo2.md#Olmo2Attention._create_k_norm), [`_create_q_norm`](../../modules/olmo2/modeling_olmo2.md#Olmo2Attention._create_q_norm), [`head_dim`](../../modules/grok_1/modeling_grok_1.md#Grok1Attention.head_dim), [`attention_performer`](../../modules/gidd/modeling_gidd.md#GiddAttention.attention_performer), [`resid_dropout`](../../modules/gpt2/modeling_gpt2.md#GPT2Attention.resid_dropout), [`head_dim`](../../modules/pixtral/modeling_pixtral.md#PixtralAttention.head_dim), [`attention_performer`](../../modules/llama4/modeling_llama4.md#Llama4VisionAttention.attention_performer)  (+30 more)

### `FlexibleAttentionModule`  ·  implements/extends Module
- def: [`easydel/layers/attention/_flexible.py:242`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L242) — documented in [easydel-layers-attention-_unified](../../../../concepts/easydel-layers-attention-_unified.md)
- doc: Unified interface for various attention mechanisms.
- signature: `class FlexibleAttentionModule(nn.Module):`
- members:
  - `__init__(self, base_config: EasyDeLBaseConfig, softmax_scale: float, dropout_prob: float = 0, *, rngs: nn.Rngs | None = None, attn_mechanism: AttentionMechanisms | None = None, requires_cache: bool | None = None)` — [`L290`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L290) — Initializes the AttentionModule.
  - `cast_to_dtype(x: tp.Any)` — [`L572`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L572)
  - `decode_operation(self)` — [`L610`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L610) — Get the decode operation instance (if different from prefill).
  - `forward(self, query_states: Float[JArray, "batch seq_q heads dim"], key_states: Float[JArray, "batch seq_k heads dim"], value_states: Float[JArray, "batch seq_v heads dim"], mode: common_types.RUNTIME_MODE_TYPES | None, mask_info: MaskInfo | None = None, bias: Float[JArray, "batch heads seq_q seq_k"] | None = None, sliding_window: int | tuple[int, int] | None = None, cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None, cache_view: TransformerCacheView | RaggedPagesCacheView | UnifiedAttentionCacheView | None = None, init_bias: tp.Callable[[], Float[JArray, "batch heads seq_q seq_k"]] | None = None, causal: bool = True, softmax_aux: Float[JArray, ...] | None = None, softmax_scale: float | None = None, logits_soft_cap: float | None = None, dropout_prob: float | None = None, dropout_rng: tp.Any | None = None, deterministic: bool | None = None, precision: lax.PrecisionLike | None = None, prevent_cse: bool = True, cum_seqlens_q: Int[JArray, batch_plus_one] | None = None, cum_seqlens_k: Int[JArray, batch_plus_one] | None = None, normalize_output: bool = True, fused_backward: bool = False, compute_dtype: jnp.dtype | None = None, optimized: bool = False, mask_value: float | None = None, vmem_limit_bytes: int | None = None, policy: tp.Any | None = None, **extra_op_kwargs: tp.Any)` — [`L359`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L359) — Performs the attention computation using the selected backend implementation. — documented in [easydel-layers-attention-_unified](../../../../concepts/easydel-layers-attention-_unified.md)
  - `has_separate_decode(self)` — [`L629`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L629) — Whether decode uses a different operation than prefill.
  - `operation(self)` — [`L605`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L605) — Get the primary (prefill) operation instance.
  - `operation_executor(self)` — [`L590`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L590) — Get an OperationExecutor for mode-bound operation access.
  - `operation_requirements(self)` — [`L615`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L615) — Get combined requirements from both prefill and decode operations.
  - `requires_cache(self)` — [`L624`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L624) — Whether this attention module requires cache.
  - `config` — [`L336`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L336)
  - `deterministic` — [`L347`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L347)
  - `dropout_prob` — [`L339`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L339)
  - `impl` — [`L342`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L342)
  - `impl_decode` — [`L348`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L348)
  - `metadata` — [`L337`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L337)
  - `softmax_scale` — [`L338`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L338)
- protocol/private: `__call__`[`L586`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L586), `_get_impl_names`[`L490`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L490), `_maybe_route_varlen_multihost_tpu_attention`[`L498`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L498), `_requires_cache`[`L340`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L340)
- uses (calls/refs, reference-scoped): [`EasyDeLBaseConfig`](../../infra/base_config.md#EasyDeLBaseConfig), [`OperationsMetadata`](../../caching/_abstracts.md#OperationsMetadata), [`RaggedPagesMetadata`](../../caching/ragged_page/cache.md#RaggedPagesMetadata), [`TransformerMetadata`](../../caching/transformer/cache.md#TransformerMetadata), [`RaggedPagesCacheView`](../../caching/ragged_page/cache.md#RaggedPagesCacheView), [`TransformerCacheView`](../../caching/transformer/cache.md#TransformerCacheView), [`AttentionOutput`](../../operations/_attention_outputs.md#AttentionOutput), [`attention_outputs`](../../operations/_attention_outputs.md#AttentionOutput.attention_outputs), [`UnifiedAttentionCacheView`](../../caching/unified_attention/cache.md#UnifiedAttentionCacheView), [`OperationMetadata`](../../operations/_operation_meta.md#OperationMetadata), [`AttentionMechanisms`](_flexible.md#AttentionMechanisms), [`mesh`](../../infra/base_config.md#EasyDeLBaseConfig.mesh), [`get_combined_requirements`](../../operations/executor.md#OperationExecutor.get_combined_requirements), [`attention_weights`](../../operations/_attention_outputs.md#AttentionOutput.attention_weights), [`from_config`](../../operations/_operation_meta.md#OperationMetadata.from_config), [`get_optimal_config`](_flexible.md#get_optimal_config), [`ScaledDotProductAttn`](../../operations/kernels/scaled_dot_product_attention.md#ScaledDotProductAttn), [`create`](../../operations/_base_operation.md#OperationRegistry.create), [`attn_dtype`](../../infra/base_config.md#EasyDeLBaseConfig.attn_dtype), [`prefill_impl`](../../operations/executor.md#OperationExecutor.prefill_impl), [`decode_impl`](../../operations/executor.md#OperationExecutor.decode_impl), [`requires_cache`](../../operations/executor.md#OperationExecutor.requires_cache), [`attn_softmax_dtype`](../../infra/base_config.md#EasyDeLBaseConfig.attn_softmax_dtype), [`has_separate_decode`](../../operations/executor.md#OperationExecutor.has_separate_decode), [`mixin_impl`](../../operations/executor.md#OperationExecutor.mixin_impl), [`OperationExecutor`](../../operations/executor.md#OperationExecutor), [`cache_view`](../../operations/_attention_outputs.md#AttentionOutput.cache_view), [`VANILLA`](_flexible.md#AttentionMechanisms.VANILLA), [`logger`](_flexible.md#logger), [`_get_jax_dtype_from_string`](_flexible.md#_get_jax_dtype_from_string), [`attn_mechanism`](../../infra/base_config.md#EasyDeLBaseConfig.attn_mechanism), [`decode_attn_mechanism`](../../infra/base_config.md#EasyDeLBaseConfig.decode_attn_mechanism), [`get_unsupported_fallback_features`](../../operations/kernels/scaled_dot_product_attention.md#ScaledDotProductAttn.get_unsupported_fallback_features), [`AUTO`](_flexible.md#AttentionMechanisms.AUTO)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`forward_mla`](../../modules/glm_moe_dsa/modeling_glm_moe_dsa.md#GlmMoeDsaAttention.forward_mla), [`forward`](_unified.md#UnifiedAttention.forward), [`forward_mla`](../../modules/glm4_moe_lite/modeling_glm4_moe_lite.md#Glm4MoeLiteAttention.forward_mla), [`get_operations_cache_info_dynamic`](../../infra/mixins/operation_cache.md#OperationCacheMixin.get_operations_cache_info_dynamic), [`_create_attention_performer`](_unified.md#UnifiedAttention._create_attention_performer), [`__all__`](../../modules/whisper/modeling_whisper.md#__all__), [`forward_mla`](_unified.md#UnifiedAttention.forward_mla), [`forward_alibi`](_unified.md#UnifiedAttention.forward_alibi), [`__call__`](../../modules/gidd/modeling_gidd.md#GiddAttention.__call__), [`__call__`](../../modules/stablelm/modeling_stablelm.md#StableLmAttention.__call__), [`forward`](../../modules/dbrx/modeling_dbrx.md#DbrxAttention.forward), [`_forward_with_kv_capture`](../../modules/gemma4/modeling_gemma4.md#Gemma4Attention._forward_with_kv_capture), [`attention_performer`](_unified.md#UnifiedAttention.attention_performer), [`__call__`](../../modules/grok_1/modeling_grok_1.md#Grok1Attention.__call__), [`forward`](../../modules/openelm/modeling_openelm.md#OpenELMMultiHeadCausalAttention.forward), [`__call__`](../../modules/phi3/modeling_phi3.md#Phi3Attention.__call__), [`__call__`](../../modules/opt/modeling_opt.md#OPTAttention.__call__), [`__call__`](../../modules/roberta/modeling_roberta.md#RobertaSelfAttention.__call__), [`forward`](../../modules/qwen3_next/modeling_qwen3_next.md#Qwen3NextFullAttention.forward), [`forward_alibi`](../../modules/mosaic_mpt/modeling_mosaic.md#MptAttention.forward_alibi), [`__call__`](../../modules/gemma4/modeling_gemma4.md#Gemma4Attention.__call__), [`__call__`](../../modules/gpt2/modeling_gpt2.md#GPT2Attention.__call__), [`__call__`](../../modules/whisper/modeling_whisper.md#WhisperAttention.__call__), [`__call__`](../../modules/gemma4/modeling_gemma4.md#Gemma4VisionAttention.__call__), [`__call__`](../../modules/pixtral/modeling_pixtral.md#PixtralAttention.__call__), [`__call__`](../../modules/llama4/modeling_llama4.md#Llama4VisionAttention.__call__), [`__call__`](../../modules/qwen3_omni_moe/modeling_qwen3_omni_moe.md#Qwen3OmniMoeCode2WavAttention.__call__), [`__call__`](../../modules/clip/modeling_clip.md#CLIPAttention.__call__), [`__call__`](../../modules/siglip/modeling_siglip.md#SiglipAttention.__call__), [`__call__`](../../modules/qwen3_omni_moe/modeling_qwen3_omni_moe.md#Qwen3OmniMoeAudioAttention.__call__), [`_attention`](../../modules/kimi_vl/modeling_kimi_vl.md#MoonVitEncoderLayer._attention), [`__call__`](../../modules/glm4v/modeling_glm4v.md#Glm4vVisionAttention.__call__), [`__call__`](../../modules/qwen3_omni_moe/modeling_qwen3_omni_moe.md#Qwen3OmniMoeVisionAttention.__call__), [`__call__`](../../modules/qwen3_vl/modeling_qwen3_vl.md#Qwen3VLVisionAttention.__call__), [`__call__`](../../modules/qwen3_vl_moe/modeling_qwen3_vl_moe.md#Qwen3VLMoeVisionAttention.__call__), [`_create_attention_performer`](../../modules/mosaic_mpt/modeling_mosaic.md#MptAttention._create_attention_performer), [`_create_attention_performer`](../../modules/deepseek_v2/modeling_deepseek.md#DeepseekV2Attention._create_attention_performer), [`_create_attention_performer`](../../modules/deepseek_v3/modeling_deepseek.md#DeepseekV3Attention._create_attention_performer), [`_create_attention_performer`](../../modules/xerxes2/modeling_xerxes2.md#Xerxes2Attention._create_attention_performer)  (+38 more)

## Functions
- `_get_jax_dtype_from_string(dtype_string: str)` — [`L85`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L85) — Convert string representation to JAX dtype.
- `get_optimal_config()` — [`L189`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L189) — Determine optimal attention configuration for hardware.
- `tpu_version_check(version: str = "v4")` — [`L161`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L161) — Check if running on specified TPU version.

## Module values
- `Cfg` — [`L634`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L634)
- `DEFAULT_ATTENTION_MECHANISM` — [`L239`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L239)
- `logger` — [`L82`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_flexible.py#L82)


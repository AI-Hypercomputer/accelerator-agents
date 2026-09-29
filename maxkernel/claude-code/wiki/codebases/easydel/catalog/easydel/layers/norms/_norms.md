---
title: 'Module: easydel/layers/norms/_norms.py'
type: catalog
provenance: extracted
module: easydel/layers/norms/_norms.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.layers.norms._norms`/
symbols:
  RMSNorm: RMSNorm#
  LayerNorm: LayerNorm#
  BatchNorm.__call__: BatchNorm#__call__().
  LayerNorm.__call__: LayerNorm#__call__().
  lowfloats: lowfloats.
  RMSNorm.__call__: RMSNorm#__call__().
  RMSNorm.kernel: RMSNorm#kernel.
  BatchNorm.scale: BatchNorm#scale.
  BatchNorm.bias: BatchNorm#bias.
  LayerNorm.scale: LayerNorm#scale.
  LayerNorm.bias: LayerNorm#bias.
  RMSNormGated: RMSNormGated#
  RMSNormGated.kernel: RMSNormGated#kernel.
  RMSNormGated.__call__: RMSNormGated#__call__().
  BatchNorm.craft_sharding: BatchNorm#craft_sharding().
  LayerNorm.craft_sharding: LayerNorm#craft_sharding().
  RMSNorm.dtype: RMSNorm#dtype.
  BatchNorm.mean: BatchNorm#mean.
  BatchNorm.var: BatchNorm#var.
  BatchNorm.momentum: BatchNorm#momentum.
  RMSNorm._norm: RMSNorm#_norm().
  BatchNorm.dtype: BatchNorm#dtype.
  LayerNorm.dtype: LayerNorm#dtype.
  BatchNorm.set_mode: BatchNorm#set_mode().
  RMSNorm.param_dtype: RMSNorm#param_dtype.
  BatchNorm.use_running_average: BatchNorm#use_running_average.
  LayerNorm.reduction_axes: LayerNorm#reduction_axes.
  RMSNorm.kernel_init: RMSNorm#kernel_init.
  RMSNorm.dim: RMSNorm#dim.
  RMSNorm.eps: RMSNorm#eps.
  RMSNormGated.kernel_init: RMSNormGated#kernel_init.
  RMSNormGated.eps: RMSNormGated#eps.
  BatchNorm: BatchNorm#
  BatchNorm.axis: BatchNorm#axis.
  BatchNorm.epsilon: BatchNorm#epsilon.
  BatchNorm.axis_name: BatchNorm#axis_name.
  BatchNorm.axis_index_groups: BatchNorm#axis_index_groups.
  BatchNorm.use_fast_variance: BatchNorm#use_fast_variance.
  BatchNorm.promote_dtype: BatchNorm#promote_dtype.
  LayerNorm.epsilon: LayerNorm#epsilon.
  LayerNorm.feature_axes: LayerNorm#feature_axes.
  LayerNorm.axis_name: LayerNorm#axis_name.
  LayerNorm.axis_index_groups: LayerNorm#axis_index_groups.
  LayerNorm.use_fast_variance: LayerNorm#use_fast_variance.
  LayerNorm.promote_dtype: LayerNorm#promote_dtype.
  RMSNorm.__init__: RMSNorm#__init__().
  RMSNorm.craft_sharding: RMSNorm#craft_sharding().
  RMSNormGated.__init__: RMSNormGated#__init__().
  RMSNormGated.hidden_size: RMSNormGated#hidden_size.
  RMSNormGated.dtype: RMSNormGated#dtype.
  RMSNormGated.param_dtype: RMSNormGated#param_dtype.
  RMSNormGated.craft_sharding: RMSNormGated#craft_sharding().
  BatchNorm.__init__: BatchNorm#__init__().
  BatchNorm.num_features: BatchNorm#num_features.
  BatchNorm.param_dtype: BatchNorm#param_dtype.
  BatchNorm.use_bias: BatchNorm#use_bias.
  BatchNorm.use_scale: BatchNorm#use_scale.
  LayerNorm.__init__: LayerNorm#__init__().
  LayerNorm.num_features: LayerNorm#num_features.
  LayerNorm.param_dtype: LayerNorm#param_dtype.
  LayerNorm.use_bias: LayerNorm#use_bias.
  LayerNorm.use_scale: LayerNorm#use_scale.
---
# Module: [`easydel/layers/norms/_norms.py`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py)

## Classes
### `BatchNorm`  ·  implements/extends Module
- def: [`easydel/layers/norms/_norms.py:343`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L343)
- doc: Batch Normalization layer.
- signature: `class BatchNorm(nn.Module):`
- members:
  - `__call__(self, x, use_running_average: bool | None = None, *, mask: jax.Array | None = None)` — [`L467`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L467) — Apply batch normalization to the input. — documented in [easydel-layers-norms-_norms](../../../../concepts/easydel-layers-norms-_norms.md)
  - `__init__(self, num_features: int, *, use_running_average: bool = False, axis: int = -1, momentum: float = 0.99, epsilon: float = 0.00001, dtype: DTypeLike | None = None, param_dtype: DTypeLike = jnp.float32, use_bias: bool = True, use_scale: bool = True, bias_init: nutil.Initializer = nn.initializers.zeros_init(), scale_init: nutil.Initializer = nn.initializers.ones_init(), axis_name: str | None = None, axis_index_groups: tp.Any = None, use_fast_variance: bool = True, promote_dtype: nutil.PromoteDtypeFn = nutil.dtypes.promote_dtype, rngs: nn.Rngs, bias_metadata: collections.abc.Mapping[str, tp.Any] = nutil.MappingProxyType({}), scale_metadata: collections.abc.Mapping[str, tp.Any] = nutil.MappingProxyType({}))` — [`L377`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L377) — Initialize the BatchNorm layer.
  - `craft_sharding(self, *, partition_manager=None, **_kwargs)` — [`L539`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L539) — Return dynamic partition specs for this module's parameters.
  - `set_mode(self, use_running_average: bool | None = None, **kwargs)` — [`L548`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L548) — Class method used by ``nnx.set_mode``.
  - `axis` — [`L455`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L455)
  - `axis_index_groups` — [`L463`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L463)
  - `axis_name` — [`L462`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L462)
  - `bias` — [`L446`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L446)
  - `dtype` — [`L458`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L458)
  - `epsilon` — [`L457`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L457)
  - `mean` — [`L436`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L436)
  - `momentum` — [`L456`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L456)
  - `num_features` — [`L453`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L453)
  - `param_dtype` — [`L459`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L459)
  - `promote_dtype` — [`L465`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L465)
  - `scale` — [`L439`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L439)
  - `use_bias` — [`L460`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L460)
  - `use_fast_variance` — [`L464`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L464)
  - `use_running_average` — [`L454`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L454)
  - `use_scale` — [`L461`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L461)
  - `var` — [`L437`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L437)

### `LayerNorm`  ·  implements/extends Module
- def: [`easydel/layers/norms/_norms.py:564`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L564) — documented in [easydel-layers-attention-_unified](../../../../concepts/easydel-layers-attention-_unified.md)
- doc: Layer Normalization layer.
- signature: `class LayerNorm(nn.Module):`
- members:
  - `__call__(self, x, *, mask: jax.Array | None = None)` — [`L679`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L679) — Apply layer normalization to the input. — documented in [easydel-layers-norms-_norms](../../../../concepts/easydel-layers-norms-_norms.md)
  - `__init__(self, num_features: int, *, epsilon: float = 0.000001, dtype: DTypeLike | None = None, param_dtype: DTypeLike = jnp.float32, use_bias: bool = True, use_scale: bool = True, bias_init: nutil.Initializer = nn.initializers.zeros_init(), scale_init: nutil.Initializer = nn.initializers.ones_init(), reduction_axes: nutil.Axes = -1, feature_axes: nutil.Axes = -1, axis_name: str | None = None, axis_index_groups: tp.Any = None, use_fast_variance: bool = True, promote_dtype: nutil.PromoteDtypeFn = nutil.dtypes.promote_dtype, rngs: nn.Rngs, bias_metadata: collections.abc.Mapping[str, tp.Any] = nutil.MappingProxyType({}), scale_metadata: collections.abc.Mapping[str, tp.Any] = nutil.MappingProxyType({}))` — [`L596`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L596) — Initialize the LayerNorm layer.
  - `craft_sharding(self, *, partition_manager=None, **_kwargs)` — [`L720`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L720) — Return dynamic partition specs for this module's parameters.
  - `axis_index_groups` — [`L675`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L675)
  - `axis_name` — [`L674`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L674)
  - `bias` — [`L659`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L659)
  - `dtype` — [`L668`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L668)
  - `epsilon` — [`L667`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L667)
  - `feature_axes` — [`L673`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L673)
  - `num_features` — [`L666`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L666)
  - `param_dtype` — [`L669`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L669)
  - `promote_dtype` — [`L677`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L677)
  - `reduction_axes` — [`L672`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L672)
  - `scale` — [`L652`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L652)
  - `use_bias` — [`L670`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L670)
  - `use_fast_variance` — [`L676`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L676)
  - `use_scale` — [`L671`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L671)
- used by: [`__all__`](../../modules/whisper/modeling_whisper.md#__all__), [`define_network`](../../modules/xerxes2/modeling_xerxes2.md#Xerxes2Attention.define_network), [`_build_transform_fn`](../../infra/base_module.md#EasyDeLBaseModule._build_transform_fn), [`final_layer_norm`](../../modules/opt/modeling_opt.md#OPTDecoder.final_layer_norm), [`encoder_attn_layer_norm`](../../modules/whisper/modeling_whisper.md#WhisperDecoderLayer.encoder_attn_layer_norm), [`input_layernorm`](../../modules/falcon/modeling_falcon.md#FalconBlock.input_layernorm), [`ln_mlp`](../../modules/falcon/modeling_falcon.md#FalconBlock.ln_mlp), [`ln_attn`](../../modules/falcon/modeling_falcon.md#FalconBlock.ln_attn), [`post_attention_layernorm`](../../modules/falcon/modeling_falcon.md#FalconBlock.post_attention_layernorm), [`layernorm`](../../modules/aya_vision/modeling_aya_vision.md#AyaVisionMultiModalProjector.layernorm), [`LayerNorm`](../../modules/roberta/modeling_roberta.md#RobertaEmbeddings.LayerNorm), [`LayerNorm`](../../modules/roberta/modeling_roberta.md#RobertaOutput.LayerNorm), [`LayerNorm`](../../modules/roberta/modeling_roberta.md#RobertaSelfOutput.LayerNorm), [`final_layer_norm`](../../modules/gpt_neox/modeling_gpt_neox.md#GPTNeoXModel.final_layer_norm), [`final_layer_norm`](../../modules/whisper/modeling_whisper.md#WhisperDecoderLayer.final_layer_norm), [`final_layer_norm`](../../modules/whisper/modeling_whisper.md#WhisperEncoderLayer.final_layer_norm), [`input_layernorm`](../../modules/stablelm/modeling_stablelm.md#StableLmDecoderLayer.input_layernorm), [`layer_norm`](../../modules/roberta/modeling_roberta.md#RobertaLMHead.layer_norm), [`layer_norm`](../../modules/whisper/modeling_whisper.md#WhisperDecoder.layer_norm), [`layer_norm`](../../modules/whisper/modeling_whisper.md#WhisperEncoder.layer_norm), [`self_attn_layer_norm`](../../modules/whisper/modeling_whisper.md#WhisperDecoderLayer.self_attn_layer_norm), [`self_attn_layer_norm`](../../modules/whisper/modeling_whisper.md#WhisperEncoderLayer.self_attn_layer_norm), [`final_layer_norm`](../../modules/opt/modeling_opt.md#OPTDecoderLayer.final_layer_norm), [`post_attention_layernorm`](../../modules/gpt_neox/modeling_gpt_neox.md#GPTNeoXBlock.post_attention_layernorm), [`post_attention_layernorm`](../../modules/stablelm/modeling_stablelm.md#StableLmDecoderLayer.post_attention_layernorm), [`self_attn_layer_norm`](../../modules/opt/modeling_opt.md#OPTDecoderLayer.self_attn_layer_norm), [`final_layer_norm`](../../modules/clip/modeling_clip.md#CLIPTextTransformer.final_layer_norm), [`final_layer_norm`](../../modules/siglip/modeling_siglip.md#SiglipTextTransformer.final_layer_norm), [`final_layernorm`](../../modules/phi/modeling_phi.md#PhiModel.final_layernorm), [`input_layernorm`](../../modules/gpt_neox/modeling_gpt_neox.md#GPTNeoXBlock.input_layernorm), [`input_layernorm`](../../modules/phi/modeling_phi.md#PhiDecoderLayer.input_layernorm), [`input_layernorm`](../../modules/phimoe/modeling_phimoe.md#PhiMoeDecoderLayer.input_layernorm), [`layer_norm1`](../../modules/clip/modeling_clip.md#CLIPEncoderLayer.layer_norm1), [`layer_norm1`](../../modules/siglip/modeling_siglip.md#SiglipEncoderLayer.layer_norm1), [`layer_norm2`](../../modules/clip/modeling_clip.md#CLIPEncoderLayer.layer_norm2), [`layer_norm2`](../../modules/siglip/modeling_siglip.md#SiglipEncoderLayer.layer_norm2), [`layernorm`](../../modules/siglip/modeling_siglip.md#SiglipMultiheadAttentionPoolingHead.layernorm), [`ln_1`](../../modules/gpt_j/modeling_gpt_j.md#GPTJBlock.ln_1), [`ln_2`](../../modules/gpt2/modeling_gpt2.md#GPT2Block.ln_2), [`ln_f`](../../modules/falcon/modeling_falcon.md#FalconModel.ln_f)  (+46 more)

### `RMSNorm`  ·  implements/extends Module
- def: [`easydel/layers/norms/_norms.py:93`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L93) — documented in [easydel-layers-attention-_unified](../../../../concepts/easydel-layers-attention-_unified.md)
- doc: Root Mean Square normalization layer.
- signature: `class RMSNorm(nn.Module):`
- members:
  - `__call__(self, x: Float[Array, "... dim"])` — [`L226`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L226) — Apply RMS normalization to input tensor.
  - `__init__(self, dim: int, eps: float = 0.000001, dtype: DTypeLike = jnp.bfloat16, param_dtype: DTypeLike = jnp.bfloat16, *, rngs: nn.Rngs | None = None)` — [`L148`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L148) — Initialize the RMSNorm layer.
  - `_norm(self, x: Float[Array, "... dim"])` — [`L200`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L200) — Compute Root Mean Square normalization without scaling.
  - `craft_sharding(self, *, partition_manager=None, **_kwargs)` — [`L269`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L269) — Return dynamic partition specs for this module's parameters.
  - `dim` — [`L192`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L192)
  - `dtype` — [`L194`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L194)
  - `eps` — [`L193`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L193)
  - `kernel` — [`L196`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L196) — documented in [easydel-layers-norms-_norms](../../../../concepts/easydel-layers-norms-_norms.md)
  - `kernel_init` — [`L145`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L145)
  - `param_dtype` — [`L195`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L195)
- uses (calls/refs, reference-scoped): [`lowfloats`](_norms.md#lowfloats), [`GptOssRMSNorm`](../../modules/gpt_oss/modeling_gpt_oss.md#GptOssRMSNorm)
- used by: [`define_network`](../../modules/deepseek_v2/modeling_deepseek.md#DeepseekV2Attention.define_network), [`define_network`](../../modules/deepseek_v3/modeling_deepseek.md#DeepseekV3Attention.define_network), [`define_network`](../../modules/glm4_moe_lite/modeling_glm4_moe_lite.md#Glm4MoeLiteAttention.define_network), [`define_network`](../../modules/glm_moe_dsa/modeling_glm_moe_dsa.md#GlmMoeDsaAttention.define_network), [`_create_k_norm`](../attention/_unified.md#UnifiedAttention._create_k_norm), [`_create_q_norm`](../attention/_unified.md#UnifiedAttention._create_q_norm), [`mlp`](../../modules/xerxes/modeling_xerxes.md#XerxesDecoderLayer.mlp), [`mlp`](../../modules/xerxes2/modeling_xerxes2.md#Xerxes2DecoderLayer.mlp), [`key_normalization`](../attention/_unified.md#UnifiedAttention.key_normalization), [`query_normalization`](../attention/_unified.md#UnifiedAttention.query_normalization), [`norm`](../../modules/falcon_mamba/modeling_falcon_mamba.md#FalconMambaBlock.norm), [`norm`](../../modules/mamba/modeling_mamba.md#MambaBlock.norm), [`norm`](../../modules/mamba2/modeling_mamba2.md#Mamba2Block.norm), [`norm`](../../modules/openelm/modeling_openelm.md#OpenELMModel.norm), [`post_layernorm`](../../modules/glm4v/modeling_glm4v.md#Glm4vVisionModel.post_layernorm), [`k_norm`](../../modules/openelm/modeling_openelm.md#OpenELMMultiHeadCausalAttention.k_norm), [`_create_k_norm`](../../modules/olmo2/modeling_olmo2.md#Olmo2Attention._create_k_norm), [`_create_k_norm`](../../modules/olmo3/modeling_olmo3.md#Olmo3Attention._create_k_norm), [`_create_q_norm`](../../modules/olmo2/modeling_olmo2.md#Olmo2Attention._create_q_norm), [`_create_q_norm`](../../modules/olmo3/modeling_olmo3.md#Olmo3Attention._create_q_norm), [`input_layernorm`](../../modules/qwen2/modeling_qwen.md#Qwen2DecoderLayer.input_layernorm), [`input_layernorm`](../../modules/qwen2_moe/modeling_qwen2_moe.md#Qwen2MoeDecoderLayer.input_layernorm), [`input_layernorm`](../../modules/qwen3/modeling_qwen3.md#Qwen3DecoderLayer.input_layernorm), [`input_layernorm`](../../modules/qwen3_moe/modeling_qwen3_moe.md#Qwen3MoeDecoderLayer.input_layernorm), [`ln_1`](../../modules/exaone/modeling_exaone.md#ExaoneDecoderLayer.ln_1), [`ln_2`](../../modules/exaone/modeling_exaone.md#ExaoneDecoderLayer.ln_2), [`norm`](../../modules/glm4_moe_lite/modeling_glm4_moe_lite.md#Glm4MoeLiteModel.norm), [`norm`](../../modules/glm_moe_dsa/modeling_glm_moe_dsa.md#GlmMoeDsaModel.norm), [`norm`](../../modules/minimax/modeling_minimax.md#MiniMaxLightningAttention.norm), [`post_attention_layernorm`](../../modules/phi3/modeling_phi3.md#Phi3DecoderLayer.post_attention_layernorm), [`post_attention_layernorm`](../../modules/qwen2/modeling_qwen.md#Qwen2DecoderLayer.post_attention_layernorm), [`post_attention_layernorm`](../../modules/qwen2_moe/modeling_qwen2_moe.md#Qwen2MoeDecoderLayer.post_attention_layernorm), [`post_attention_layernorm`](../../modules/qwen3/modeling_qwen3.md#Qwen3DecoderLayer.post_attention_layernorm), [`post_attention_layernorm`](../../modules/qwen3_moe/modeling_qwen3_moe.md#Qwen3MoeDecoderLayer.post_attention_layernorm), [`post_attn_norm`](../../modules/grok_1/modeling_grok_1.md#Grok1DecoderLayer.post_attn_norm), [`post_moe_norm`](../../modules/grok_1/modeling_grok_1.md#Grok1DecoderLayer.post_moe_norm), [`pre_attn_norm`](../../modules/grok_1/modeling_grok_1.md#Grok1DecoderLayer.pre_attn_norm), [`pre_moe_norm`](../../modules/grok_1/modeling_grok_1.md#Grok1DecoderLayer.pre_moe_norm), [`_create_k_norm`](../../modules/exaone4/modeling_exaone4.md#Exaone4Attention._create_k_norm), [`_create_q_norm`](../../modules/exaone4/modeling_exaone4.md#Exaone4Attention._create_q_norm)  (+123 more)

### `RMSNormGated`  ·  implements/extends Module
- def: [`easydel/layers/norms/_norms.py:274`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L274)
- doc: Gated Root Mean Square normalization layer.
- signature: `class RMSNormGated(nn.Module):`
- members:
  - `__call__(self, hidden_states: Float[Array, "... hidden_size"], gate: Float[Array, "... hidden_size"])` — [`L320`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L320) — Apply gated RMSNorm normalization.
  - `craft_sharding(self, *, partition_manager=None, **_kwargs)` — [`L315`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L315) — Return sharding specs for normalization parameters.
  - `dtype` — [`L309`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L309)
  - `eps` — [`L308`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L308)
  - `hidden_size` — [`L307`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L307)
  - `kernel` — [`L311`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L311)
  - `kernel_init` — [`L296`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L296)
  - `param_dtype` — [`L310`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L310)
- protocol/private: `__init__`[`L298`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L298)
- used by: [`norm`](../../modules/qwen3_next/modeling_qwen3_next.md#Qwen3NextLinearAttention.norm), [`o_norm`](../../modules/kimi_linear/modeling_kimi_linear.md#KimiDeltaAttention.o_norm)

## Module values
- `lowfloats` — [`L58`](../../../../../../../raw/code/EasyDeL/easydel/layers/norms/_norms.py#L58)


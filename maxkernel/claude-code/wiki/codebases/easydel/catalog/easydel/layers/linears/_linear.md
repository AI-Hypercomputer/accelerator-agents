---
title: 'Module: easydel/layers/linears/_linear.py'
type: catalog
provenance: extracted
module: easydel/layers/linears/_linear.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.layers.linears._linear`/
symbols:
  ColumnParallelLinear: ColumnParallelLinear#
  RowParallelLinear: RowParallelLinear#
  ParallelLinear.to_quantized: ParallelLinear#to_quantized().
  ParallelLinear.kernel: ParallelLinear#kernel.
  ParallelLinear: ParallelLinear#
  ParallelLinear.native_forward: ParallelLinear#native_forward().
  ParallelLinear.__init__: ParallelLinear#__init__().
  ParallelLinear.craft_sharding: ParallelLinear#craft_sharding().
  ParallelLinear.bias: ParallelLinear#bias.
  ParallelLinear._quantized_friend: ParallelLinear#_quantized_friend().
  ParallelLinear.dtype: ParallelLinear#dtype.
  ParallelLinear.kernel_init: ParallelLinear#kernel_init.
  ParallelLinear._direction: ParallelLinear#_direction.
  ParallelLinear.param_dtype: ParallelLinear#param_dtype.
  ParallelLinear.precision: ParallelLinear#precision.
  ParallelLinear.use_bias: ParallelLinear#use_bias.
  Dtype: Dtype.
  Initializer: Initializer.
  ParallelLinear._scale_operator: ParallelLinear#_scale_operator.
  ParallelLinear.bias_init: ParallelLinear#bias_init.
  ParallelLinear.tp_merged: ParallelLinear#tp_merged.
  ParallelLinear.__call__: ParallelLinear#__call__().
  ParallelLinear.in_features: ParallelLinear#in_features.
  ParallelLinear.out_features: ParallelLinear#out_features.
  PrecisionLike: PrecisionLike.
  default_kernel_init: default_kernel_init.
  default_bias_init: default_bias_init.
  Shape: Shape.
  AxisNames: AxisNames.
  ParallelLinear.distributed_matmul: ParallelLinear#distributed_matmul.
  RowParallelLinear._direction: RowParallelLinear#_direction.
  ColumnParallelLinear._direction: ColumnParallelLinear#_direction.
---
# Module: [`easydel/layers/linears/_linear.py`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear.py)

## Classes
### `ColumnParallelLinear`  ·  implements/extends ParallelLinear
- def: [`easydel/layers/linears/_linear.py:484`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear.py#L484) — documented in [easydel-layers-attention-_unified](../../../../concepts/easydel-layers-attention-_unified.md)
- doc: Column-parallel variant of ParallelLinear.
- signature: `class ColumnParallelLinear(ParallelLinear):`
- protocol/private: `_direction`[`L510`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear.py#L510)
- uses (calls/refs, reference-scoped): [`ParallelLinear`](_linear.md#ParallelLinear)
- used by: [`__all__`](../../modules/whisper/modeling_whisper.md#__all__), [`__init__`](../../modules/_base/causal_lm_module.md#BaseCausalLMModule.__init__), [`define_network`](../../modules/deepseek_v2/modeling_deepseek.md#DeepseekV2Attention.define_network), [`define_network`](../../modules/deepseek_v3/modeling_deepseek.md#DeepseekV3Attention.define_network), [`define_network`](../../modules/glm4_moe_lite/modeling_glm4_moe_lite.md#Glm4MoeLiteAttention.define_network), [`from_quantized`](_linear_quantized.md#ParallelLinearQuantized.from_quantized), [`define_network`](../../modules/glm_moe_dsa/modeling_glm_moe_dsa.md#GlmMoeDsaAttention.define_network), [`define_network`](../../modules/kimi_linear/modeling_kimi_linear.md#KimiMLAAttention.define_network), [`define_network`](../../modules/xerxes2/modeling_xerxes2.md#Xerxes2Attention.define_network), [`qk_scale`](../../modules/gidd/modeling_gidd.md#GiddAttention.qk_scale), [`_create_q_proj`](../attention/_unified.md#UnifiedAttention._create_q_proj), [`in_proj`](../../modules/mamba2/modeling_mamba2.md#Mamba2Mixer.in_proj), [`__init__`](../../modules/_base/image_classification_module.md#BaseImageClassificationModule.__init__), [`__init__`](../../modules/_base/question_answering_module.md#BaseQuestionAnsweringModule.__init__), [`_create_k_proj`](../attention/_unified.md#UnifiedAttention._create_k_proj), [`_create_v_proj`](../attention/_unified.md#UnifiedAttention._create_v_proj), [`encoder_attn_layer_norm`](../../modules/whisper/modeling_whisper.md#WhisperDecoderLayer.encoder_attn_layer_norm), [`_score_head_name`](../../modules/_base/sequence_classification_module.md#BaseSequenceClassificationModule._score_head_name), [`ffn_with_glu`](../../modules/openelm/modeling_openelm.md#OpenELMFeedForwardNetwork.ffn_with_glu), [`_create_fused_qkv_proj`](../attention/_unified.md#UnifiedAttention._create_fused_qkv_proj), [`self_attn`](../../modules/whisper/modeling_whisper.md#WhisperEncoderLayer.self_attn), [`_create_fused_qkv_proj`](../../modules/falcon/modeling_falcon.md#FalconAttention._create_fused_qkv_proj), [`qkv_proj`](../../modules/openelm/modeling_openelm.md#OpenELMMultiHeadCausalAttention.qkv_proj), [`ParallelLinear`](_linear.md#ParallelLinear), [`config`](../../modules/mosaic_mpt/modeling_mosaic.md#MptMLP.config), [`in_proj`](../../modules/falcon_h1/modeling_falcon_h1.md#FalconH1Mixer.in_proj), [`in_proj_a`](../../modules/qwen3_next/modeling_qwen3_next.md#Qwen3NextLinearAttention.in_proj_a), [`k_proj`](../../modules/grok_1/modeling_grok_1.md#Grok1Attention.k_proj), [`q_proj`](../../modules/grok_1/modeling_grok_1.md#Grok1Attention.q_proj), [`v_proj`](../../modules/grok_1/modeling_grok_1.md#Grok1Attention.v_proj), [`value_dim`](../../modules/kimi_linear/modeling_kimi_linear.md#KimiDeltaAttention.value_dim), [`dropout`](../../modules/_base/token_classification_module.md#BaseTokenClassificationModule.dropout), [`head_dim`](../../modules/whisper/modeling_whisper.md#WhisperAttention.head_dim), [`__init__`](../../modules/_base/conditional_generation_module.md#BaseConditionalGenerationModule.__init__), [`lm_head`](../../modules/qwen3_omni_moe/modeling_qwen3_omni_moe.md#Qwen3OmniMoeTalkerCodePredictorForConditionalGeneration.lm_head), [`act`](../../modules/gemma2/modeling_gemma2.md#Gemma2MLP.act), [`act`](../../modules/gemma3/modeling_gemma3.md#Gemma3MLP.act), [`dense`](../../modules/roberta/modeling_roberta.md#RobertaIntermediate.dense), [`gate`](../../modules/xerxes/modeling_xerxes.md#XerxesSparseMoeBlock.gate), [`in_proj_qkv`](../../modules/qwen3_next/modeling_qwen3_next.md#Qwen3NextLinearAttention.in_proj_qkv)  (+166 more)

### `ParallelLinear`  ·  implements/extends Module
- def: [`easydel/layers/linears/_linear.py:78`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear.py#L78) — documented in [easydel-layers-linears-_linear](../../../../concepts/easydel-layers-linears-_linear.md)
- doc: A linear transformation layer with optional parallelism support.
- signature: `class ParallelLinear(nn.Module):`
- members:
  - `__call__(self, inputs: Shaped[Array, "... in_features"], w: Array | None = None)` — [`L324`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear.py#L324) — Apply the linear transformation to inputs.
  - `__init__(self, in_features: int, out_features: int, *, scale: float | tp.Literal["fan_in", "fan_out"] = 1, use_bias: bool = True, dtype: Dtype | None = None, param_dtype: Dtype = jnp.float32, precision: PrecisionLike = None, kernel_init: Initializer = default_kernel_init, bias_init: Initializer = default_bias_init, rngs: nn.Rngs | None = None)` — [`L134`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear.py#L134) — Initialize a parallel linear layer.
  - `_quantized_friend(self)` — [`L435`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear.py#L435) — Get the corresponding quantized layer class.
  - `craft_sharding(self, *, partition_manager=None, **_kwargs)` — [`L348`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear.py#L348) — Return dynamic partition specs for this module's parameters.
  - `native_forward(self, inputs: Shaped[Array, "... in_features"], w: Array | None = None)` — [`L245`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear.py#L245) — Apply the linear transformation using native JAX operations. — documented in [easydel-layers-linears-_linear](../../../../concepts/easydel-layers-linears-_linear.md)
  - `to_quantized(self, config: QuantizationConfig, **kwargs)` — [`L380`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear.py#L380) — Convert this layer to a quantized version. — documented in [easydel-layers-linears-_linear](../../../../concepts/easydel-layers-linears-_linear.md)
  - `bias` — [`L240`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear.py#L240)
  - `bias_init` — [`L214`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear.py#L214)
  - `distributed_matmul` — [`L243`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear.py#L243)
  - `dtype` — [`L210`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear.py#L210)
  - `in_features` — [`L206`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear.py#L206)
  - `kernel` — [`L234`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear.py#L234) — documented in [easydel-layers-linears-_linear](../../../../concepts/easydel-layers-linears-_linear.md)
  - `kernel_init` — [`L213`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear.py#L213)
  - `out_features` — [`L207`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear.py#L207)
  - `param_dtype` — [`L211`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear.py#L211)
  - `precision` — [`L212`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear.py#L212)
  - `tp_merged` — [`L222`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear.py#L222)
  - `use_bias` — [`L209`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear.py#L209)
- protocol/private: `_direction`[`L132`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear.py#L132), `_scale_operator`[`L196`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear.py#L196), `_scale_operator`[`L205`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear.py#L205)
- uses (calls/refs, reference-scoped): [`ColumnParallelLinear`](_linear.md#ColumnParallelLinear), [`RowParallelLinear`](_linear.md#RowParallelLinear), [`QuantizationConfig`](../quantization/_configs.md#QuantizationConfig), [`resolve_safe_sharding`](../_sharding.md#resolve_safe_sharding), [`ColumnParallelLinearQuantized`](_linear_quantized.md#ColumnParallelLinearQuantized), [`RowParallelLinearQuantized`](_linear_quantized.md#RowParallelLinearQuantized), [`Dtype`](_linear.md#Dtype), [`Initializer`](_linear.md#Initializer), [`PrecisionLike`](_linear.md#PrecisionLike), [`default_bias_init`](_linear.md#default_bias_init), [`default_kernel_init`](_linear.md#default_kernel_init)
- used by: [`ColumnParallelLinear`](_linear.md#ColumnParallelLinear), [`get_lm_head`](../../infra/base_module.md#EasyDeLBaseModule.get_lm_head), [`RowParallelLinear`](_linear.md#RowParallelLinear), [`forward_mla`](../../modules/glm_moe_dsa/modeling_glm_moe_dsa.md#GlmMoeDsaAttention.forward_mla), [`forward_mla`](../../modules/glm4_moe_lite/modeling_glm4_moe_lite.md#Glm4MoeLiteAttention.forward_mla), [`forward_mla`](../attention/_unified.md#UnifiedAttention.forward_mla), [`from_quantized`](_linear_quantized.md#ParallelLinearQuantized.from_quantized), [`apply_lora_to_layers`](../../infra/utils.md#apply_lora_to_layers), [`make_lm_head_fn`](../../infra/base_module.md#EasyDeLBaseModule.make_lm_head_fn), [`__call__`](../../modules/roberta/modeling_roberta.md#RobertaLMHead.__call__), [`get_module_repr`](../../infra/mixins/protocol.md#get_module_repr), [`get_dtype`](../../modules/qwen3_vl/modeling_qwen3_vl.md#Qwen3VisionTransformerPretrainedModel.get_dtype), [`get_dtype`](../../modules/qwen3_vl_moe/modeling_qwen3_vl_moe.md#Qwen3VLMoeVisionTransformerPretrainedModel.get_dtype), [`get_dtype`](../../modules/qwen2_vl/modeling_qwen2_vl.md#Qwen2VLVisionTransformer.get_dtype), [`__call__`](../../modules/gemma4/modeling_gemma4.md#Gemma4VisionPatchEmbedder.__call__)

### `RowParallelLinear`  ·  implements/extends ParallelLinear
- def: [`easydel/layers/linears/_linear.py:455`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear.py#L455) — documented in [easydel-layers-attention-_unified](../../../../concepts/easydel-layers-attention-_unified.md)
- doc: Row-parallel variant of ParallelLinear.
- signature: `class RowParallelLinear(ParallelLinear):`
- protocol/private: `_direction`[`L481`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear.py#L481)
- uses (calls/refs, reference-scoped): [`ParallelLinear`](_linear.md#ParallelLinear)
- used by: [`__all__`](../../modules/whisper/modeling_whisper.md#__all__), [`define_network`](../../modules/deepseek_v2/modeling_deepseek.md#DeepseekV2Attention.define_network), [`define_network`](../../modules/deepseek_v3/modeling_deepseek.md#DeepseekV3Attention.define_network), [`define_network`](../../modules/glm4_moe_lite/modeling_glm4_moe_lite.md#Glm4MoeLiteAttention.define_network), [`from_quantized`](_linear_quantized.md#ParallelLinearQuantized.from_quantized), [`define_network`](../../modules/glm_moe_dsa/modeling_glm_moe_dsa.md#GlmMoeDsaAttention.define_network), [`define_network`](../../modules/kimi_linear/modeling_kimi_linear.md#KimiMLAAttention.define_network), [`define_network`](../../modules/xerxes2/modeling_xerxes2.md#Xerxes2Attention.define_network), [`_create_o_proj`](../attention/_unified.md#UnifiedAttention._create_o_proj), [`qk_scale`](../../modules/gidd/modeling_gidd.md#GiddAttention.qk_scale), [`ffn_with_glu`](../../modules/openelm/modeling_openelm.md#OpenELMFeedForwardNetwork.ffn_with_glu), [`_create_o_proj`](../../modules/seed_oss/modeling_seed_oss.md#SeedOssAttention._create_o_proj), [`ParallelLinear`](_linear.md#ParallelLinear), [`dense`](../../modules/roberta/modeling_roberta.md#RobertaClassificationHead.dense), [`o_proj`](../../modules/grok_1/modeling_grok_1.md#Grok1Attention.o_proj), [`out_proj`](../../modules/mamba2/modeling_mamba2.md#Mamba2Mixer.out_proj), [`_create_o_proj`](../../modules/falcon/modeling_falcon.md#FalconAttention._create_o_proj), [`out_proj`](../../modules/openelm/modeling_openelm.md#OpenELMMultiHeadCausalAttention.out_proj), [`output_projection`](../attention/_unified.md#UnifiedAttention.output_projection), [`decoder`](../../modules/roberta/modeling_roberta.md#RobertaLMHead.decoder), [`out_proj`](../../modules/falcon_h1/modeling_falcon_h1.md#FalconH1Mixer.out_proj), [`act`](../../modules/gemma2/modeling_gemma2.md#Gemma2MLP.act), [`act`](../../modules/gemma3/modeling_gemma3.md#Gemma3MLP.act), [`dense`](../../modules/roberta/modeling_roberta.md#RobertaOutput.dense), [`linear_1`](../../modules/aya_vision/modeling_aya_vision.md#AyaVisionMultiModalProjector.linear_1), [`linear_1`](../../modules/grok_1/modeling_grok_1.md#Grok1BLockSparseMLP.linear_1), [`num_key_value_groups`](../../modules/pixtral/modeling_pixtral.md#PixtralAttention.num_key_value_groups), [`o_proj`](../../modules/qwen3_omni_moe/modeling_qwen3_omni_moe.md#Qwen3OmniMoeCode2WavAttention.o_proj), [`out_proj`](../../modules/minimax/modeling_minimax.md#MiniMaxLightningAttention.out_proj), [`intermediate_size`](../../modules/glm4_moe/modeling_glm4_moe.md#Glm4MoeMLP.intermediate_size), [`intermediate_size`](../../modules/glm4_moe_lite/modeling_glm4_moe_lite.md#Glm4MoeLiteMLP.intermediate_size), [`intermediate_size`](../../modules/glm_moe_dsa/modeling_glm_moe_dsa.md#GlmMoeDsaMLP.intermediate_size), [`project_out`](../../modules/opt/modeling_opt.md#OPTDecoder.project_out), [`__init__`](../../modules/whisper/modeling_whisper.md#WhisperForConditionalGeneration.__init__), [`_create_o_proj`](../../modules/glm4v/modeling_glm4v.md#Glm4vTextAttention._create_o_proj), [`_create_o_proj`](../../modules/glm4v_moe/modeling_glm4v_moe.md#Glm4vMoeTextAttention._create_o_proj), [`precision`](../../modules/gidd/modeling_gidd.md#GiddMLP.precision), [`precision`](../../modules/llama/modeling_llama.md#LlamaMLP.precision), [`precision`](../../modules/phimoe/modeling_phimoe.md#PhiMoEBlockSparseTop2MLP.precision), [`mlp`](../../modules/qwen3_omni_moe/modeling_qwen3_omni_moe.md#Qwen3OmniMoeVisionPatchMerger.mlp)  (+81 more)

## Module values
- `AxisNames` — [`L71`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear.py#L71)
- `Dtype` — [`L67`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear.py#L67)
- `Initializer` — [`L68`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear.py#L68)
- `PrecisionLike` — [`L69`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear.py#L69)
- `Shape` — [`L70`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear.py#L70)
- `default_bias_init` — [`L75`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear.py#L75)
- `default_kernel_init` — [`L74`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear.py#L74)


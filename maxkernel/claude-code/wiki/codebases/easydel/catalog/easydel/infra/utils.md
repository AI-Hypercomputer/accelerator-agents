---
title: 'Module: easydel/infra/utils.py'
type: catalog
provenance: extracted
module: easydel/infra/utils.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.infra.utils`/
symbols:
  auto_remat: auto_remat().
  ACT2FN: ACT2FN.
  ArrayParam: ArrayParam#
  ProcessingClassType: ProcessingClassType.
  block_wise_ffn: block_wise_ffn().
  flops_per_token: flops_per_token().
  count_flop_jaxpr: count_flop_jaxpr().
  AttnMaskDetail: AttnMaskDetail#
  ArrayParam.bound: ArrayParam#bound().
  AttnMaskType: AttnMaskType#
  flop_seq2seq: flop_seq2seq().
  flop_activation: flop_activation().
  apply_lora_to_layers: apply_lora_to_layers().
  AttnMaskType.from_hf: AttnMaskType#from_hf().
  flop_transformer_body: flop_transformer_body().
  count_flop_jaxpr.compute_binary_op_flops: count_flop_jaxpr().compute_binary_op_flops().
  flop_mlp: flop_mlp().
  ActivationType: ActivationType#
  flop_vision_tower: flop_vision_tower().
  merge_lora_params: merge_lora_params().
  unwrap_lora_to_layers: unwrap_lora_to_layers().
  ModuleCaches: ModuleCaches#
  count_flop_jaxpr.compute_unary_op_flops: count_flop_jaxpr().compute_unary_op_flops().
  FlopCalcConfig.hidden_dim: FlopCalcConfig#hidden_dim.
  split_lora_params: split_lora_params().
  FlopCalcConfig.task: FlopCalcConfig#task.
  AttnMaskType.SLIDING: AttnMaskType#SLIDING.
  apply_sparsity_to_params: apply_sparsity_to_params().
  sanitize_partition_specs_for_shape_tree: sanitize_partition_specs_for_shape_tree().
  get_gradient_checkpoint_policy: get_gradient_checkpoint_policy().
  M: M.
  count_flop_jaxpr.get_shape_size: count_flop_jaxpr().get_shape_size().
  trace_functions: trace_functions().
  ArrayParam.resure: ArrayParam#resure().
  materialize_meta_leaves: materialize_meta_leaves().
  CompilationTracker.trace_compilation: CompilationTracker#trace_compilation().
  sanitize_partition_spec_for_shape: sanitize_partition_spec_for_shape().
  quantize_linear_layers: quantize_linear_layers().
  FlopCalcConfig: FlopCalcConfig#
  FlopCalcConfig.intermediate_dim: FlopCalcConfig#intermediate_dim.
  FlopCalcConfig.seq_len: FlopCalcConfig#seq_len.
  FlopCalcConfig.activation_type: FlopCalcConfig#activation_type.
  FlopCalcConfig.vocab_size: FlopCalcConfig#vocab_size.
  apply_sparsity_to_params._with_progress: apply_sparsity_to_params()._with_progress().
  count_flop_jaxpr.compute_max_flops: count_flop_jaxpr().compute_max_flops().
  count_flop_jaxpr.compute_min_flops: count_flop_jaxpr().compute_min_flops().
  CompilationTracker: CompilationTracker#
  FlopCalcConfig.num_layers: FlopCalcConfig#num_layers.
  _select_checkpoint_names_by_regex: _select_checkpoint_names_by_regex().
  hashable_dict: hashable_dict#
  FlopCalcConfig.include_loss: FlopCalcConfig#include_loss.
  count_flop_jaxpr.visit_jaxpr: count_flop_jaxpr().visit_jaxpr().
  FunctionTracer.new_executables: FunctionTracer#new_executables.
  FlopCalcConfig.num_heads: FlopCalcConfig#num_heads.
  FlopCalcConfig.num_labels: FlopCalcConfig#num_labels.
  flop_loss: flop_loss().
  apply_sparsity_to_params.filter_params: apply_sparsity_to_params().filter_params().
  count_flop_jaxpr.count_scan_flops: count_flop_jaxpr().count_scan_flops().
  count_flop_jaxpr.count_cond_flops: count_flop_jaxpr().count_cond_flops().
  count_flop_jaxpr.get_scatter_flops: count_flop_jaxpr().get_scatter_flops().
  count_flop_jaxpr.compute_select_n_flops: count_flop_jaxpr().compute_select_n_flops().
  count_flop_jaxpr.compute_cumsum_flops: count_flop_jaxpr().compute_cumsum_flops().
  count_flop_jaxpr.compute_pow_flops: count_flop_jaxpr().compute_pow_flops().
  count_flop_jaxpr.compute_integer_pow_flops: count_flop_jaxpr().compute_integer_pow_flops().
  count_flop_jaxpr.compute_and_flops: count_flop_jaxpr().compute_and_flops().
  count_flop_jaxpr.count_custom_vjp_flops: count_flop_jaxpr().count_custom_vjp_flops().
  count_flop_jaxpr.compute_sqrt_flops: count_flop_jaxpr().compute_sqrt_flops().
  count_flop_jaxpr.compute_argmax_flops: count_flop_jaxpr().compute_argmax_flops().
  count_flop_jaxpr.compute_rem_flops: count_flop_jaxpr().compute_rem_flops().
  count_flop_jaxpr.compute_square_flops: count_flop_jaxpr().compute_square_flops().
  count_flop_jaxpr.compute_erf_inv_flops: count_flop_jaxpr().compute_erf_inv_flops().
  count_flop_jaxpr.compute_or_flops: count_flop_jaxpr().compute_or_flops().
  count_flop_jaxpr.compute_shift_right_logical_flops: count_flop_jaxpr().compute_shift_right_logical_flops().
  TraceResult.cost_analysis: TraceResult#cost_analysis().
  AttnMaskDetail.mask_type: AttnMaskDetail#mask_type.
  ArrayParam.init_kwargs: ArrayParam#init_kwargs.
  count_flop_jaxpr.compute_reduce_flops: count_flop_jaxpr().compute_reduce_flops().
  CompilationTracker.functions: CompilationTracker#functions.
  AttnMaskType.FULL: AttnMaskType#FULL.
  FlopCalcConfig.kv_heads: FlopCalcConfig#kv_heads.
  FlopCalcConfig.head_dim: FlopCalcConfig#head_dim.
  FlopCalcConfig.num_experts: FlopCalcConfig#num_experts.
  FlopCalcConfig.num_experts_per_tok: FlopCalcConfig#num_experts_per_tok.
  flop_lm_head: flop_lm_head().
  ArrayParam.init_method: ArrayParam#init_method.
  add_start_docstrings: add_start_docstrings().
  TraceResult.flops: TraceResult#flops().
  FunctionTracer.__getitem__: FunctionTracer#__getitem__().
  CompilationTracker.online_flops: CompilationTracker#online_flops().
  AttnMaskDetail.bricks: AttnMaskDetail#bricks.
  _ATTN_CHECKPOINT_NAME_PATTERN: _ATTN_CHECKPOINT_NAME_PATTERN.
  _MLP_CHECKPOINT_NAME_PATTERN: _MLP_CHECKPOINT_NAME_PATTERN.
  is_flatten: is_flatten().
  apply_sparsity_to_params._path_to_str: apply_sparsity_to_params()._path_to_str().
  TraceResult: TraceResult#
  FunctionTracer._before: FunctionTracer#_before.
  CompilationTracker.first_time: CompilationTracker#first_time.
  ActivationType.GELU: ActivationType#GELU.
  ActivationType.SILU: ActivationType#SILU.
  AttnMaskType.CHUNK: AttnMaskType#CHUNK.
  FlopCalcConfig.enc_seq_len: FlopCalcConfig#enc_seq_len.
  FlopCalcConfig.glu: FlopCalcConfig#glu.
  FlopCalcConfig.num_shared_experts: FlopCalcConfig#num_shared_experts.
  FlopCalcConfig.vision_hidden_dim: FlopCalcConfig#vision_hidden_dim.
  FlopCalcConfig.vision_intermediate_dim: FlopCalcConfig#vision_intermediate_dim.
  FlopCalcConfig.vision_num_layers: FlopCalcConfig#vision_num_layers.
  FlopCalcConfig.vision_seq_len: FlopCalcConfig#vision_seq_len.
  flop_layernorm: flop_layernorm().
  flop_attention: flop_attention().
  flop_cls_head: flop_cls_head().
  quick_gelu: quick_gelu().
  add_start_docstrings.docstring_decorator: add_start_docstrings().docstring_decorator().
  extract_static_parameters: extract_static_parameters().
  count_flop_jaxpr.compute_dot_general_flops: count_flop_jaxpr().compute_dot_general_flops().
  count_flop_jaxpr.compute_conv_flops: count_flop_jaxpr().compute_conv_flops().
  count_flop_jaxpr.compute_attention_flops: count_flop_jaxpr().compute_attention_flops().
  count_flop_jaxpr.compute_triangular_solve_flops: count_flop_jaxpr().compute_triangular_solve_flops().
  TraceResult._executable: TraceResult#_executable.
  FunctionTracer: FunctionTracer#
  CompilationTracker.cached_flops: CompilationTracker#cached_flops.
  ActivationType.RELU: ActivationType#RELU.
  ActivationType.SWISH: ActivationType#SWISH.
  ActivationType.GELU_NEW: ActivationType#GELU_NEW.
  ActivationType.GELU_PYTORCH_TANH: ActivationType#GELU_PYTORCH_TANH.
  ActivationType.TANH: ActivationType#TANH.
  ActivationType.SIGMOID: ActivationType#SIGMOID.
  ActivationType.LEAKY_RELU: ActivationType#LEAKY_RELU.
  ActivationType.GLU: ActivationType#GLU.
  ActivationType.ELU: ActivationType#ELU.
  ActivationType.SOFTMAX: ActivationType#SOFTMAX.
  ActivationType.QUICK_GELU: ActivationType#QUICK_GELU.
  AttnMaskDetail.size: AttnMaskDetail#size.
  AttnMaskDetail.chunks: AttnMaskDetail#chunks.
  FlopCalcConfig.enc_num_layers: FlopCalcConfig#enc_num_layers.
  FlopCalcConfig.vision_num_heads: FlopCalcConfig#vision_num_heads.
  flop_cross_attention: flop_cross_attention().
  ArrayParam.shape: ArrayParam#shape.
  ArrayParam.dtype: ArrayParam#dtype.
  mesh_partition_product: mesh_partition_product().
  logger: logger.
  ROPE_TYPES: ROPE_TYPES.
  canonicalize_dtype: canonicalize_dtype().
  create_transformer_checkpoint_policy: create_transformer_checkpoint_policy().
  TraceResult.__init__: TraceResult#__init__().
  TraceResult._cached_cost: TraceResult#_cached_cost.
  FunctionTracer.__init__: FunctionTracer#__init__().
  CompilationTracker.__init__: CompilationTracker#__init__().
  AttnMaskType.LINEAR: AttnMaskType#LINEAR.
  AttnMaskDetail.offset: AttnMaskDetail#offset.
  OverWriteWithGradient: OverWriteWithGradient#
  hashable_dict.__hash__: hashable_dict#__hash__.
---
# Module: [`easydel/infra/utils.py`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py)

## Classes
### `ActivationType`  ·  implements/extends StrEnum
- def: [`easydel/infra/utils.py:1260`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1260)
- signature: `class ActivationType(StrEnum):`
- members:
  - `ELU` — [`L1271`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1271)
  - `GELU` — [`L1261`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1261)
  - `GELU_NEW` — [`L1265`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1265)
  - `GELU_PYTORCH_TANH` — [`L1266`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1266)
  - `GLU` — [`L1270`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1270)
  - `LEAKY_RELU` — [`L1269`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1269)
  - `QUICK_GELU` — [`L1273`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1273)
  - `RELU` — [`L1262`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1262)
  - `SIGMOID` — [`L1268`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1268)
  - `SILU` — [`L1263`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1263)
  - `SOFTMAX` — [`L1272`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1272)
  - `SWISH` — [`L1264`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1264)
  - `TANH` — [`L1267`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1267)
- used by: [`flops_per_token`](base_module.md#EasyDeLBaseModule.flops_per_token), [`flop_activation`](utils.md#flop_activation), [`activation_type`](utils.md#FlopCalcConfig.activation_type)

### `ArrayParam`  ·  implements/extends Param
- def: [`easydel/infra/utils.py:1688`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1688) — documented in [easydel-infra-utils](../../../concepts/easydel-infra-utils.md)
- doc: Parameterized array with serializable initialization.
- signature: `class ArrayParam(nn.Param):`
- members:
  - `bound(cls, shape: Sequence[int], dtype: DTypeLike, init_method: str, init_kwargs: hashable_dict | None = None, *, key: PRNGKeyArray | None = None, value: Array | None = None, use_ref: bool | None = None, **metadata)` — [`L1709`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1709) — Create an ArrayParam with initialized value.
  - `resure(self, key: PRNGKeyArray, shard_fn: tp.Callable[[Array], Array] | None = None)` — [`L1759`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1759) — Reinitialize the parameter value with a new random key.
  - `dtype` — [`L1704`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1704)
  - `init_kwargs` — [`L1706`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1706)
  - `init_method` — [`L1705`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1705)
  - `shape` — [`L1703`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1703)
- uses (calls/refs, reference-scoped): [`hashable_dict`](utils.md#hashable_dict)
- used by: [`qk_scale`](../modules/gidd/modeling_gidd.md#GiddAttention.qk_scale), [`sequential_init`](base_module.md#EasyDeLBaseModule.sequential_init), [`dt_bias`](../modules/kimi_linear/modeling_kimi_linear.md#KimiDeltaAttention.dt_bias), [`bias`](../modules/roberta/modeling_roberta.md#RobertaLMHead.bias), [`kernel`](../modules/deepseek_v2/modeling_deepseek.md#MoEGate.kernel), [`kernel`](../modules/glm4_moe/modeling_glm4_moe.md#Glm4MoeTopKRouter.kernel), [`positional_embedding_vlm`](../modules/llama4/modeling_llama4.md#Llama4VisionModel.positional_embedding_vlm), [`D`](../modules/falcon_h1/modeling_falcon_h1.md#FalconH1Mixer.D), [`A_log`](../modules/mamba2/modeling_mamba2.md#Mamba2Mixer.A_log), [`D`](../modules/mamba2/modeling_mamba2.md#Mamba2Mixer.D), [`class_embedding`](../modules/llama4/modeling_llama4.md#Llama4VisionModel.class_embedding), [`dt_bias`](../modules/mamba2/modeling_mamba2.md#Mamba2Mixer.dt_bias), [`kernel`](../modules/cohere/modeling_cohere.md#RMSNorm.kernel), [`kernel`](../modules/cohere2/modeling_cohere2.md#Cohere2LayerNorm.kernel), [`kernel`](../modules/gemma/modeling_gemma.md#GemmaRMSNorm.kernel), [`kernel`](../modules/gemma2/modeling_gemma2.md#Gemma2RMSNorm.kernel), [`kernel`](../modules/gidd/modeling_gidd.md#GiddRMSNorm.kernel), [`kernel`](../modules/kimi_linear/modeling_kimi_linear.md#KimiMoEGate.kernel), [`kernel`](../modules/mamba2/modeling_mamba2.md#MambaRMSNormGated.kernel), [`mm_input_projection_weight`](../modules/gemma3/modeling_gemma3.md#Gemma3MultiModalProjector.mm_input_projection_weight), [`position_embedding_table`](../modules/gemma4/modeling_gemma4.md#Gemma4VisionPatchEmbedder.position_embedding_table), [`sinks`](../modules/gpt_oss/modeling_gpt_oss.md#GptOssAttention.sinks), [`A_log`](../modules/falcon_h1/modeling_falcon_h1.md#FalconH1Mixer.A_log), [`dt_bias`](../modules/falcon_h1/modeling_falcon_h1.md#FalconH1Mixer.dt_bias), [`kernel`](../modules/opt/modeling_opt.md#OPTLearnedPositionalEmbedding.kernel), [`A_log`](../modules/kimi_linear/modeling_kimi_linear.md#KimiDeltaAttention.A_log), [`e_score_correction_bias`](../modules/deepseek_v3/modeling_deepseek.md#MoEGate.e_score_correction_bias), [`e_score_correction_bias`](../modules/glm4_moe/modeling_glm4_moe.md#Glm4MoeTopKRouter.e_score_correction_bias), [`e_score_correction_bias`](../modules/kimi_linear/modeling_kimi_linear.md#KimiMoEGate.e_score_correction_bias), [`kernel`](../modules/deepseek_v3/modeling_deepseek.md#MoEGate.kernel), [`logit_scale`](../modules/clip/modeling_clip.md#CLIPModel.logit_scale), [`per_expert_scale`](../modules/gemma4/modeling_gemma4.md#Gemma4TextRouter.per_expert_scale), [`probe`](../modules/siglip/modeling_siglip.md#SiglipMultiheadAttentionPoolingHead.probe), [`scale`](../modules/gemma4/modeling_gemma4.md#Gemma4TextRouter.scale), [`std_bias`](../modules/gemma4/modeling_gemma4.md#Gemma4VisionModel.std_bias), [`std_scale`](../modules/gemma4/modeling_gemma4.md#Gemma4VisionModel.std_scale), [`v1`](../modules/dbrx/modeling_dbrx.md#DbrxExpertGLU.v1), [`w1`](../modules/dbrx/modeling_dbrx.md#DbrxExpertGLU.w1), [`w2`](../modules/dbrx/modeling_dbrx.md#DbrxExpertGLU.w2), [`A_log`](../modules/falcon_mamba/modeling_falcon_mamba.md#FalconMambaMixer.A_log)  (+23 more)

### `AttnMaskDetail`
- def: [`easydel/infra/utils.py:1332`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1332)
- doc: Details for attention mask configuration.
- signature: `class AttnMaskDetail:`
- members:
  - `bricks` — [`L1357`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1357)
  - `chunks` — [`L1356`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1356)
  - `mask_type` — [`L1353`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1353)
  - `offset` — [`L1355`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1355)
  - `size` — [`L1354`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1354)
- uses (calls/refs, reference-scoped): [`TaskType`](factory.md#TaskType), [`AttnMaskType`](utils.md#AttnMaskType)
- used by: [`concatenate`](../layers/attention/_flexible.md#AttentionModule.concatenate), [`create_kv_cache_specs_from_config`](../inference/esurge/core/interface.md#create_kv_cache_specs_from_config), [`get_mask_details`](base_config.md#EasyDeLBaseConfig.get_mask_details), [`get_mask_details`](../modules/minimax/minimax_configuration.md#MiniMaxConfig.get_mask_details), [`_apply_sliding_window`](../layers/attention/_flexible.md#AttentionModule._apply_sliding_window), [`_handle_cache_concat`](../layers/attention/_flexible.md#AttentionModule._handle_cache_concat), [`get_mask_details`](../modules/qwen3_moe/qwen3_moe_configuration.md#Qwen3MoeConfig.get_mask_details), [`get_mask_details`](../modules/qwen3_omni_moe/qwen3_omni_moe_configuration.md#Qwen3OmniMoeTextConfig.get_mask_details), [`get_mask_details`](../modules/qwen3_vl/qwen3_vl_configuration.md#Qwen3VLTextConfig.get_mask_details), [`get_mask_details`](../modules/qwen3_vl_moe/qwen3_vl_moe_configuration.md#Qwen3VLMoeTextConfig.get_mask_details), [`get_mask_details`](../modules/arctic/arctic_configuration.md#ArcticConfig.get_mask_details), [`get_mask_details`](../modules/cohere2/cohere2_configuration.md#Cohere2Config.get_mask_details), [`get_mask_details`](../modules/gemma2/gemma2_configuration.md#Gemma2Config.get_mask_details), [`get_mask_details`](../modules/gemma3/gemma3_configuration.md#Gemma3TextConfig.get_mask_details), [`get_mask_details`](../modules/gemma4/gemma4_configuration.md#Gemma4TextConfig.get_mask_details), [`get_mask_details`](../modules/mistral/mistral_configuration.md#MistralConfig.get_mask_details), [`get_mask_details`](../modules/mixtral/mixtral_configuration.md#MixtralConfig.get_mask_details), [`get_mask_details`](../modules/phi3/phi3_configuration.md#Phi3Config.get_mask_details), [`get_mask_details`](../modules/phimoe/phimoe_configuration.md#PhiMoeConfig.get_mask_details), [`get_mask_details`](../modules/qwen2/qwen_configuration.md#Qwen2Config.get_mask_details), [`get_mask_details`](../modules/qwen2_moe/qwen2_moe_configuration.md#Qwen2MoeConfig.get_mask_details), [`get_mask_details`](../modules/qwen3/qwen3_configuration.md#Qwen3Config.get_mask_details), [`get_mask_details`](../modules/xerxes/xerxes_configuration.md#XerxesConfig.get_mask_details), [`get_mask_details`](../modules/seed_oss/seed_oss_configuration.md#SeedOssConfig.get_mask_details), [`get_mask_details`](../modules/qwen2_vl/qwen2_vl_configuration.md#Qwen2VLConfig.get_mask_details)

### `AttnMaskType`  ·  implements/extends StrEnum
- def: [`easydel/infra/utils.py:1298`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1298)
- signature: `class AttnMaskType(StrEnum):`
- members:
  - `from_hf(cls, hf_type: tp.Literal["sliding_attention", "full_attention", "chunk_attention", "chunked_attention", "linear_attention", "kda_linear_attention", "hybrid", "parallel_hybrid",])` — [`L1305`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1305)
  - `CHUNK` — [`L1301`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1301)
  - `FULL` — [`L1299`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1299)
  - `LINEAR` — [`L1302`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1302)
  - `SLIDING` — [`L1300`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1300)
- used by: [`concatenate`](../layers/attention/_flexible.md#AttentionModule.concatenate), [`create_kv_cache_specs_from_config`](../inference/esurge/core/interface.md#create_kv_cache_specs_from_config), [`get_mask_details`](base_config.md#EasyDeLBaseConfig.get_mask_details), [`concatenate_to_cache`](../caching/transformer/cache.md#TransformerCacheView.concatenate_to_cache), [`init`](../caching/transformer/cache.md#TransformerCacheView.init), [`get_mask_details`](../modules/minimax/minimax_configuration.md#MiniMaxConfig.get_mask_details), [`get_mask_details`](../modules/qwen3_moe/qwen3_moe_configuration.md#Qwen3MoeConfig.get_mask_details), [`get_mask_details`](../modules/qwen3_omni_moe/qwen3_omni_moe_configuration.md#Qwen3OmniMoeTextConfig.get_mask_details), [`get_mask_details`](../modules/qwen3_vl/qwen3_vl_configuration.md#Qwen3VLTextConfig.get_mask_details), [`get_mask_details`](../modules/qwen3_vl_moe/qwen3_vl_moe_configuration.md#Qwen3VLMoeTextConfig.get_mask_details), [`get_mask_details`](../modules/arctic/arctic_configuration.md#ArcticConfig.get_mask_details), [`get_mask_details`](../modules/cohere2/cohere2_configuration.md#Cohere2Config.get_mask_details), [`get_mask_details`](../modules/gemma2/gemma2_configuration.md#Gemma2Config.get_mask_details), [`get_mask_details`](../modules/gemma3/gemma3_configuration.md#Gemma3TextConfig.get_mask_details), [`get_mask_details`](../modules/gemma4/gemma4_configuration.md#Gemma4TextConfig.get_mask_details), [`get_mask_details`](../modules/mistral/mistral_configuration.md#MistralConfig.get_mask_details), [`get_mask_details`](../modules/mixtral/mixtral_configuration.md#MixtralConfig.get_mask_details), [`get_mask_details`](../modules/phi3/phi3_configuration.md#Phi3Config.get_mask_details), [`get_mask_details`](../modules/phimoe/phimoe_configuration.md#PhiMoeConfig.get_mask_details), [`get_mask_details`](../modules/qwen2/qwen_configuration.md#Qwen2Config.get_mask_details), [`get_mask_details`](../modules/qwen2_moe/qwen2_moe_configuration.md#Qwen2MoeConfig.get_mask_details), [`get_mask_details`](../modules/qwen3/qwen3_configuration.md#Qwen3Config.get_mask_details), [`get_mask_details`](../modules/xerxes/xerxes_configuration.md#XerxesConfig.get_mask_details), [`get_mask_details`](../modules/seed_oss/seed_oss_configuration.md#SeedOssConfig.get_mask_details), [`get_mask_details`](../modules/qwen2_vl/qwen2_vl_configuration.md#Qwen2VLConfig.get_mask_details), [`mask_type`](utils.md#AttnMaskDetail.mask_type)

### `CompilationTracker`
- def: [`easydel/infra/utils.py:1198`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1198)
- doc: Tracks XLA compilation and FLOP counts across function calls.
- signature: `class CompilationTracker:`
- members:
  - `online_flops(self)` — [`L1229`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1229)
  - `trace_compilation(self)` — [`L1241`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1241)
  - `cached_flops` — [`L1225`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1225)
  - `first_time` — [`L1224`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1224)
  - `functions` — [`L1226`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1226)
- protocol/private: `__init__`[`L1223`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1223)
- used by: [`_train_epoch`](../trainers/trainer/trainer.md#Trainer._train_epoch), [`_eval_dataset_steps_auto_clamped`](../trainers/base_trainer.md#BaseTrainer._eval_dataset_steps_auto_clamped), [`_eval_epoch`](../trainers/trainer/trainer.md#Trainer._eval_epoch), [`evalu_tracker`](../trainers/trainer_protocol.md#BaseTrainerProtocol.evalu_tracker), [`train_tracker`](../trainers/trainer_protocol.md#BaseTrainerProtocol.train_tracker)

### `FlopCalcConfig`
- def: [`easydel/infra/utils.py:1364`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1364)
- doc: Configuration for calculating FLOPs in transformer models.
- signature: `class FlopCalcConfig:`
- members:
  - `activation_type` — [`L1431`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1431)
  - `enc_num_layers` — [`L1421`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1421)
  - `enc_seq_len` — [`L1422`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1422)
  - `glu` — [`L1425`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1425)
  - `head_dim` — [`L1417`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1417)
  - `hidden_dim` — [`L1412`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1412)
  - `include_loss` — [`L1443`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1443)
  - `intermediate_dim` — [`L1413`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1413)
  - `kv_heads` — [`L1416`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1416)
  - `num_experts` — [`L1426`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1426)
  - `num_experts_per_tok` — [`L1428`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1428)
  - `num_heads` — [`L1415`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1415)
  - `num_labels` — [`L1434`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1434)
  - `num_layers` — [`L1414`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1414)
  - `num_shared_experts` — [`L1427`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1427)
  - `seq_len` — [`L1418`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1418)
  - `task` — [`L1432`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1432)
  - `vision_hidden_dim` — [`L1437`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1437)
  - `vision_intermediate_dim` — [`L1438`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1438)
  - `vision_num_heads` — [`L1440`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1440)
  - `vision_num_layers` — [`L1439`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1439)
  - `vision_seq_len` — [`L1441`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1441)
  - `vocab_size` — [`L1433`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1433)
- uses (calls/refs, reference-scoped): [`TaskType`](factory.md#TaskType), [`ActivationType`](utils.md#ActivationType), [`AUTO_BIND`](factory.md#TaskType.AUTO_BIND), [`GELU`](utils.md#ActivationType.GELU)
- used by: [`flops_per_token`](base_module.md#EasyDeLBaseModule.flops_per_token), [`flops_per_token`](utils.md#flops_per_token), [`flop_seq2seq`](utils.md#flop_seq2seq), [`flop_transformer_body`](utils.md#flop_transformer_body), [`flop_mlp`](utils.md#flop_mlp), [`flop_vision_tower`](utils.md#flop_vision_tower)

### `FunctionTracer`
- def: [`easydel/infra/utils.py:1173`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1173)
- doc: Tracer for capturing new XLA executables during compilation.
- signature: `class FunctionTracer:`
- members:
  - `new_executables` — [`L1191`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1191)
- protocol/private: `__getitem__`[`L1194`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1194), `__init__`[`L1190`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1190), `_before`[`L1192`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1192)
- uses (calls/refs, reference-scoped): [`TraceResult`](utils.md#TraceResult)
- used by: [`trace_functions`](utils.md#trace_functions)

### `ModuleCaches`  ·  implements/extends Cache
- def: [`easydel/infra/utils.py:1666`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1666)
- doc: Cache container for module-level cached values.
- signature: `class ModuleCaches(nn.Cache):`
- used by: [`get_basic_frequencies`](base_config.md#EasyDeLBaseConfig.get_basic_frequencies), [`get_basic_causal_mask`](base_config.md#EasyDeLBaseConfig.get_basic_causal_mask), [`default_frequencies`](../modules/gemma4/modeling_gemma4.md#Gemma4TextModel.default_frequencies), [`get_basic_inv_frequencies`](base_config.md#EasyDeLBaseConfig.get_basic_inv_frequencies), [`global_frequencies`](../modules/gemma4/modeling_gemma4.md#Gemma4TextModel.global_frequencies), [`default_frequencies`](../modules/gemma3/modeling_gemma3.md#Gemma3TextModel.default_frequencies), [`default_frequencies`](../modules/xerxes/modeling_xerxes.md#XerxesModel.default_frequencies)

### `OverWriteWithGradient`  ·  implements/extends Param
- def: [`easydel/infra/utils.py:1675`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1675)
- doc: Parameter type that allows gradient overwrites.
- signature: `class OverWriteWithGradient(nn.Param):`

### `TraceResult`
- def: [`easydel/infra/utils.py:1144`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1144)
- doc: Container for XLA executable trace results with cost analysis.
- signature: `class TraceResult:`
- members:
  - `cost_analysis(self)` — [`L1165`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1165)
  - `flops(self)` — [`L1169`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1169)
- protocol/private: `__init__`[`L1159`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1159), `_cached_cost`[`L1161`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1161), `_executable`[`L1160`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1160)
- used by: [`trace_functions`](utils.md#trace_functions), [`new_executables`](utils.md#FunctionTracer.new_executables)

### `hashable_dict`  ·  implements/extends dict
- def: [`easydel/infra/utils.py:1684`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1684)
- signature: `class hashable_dict(dict):`
- protocol/private: `__hash__`[`L1685`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1685)
- uses (calls/refs, reference-scoped): [`hash_fn`](../utils/compiling_utils.md#hash_fn)
- used by: [`bound`](utils.md#ArrayParam.bound), [`init_kwargs`](utils.md#ArrayParam.init_kwargs)

## Functions
- `_path_to_str(path)` — [`L610`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L610)
- `_select_checkpoint_names_by_regex(*, include_patterns: Sequence[re.Pattern[str]] | None = None, exclude_patterns: Sequence[re.Pattern[str]] | None = None)` — [`L91`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L91) — Resolve known checkpoint names using include/exclude regex filters.
- `_with_progress(path, array)` — [`L636`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L636)
- `add_start_docstrings(*docstr)` — [`L329`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L329) — The add_start_docstrings function is a decorator that adds the docstrings to the beginning of a function.
- `apply_lora_to_layers(model: nn.Module,, *, lora_rank: int, lora_pattern: str | None = None, verbose: bool = True, rngs: nn.Rngs | None = None)` — [`L443`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L443) — Wrap matching ``ParallelLinear`` modules with EasyDeL's LoRA adapter.
- `apply_sparsity_to_params(params: dict[str, tp.Any] | tp.Any, sparsify_module: AVAILABLE_SPARSE_MODULE_TYPES = "bcoo", verbose: bool = True)` — [`L591`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L591)
- `auto_remat(module: type[M],, *, policy: EasyDeLGradientCheckPointers | str | tp.Callable = EasyDeLGradientCheckPointers.NONE, prevent_cse: bool = True, save_names: list[str] | None = None, exclude_names: list[str] | None = None)` — [`L689`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L689) — documented in [easydel-infra-utils](../../../concepts/easydel-infra-utils.md)
- `block_wise_ffn(remat_ffn: tp.Callable, inputs: jax.Array, chunk_size: int)` — [`L350`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L350) — Apply a feed-forward network block-wise to reduce memory usage.
- `canonicalize_dtype(*args, dtype: jax.numpy.dtype | None = None, inexact: bool = True)` — [`L148`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L148) — Canonicalize an optional dtype to the definitive dtype.
- `compute_and_flops(eqn)` — [`L971`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L971) — Compute FLOPs for logical and operation.
- `compute_argmax_flops(eqn)` — [`L987`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L987) — Compute FLOPs for argmax operation.
- `compute_attention_flops(eqn)` — [`L896`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L896) — Compute FLOPs for attention operation.
- `compute_binary_op_flops(eqn)` — [`L820`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L820) — Generic FLOP counter for binary operations with broadcasting.
- `compute_conv_flops(eqn)` — [`L856`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L856) — Compute FLOPs for convolution operation.
- `compute_cumsum_flops(eqn)` — [`L941`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L941) — Compute FLOPs for cumulative sum.
- `compute_dot_general_flops(eqn)` — [`L832`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L832) — Compute FLOPs for dot_general operation.
- `compute_erf_inv_flops(eqn)` — [`L1024`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1024) — Compute FLOPs for inverse error function.
- `compute_integer_pow_flops(eqn)` — [`L965`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L965) — Compute FLOPs for integer power.
- `compute_max_flops(eqn)` — [`L948`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L948) — Compute FLOPs for max operation.
- `compute_min_flops(eqn)` — [`L995`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L995) — Compute FLOPs for min operation.
- `compute_or_flops(eqn)` — [`L1030`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1030) — Compute FLOPs for logical or operation.
- `compute_pow_flops(eqn)` — [`L956`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L956) — Compute FLOPs for power operation.
- `compute_reduce_flops(eqn)` — [`L882`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L882) — Compute FLOPs for reduction operations.
- `compute_rem_flops(eqn)` — [`L1003`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1003) — Compute FLOPs for remainder operation.
- `compute_select_n_flops(eqn)` — [`L936`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L936) — Compute FLOPs for select_n operation.
- `compute_shift_right_logical_flops(eqn)` — [`L1034`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1034) — Compute FLOPs for logical right shift.
- `compute_sqrt_flops(eqn)` — [`L982`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L982) — Compute FLOPs for square root operation.
- `compute_square_flops(eqn)` — [`L1008`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1008) — Compute FLOPs for square operation (x * x).
- `compute_triangular_solve_flops(eqn)` — [`L1014`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1014) — Compute FLOPs for triangular solve operation.
- `compute_unary_op_flops(eqn)` — [`L827`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L827) — FLOP counter for unary operations.
- `count_cond_flops(eqn)` — [`L919`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L919) — Count FLOPs in a conditional operation.
- `count_custom_vjp_flops(eqn)` — [`L975`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L975) — Count FLOPs in custom VJP operation.
- `count_flop_jaxpr(jaxpr)` — [`L813`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L813) — Count flops in a Jaxpr.
- `count_scan_flops(eqn)` — [`L910`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L910) — Count FLOPs in a scan operation.
- `create_transformer_checkpoint_policy(save_attention: bool = True, save_mlp: bool = True, save_residuals: bool = True, save_layer_outputs: bool = False, save_embeddings: bool = False, custom_names: list[str] | None = None)` — [`L270`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L270) — Create a checkpoint policy optimized for transformer models.
- `docstring_decorator(fn)` — [`L343`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L343)
- `extract_static_parameters(module)` — [`L649`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L649) — Extract static_argnums for specified parameters across functions in a module.
- `filter_params(path, array)` — [`L623`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L623)
- `flop_activation(activation_type: ActivationType, dim: int)` — [`L1276`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1276) — Calculate FLOPs for different activation functions.
- `flop_attention(hidden_dim: int, num_heads: int, num_kv_heads: int, head_dim: int | None, seq_len: int)` — [`L1450`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1450)
- `flop_cls_head(hidden_dim: int, num_labels: int)` — [`L1505`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1505)
- `flop_cross_attention(hidden_dim: int, num_heads: int, enc_seq_len: int, dec_seq_len: int)` — [`L1469`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1469)
- `flop_layernorm(hidden_dim: int)` — [`L1446`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1446)
- `flop_lm_head(hidden_dim: int, vocab_size: int)` — [`L1501`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1501)
- `flop_loss(num_classes: int)` — [`L1509`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1509)
- `flop_mlp(cfg: FlopCalcConfig, hidden_dim: int, intermediate_dim: int)` — [`L1484`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1484)
- `flop_seq2seq(cfg: FlopCalcConfig)` — [`L1532`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1532)
- `flop_transformer_body(layers: int, seq_len: int, hidden_dim: int, intermediate_dim: int, cfg: FlopCalcConfig)` — [`L1513`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1513)
- `flop_vision_tower(cfg: FlopCalcConfig)` — [`L1559`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1559)
- `flops_per_token(cfg: FlopCalcConfig)` — [`L1569`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1569)
- `get_gradient_checkpoint_policy(name: str | EasyDeLGradientCheckPointers, save_names: list[str] | None = None, exclude_names: list[str] | None = None)` — [`L180`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L180) — Get a gradient checkpointing policy by name or create a custom one.
- `get_scatter_flops(eqn)` — [`L931`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L931) — Count FLOPs in a scatter operation.
- `get_shape_size(shape)` — [`L816`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L816) — Calculate total size of an array shape.
- `is_flatten(pytree: dict)` — [`L403`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L403) — The is_flatten function checks if the pytree is flattened.
- `materialize_meta_leaves(tree: tp.Any, *, seed: int = 0)` — [`L1871`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1871) — Replace ShapeDtypeStruct placeholder leaves with concrete values.
- `merge_lora_params(model: nn.Module, lora_tree: dict)` — [`L532`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L532) — get LoRA (Low-Rank Adaptation) from layers within a model.
- `mesh_partition_product(mesh: Mesh, axis_spec: object)` — [`L1797`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1797) — Return shard multiplicity implied by a PartitionSpec entry.
- `quantize_linear_layers(model: nn.Module,, *, quantization_config: QuantizationConfig | None = None, verbose: bool = True)` — [`L418`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L418) — Quantize parameters to requested precision, excluding specified layers.
- `quick_gelu(x)` — [`L107`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L107) — Quick GELU activation function.
- `sanitize_partition_spec_for_shape(spec: PartitionSpec, shape: tuple[int, ...], mesh: Mesh)` — [`L1817`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1817) — Drop non-divisible sharding axes for a concrete tensor shape.
- `sanitize_partition_specs_for_shape_tree(partition_specs: tp.Any, shape_tree: tp.Any, mesh: Mesh)` — [`L1844`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1844) — Sanitize a partition-spec tree against concrete parameter shapes.
- `split_lora_params(model: nn.Module)` — [`L509`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L509) — get LoRA (Low-Rank Adaptation) from layers within a model.
- `trace_functions()` — [`L1654`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1654)
- `unwrap_lora_to_layers(model: nn.Module,, *, verbose: bool = True)` — [`L557`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L557) — UnWrap LoRA (Low-Rank Adaptation) from specified linear layers within a model.
- `visit_jaxpr(jaxpr)` — [`L1127`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1127)

## Module values
- `ACT2FN` — [`L121`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L121) — documented in [easydel-infra-utils](../../../concepts/easydel-infra-utils.md)
- `M` — [`L685`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L685)
- `ProcessingClassType` — [`L1792`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L1792) — documented in [easydel-infra-utils](../../../concepts/easydel-infra-utils.md)
- `ROPE_TYPES` — [`L142`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L142)
- `_ATTN_CHECKPOINT_NAME_PATTERN` — [`L87`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L87)
- `_MLP_CHECKPOINT_NAME_PATTERN` — [`L88`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L88)
- `logger` — [`L85`](../../../../../../raw/code/EasyDeL/easydel/infra/utils.py#L85)


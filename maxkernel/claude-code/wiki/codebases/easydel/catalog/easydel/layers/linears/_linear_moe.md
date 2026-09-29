---
title: 'Module: easydel/layers/linears/_linear_moe.py'
type: catalog
provenance: extracted
module: easydel/layers/linears/_linear_moe.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.layers.linears._linear_moe`/
symbols:
  ColumnParallelMoELinear: ColumnParallelMoELinear#
  ParallelMoELinear.kernel: ParallelMoELinear#kernel.
  RowParallelMoELinear: RowParallelMoELinear#
  ParallelMoELinear._output_axes: ParallelMoELinear#_output_axes().
  ParallelMoELinear.__call__: ParallelMoELinear#__call__().
  ParallelMoELinear: ParallelMoELinear#
  ParallelMoELinear.bias: ParallelMoELinear#bias.
  ParallelMoELinear.craft_sharding: ParallelMoELinear#craft_sharding().
  ParallelMoELinear.alt_sharding: ParallelMoELinear#alt_sharding().
  ParallelMoELinear.direction: ParallelMoELinear#direction().
  ParallelMoELinear._input_axes: ParallelMoELinear#_input_axes().
  ParallelMoELinear.use_expert_tensor_mode: ParallelMoELinear#use_expert_tensor_mode.
  ParallelMoELinear.__init__: ParallelMoELinear#__init__().
  ParallelMoELinear.expert_axis: ParallelMoELinear#expert_axis().
  DP: DP.
  ParallelMoELinear._direction: ParallelMoELinear#_direction.
  EMPTY: EMPTY.
  ParallelMoELinear.partition_manager: ParallelMoELinear#partition_manager.
  ParallelMoELinear.can_use_shard_map: ParallelMoELinear#can_use_shard_map().
  ParallelMoELinear.weight_modif_fn: ParallelMoELinear#weight_modif_fn.
  TP: TP.
  ParallelMoELinear._expand_bias_ragged: ParallelMoELinear#_expand_bias_ragged().
  ParallelMoELinear.alt_sharding_axis: ParallelMoELinear#alt_sharding_axis().
  ParallelMoELinear._group_axes: ParallelMoELinear#_group_axes().
  EP: EP.
  Initializer: Initializer.
  default_kernel_init: default_kernel_init.
  default_bias_init: default_bias_init.
  ParallelMoELinear.out_first: ParallelMoELinear#out_first.
  ParallelMoELinear.dtype: ParallelMoELinear#dtype.
  ParallelMoELinear.param_dtype: ParallelMoELinear#param_dtype.
  BATCH: BATCH.
  EMBED: EMBED.
  EXPERT: EXPERT.
  MODE_TRAIN: MODE_TRAIN.
  FSDP: FSDP.
  SP: SP.
  ParallelMoELinear.num_experts: ParallelMoELinear#num_experts.
  ParallelMoELinear.in_features: ParallelMoELinear#in_features.
  ParallelMoELinear.out_features: ParallelMoELinear#out_features.
  ParallelMoELinear.kernel_init: ParallelMoELinear#kernel_init.
  ParallelMoELinear.bias_init: ParallelMoELinear#bias_init.
  RowParallelMoELinear._direction: RowParallelMoELinear#_direction.
  ColumnParallelMoELinear._direction: ColumnParallelMoELinear#_direction.
---
# Module: [`easydel/layers/linears/_linear_moe.py`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_moe.py)

## Classes
### `ColumnParallelMoELinear`  ·  implements/extends ParallelMoELinear
- def: [`easydel/layers/linears/_linear_moe.py:519`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_moe.py#L519)
- doc: Column-parallel variant of ParallelMoELinear.
- signature: `class ColumnParallelMoELinear(ParallelMoELinear):`
- protocol/private: `_direction`[`L542`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_moe.py#L542)
- uses (calls/refs, reference-scoped): [`ParallelMoELinear`](_linear_moe.md#ParallelMoELinear)
- used by: [`gate_proj`](../../modules/gpt_oss/modeling_gpt_oss.md#GptOssExperts.gate_proj), [`up_proj`](../../modules/gpt_oss/modeling_gpt_oss.md#GptOssExperts.up_proj), [`gate_proj`](../../modules/glm4_moe/modeling_glm4_moe.md#Glm4MoeMLPStack.gate_proj), [`gate_proj`](../../modules/glm4_moe_lite/modeling_glm4_moe_lite.md#Glm4MoeLiteMLPStack.gate_proj), [`gate_proj`](../../modules/glm_moe_dsa/modeling_glm_moe_dsa.md#GlmMoeDsaMLPStack.gate_proj), [`gate_proj`](../../modules/qwen2_moe/modeling_qwen2_moe.md#Qwen2MoeMLPStack.gate_proj), [`gate_proj`](../../modules/qwen3_moe/modeling_qwen3_moe.md#Qwen3MoeMLPStack.gate_proj), [`gate_proj`](../../modules/qwen3_next/modeling_qwen3_next.md#Qwen3NextMLPStack.gate_proj), [`gate_proj`](../../modules/qwen3_omni_moe/modeling_qwen3_omni_moe.md#Qwen3OmniMoeMLPStack.gate_proj), [`gate_proj`](../../modules/qwen3_omni_moe/modeling_qwen3_omni_moe.md#Qwen3OmniMoeTalkerMLPStack.gate_proj), [`gate_proj`](../../modules/qwen3_vl_moe/modeling_qwen3_vl_moe.md#Qwen3VLMoeMLPStack.gate_proj), [`gate_proj`](../../modules/xerxes2/modeling_xerxes2.md#Xerxes2MoeMLPStack.gate_proj), [`up_proj`](../../modules/glm4_moe/modeling_glm4_moe.md#Glm4MoeMLPStack.up_proj), [`up_proj`](../../modules/glm4_moe_lite/modeling_glm4_moe_lite.md#Glm4MoeLiteMLPStack.up_proj), [`up_proj`](../../modules/glm_moe_dsa/modeling_glm_moe_dsa.md#GlmMoeDsaMLPStack.up_proj), [`up_proj`](../../modules/qwen2_moe/modeling_qwen2_moe.md#Qwen2MoeMLPStack.up_proj), [`up_proj`](../../modules/qwen3_moe/modeling_qwen3_moe.md#Qwen3MoeMLPStack.up_proj), [`up_proj`](../../modules/qwen3_next/modeling_qwen3_next.md#Qwen3NextMLPStack.up_proj), [`up_proj`](../../modules/qwen3_omni_moe/modeling_qwen3_omni_moe.md#Qwen3OmniMoeMLPStack.up_proj), [`up_proj`](../../modules/qwen3_omni_moe/modeling_qwen3_omni_moe.md#Qwen3OmniMoeTalkerMLPStack.up_proj), [`up_proj`](../../modules/qwen3_vl_moe/modeling_qwen3_vl_moe.md#Qwen3VLMoeMLPStack.up_proj), [`up_proj`](../../modules/xerxes2/modeling_xerxes2.md#Xerxes2MoeMLPStack.up_proj), [`w1`](../../modules/arctic/modeling_arctic.md#ArcticMLPMoE.w1), [`w1`](../../modules/minimax/modeling_minimax.md#MiniMaxExperts.w1), [`w1`](../../modules/mixtral/modeling_mixtral.md#MixtralMoEMlp.w1), [`w3`](../../modules/arctic/modeling_arctic.md#ArcticMLPMoE.w3), [`w3`](../../modules/minimax/modeling_minimax.md#MiniMaxExperts.w3), [`w3`](../../modules/mixtral/modeling_mixtral.md#MixtralMoEMlp.w3), [`gate_proj`](../../modules/kimi_linear/modeling_kimi_linear.md#KimiMLPMoE.gate_proj), [`up_proj`](../../modules/kimi_linear/modeling_kimi_linear.md#KimiMLPMoE.up_proj), [`ParallelMoELinear`](_linear_moe.md#ParallelMoELinear), [`gate_proj`](../../modules/deepseek_v2/modeling_deepseek.md#DeepseekV2MLPMoE.gate_proj), [`gate_proj`](../../modules/deepseek_v3/modeling_deepseek.md#DeepseekV3MLPMoE.gate_proj), [`gate_proj`](../../modules/llama4/modeling_llama4.md#Llama4TextExperts.gate_proj), [`up_proj`](../../modules/deepseek_v2/modeling_deepseek.md#DeepseekV2MLPMoE.up_proj), [`up_proj`](../../modules/deepseek_v3/modeling_deepseek.md#DeepseekV3MLPMoE.up_proj), [`up_proj`](../../modules/llama4/modeling_llama4.md#Llama4TextExperts.up_proj), [`precision`](../../modules/gemma4/modeling_gemma4.md#Gemma4TextMLPStack.precision)

### `ParallelMoELinear`  ·  implements/extends Module
- def: [`easydel/layers/linears/_linear_moe.py:90`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_moe.py#L90)
- doc: A batched linear transformation layer for Mixture of Experts (MoE) models.
- signature: `class ParallelMoELinear(nn.Module):`
- members:
  - `__call__(self, inputs: Float[Array, "tokens_ragged hidden_dim"], group_sizes: Int[Array, num_groups], sorted_experts: Int[Array, tokens_ragged] | None = None)` — [`L389`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_moe.py#L389) — Applies the batched linear transformation.
  - `__init__(self, num_experts: int, in_features: int, out_features: int, *, use_bias: bool = True, out_first: bool = False, kernel_init: Initializer = default_kernel_init, bias_init: Initializer = default_bias_init, dtype: jnp.dtype = jnp.bfloat16, param_dtype: jnp.dtype = jnp.bfloat16, partition_manager: PartitionManager | None = None, direction: typing.Literal["row", "column"] | None = None, use_expert_tensor_mode: bool = False, weight_modif_fn: typing.Callable[[Array], Array] | None = None, rngs: nn.Rngs)` — [`L160`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_moe.py#L160) — Initialize a ParallelMoELinear layer.
  - `_expand_bias_ragged(self, group_sizes: Int[Array, num_groups], sorted_experts: Int[Array, tokens_ragged] | None = None)` — [`L438`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_moe.py#L438) — Expands bias to match the ragged token batch structure.
  - `_group_axes(self)` — [`L332`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_moe.py#L332) — Returns sharding axes for expert group sizes array.
  - `_input_axes(self)` — [`L343`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_moe.py#L343) — Returns sharding axes for input activations based on parallelism direction.
  - `_output_axes(self)` — [`L364`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_moe.py#L364) — Returns sharding axes for output activations based on parallelism direction.
  - `alt_sharding(self)` — [`L261`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_moe.py#L261) — Returns the ALT (Alternative) sharding configuration for this layer.
  - `alt_sharding_axis(self)` — [`L283`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_moe.py#L283) — Returns the axis names for ALT sharding configuration.
  - `can_use_shard_map(self)` — [`L251`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_moe.py#L251) — Checks if this layer can use shard_map for distributed execution.
  - `craft_sharding(self, *, partition_manager=None, **kwargs)` — [`L294`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_moe.py#L294) — Return dynamic partition specs for this module's parameters.
  - `direction(self)` — [`L240`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_moe.py#L240) — Returns the parallelism direction for this layer.
  - `expert_axis(self)` — [`L328`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_moe.py#L328) — Semantic axis name representing the expert dimension.
  - `bias` — [`L235`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_moe.py#L235)
  - `bias_init` — [`L225`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_moe.py#L225)
  - `dtype` — [`L219`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_moe.py#L219)
  - `in_features` — [`L216`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_moe.py#L216)
  - `kernel` — [`L232`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_moe.py#L232)
  - `kernel_init` — [`L224`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_moe.py#L224)
  - `num_experts` — [`L215`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_moe.py#L215)
  - `out_features` — [`L217`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_moe.py#L217)
  - `out_first` — [`L218`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_moe.py#L218)
  - `param_dtype` — [`L220`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_moe.py#L220)
  - `partition_manager` — [`L221`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_moe.py#L221)
  - `use_expert_tensor_mode` — [`L222`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_moe.py#L222)
  - `weight_modif_fn` — [`L226`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_moe.py#L226)
- protocol/private: `_direction`[`L158`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_moe.py#L158)
- uses (calls/refs, reference-scoped): [`ColumnParallelMoELinear`](_linear_moe.md#ColumnParallelMoELinear), [`RowParallelMoELinear`](_linear_moe.md#RowParallelMoELinear), [`get_moe_partition_spec`](../moe/_communication_utils.md#get_moe_partition_spec), [`DP`](_linear_moe.md#DP), [`EMPTY`](_linear_moe.md#EMPTY), [`TP`](_linear_moe.md#TP), [`EP`](_linear_moe.md#EP), [`Initializer`](_linear_moe.md#Initializer), [`default_bias_init`](_linear_moe.md#default_bias_init), [`default_kernel_init`](_linear_moe.md#default_kernel_init)
- used by: [`ColumnParallelMoELinear`](_linear_moe.md#ColumnParallelMoELinear), [`RowParallelMoELinear`](_linear_moe.md#RowParallelMoELinear), [`easydel_to_torch`](../../utils/parameters_transformation.md#StateDictConverter.easydel_to_torch), [`__call__`](../../modules/gemma4/modeling_gemma4.md#Gemma4TextRouter.__call__), [`__call__`](../../modules/qwen3_next/modeling_qwen3_next.md#Qwen3NextSparseMoeBlock.__call__), [`__call__`](../../modules/deepseek_v2/modeling_deepseek.md#DeepseekV2MoE.__call__), [`__call__`](../../modules/deepseek_v3/modeling_deepseek.md#DeepseekV3MoE.__call__), [`__call__`](../../modules/kimi_linear/modeling_kimi_linear.md#KimiSparseMoeBlock.__call__), [`__call__`](../../modules/arctic/modeling_arctic.md#ArcticMoeBlock.__call__), [`__call__`](../../modules/gpt_oss/modeling_gpt_oss.md#GptOssMLP.__call__), [`__call__`](../../modules/llama4/modeling_llama4.md#Llama4TextMoe.__call__), [`__call__`](../../modules/minimax/modeling_minimax.md#MiniMaxSparseMoeBlock.__call__), [`__call__`](../../modules/qwen2_moe/modeling_qwen2_moe.md#Qwen2MoeSparseBlock.__call__), [`__call__`](../../modules/qwen3_omni_moe/modeling_qwen3_omni_moe.md#Qwen3OmniMoeTalkerTextSparseMoeBlock.__call__), [`_build_transform_fn`](../../infra/base_module.md#EasyDeLBaseModule._build_transform_fn), [`__call__`](../../modules/glm4_moe/modeling_glm4_moe.md#Glm4MoeMoE.__call__), [`__call__`](../../modules/glm4_moe_lite/modeling_glm4_moe_lite.md#Glm4MoeLiteMoE.__call__), [`__call__`](../../modules/glm_moe_dsa/modeling_glm_moe_dsa.md#GlmMoeDsaMoE.__call__), [`__call__`](../../modules/mixtral/modeling_mixtral.md#MixtralSparseMoeBlock.__call__), [`__call__`](../../modules/qwen3_moe/modeling_qwen3_moe.md#Qwen3MoeSparseBlock.__call__), [`__call__`](../../modules/qwen3_omni_moe/modeling_qwen3_omni_moe.md#Qwen3OmniMoeTextSparseBlock.__call__), [`__call__`](../../modules/qwen3_vl_moe/modeling_qwen3_vl_moe.md#Qwen3VLMoeTextSparseBlock.__call__), [`__call__`](../../modules/xerxes2/modeling_xerxes2.md#Xerxes2MoeSparseBlock.__call__)

### `RowParallelMoELinear`  ·  implements/extends ParallelMoELinear
- def: [`easydel/layers/linears/_linear_moe.py:491`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_moe.py#L491)
- doc: Row-parallel variant of ParallelMoELinear.
- signature: `class RowParallelMoELinear(ParallelMoELinear):`
- protocol/private: `_direction`[`L516`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_moe.py#L516)
- uses (calls/refs, reference-scoped): [`ParallelMoELinear`](_linear_moe.md#ParallelMoELinear)
- used by: [`down_proj`](../../modules/gpt_oss/modeling_gpt_oss.md#GptOssExperts.down_proj), [`down_proj`](../../modules/glm4_moe/modeling_glm4_moe.md#Glm4MoeMLPStack.down_proj), [`down_proj`](../../modules/glm4_moe_lite/modeling_glm4_moe_lite.md#Glm4MoeLiteMLPStack.down_proj), [`down_proj`](../../modules/glm_moe_dsa/modeling_glm_moe_dsa.md#GlmMoeDsaMLPStack.down_proj), [`down_proj`](../../modules/qwen2_moe/modeling_qwen2_moe.md#Qwen2MoeMLPStack.down_proj), [`down_proj`](../../modules/qwen3_moe/modeling_qwen3_moe.md#Qwen3MoeMLPStack.down_proj), [`down_proj`](../../modules/qwen3_next/modeling_qwen3_next.md#Qwen3NextMLPStack.down_proj), [`down_proj`](../../modules/qwen3_omni_moe/modeling_qwen3_omni_moe.md#Qwen3OmniMoeMLPStack.down_proj), [`down_proj`](../../modules/qwen3_omni_moe/modeling_qwen3_omni_moe.md#Qwen3OmniMoeTalkerMLPStack.down_proj), [`down_proj`](../../modules/qwen3_vl_moe/modeling_qwen3_vl_moe.md#Qwen3VLMoeMLPStack.down_proj), [`down_proj`](../../modules/xerxes2/modeling_xerxes2.md#Xerxes2MoeMLPStack.down_proj), [`w2`](../../modules/arctic/modeling_arctic.md#ArcticMLPMoE.w2), [`w2`](../../modules/minimax/modeling_minimax.md#MiniMaxExperts.w2), [`w2`](../../modules/mixtral/modeling_mixtral.md#MixtralMoEMlp.w2), [`down_proj`](../../modules/kimi_linear/modeling_kimi_linear.md#KimiMLPMoE.down_proj), [`ParallelMoELinear`](_linear_moe.md#ParallelMoELinear), [`down_proj`](../../modules/deepseek_v2/modeling_deepseek.md#DeepseekV2MLPMoE.down_proj), [`down_proj`](../../modules/deepseek_v3/modeling_deepseek.md#DeepseekV3MLPMoE.down_proj), [`down_proj`](../../modules/llama4/modeling_llama4.md#Llama4TextExperts.down_proj), [`precision`](../../modules/gemma4/modeling_gemma4.md#Gemma4TextMLPStack.precision)

## Module values
- `BATCH` — [`L73`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_moe.py#L73)
- `DP` — [`L79`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_moe.py#L79)
- `EMBED` — [`L75`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_moe.py#L75)
- `EMPTY` — [`L74`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_moe.py#L74)
- `EP` — [`L78`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_moe.py#L78)
- `EXPERT` — [`L76`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_moe.py#L76)
- `FSDP` — [`L80`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_moe.py#L80)
- `Initializer` — [`L87`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_moe.py#L87)
- `MODE_TRAIN` — [`L77`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_moe.py#L77)
- `SP` — [`L82`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_moe.py#L82)
- `TP` — [`L81`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_moe.py#L81)
- `default_bias_init` — [`L86`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_moe.py#L86)
- `default_kernel_init` — [`L85`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_moe.py#L85)


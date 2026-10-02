---
title: 'Module: easydel/layers/moe/_communication_utils.py'
type: catalog
provenance: extracted
module: easydel/layers/moe/_communication_utils.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.layers.moe._communication_utils`/
symbols:
  MoeRoutingStrategy: MoeRoutingStrategy#
  MoeLoadBalancingStrategy: MoeLoadBalancingStrategy#
  MoeFusedHooks.__hash__: MoeFusedHooks#__hash__().
  get_moe_partition_spec: get_moe_partition_spec().
  MoeRoutingStrategy.TOP_K: MoeRoutingStrategy#TOP_K.
  MoeLoadBalancingStrategy.STANDARD: MoeLoadBalancingStrategy#STANDARD.
  resolve_eformer_axis: resolve_eformer_axis().
  get_all_to_all_params: get_all_to_all_params().
  get_all_to_all_params.transform: get_all_to_all_params().transform().
  MoeFusedHooks: MoeFusedHooks#
  _Transform: _Transform#
  MoeFusedHooks.select_hook: MoeFusedHooks#select_hook.
  MoeFusedHooks.replace: MoeFusedHooks#replace().
  sort_activations: sort_activations().
  permute: permute().
  MoEMethods: MoEMethods#
  MoeFusedHooks.before_gate: MoeFusedHooks#before_gate.
  MoeFusedHooks.after_gate: MoeFusedHooks#after_gate.
  MoeFusedHooks.before_topk: MoeFusedHooks#before_topk.
  MoeRoutingStrategy.TOP_K_NDIV: MoeRoutingStrategy#TOP_K_NDIV.
  MoeMetrics: MoeMetrics#
  unpermute: unpermute().
  MoeFusedHooks.normalize_gate_logits: MoeFusedHooks#normalize_gate_logits.
  MoeFusedHooks.refine_weights_hook: MoeFusedHooks#refine_weights_hook.
  MoeFusedHooks.before_combine: MoeFusedHooks#before_combine.
  MoeFusedHooks.finalize_output: MoeFusedHooks#finalize_output.
  local_permute: local_permute().
  MoeFusedHooks.refine_inputs_hook: MoeFusedHooks#refine_inputs_hook.
  get_experts_location: get_experts_location().
  MoeFusedHooks.before_wo: MoeFusedHooks#before_wo.
  MoeFusedHooks.after_wo: MoeFusedHooks#after_wo.
  _Transform.INPUT_OFFSET: _Transform#INPUT_OFFSET.
  _Transform.SEND_SIZE: _Transform#SEND_SIZE.
  _Transform.OUTPUT_OFFSET: _Transform#OUTPUT_OFFSET.
  _Transform.RECV_SIZE: _Transform#RECV_SIZE.
  rsum_scatter_maybe: rsum_scatter_maybe().
  sort_activations_custom: sort_activations_custom().
  MoeFusedHooks.scale_replicated_inputs: MoeFusedHooks#scale_replicated_inputs.
  MoeFusedHooks.output_weights_hook: MoeFusedHooks#output_weights_hook.
  MoeLoadBalancingStrategy.NONE: MoeLoadBalancingStrategy#NONE.
  MODE_TRAIN: MODE_TRAIN.
  EP: EP.
  FSDP: FSDP.
  TP: TP.
  SP: SP.
  MoEMethods.FUSED_MOE: MoEMethods#FUSED_MOE.
  MoEMethods.STANDARD_MOE: MoEMethods#STANDARD_MOE.
  MoEMethods.DENSE_MOE: MoEMethods#DENSE_MOE.
  sort_activations_custom_fwd: sort_activations_custom_fwd().
  sort_activations_custom_bwd: sort_activations_custom_bwd().
  MoeFusedHooks.after_ep_receive: MoeFusedHooks#after_ep_receive.
  MoeFusedHooks.after_wiwu: MoeFusedHooks#after_wiwu.
  canon_dim: canon_dim().
  MoeRoutingStrategy.SWITCH: MoeRoutingStrategy#SWITCH.
  MoeRoutingStrategy.EMPTY_CHOICE: MoeRoutingStrategy#EMPTY_CHOICE.
  MoeRoutingStrategy.HASH: MoeRoutingStrategy#HASH.
  MoeLoadBalancingStrategy.SWITCH_TRANSFORMER: MoeLoadBalancingStrategy#SWITCH_TRANSFORMER.
  MoeLoadBalancingStrategy.EMPTY_CHOICE: MoeLoadBalancingStrategy#EMPTY_CHOICE.
  MoeMetrics.expert_loads: MoeMetrics#expert_loads.
  MoeMetrics.router_probs: MoeMetrics#router_probs.
  MoeMetrics.selected_experts: MoeMetrics#selected_experts.
  MoeMetrics.selected_weights: MoeMetrics#selected_weights.
  MoeMetrics.load_balancing_loss: MoeMetrics#load_balancing_loss.
  MoeMetrics.router_z_loss: MoeMetrics#router_z_loss.
  MoeMetrics.expert_utilization: MoeMetrics#expert_utilization.
  MoeMetrics.routing_entropy: MoeMetrics#routing_entropy.
  BATCH: BATCH.
  EMPTY: EMPTY.
  EMBED: EMBED.
  EXPERT: EXPERT.
  DP: DP.
  EP_DISPATCH: EP_DISPATCH.
  EP_AUTO_THRESHOLD: EP_AUTO_THRESHOLD.
  GMM_PLATFORM: GMM_PLATFORM.
  rsum_scatter: rsum_scatter().
  argsort: argsort().
  take1d: take1d().
  repeat_take_sorted: repeat_take_sorted().
  bincount: bincount().
  all_i32: all_i32().
  psum_maybe: psum_maybe().
  slice_k_for_param_shards: slice_k_for_param_shards().
  tp_global_topk: tp_global_topk().
---
# Module: [`easydel/layers/moe/_communication_utils.py`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py)

## Classes
### `MoEMethods`  ·  implements/extends StrEnum
- def: [`easydel/layers/moe/_communication_utils.py:54`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L54)
- doc: Enumeration of available MoE execution methods.
- signature: `class MoEMethods(enum.StrEnum):`
- members:
  - `DENSE_MOE` — [`L84`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L84)
  - `FUSED_MOE` — [`L82`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L82)
  - `STANDARD_MOE` — [`L83`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L83)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`moe_call`](_moe_module.md#BaseMoeModule.moe_call)

### `MoeFusedHooks`
- def: [`easydel/layers/moe/_communication_utils.py:335`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L335)
- doc: Optional callbacks executed at key points of the fused MoE pipeline.
- signature: `class MoeFusedHooks:`
- members:
  - `__hash__(self)` — [`L418`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L418) — Makes the hooks dataclass hashable for NNX graph hashing.
  - `replace(self, **kws)` — [`L447`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L447) — Create a new MoeFusedHooks instance with some fields replaced.
  - `after_ep_receive` — [`L409`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L409)
  - `after_gate` — [`L403`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L403)
  - `after_wiwu` — [`L412`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L412)
  - `after_wo` — [`L414`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L414)
  - `before_combine` — [`L415`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L415)
  - `before_gate` — [`L402`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L402)
  - `before_topk` — [`L405`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L405)
  - `before_wo` — [`L413`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L413)
  - `finalize_output` — [`L416`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L416)
  - `normalize_gate_logits` — [`L404`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L404)
  - `output_weights_hook` — [`L411`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L411)
  - `refine_inputs_hook` — [`L407`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L407)
  - `refine_weights_hook` — [`L410`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L410)
  - `scale_replicated_inputs` — [`L408`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L408)
  - `select_hook` — [`L406`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L406)
- used by: [`_sparse_moe_call`](_moe_module.md#BaseMoeModule._sparse_moe_call), [`moe_call`](_moe_module.md#BaseMoeModule.moe_call), [`_moe_call_standard`](_moe_module.md#BaseMoeModule._moe_call_standard), [`_moe_call_dense`](_moe_module.md#BaseMoeModule._moe_call_dense), [`_configure_hooks_for_routing_strategy`](_moe_module.md#BaseMoeModule._configure_hooks_for_routing_strategy), [`shared_experts`](../../modules/glm4_moe_lite/modeling_glm4_moe_lite.md#Glm4MoeLiteMoE.shared_experts), [`shared_experts`](../../modules/glm_moe_dsa/modeling_glm_moe_dsa.md#GlmMoeDsaMoE.shared_experts), [`__init__`](_moe_module.md#BaseMoeModule.__init__), [`shared_experts`](../../modules/glm4_moe/modeling_glm4_moe.md#Glm4MoeMoE.shared_experts), [`__call__`](../../modules/gemma4/modeling_gemma4.md#Gemma4TextRouter.__call__), [`__init__`](../../modules/gpt_oss/modeling_gpt_oss.md#GptOssMLP.__init__), [`__init__`](../../modules/llama4/modeling_llama4.md#Llama4TextMoe.__init__), [`__init__`](../../modules/minimax/modeling_minimax.md#MiniMaxSparseMoeBlock.__init__), [`moe_hooks`](_moe_module.md#BaseMoeModule.moe_hooks), [`shared_experts`](../../modules/deepseek_v2/modeling_deepseek.md#DeepseekV2MoE.shared_experts), [`shared_expert_gate`](../../modules/qwen2_moe/modeling_qwen2_moe.md#Qwen2MoeSparseBlock.shared_expert_gate)

### `MoeLoadBalancingStrategy`  ·  implements/extends Enum
- def: [`easydel/layers/moe/_communication_utils.py:1084`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L1084)
- doc: Defines the available strategies for calculating the load balancing loss.
- signature: `class MoeLoadBalancingStrategy(enum.Enum):`
- members:
  - `EMPTY_CHOICE` — [`L1099`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L1099)
  - `NONE` — [`L1100`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L1100)
  - `STANDARD` — [`L1097`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L1097)
  - `SWITCH_TRANSFORMER` — [`L1098`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L1098)
- used by: [`__init__`](_moe_module.md#BaseMoeModule.__init__), [`__init__`](../../modules/gpt_oss/modeling_gpt_oss.md#GptOssMLP.__init__), [`__init__`](../../modules/llama4/modeling_llama4.md#Llama4TextMoe.__init__), [`__init__`](../../modules/minimax/modeling_minimax.md#MiniMaxSparseMoeBlock.__init__), [`__init__`](../../modules/qwen2_moe/modeling_qwen2_moe.md#Qwen2MoeSparseBlock.__init__), [`__init__`](../../modules/qwen3_moe/modeling_qwen3_moe.md#Qwen3MoeSparseBlock.__init__), [`__init__`](../../modules/qwen3_next/modeling_qwen3_next.md#Qwen3NextSparseMoeBlock.__init__), [`__init__`](../../modules/qwen3_omni_moe/modeling_qwen3_omni_moe.md#Qwen3OmniMoeTalkerTextSparseMoeBlock.__init__), [`__init__`](../../modules/qwen3_omni_moe/modeling_qwen3_omni_moe.md#Qwen3OmniMoeTextSparseBlock.__init__), [`__init__`](../../modules/xerxes2/modeling_xerxes2.md#Xerxes2MoeSparseBlock.__init__), [`_compute_load_balancing_loss`](_moe_module.md#BaseMoeModule._compute_load_balancing_loss), [`__init__`](../../modules/arctic/modeling_arctic.md#ArcticMoeBlock.__init__), [`__init__`](../../modules/deepseek_v2/modeling_deepseek.md#DeepseekV2MoE.__init__), [`__init__`](../../modules/deepseek_v3/modeling_deepseek.md#DeepseekV3MoE.__init__), [`__init__`](../../modules/gemma4/modeling_gemma4.md#Gemma4TextRouter.__init__), [`__init__`](../../modules/glm4_moe/modeling_glm4_moe.md#Glm4MoeMoE.__init__), [`__init__`](../../modules/glm4_moe_lite/modeling_glm4_moe_lite.md#Glm4MoeLiteMoE.__init__), [`__init__`](../../modules/glm_moe_dsa/modeling_glm_moe_dsa.md#GlmMoeDsaMoE.__init__), [`__init__`](../../modules/kimi_linear/modeling_kimi_linear.md#KimiSparseMoeBlock.__init__), [`__init__`](../../modules/mixtral/modeling_mixtral.md#MixtralSparseMoeBlock.__init__), [`__init__`](../../modules/qwen3_vl_moe/modeling_qwen3_vl_moe.md#Qwen3VLMoeTextSparseBlock.__init__)

### `MoeMetrics`
- def: [`easydel/layers/moe/_communication_utils.py:1104`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L1104)
- doc: A container for storing metrics and auxiliary losses from an MoE layer.
- signature: `class MoeMetrics:`
- members:
  - `expert_loads` — [`L1126`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L1126)
  - `expert_utilization` — [`L1132`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L1132)
  - `load_balancing_loss` — [`L1130`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L1130)
  - `router_probs` — [`L1127`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L1127)
  - `router_z_loss` — [`L1131`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L1131)
  - `routing_entropy` — [`L1133`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L1133)
  - `selected_experts` — [`L1128`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L1128)
  - `selected_weights` — [`L1129`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L1129)
- used by: [`_moe_call_standard`](_moe_module.md#BaseMoeModule._moe_call_standard), [`_compute_metrics`](_moe_module.md#BaseMoeModule._compute_metrics), [`__call__`](_moe_module.md#BaseMoeModule.__call__)

### `MoeRoutingStrategy`  ·  implements/extends Enum
- def: [`easydel/layers/moe/_communication_utils.py:1024`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L1024)
- doc: Defines the available strategies for routing tokens to experts in an MoE layer.
- signature: `class MoeRoutingStrategy(enum.Enum):`
- members:
  - `EMPTY_CHOICE` — [`L1080`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L1080)
  - `HASH` — [`L1081`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L1081)
  - `SWITCH` — [`L1079`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L1079)
  - `TOP_K` — [`L1077`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L1077)
  - `TOP_K_NDIV` — [`L1078`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L1078)
- used by: [`_configure_hooks_for_routing_strategy`](_moe_module.md#BaseMoeModule._configure_hooks_for_routing_strategy), [`__init__`](_moe_module.md#BaseMoeModule.__init__), [`__init__`](../../modules/gpt_oss/modeling_gpt_oss.md#GptOssMLP.__init__), [`__init__`](../../modules/llama4/modeling_llama4.md#Llama4TextMoe.__init__), [`__init__`](../../modules/minimax/modeling_minimax.md#MiniMaxSparseMoeBlock.__init__), [`__init__`](../../modules/qwen2_moe/modeling_qwen2_moe.md#Qwen2MoeSparseBlock.__init__), [`__init__`](../../modules/qwen3_moe/modeling_qwen3_moe.md#Qwen3MoeSparseBlock.__init__), [`__init__`](../../modules/qwen3_next/modeling_qwen3_next.md#Qwen3NextSparseMoeBlock.__init__), [`__init__`](../../modules/qwen3_omni_moe/modeling_qwen3_omni_moe.md#Qwen3OmniMoeTalkerTextSparseMoeBlock.__init__), [`__init__`](../../modules/qwen3_omni_moe/modeling_qwen3_omni_moe.md#Qwen3OmniMoeTextSparseBlock.__init__), [`__init__`](../../modules/xerxes2/modeling_xerxes2.md#Xerxes2MoeSparseBlock.__init__), [`__init__`](../../modules/arctic/modeling_arctic.md#ArcticMoeBlock.__init__), [`__init__`](../../modules/deepseek_v2/modeling_deepseek.md#DeepseekV2MoE.__init__), [`__init__`](../../modules/deepseek_v3/modeling_deepseek.md#DeepseekV3MoE.__init__), [`__init__`](../../modules/gemma4/modeling_gemma4.md#Gemma4TextRouter.__init__), [`__init__`](../../modules/glm4_moe/modeling_glm4_moe.md#Glm4MoeMoE.__init__), [`__init__`](../../modules/glm4_moe_lite/modeling_glm4_moe_lite.md#Glm4MoeLiteMoE.__init__), [`__init__`](../../modules/glm_moe_dsa/modeling_glm_moe_dsa.md#GlmMoeDsaMoE.__init__), [`__init__`](../../modules/kimi_linear/modeling_kimi_linear.md#KimiSparseMoeBlock.__init__), [`__init__`](../../modules/mixtral/modeling_mixtral.md#MixtralSparseMoeBlock.__init__), [`__init__`](../../modules/qwen3_vl_moe/modeling_qwen3_vl_moe.md#Qwen3VLMoeTextSparseBlock.__init__)

### `_Transform`  ·  implements/extends Enum
- def: [`easydel/layers/moe/_communication_utils.py:589`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L589)
- doc: Enumeration of transformation strategies for all-to-all communication parameters.
- signature: `class _Transform(enum.Enum):`
- members:
  - `INPUT_OFFSET` — [`L619`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L619)
  - `OUTPUT_OFFSET` — [`L621`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L621)
  - `RECV_SIZE` — [`L622`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L622)
  - `SEND_SIZE` — [`L620`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L620)
- used by: [`get_all_to_all_params`](_communication_utils.md#get_all_to_all_params), [`transform`](_communication_utils.md#get_all_to_all_params.transform)

## Functions
- `all_i32(*xs: jax.Array)` — [`L210`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L210) — Casts all input arrays to int32 dtype.
- `argsort(x: jax.Array)` — [`L115`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L115) — Returns indices that would sort the input array along its last axis.
- `bincount(x: jax.Array, length: int)` — [`L187`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L187) — Counts occurrences of non-negative integers in an array.
- `canon_dim(ndim: int, dim: int)` — [`L473`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L473) — Canonicalizes a dimension index to be non-negative.
- `get_all_to_all_params(all_shards_group_sizes: jax.Array, shard_id: int, num_expert_parallelism: int, is_batch_sharded: bool)` — [`L625`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L625) — Computes parameters for ragged all-to-all communication in expert parallelism.
- `get_experts_location(gate_logits, pre_bias_logits, select_hook: typing.Callable[[jax.Array, jax.Array, int], tuple[jax.Array, jax.Array]] | None = None, refine_weights_hook: typing.Callable[[jax.Array], jax.Array] | None = None, *, num_experts_per_tok: int)` — [`L724`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L724) — Compute top-k experts and weights with optional overrides.
- `get_moe_partition_spec(partition_manager: PartitionManager, direction: typing.Literal["row", "column"], tensors_are_expert: bool, is_bias: bool = False, fsdp_is_ep_bound: bool = True, sp_is_ep_bound: bool = True, module_view: bool = False)` — [`L1175`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L1175) — Generate JAX PartitionSpec for MoE expert weight tensors.
- `local_permute(inputs: jax.Array, global_group_sizes: jax.Array, local_expert_size: int, shard_index: int, is_offset: bool = False, global_sorted_experts: jax.Array | None = None, use_custom_sort_vjp: bool = True)` — [`L966`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L966) — Performs local permutation of tokens to group them by expert on a single shard.
- `permute(inputs: jax.Array, gate_logits: jax.Array, pre_bias_logits: jax.Array | None = None, use_custom_sort_vjp: bool = True, roll_to_expert_id=None, *, num_experts_per_tok: int, num_experts: int, dtype: jnp.dtype, select_hook: typing.Callable[[jax.Array, jax.Array, int], tuple[jax.Array, jax.Array]] | None = None, refine_weights_hook: typing.Callable[[jax.Array], jax.Array] | None = None, refine_inputs_hook: typing.Callable[[jax.Array, jax.Array, tuple[int]], jax.Array] | None = None, scale_replicated_inputs: typing.Callable[[jax.Array, jax.Array], jax.Array] | None = None)` — [`L764`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L764) — Permute tokens by expert assignment for grouped matmul.
- `psum_maybe(x: jax.Array, axes: tuple[str, ...], mesh: jax.sharding.Mesh, dtype: jnp.dtype = jnp.float32)` — [`L499`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L499) — Conditionally performs parallel sum across specified axes if they exist in mesh.
- `repeat_take_sorted(x: jax.Array, sort_idx: jax.Array, k: int)` — [`L160`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L160) — Repeats and reorders rows of an array based on sorted indices.
- `resolve_eformer_axis(axis: str | list[str], manager: PartitionManager)` — [`L1136`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L1136) — Resolves logical axis name(s) to physical mesh axis names.
- `rsum_scatter(x: jax.Array, axis_name: str, scatter_dimension: int, tiled: bool = True)` — [`L87`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L87) — Performs reduce-scatter collective operation with float32 accumulation.
- `rsum_scatter_maybe(x: jax.Array, axis_name: str, dim: int, mesh: jax.sharding.Mesh, dtype: jnp.dtype = jnp.float32)` — [`L527`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L527) — Conditionally performs reduce-scatter if the axis exists and has size > 1.
- `slice_k_for_param_shards(x_mat: jax.Array, chunk: int, axes: tuple[str, ...], mesh: jax.sharding.Mesh, axis: int = 1)` — [`L553`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L553) — Slices activation tensor to match the chunk size of parameter shards.
- `sort_activations(inputs: jax.Array, sort_indices: jax.Array, use_custom_vjp: bool = True)` — [`L232`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L232) — Reorders activations using provided sort indices with optional custom gradient.
- `sort_activations_custom(inputs: jax.Array, sort_indices: jax.Array)` — [`L269`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L269) — Custom VJP implementation for sorting activations.
- `sort_activations_custom_bwd(residuals: jax.Array, grads: jax.Array)` — [`L310`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L310) — Backward pass for custom VJP sorting.
- `sort_activations_custom_fwd(inputs: jax.Array, sort_indices: jax.Array)` — [`L291`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L291) — Forward pass for custom VJP sorting.
- `take1d(x: jax.Array, idx: jax.Array)` — [`L137`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L137) — Indexes an array along axis 0 using the provided indices.
- `tp_global_topk(logits_shard: jax.Array, k: int, tp_axis: str)` — [`L688`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L688) — Computes global top-k selection across tensor-parallel shards.
- `transform(inp, shard_id, strategy, is_batch_sharded)` — [`L656`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L656)
- `unpermute(intermediate, sorted_selected_experts, weights, batch_size, sequence_length, use_custom_sort_vjp=True, weight_modif_fn: typing.Callable[[jax.Array], jax.Array] | None = None, *, num_experts_per_tok: int, dtype: jnp.dtype)` — [`L886`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L886) — Invert expert permutation and combine expert outputs per token.

## Module values
- `BATCH` — [`L38`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L38)
- `DP` — [`L44`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L44)
- `EMBED` — [`L40`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L40)
- `EMPTY` — [`L39`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L39)
- `EP` — [`L43`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L43)
- `EP_AUTO_THRESHOLD` — [`L50`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L50)
- `EP_DISPATCH` — [`L49`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L49)
- `EXPERT` — [`L41`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L41)
- `FSDP` — [`L45`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L45)
- `GMM_PLATFORM` — [`L51`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L51)
- `MODE_TRAIN` — [`L42`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L42)
- `SP` — [`L47`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L47)
- `TP` — [`L46`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_communication_utils.py#L46)


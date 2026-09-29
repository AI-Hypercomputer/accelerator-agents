---
title: 'Module: easydel/layers/moe/_moe_module.py'
type: catalog
provenance: extracted
module: easydel/layers/moe/_moe_module.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.layers.moe._moe_module`/
symbols:
  BaseMoeModule: BaseMoeModule#
  BaseMoeModule._sparse_moe_call: BaseMoeModule#_sparse_moe_call().
  BaseMoeModule.moe_call: BaseMoeModule#moe_call().
  BaseMoeModule._moe_call_standard: BaseMoeModule#_moe_call_standard().
  BaseMoeModule._moe_call_dense: BaseMoeModule#_moe_call_dense().
  BaseMoeModule._configure_hooks_for_routing_strategy: BaseMoeModule#_configure_hooks_for_routing_strategy().
  BaseMoeModule.__init__: BaseMoeModule#__init__().
  BaseMoeModule._compute_metrics: BaseMoeModule#_compute_metrics().
  BaseMoeModule.moe_hooks: BaseMoeModule#moe_hooks.
  EMPTY: EMPTY.
  BaseMoeModule.n_routed_experts: BaseMoeModule#n_routed_experts.
  BaseMoeModule._get_sharding_status: BaseMoeModule#_get_sharding_status().
  BaseMoeModule._compute_load_balancing_loss: BaseMoeModule#_compute_load_balancing_loss().
  BaseMoeModule._sparse_call: BaseMoeModule#_sparse_call().
  BaseMoeModule.dtype: BaseMoeModule#dtype.
  BaseMoeModule.get_moe_spec: BaseMoeModule#get_moe_spec().
  BaseMoeModule.config: BaseMoeModule#config.
  BaseMoeModule.num_experts_per_tok: BaseMoeModule#num_experts_per_tok.
  MODE_TRAIN: MODE_TRAIN.
  BaseMoeModule._apply_expert_sharding: BaseMoeModule#_apply_expert_sharding().
  BaseMoeModule.mesh: BaseMoeModule#mesh.
  BaseMoeModule.partition_manager: BaseMoeModule#partition_manager.
  BaseMoeModule._get_gate_layer_sharding: BaseMoeModule#_get_gate_layer_sharding().
  BaseMoeModule._get_gate_layer_bias_sharding: BaseMoeModule#_get_gate_layer_bias_sharding().
  BaseMoeModule.auto_expert_mesh: BaseMoeModule#auto_expert_mesh.
  BaseMoeModule.module_moe_method: BaseMoeModule#module_moe_method.
  BaseMoeModule._replicate_and_sort_tokens: BaseMoeModule#_replicate_and_sort_tokens().
  BaseMoeModule._validate_routing_inputs: BaseMoeModule#_validate_routing_inputs().
  BaseMoeModule.routing_strategy: BaseMoeModule#routing_strategy.
  BaseMoeModule.expert_mesh: BaseMoeModule#expert_mesh.
  BaseMoeModule.expert_abstract_mesh: BaseMoeModule#expert_abstract_mesh.
  DP: DP.
  TP: TP.
  BaseMoeModule.lbl_coef: BaseMoeModule#lbl_coef.
  BaseMoeModule._apply_capacity_mask: BaseMoeModule#_apply_capacity_mask().
  BaseMoeModule._compute_router_z_loss: BaseMoeModule#_compute_router_z_loss().
  BaseMoeModule._apply_capacity_constraint: BaseMoeModule#_apply_capacity_constraint().
  EP: EP.
  BaseMoeModule._expert_group_mask: BaseMoeModule#_expert_group_mask().
  BaseMoeModule.__call__: BaseMoeModule#__call__().
  logger: logger.
  BaseMoeModule.hidden_size: BaseMoeModule#hidden_size.
  BaseMoeModule.rzl_coef: BaseMoeModule#rzl_coef.
  FSDP: FSDP.
  SP: SP.
  BaseMoeModule.load_balancing_strategy: BaseMoeModule#load_balancing_strategy.
  BaseMoeModule.normalize_selected_weights: BaseMoeModule#normalize_selected_weights().
  BaseMoeModule.passthrough_weights: BaseMoeModule#passthrough_weights().
  BaseMoeModule.hard_assignment_weights: BaseMoeModule#hard_assignment_weights().
  BaseMoeModule.expert_choice_weights: BaseMoeModule#expert_choice_weights().
  BaseMoeModule.uniform_weights: BaseMoeModule#uniform_weights().
  BATCH: BATCH.
  EMBED: EMBED.
  EXPERT: EXPERT.
  BaseMoeModule.craft_sharding: BaseMoeModule#craft_sharding().
  BaseMoeModule._create_expert_mask: BaseMoeModule#_create_expert_mask().
---
# Module: [`easydel/layers/moe/_moe_module.py`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py)

## Classes
### `BaseMoeModule`  ·  implements/extends ABC, Module
- def: [`easydel/layers/moe/_moe_module.py:108`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L108)
- doc: An abstract base class for Mixture of Experts (MoE) modules.
- signature: `class BaseMoeModule(nn.Module, ABC):`
- members:
  - `__call__(self, hidden_states: Float[Array, "batch seq hidden_dim"], **kwargs)` — [`L1771`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L1771) — Performs the forward pass of the MoE module.
  - `__init__(self, config: EasyDeLBaseConfig, n_routed_experts: int | None = None, num_experts_per_tok: int | None = None, hidden_size: int | None = None, lbl_coef: float | None = None, rzl_coef: float | None = None, routing_strategy: MoeRoutingStrategy = MoeRoutingStrategy.TOP_K, load_balancing_strategy: MoeLoadBalancingStrategy = MoeLoadBalancingStrategy.STANDARD, moe_hooks: MoeFusedHooks | None = None)` — [`L131`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L131) — Initializes the BaseMoeModule.
  - `_apply_capacity_constraint(self, selected_experts: jax.Array, selected_weights: jax.Array, capacity_factor: float | None = None)` — [`L759`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L759) — Apply soft capacity constraint to limit tokens per expert.
  - `_apply_capacity_mask(self, selected_experts: jax.Array, weights: jax.Array, capacity_factor: float)` — [`L345`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L345) — Applies soft capacity constraints to expert assignments.
  - `_apply_expert_sharding(self, tensor: Float[Array, ...], tensor_type: str = "weight")` — [`L614`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L614) — Applies expert parallel sharding to a tensor for distributed training.
  - `_compute_load_balancing_loss(self, router_probs: jax.Array, expert_loads: jax.Array, strategy: MoeLoadBalancingStrategy | None = None)` — [`L434`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L434) — Compute the load balancing auxiliary loss for even expert utilization.
  - `_compute_metrics(self, router_logits: jax.Array, router_probs: jax.Array, selected_experts: jax.Array, selected_weights: jax.Array, expert_loads: jax.Array)` — [`L549`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L549) — Compute and aggregate all MoE-related metrics and auxiliary losses.
  - `_compute_router_z_loss(self, router_logits: Float[Array, "batch_seq num_experts"])` — [`L509`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L509) — Compute router z-loss to encourage numerical stability.
  - `_configure_hooks_for_routing_strategy(self)` — [`L1515`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L1515) — Configure default hooks based on the current routing strategy.
  - `_create_expert_mask(self, selected_experts: Int[Array, "batch_seq k"], expert_id: int)` — [`L819`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L819) — Creates a boolean mask identifying tokens assigned to a specific expert.
  - `_expert_group_mask(self, gate_logits: jax.Array, n_groups: int, topk_groups: int)` — [`L393`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L393) — Creates a mask for hierarchical routing with grouped experts.
  - `_get_gate_layer_bias_sharding(self, bias_shape: tuple)` — [`L697`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L697) — Returns the partition specification for gate/router layer bias.
  - `_get_gate_layer_sharding(self, weight_shape: tuple)` — [`L677`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L677) — Returns the partition specification for gate/router layer weights.
  - `_get_sharding_status(self)` — [`L228`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L228) — Resolves and returns all parallelism axis names and sizes for this MoE layer.
  - `_moe_call_dense(self, hidden_state: jax.Array, gate_layer: nn.Module, wi_kernel: jax.Array, wu_kernel: jax.Array, wd_kernel: jax.Array, wi_bias: jax.Array | None = None, wu_bias: jax.Array | None = None, wd_bias: jax.Array | None = None, ffn_activation: Callable[[jax.Array, jax.Array], jax.Array] | None = None, gate_hidden_state: jax.Array | None = None, hooks: MoeFusedHooks | None = None, *, act_fn: Callable[[jax.Array], jax.Array], capacity_factor: float | None = None, output_metrics: bool = False)` — [`L1340`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L1340) — Dense MoE forward pass using per-token batched matmuls.
  - `_moe_call_standard(self, gate_layer: nn.Module, expert_layer: nn.Module, hidden_state: jax.Array, output_metrics: bool = False, validate_inputs: bool = False, apply_capacity_constraint: bool = False, reform_router_probs_fn: typing.Callable[[jax.Array], jax.Array] | None = None, layer_idx: int | None = None)` — [`L1627`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L1627) — Standard MoE forward pass: routing, permutation, expert computation, and combining.
  - `_replicate_and_sort_tokens(self, inputs_flat: jax.Array, selected_experts: jax.Array, use_custom_sort_vjp: bool = True)` — [`L293`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L293) — Replicates tokens k times and sorts them by assigned expert ID.
  - `_sparse_moe_call(self, hidden_state: jax.Array, gate_layer: nn.Module, wi_kernel: jax.Array, wu_kernel: jax.Array, wd_kernel: jax.Array, wi_bias: jax.Array | None = None, wu_bias: jax.Array | None = None, wd_bias: jax.Array | None = None, ffn_activation: Callable[[jax.Array, jax.Array], jax.Array] | None = None, gate_hidden_state: jax.Array | None = None, hooks: MoeFusedHooks | None = None, *, act_fn: Callable[[jax.Array], jax.Array])` — [`L845`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L845) — Fused MoE path using grouped matmul and shard_map.
  - `_validate_routing_inputs(self, hidden_states: Float[Array, "batch seq hidden_dim"], router_logits: Float[Array, "batch_seq num_experts"])` — [`L713`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L713) — Validate input tensor shapes for routing operations.
  - `craft_sharding(self, *, partition_manager=None, **_kwargs)` — [`L180`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L180) — Return dynamic partition specs for this module's parameters.
  - `expert_choice_weights(weights: jax.Array)` — [`L1595`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L1595) — Normalize weights for Expert Choice routing.
  - `get_moe_spec(self, direction: tp.Literal["row", "column"], tensors_are_expert: bool, is_bias: bool = False)` — [`L184`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L184) — Generate partition spec for MoE weight tensors.
  - `hard_assignment_weights(weights: jax.Array)` — [`L1580`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L1580) — Enforce hard assignment for SWITCH routing.
  - `moe_call(self, hidden_state: jax.Array, gate_layer: nn.Module, expert_layer: nn.Module, wi_kernel: jax.Array, wu_kernel: jax.Array, wd_kernel: jax.Array, wi_bias: jax.Array | None = None, wu_bias: jax.Array | None = None, wd_bias: jax.Array | None = None, ffn_activation: Callable[[jax.Array, jax.Array], jax.Array] | None = None, reform_router_probs_fn: typing.Callable[[jax.Array], jax.Array] | None = None, hooks: MoeFusedHooks | None = None, gate_hidden_state: jax.Array | None = None, *, act_fn: Callable[[jax.Array], jax.Array], output_metrics: bool = False, layer_idx: int | None = None)` — [`L1226`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L1226) — Wrapper for fused MoE call with automatic hook configuration.
  - `normalize_selected_weights(weights: jax.Array)` — [`L1548`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L1548) — Normalize top-k expert weights by their sum.
  - `passthrough_weights(weights: jax.Array)` — [`L1566`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L1566) — Pass through weights unchanged.
  - `uniform_weights(weights: jax.Array)` — [`L1613`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L1613) — Uniform weights for hash-based routing.
  - `auto_expert_mesh` — [`L175`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L175)
  - `config` — [`L161`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L161)
  - `dtype` — [`L178`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L178)
  - `expert_abstract_mesh` — [`L176`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L176)
  - `expert_mesh` — [`L174`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L174)
  - `hidden_size` — [`L166`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L166)
  - `lbl_coef` — [`L167`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L167)
  - `load_balancing_strategy` — [`L170`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L170)
  - `mesh` — [`L162`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L162)
  - `module_moe_method` — [`L172`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L172)
  - `moe_hooks` — [`L171`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L171)
  - `n_routed_experts` — [`L164`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L164)
  - `num_experts_per_tok` — [`L165`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L165)
  - `partition_manager` — [`L163`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L163)
  - `routing_strategy` — [`L169`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L169)
  - `rzl_coef` — [`L168`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L168)
- protocol/private: `_sparse_call`[`L1034`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L1034)
- uses (calls/refs, reference-scoped): [`EasyDeLBaseConfig`](../../infra/base_config.md#EasyDeLBaseConfig), [`partition_manager`](../../infra/base_config.md#EasyDeLBaseConfig.partition_manager), [`use_expert_tensor_mode`](../../infra/base_config.md#EasyDeLBaseConfig.use_expert_tensor_mode), [`MoeRoutingStrategy`](_communication_utils.md#MoeRoutingStrategy), [`MoeLoadBalancingStrategy`](_communication_utils.md#MoeLoadBalancingStrategy), [`check_bool_flag`](../../utils/helpers.md#check_bool_flag), [`EMPTY`](_moe_module.md#EMPTY), [`get_moe_partition_spec`](_communication_utils.md#get_moe_partition_spec), [`mesh`](../../infra/base_config.md#EasyDeLBaseConfig.mesh), [`TOP_K`](_communication_utils.md#MoeRoutingStrategy.TOP_K), [`STANDARD`](_communication_utils.md#MoeLoadBalancingStrategy.STANDARD), [`resolve_eformer_axis`](_communication_utils.md#resolve_eformer_axis), [`get_all_to_all_params`](_communication_utils.md#get_all_to_all_params), [`MoeFusedHooks`](_communication_utils.md#MoeFusedHooks), [`expert_abstract_mesh`](../../infra/base_config.md#EasyDeLBaseConfig.expert_abstract_mesh), [`expert_mesh`](../../infra/base_config.md#EasyDeLBaseConfig.expert_mesh), [`MODE_TRAIN`](_moe_module.md#MODE_TRAIN), [`auto_expert_mesh`](../../infra/base_config.md#EasyDeLBaseConfig.auto_expert_mesh), [`select_hook`](_communication_utils.md#MoeFusedHooks.select_hook), [`replace`](_communication_utils.md#MoeFusedHooks.replace), [`sort_activations`](_communication_utils.md#sort_activations), [`permute`](_communication_utils.md#permute), [`MoEMethods`](_communication_utils.md#MoEMethods), [`MoeMetrics`](_communication_utils.md#MoeMetrics), [`TOP_K_NDIV`](_communication_utils.md#MoeRoutingStrategy.TOP_K_NDIV), [`after_gate`](_communication_utils.md#MoeFusedHooks.after_gate), [`before_gate`](_communication_utils.md#MoeFusedHooks.before_gate), [`before_topk`](_communication_utils.md#MoeFusedHooks.before_topk), [`Glm4MoeMoE`](../../modules/glm4_moe/modeling_glm4_moe.md#Glm4MoeMoE), [`unpermute`](_communication_utils.md#unpermute), [`before_combine`](_communication_utils.md#MoeFusedHooks.before_combine), [`finalize_output`](_communication_utils.md#MoeFusedHooks.finalize_output), [`normalize_gate_logits`](_communication_utils.md#MoeFusedHooks.normalize_gate_logits), [`refine_weights_hook`](_communication_utils.md#MoeFusedHooks.refine_weights_hook), [`local_permute`](_communication_utils.md#local_permute), [`DP`](_moe_module.md#DP), [`TP`](_moe_module.md#TP), [`fsdp_is_ep_bound`](../../infra/base_config.md#EasyDeLBaseConfig.fsdp_is_ep_bound), [`get_experts_location`](_communication_utils.md#get_experts_location), [`refine_inputs_hook`](_communication_utils.md#MoeFusedHooks.refine_inputs_hook)  (+47 more)
- used by: [`easydel_to_torch`](../../utils/parameters_transformation.md#StateDictConverter.easydel_to_torch), [`shared_experts`](../../modules/glm4_moe_lite/modeling_glm4_moe_lite.md#Glm4MoeLiteMoE.shared_experts), [`shared_experts`](../../modules/glm_moe_dsa/modeling_glm_moe_dsa.md#GlmMoeDsaMoE.shared_experts), [`shared_experts`](../../modules/glm4_moe/modeling_glm4_moe.md#Glm4MoeMoE.shared_experts), [`__call__`](../../modules/gemma4/modeling_gemma4.md#Gemma4TextRouter.__call__), [`__init__`](../../modules/gpt_oss/modeling_gpt_oss.md#GptOssMLP.__init__), [`__init__`](../../modules/llama4/modeling_llama4.md#Llama4TextMoe.__init__), [`__call__`](../../modules/qwen3_next/modeling_qwen3_next.md#Qwen3NextSparseMoeBlock.__call__), [`__init__`](../../modules/minimax/modeling_minimax.md#MiniMaxSparseMoeBlock.__init__), [`__call__`](../../modules/deepseek_v2/modeling_deepseek.md#DeepseekV2MoE.__call__), [`__call__`](../../modules/deepseek_v3/modeling_deepseek.md#DeepseekV3MoE.__call__), [`__call__`](../../modules/kimi_linear/modeling_kimi_linear.md#KimiSparseMoeBlock.__call__), [`__init__`](../../modules/qwen2_moe/modeling_qwen2_moe.md#Qwen2MoeSparseBlock.__init__), [`__init__`](../../modules/qwen3_moe/modeling_qwen3_moe.md#Qwen3MoeSparseBlock.__init__), [`__init__`](../../modules/qwen3_next/modeling_qwen3_next.md#Qwen3NextSparseMoeBlock.__init__), [`__init__`](../../modules/qwen3_omni_moe/modeling_qwen3_omni_moe.md#Qwen3OmniMoeTalkerTextSparseMoeBlock.__init__), [`__init__`](../../modules/qwen3_omni_moe/modeling_qwen3_omni_moe.md#Qwen3OmniMoeTextSparseBlock.__init__), [`__init__`](../../modules/xerxes2/modeling_xerxes2.md#Xerxes2MoeSparseBlock.__init__), [`shared_experts`](../../modules/deepseek_v2/modeling_deepseek.md#DeepseekV2MoE.shared_experts), [`__call__`](../../modules/arctic/modeling_arctic.md#ArcticMoeBlock.__call__), [`__call__`](../../modules/gpt_oss/modeling_gpt_oss.md#GptOssMLP.__call__), [`__call__`](../../modules/llama4/modeling_llama4.md#Llama4TextMoe.__call__), [`__call__`](../../modules/minimax/modeling_minimax.md#MiniMaxSparseMoeBlock.__call__), [`__call__`](../../modules/qwen2_moe/modeling_qwen2_moe.md#Qwen2MoeSparseBlock.__call__), [`__call__`](../../modules/qwen3_omni_moe/modeling_qwen3_omni_moe.md#Qwen3OmniMoeTalkerTextSparseMoeBlock.__call__), [`_build_transform_fn`](../../infra/base_module.md#EasyDeLBaseModule._build_transform_fn), [`__call__`](../../modules/glm4_moe/modeling_glm4_moe.md#Glm4MoeMoE.__call__), [`__call__`](../../modules/glm4_moe_lite/modeling_glm4_moe_lite.md#Glm4MoeLiteMoE.__call__), [`__call__`](../../modules/glm_moe_dsa/modeling_glm_moe_dsa.md#GlmMoeDsaMoE.__call__), [`__init__`](../../modules/arctic/modeling_arctic.md#ArcticMoeBlock.__init__), [`__init__`](../../modules/deepseek_v2/modeling_deepseek.md#DeepseekV2MoE.__init__), [`__init__`](../../modules/deepseek_v3/modeling_deepseek.md#DeepseekV3MoE.__init__), [`__init__`](../../modules/gemma4/modeling_gemma4.md#Gemma4TextRouter.__init__), [`__init__`](../../modules/glm4_moe/modeling_glm4_moe.md#Glm4MoeMoE.__init__), [`__init__`](../../modules/glm4_moe_lite/modeling_glm4_moe_lite.md#Glm4MoeLiteMoE.__init__), [`__init__`](../../modules/glm_moe_dsa/modeling_glm_moe_dsa.md#GlmMoeDsaMoE.__init__), [`__init__`](../../modules/kimi_linear/modeling_kimi_linear.md#KimiSparseMoeBlock.__init__), [`__init__`](../../modules/mixtral/modeling_mixtral.md#MixtralSparseMoeBlock.__init__), [`__init__`](../../modules/qwen3_vl_moe/modeling_qwen3_vl_moe.md#Qwen3VLMoeTextSparseBlock.__init__), [`__call__`](../../modules/mixtral/modeling_mixtral.md#MixtralSparseMoeBlock.__call__)  (+26 more)

## Module values
- `BATCH` — [`L96`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L96)
- `DP` — [`L102`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L102)
- `EMBED` — [`L98`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L98)
- `EMPTY` — [`L97`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L97)
- `EP` — [`L101`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L101)
- `EXPERT` — [`L99`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L99)
- `FSDP` — [`L103`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L103)
- `MODE_TRAIN` — [`L100`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L100)
- `SP` — [`L105`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L105)
- `TP` — [`L104`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L104)
- `logger` — [`L93`](../../../../../../../raw/code/EasyDeL/easydel/layers/moe/_moe_module.py#L93)


---
title: 'Module: axlearn/common/mixture_of_experts.py'
type: catalog
provenance: extracted
module: axlearn/common/mixture_of_experts.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.mixture_of_experts`/
symbols:
  GateNoise: GateNoise#
  GateNoise.GUMBEL: GateNoise#GUMBEL.
  GMMBackend: GMMBackend#
  GMMBackend.PALLAS: GMMBackend#PALLAS.
  GMMBackend.RAGGED_DOT: GMMBackend#RAGGED_DOT.
  GMMBackend.TOKAMAX: GMMBackend#TOKAMAX.
  TopKFn: TopKFn.
  ScoreFn: ScoreFn.
  approx_max_k: approx_max_k().
  sigmoid: sigmoid().
  _router_z_loss: _router_z_loss().
  _cum_sum: _cum_sum().
  _create_over_capacity_ratio_summary: _create_over_capacity_ratio_summary().
  _compute_expert_capacity: _compute_expert_capacity().
  _cap_logits: _cap_logits().
  get_outer_batch_from_mesh: get_outer_batch_from_mesh().
  AdaptiveLoadBalanceLoss: AdaptiveLoadBalanceLoss#
  AdaptiveLoadBalanceLoss.Config: AdaptiveLoadBalanceLoss#Config#
  AdaptiveLoadBalanceLoss.Config.moving_average: AdaptiveLoadBalanceLoss#Config#moving_average.
  AdaptiveLoadBalanceLoss.Config.max_value: AdaptiveLoadBalanceLoss#Config#max_value.
  AdaptiveLoadBalanceLoss.Config.min_value: AdaptiveLoadBalanceLoss#Config#min_value.
  AdaptiveLoadBalanceLoss.Config.log_step: AdaptiveLoadBalanceLoss#Config#log_step.
  AdaptiveLoadBalanceLoss.__init__: AdaptiveLoadBalanceLoss#__init__().
  AdaptiveLoadBalanceLoss._create_layer_parameter_specs: AdaptiveLoadBalanceLoss#_create_layer_parameter_specs().
  AdaptiveLoadBalanceLoss.forward: AdaptiveLoadBalanceLoss#forward().
  BaseGating: BaseGating#
  BaseGating.Config: BaseGating#Config#
  BaseGating.Config.num_experts: BaseGating#Config#num_experts.
  BaseGating.Output: BaseGating#Output#
  BaseGating.Output.combine_tensor: BaseGating#Output#combine_tensor.
  BaseGating.Output.dispatch_tensor: BaseGating#Output#dispatch_tensor.
  BaseGating.Output.load_balance_loss: BaseGating#Output#load_balance_loss.
  BaseGating.Output.router_z_loss: BaseGating#Output#router_z_loss.
  BaseGating.forward: BaseGating#forward().
  BaseGating.dispatch: BaseGating#dispatch().
  BaseGating.combine: BaseGating#combine().
  Top2Gating: Top2Gating#
  Top2Gating.Config: Top2Gating#Config#
  Top2Gating.Config.gating_logit_cap: Top2Gating#Config#gating_logit_cap.
  Top2Gating.Config.mask_dtype: Top2Gating#Config#mask_dtype.
  Top2Gating.Config.train_capacity_factor: Top2Gating#Config#train_capacity_factor.
  Top2Gating.Config.eval_capacity_factor: Top2Gating#Config#eval_capacity_factor.
  Top2Gating.Config.expert_capacity: Top2Gating#Config#expert_capacity.
  Top2Gating.Config.adaptive_load_balance_loss: Top2Gating#Config#adaptive_load_balance_loss.
  Top2Gating.__init__: Top2Gating#__init__().
  Top2Gating.forward: Top2Gating#forward().
  Top2Gating.dispatch: Top2Gating#dispatch().
  Top2Gating.combine: Top2Gating#combine().
  TopKGating: TopKGating#
  TopKGating.Config: TopKGating#Config#
  TopKGating.Config.num_experts_per_token: TopKGating#Config#num_experts_per_token.
  TopKGating.Config.gating_logit_cap: TopKGating#Config#gating_logit_cap.
  TopKGating.Config.train_capacity_factor: TopKGating#Config#train_capacity_factor.
  TopKGating.Config.eval_capacity_factor: TopKGating#Config#eval_capacity_factor.
  TopKGating.Config.expert_capacity: TopKGating#Config#expert_capacity.
  TopKGating.Config.adaptive_load_balance_loss: TopKGating#Config#adaptive_load_balance_loss.
  TopKGating.Config.noisy_gating: TopKGating#Config#noisy_gating.
  TopKGating.Config.topk_fn: TopKGating#Config#topk_fn.
  TopKGating.Config.score_fn: TopKGating#Config#score_fn.
  TopKGating.dispatch_tensor_shape: TopKGating#dispatch_tensor_shape().
  TopKGating.combine_tensor_shape: TopKGating#combine_tensor_shape().
  TopKGating.__init__: TopKGating#__init__().
  TopKGating._add_gumbel_noise: TopKGating#_add_gumbel_noise().
  TopKGating._top_k: TopKGating#_top_k().
  TopKGating._score: TopKGating#_score().
  TopKGating._get_normalized_gates: TopKGating#_get_normalized_gates().
  TopKGating._process_logits: TopKGating#_process_logits().
  TopKGating._get_expert_capacity: TopKGating#_get_expert_capacity().
  TopKGating._load_balance_loss: TopKGating#_load_balance_loss().
  TopKGating.forward: TopKGating#forward().
  TopKGating.dispatch: TopKGating#dispatch().
  TopKGating.combine: TopKGating#combine().
  TopKDropFreeGating: TopKDropFreeGating#
  TopKDropFreeGating.Output: TopKDropFreeGating#Output#
  TopKDropFreeGating.Output.gate_assignment: TopKDropFreeGating#Output#gate_assignment.
  TopKDropFreeGating.Output.expert_weights: TopKDropFreeGating#Output#expert_weights.
  TopKDropFreeGating.Output.load_balance_loss: TopKDropFreeGating#Output#load_balance_loss.
  TopKDropFreeGating.Output.router_z_loss: TopKDropFreeGating#Output#router_z_loss.
  TopKDropFreeGating.Output.seq_load_balance_loss: TopKDropFreeGating#Output#seq_load_balance_loss.
  TopKDropFreeGating._load_balance_loss: TopKDropFreeGating#_load_balance_loss().
  TopKDropFreeGating._seq_load_balance_loss: TopKDropFreeGating#_seq_load_balance_loss().
  TopKDropFreeGating.forward: TopKDropFreeGating#forward().
  TopKBiasGating: TopKBiasGating#
  TopKBiasGating.Config: TopKBiasGating#Config#
  TopKBiasGating.Config.gating_update_rate: TopKBiasGating#Config#gating_update_rate.
  TopKBiasGating.Config.routed_scaling_factor: TopKBiasGating#Config#routed_scaling_factor.
  TopKBiasGating.Config.num_group_of_experts: TopKBiasGating#Config#num_group_of_experts.
  TopKBiasGating.Config.topk_group: TopKBiasGating#Config#topk_group.
  TopKBiasGating._create_layer_parameter_specs: TopKBiasGating#_create_layer_parameter_specs().
  TopKBiasGating._adjust_gating_scores: TopKBiasGating#_adjust_gating_scores().
  TopKBiasGating._update_gating_bias: TopKBiasGating#_update_gating_bias().
  TopKBiasGating._top_k: TopKBiasGating#_top_k().
  TopKBiasGating._top_k_with_two_gates: TopKBiasGating#_top_k_with_two_gates().
  TopKBiasGating.forward: TopKBiasGating#forward().
  TransformerFeedForwardMoE: TransformerFeedForwardMoE#
  TransformerFeedForwardMoE.Config: TransformerFeedForwardMoE#Config#
  TransformerFeedForwardMoE.Config.input_dim: TransformerFeedForwardMoE#Config#input_dim.
  TransformerFeedForwardMoE.Config.hidden_dim: TransformerFeedForwardMoE#Config#hidden_dim.
  TransformerFeedForwardMoE.Config.outer_batch: TransformerFeedForwardMoE#Config#outer_batch.
  TransformerFeedForwardMoE.Config.norm: TransformerFeedForwardMoE#Config#norm.
  TransformerFeedForwardMoE.Config.activation: TransformerFeedForwardMoE#Config#activation.
  TransformerFeedForwardMoE.Config.dropout: TransformerFeedForwardMoE#Config#dropout.
  TransformerFeedForwardMoE.Config.stochastic_depth: TransformerFeedForwardMoE#Config#stochastic_depth.
  TransformerFeedForwardMoE.Config.structure: TransformerFeedForwardMoE#Config#structure.
  TransformerFeedForwardMoE.Config.residual_weight: TransformerFeedForwardMoE#Config#residual_weight.
  TransformerFeedForwardMoE.Config.num_experts: TransformerFeedForwardMoE#Config#num_experts.
  TransformerFeedForwardMoE.Config.num_groups: TransformerFeedForwardMoE#Config#num_groups.
  TransformerFeedForwardMoE.Config.gating: TransformerFeedForwardMoE#Config#gating.
  TransformerFeedForwardMoE.Config.load_balance_loss_weight: TransformerFeedForwardMoE#Config#load_balance_loss_weight.
  TransformerFeedForwardMoE.Config.router_z_loss_weight: TransformerFeedForwardMoE#Config#router_z_loss_weight.
  TransformerFeedForwardMoE.Config.corr_loss_weight: TransformerFeedForwardMoE#Config#corr_loss_weight.
  TransformerFeedForwardMoE.Config.dim_to_mesh_axis_map: TransformerFeedForwardMoE#Config#dim_to_mesh_axis_map.
  TransformerFeedForwardMoE.Config.residual_gate_init: TransformerFeedForwardMoE#Config#residual_gate_init.
  TransformerFeedForwardMoE.default_config: TransformerFeedForwardMoE#default_config().
  TransformerFeedForwardMoE._create_layer_parameter_specs: TransformerFeedForwardMoE#_create_layer_parameter_specs().
  TransformerFeedForwardMoE.__init__: TransformerFeedForwardMoE#__init__().
  TransformerFeedForwardMoE._expert_correlation_loss: TransformerFeedForwardMoE#_expert_correlation_loss().
  TransformerFeedForwardMoE.forward: TransformerFeedForwardMoE#forward().
  TransformerFeedForwardMoE._dispatch_and_combine: TransformerFeedForwardMoE#_dispatch_and_combine().
  TransformerFeedForwardMoE._wi_activation: TransformerFeedForwardMoE#_wi_activation().
  _convert_feedforward_to_moe_parameters: _convert_feedforward_to_moe_parameters().
  convert_dense_to_moe_parameters: convert_dense_to_moe_parameters().
  _custom_gather: _custom_gather().
  _custom_gather_fwd: _custom_gather_fwd().
  _custom_gather_bwd: _custom_gather_bwd().
  _get_all_to_all_params: _get_all_to_all_params().
  _drop_tokens: _drop_tokens().
  _RaggedA2AParams: _RaggedA2AParams.
  _ragged_all_to_all: _ragged_all_to_all().
  _ragged_all_to_all_fwd: _ragged_all_to_all_fwd().
  _ragged_all_to_all_bwd: _ragged_all_to_all_bwd().
  _all_to_all_dispatch: _all_to_all_dispatch().
  _all_to_all_combine: _all_to_all_combine().
  TransformerFeedForwardDropFreeMoE: TransformerFeedForwardDropFreeMoE#
  TransformerFeedForwardDropFreeMoE.Config: TransformerFeedForwardDropFreeMoE#Config#
  TransformerFeedForwardDropFreeMoE.Config.gmm_backend: TransformerFeedForwardDropFreeMoE#Config#gmm_backend.
  TransformerFeedForwardDropFreeMoE.Config.tiling: TransformerFeedForwardDropFreeMoE#Config#tiling.
  TransformerFeedForwardDropFreeMoE.Config.input_dim_to_partition_spec: TransformerFeedForwardDropFreeMoE#Config#input_dim_to_partition_spec.
  TransformerFeedForwardDropFreeMoE.Config.output_dim_to_partition_spec: TransformerFeedForwardDropFreeMoE#Config#output_dim_to_partition_spec.
  TransformerFeedForwardDropFreeMoE.Config.interpret: TransformerFeedForwardDropFreeMoE#Config#interpret.
  TransformerFeedForwardDropFreeMoE.Config.preferred_element_type: TransformerFeedForwardDropFreeMoE#Config#preferred_element_type.
  TransformerFeedForwardDropFreeMoE.Config.seq_load_balance_loss_weight: TransformerFeedForwardDropFreeMoE#Config#seq_load_balance_loss_weight.
  TransformerFeedForwardDropFreeMoE.Config.tokamax_implementation: TransformerFeedForwardDropFreeMoE#Config#tokamax_implementation.
  TransformerFeedForwardDropFreeMoE.default_config: TransformerFeedForwardDropFreeMoE#default_config().
  TransformerFeedForwardDropFreeMoE.__init__: TransformerFeedForwardDropFreeMoE#__init__().
  TransformerFeedForwardDropFreeMoE._padded_gmm: TransformerFeedForwardDropFreeMoE#_padded_gmm().
  TransformerFeedForwardDropFreeMoE._ragged_dot_gmm: TransformerFeedForwardDropFreeMoE#_ragged_dot_gmm().
  TransformerFeedForwardDropFreeMoE._tokamax_gmm: TransformerFeedForwardDropFreeMoE#_tokamax_gmm().
  TransformerFeedForwardDropFreeMoE._pallas_gmm: TransformerFeedForwardDropFreeMoE#_pallas_gmm().
  TransformerFeedForwardDropFreeMoE._dispatch_hook: TransformerFeedForwardDropFreeMoE#_dispatch_hook().
  TransformerFeedForwardDropFreeMoE._combine_hook: TransformerFeedForwardDropFreeMoE#_combine_hook().
  TransformerFeedForwardDropFreeMoE._additional_shmap_output_sharding: TransformerFeedForwardDropFreeMoE#_additional_shmap_output_sharding().
  TransformerFeedForwardDropFreeMoE._additional_shmap_output_hook: TransformerFeedForwardDropFreeMoE#_additional_shmap_output_hook().
  TransformerFeedForwardDropFreeMoE._dispatch_and_combine: TransformerFeedForwardDropFreeMoE#_dispatch_and_combine().
  ApproximateTokenDropFreeMoE: ApproximateTokenDropFreeMoE#
  ApproximateTokenDropFreeMoE.Config: ApproximateTokenDropFreeMoE#Config#
  ApproximateTokenDropFreeMoE.Config.expert_parallel_capacity: ApproximateTokenDropFreeMoE#Config#expert_parallel_capacity.
  ApproximateTokenDropFreeMoE.default_config: ApproximateTokenDropFreeMoE#default_config().
  ApproximateTokenDropFreeMoE._has_track_axis: ApproximateTokenDropFreeMoE#_has_track_axis().
  ApproximateTokenDropFreeMoE._dispatch_hook: ApproximateTokenDropFreeMoE#_dispatch_hook().
  ApproximateTokenDropFreeMoE._combine_hook: ApproximateTokenDropFreeMoE#_combine_hook().
  ApproximateTokenDropFreeMoE._additional_shmap_output_sharding: ApproximateTokenDropFreeMoE#_additional_shmap_output_sharding().
  ApproximateTokenDropFreeMoE._additional_shmap_output_hook: ApproximateTokenDropFreeMoE#_additional_shmap_output_hook().
  set_interpret_in_moe_config_recursively: set_interpret_in_moe_config_recursively().
  V6eGMMTilingModifier: V6eGMMTilingModifier#
  V6eGMMTilingModifier.__call__: V6eGMMTilingModifier#__call__().
---
# Module: [`axlearn/common/mixture_of_experts.py`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py)

## Classes
### `AdaptiveLoadBalanceLoss`
- def: [`axlearn/common/mixture_of_experts.py:306`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L306)
- doc: A layer to adjust the aux loss weight based on the overcapacity ratio.
- signature: `class AdaptiveLoadBalanceLoss(BaseLayer):`
- members:
  - `forward(self, value: Tensor)` — [`L337`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L337) — Adjusts and returns the loss scale based on a moving average of `value`.
- protocol/private: `__init__`[`L321`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L321), `_create_layer_parameter_specs`[`L326`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L326)

### `ApproximateTokenDropFreeMoE`
- def: [`axlearn/common/mixture_of_experts.py:2694`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L2694)
- doc: Mostly the same as `TransformerFeedForwardDropFreeMoE`, but allows expert parallel training.
- signature: `class ApproximateTokenDropFreeMoE(TransformerFeedForwardDropFreeMoE):`
- members:
  - `_has_track_axis(self)` — [`L2730`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L2730) — Check if we're in a vmap context with track axis (VectorizedTrackTransformerLayer).
  - `default_config(cls)` — [`L2712`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L2712)
- protocol/private: `_additional_shmap_output_hook`[`L2759`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L2759), `_additional_shmap_output_sharding`[`L2750`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L2750), `_combine_hook`[`L2745`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L2745), `_dispatch_hook`[`L2735`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L2735)

### `BaseGating`
- def: [`axlearn/common/mixture_of_experts.py:364`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L364)
- doc: An abstract class to define the common interface of gating layers.
- signature: `class BaseGating(BaseLayer):`
- members:
  - `combine(self, inputs: Tensor, *, combine_tensor: Tensor, dtype: jnp.dtype=jnp.bfloat16, partition_spec: PartitionSpec=PartitionSpec(None))` — [`L430`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L430) — Take weighted average / combination of dispatched tensors.
  - `dispatch(self, inputs: Tensor, *, dispatch_tensor: Tensor, dtype: jnp.dtype=jnp.bfloat16, partition_spec: PartitionSpec=PartitionSpec(None), combine_tensor: Optional[Tensor]=None)` — [`L406`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L406) — Dispatch the input tensors according to dispatch_tensor.
  - `forward(self, logits: Tensor)` — [`L394`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L394) — Forward pass of gating.

### `Config`
- def: [`axlearn/common/mixture_of_experts.py:2705`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L2705)
- signature: `class Config(TransformerFeedForwardDropFreeMoE.Config):`
- members:
  - `activation` — [`L1402`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1402)
  - `adaptive_load_balance_loss` — [`L487`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L487)
  - `adaptive_load_balance_loss` — [`L722`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L722)
  - `corr_loss_weight` — [`L1431`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1431)
  - `dim_to_mesh_axis_map` — [`L1441`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1441)
  - `dropout` — [`L1403`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1403)
  - `eval_capacity_factor` — [`L481`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L481)
  - `eval_capacity_factor` — [`L713`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L713)
  - `expert_capacity` — [`L484`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L484)
  - `expert_capacity` — [`L719`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L719)
  - `expert_parallel_capacity` — [`L2709`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L2709)
  - `gating` — [`L1423`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1423)
  - `gating_logit_cap` — [`L474`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L474)
  - `gating_logit_cap` — [`L708`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L708)
  - `gating_update_rate` — [`L1200`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1200)
  - `gmm_backend` — [`L2258`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L2258)
  - `hidden_dim` — [`L1395`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1395)
  - `input_dim` — [`L1394`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1394)
  - `input_dim_to_partition_spec` — [`L2268`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L2268)
  - `interpret` — [`L2281`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L2281)
  - `load_balance_loss_weight` — [`L1426`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1426)
  - `log_step` — [`L319`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L319)
  - `mask_dtype` — [`L477`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L477)
  - `max_value` — [`L316`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L316)
  - `min_value` — [`L317`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L317)
  - `moving_average` — [`L315`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L315)
  - `noisy_gating` — [`L725`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L725)
  - `norm` — [`L1401`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1401)
  - `num_experts` — [`L382`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L382)
  - `num_experts` — [`L1419`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1419)
  - `num_experts_per_token` — [`L704`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L704)
  - `num_group_of_experts` — [`L1207`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1207)
  - `num_groups` — [`L1421`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1421)
  - `outer_batch` — [`L1399`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1399)
  - `output_dim_to_partition_spec` — [`L2275`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L2275)
  - `preferred_element_type` — [`L2283`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L2283)
  - `residual_gate_init` — [`L1445`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1445)
  - `residual_weight` — [`L1418`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1418)
  - `routed_scaling_factor` — [`L1202`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1202)
  - `router_z_loss_weight` — [`L1428`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1428)
  - `score_fn` — [`L732`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L732)
  - `seq_load_balance_loss_weight` — [`L2290`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L2290)
  - `stochastic_depth` — [`L1404`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1404)
  - `structure` — [`L1415`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1415)
  - `tiling` — [`L2266`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L2266)
  - `tokamax_implementation` — [`L2296`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L2296)
  - `topk_fn` — [`L729`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L729)
  - `topk_group` — [`L1208`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1208)
  - `train_capacity_factor` — [`L480`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L480)
  - `train_capacity_factor` — [`L712`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L712)

### `GMMBackend`
- def: [`axlearn/common/mixture_of_experts.py:84`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L84)
- doc: Backend implementations for Grouped Matrix Multiplication (GMM).
- signature: `class GMMBackend(enum.Enum):`
- members:
  - `PALLAS` — [`L93`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L93)
  - `RAGGED_DOT` — [`L97`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L97)
  - `TOKAMAX` — [`L101`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L101)

### `GateNoise`
- def: [`axlearn/common/mixture_of_experts.py:77`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L77)
- doc: Types of noise that can be added to gating logits.
- signature: `class GateNoise(enum.Enum):`
- members:
  - `GUMBEL` — [`L81`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L81)

### `Output`
- def: [`axlearn/common/mixture_of_experts.py:1054`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1054)
- doc: Output of TopKDropFreeGating.
- signature: `class Output(NamedTuple):`
- members:
  - `combine_tensor` — [`L386`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L386)
  - `dispatch_tensor` — [`L388`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L388)
  - `expert_weights` — [`L1060`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1060)
  - `gate_assignment` — [`L1058`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1058)
  - `load_balance_loss` — [`L390`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L390)
  - `load_balance_loss` — [`L1062`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1062)
  - `router_z_loss` — [`L392`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L392)
  - `router_z_loss` — [`L1064`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1064)
  - `seq_load_balance_loss` — [`L1066`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1066)

### `Top2Gating`
- def: [`axlearn/common/mixture_of_experts.py:452`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L452)
- doc: Computes Top-2 gating for Mixture-of-Experts.
- signature: `class Top2Gating(BaseGating):`
- members:
  - `combine(self, inputs: Tensor, *, combine_tensor: Tensor, dtype: jnp.dtype=jnp.bfloat16, partition_spec: PartitionSpec=PartitionSpec(None))` — [`L669`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L669) — Take weighted average / combination of dispatched tensors.
  - `dispatch(self, inputs: Tensor, *, dispatch_tensor: Tensor, dtype: jnp.dtype=jnp.bfloat16, partition_spec: PartitionSpec=PartitionSpec(None), combine_tensor: Optional[Tensor]=None)` — [`L642`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L642) — Dispatch the input tensors according to dispatch_tensor.
  - `forward(self, logits: Tensor)` — [`L496`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L496) — Please see comments of BaseGating.forward.
- protocol/private: `__init__`[`L489`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L489)

### `TopKBiasGating`
- def: [`axlearn/common/mixture_of_experts.py:1189`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1189)
- doc: An implementation of gate with Auxiliary-Loss-Free Load Balancing strategy.
- signature: `class TopKBiasGating(TopKDropFreeGating):`
- members:
  - `_adjust_gating_scores(self, *, raw_gates: Tensor)` — [`L1221`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1221) — Adjusts the gating scores after the score_fn called with the loading bias.
  - `forward(self, logits: Tensor, seq_load_balance_loss_weight: Optional[float]=None)` — [`L1262`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1262)
- protocol/private: `_create_layer_parameter_specs`[`L1210`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1210), `_top_k`[`L1235`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1235), `_top_k_with_two_gates`[`L1249`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1249), `_update_gating_bias`[`L1225`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1225)

### `TopKDropFreeGating`
- def: [`axlearn/common/mixture_of_experts.py:1051`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1051)
- doc: Computes Token-drop free Top-K gating for Mixture-of-Experts.
- signature: `class TopKDropFreeGating(TopKGating):`
- members:
  - `_load_balance_loss(self, *, raw_gates: Tensor, gate_assignment: Tensor, num_experts_per_token: int)` — [`L1069`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1069) — Calculates the load balance loss.
  - `_seq_load_balance_loss(self, *, raw_gates: Tensor, gate_assignment: Tensor)` — [`L1100`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1100) — Calculates the sequence wise load balance loss.
  - `forward(self, logits: Tensor, seq_load_balance_loss_weight: Optional[float]=None)` — [`L1139`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1139)

### `TopKGating`
- def: [`axlearn/common/mixture_of_experts.py:693`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L693)
- doc: Generalized Top-K gating for Mixture-of-Experts.
- signature: `class TopKGating(BaseGating):`
- members:
  - `_add_gumbel_noise(self, logits: Tensor)` — [`L750`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L750) — Adds Gumbel noise to logits for noisy gating.
  - `_get_expert_capacity(self, *, group_size: int)` — [`L810`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L810) — Computes expert capacity based on configuration.
  - `_get_normalized_gates(self, raw_gates: Tensor)` — [`L780`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L780) — Gets normalized gates. If using softmax, already normalized; otherwise normalize.
  - `_load_balance_loss(self, *, raw_gates: Tensor, gate_assignment: Tensor, num_experts_per_token: int)` — [`L827`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L827) — Calculates the load balance loss.
  - `_process_logits(self, logits: Tensor)` — [`L789`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L789) — Converts input logits into float32, caps values, and optionally adds noise.
  - `_score(self, logits: Tensor, axis: int=-1)` — [`L772`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L772) — Computes scores from logits using configured score_fn or default softmax.
  - `_top_k(self, raw_gates: Tensor, k: int)` — [`L757`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L757) — Selects top-k experts, using configured topk_fn if available.
  - `combine(self, inputs: Tensor, *, combine_tensor: Tensor, dtype: jnp.dtype=jnp.bfloat16, partition_spec: PartitionSpec=PartitionSpec(None))` — [`L1027`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1027) — Take weighted average / combination of dispatched tensors.
  - `combine_tensor_shape(cls)` — [`L740`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L740) — Returns the shape specification for the combine tensor.
  - `dispatch(self, inputs: Tensor, *, dispatch_tensor: Tensor, dtype: jnp.dtype=jnp.bfloat16, partition_spec: PartitionSpec=PartitionSpec(None), combine_tensor: Optional[Tensor]=None)` — [`L1000`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1000) — Dispatch the input tensors according to dispatch_tensor.
  - `dispatch_tensor_shape(cls)` — [`L735`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L735) — Returns the shape specification for the dispatch tensor.
  - `forward(self, logits: Tensor)` — [`L864`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L864) — Please see comments of BaseGating.forward.
- protocol/private: `__init__`[`L744`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L744)

### `TransformerFeedForwardDropFreeMoE`
- def: [`axlearn/common/mixture_of_experts.py:2246`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L2246)
- doc: A Transformer feed-forward layer with mixture of experts with NO token drop.
- signature: `class TransformerFeedForwardDropFreeMoE(TransformerFeedForwardMoE):`
- members:
  - `_additional_shmap_output_hook(self, out: Sequence[Tensor])` — [`L2479`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L2479) — Hook for processing additional shmap output.
  - `_additional_shmap_output_sharding(self, mesh: jax.sharding.Mesh)` — [`L2474`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L2474) — Specifies the sharding for _dispatch_hook(...)[2].
  - `_combine_hook(self, *, sorted_output: Tensor, residuals: Sequence[Any])` — [`L2469`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L2469) — Hook for subclasses to perform additional processing during combine.
  - `_dispatch_and_combine(self, x: Tensor)` — [`L2484`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L2484) — Runs forward pass on the linear layers and dispatching and combining.
  - `_dispatch_hook(self, *, sorted_inputs: Tensor, tokens_per_expert: Tensor)` — [`L2451`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L2451) — Hook for subclasses to perform additional processing during dispatch.
  - `_pallas_gmm(self, lhs, rhs, tokens_per_expert)` — [`L2400`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L2400) — Performs grouped matrix multiplication using Pallas/Triton-based GMM kernel.
  - `_ragged_dot_gmm(self, lhs, rhs, tokens_per_expert)` — [`L2333`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L2333) — Performs grouped matrix multiplication using jax.lax.ragged_dot.
  - `_tokamax_gmm(self, lhs, rhs, tokens_per_expert)` — [`L2365`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L2365) — Performs grouped matrix multiplication using tokamax.ragged_dot.
  - `default_config(cls)` — [`L2299`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L2299)
- protocol/private: `__init__`[`L2313`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L2313), `_padded_gmm`[`L2321`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L2321)

### `TransformerFeedForwardMoE`
- def: [`axlearn/common/mixture_of_experts.py:1381`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1381)
- doc: A Transformer feed-forward layer with mixture of experts.
- signature: `class TransformerFeedForwardMoE(DenseGeneralBaseLayer):`
- members:
  - `_dispatch_and_combine(self, x: Tensor)` — [`L1622`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1622) — Runs forward pass on the linear layers and dispatching and combining.
  - `_expert_correlation_loss(self)` — [`L1540`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1540) — Computes the correlation among the experts based on gate_weight.
  - `_wi_activation(self, x: Tensor, dispatch_tensor: Tensor)` — [`L1723`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1723) — Applies activation functions to the input projection.
  - `default_config(cls)` — [`L1448`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1448)
  - `forward(self, inputs: Tensor)` — [`L1559`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1559)
- protocol/private: `__init__`[`L1505`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1505), `_create_layer_parameter_specs`[`L1463`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1463)

### `V6eGMMTilingModifier`
- def: [`axlearn/common/mixture_of_experts.py:2792`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L2792)
- doc: Modifies the tiling config of TransformerFeedForwardDropFreeMoE for V6e TPU.
- signature: `class V6eGMMTilingModifier(ConfigModifier):`
- protocol/private: `__call__`[`L2795`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L2795)

## Functions
- `_all_to_all_combine(sorted_output: Tensor, residuals: Sequence[Tensor], has_track_axis: bool=False)` — [`L2203`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L2203) — Perform all-to-all combine for expert parallelism.
- `_all_to_all_dispatch(sorted_inputs: Tensor, tokens_per_expert: Tensor, expert_parallel_capacity: float, has_track_axis: bool=False)` — [`L2106`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L2106) — Perform all-to-all dispatch for expert parallelism.
- `_cap_logits(logits: Tensor, gating_logit_cap: float)` — [`L253`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L253)
- `_compute_expert_capacity(*, group_size: int, num_experts: int, expert_capacity: Optional[int], capacity_factor: Optional[float])` — [`L223`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L223) — Computes the final expert capacity.
- `_convert_feedforward_to_moe_parameters(source_parameters: Nested[Tensor], *, num_experts: int, moe_parameter_specs: Nested[ParameterSpec])` — [`L1781`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1781) — Converts parameters of a TransformerFeedForwardLayer to those of a TransformerFeedForwardMoE.
- `_create_over_capacity_ratio_summary(*, mask: Tensor, position_in_expert: Tensor, capacity: float)` — [`L201`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L201) — Computes the capacity ratio of tokens that were not dispatched due to lack of capcity.
- `_cum_sum(elements: Tensor, *, axis: int=0, exclusive: bool=False, reverse: bool=False)` — [`L174`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L174) — Same as jax.np.cumsum but with the extra options from tf.cumsum.
- `_custom_gather(x: Tensor, idx: Tensor, argsort_idx: Tensor, unique_indices: bool=True)` — [`L1944`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1944) — Equivalent to `x.at[idx].get(unique_indices=unique_indices)`, but with a gather-based
- `_custom_gather_bwd(unique_indices: bool, res: tuple[Tensor, Tensor], g: Tensor)` — [`L1982`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1982)
- `_custom_gather_fwd(x: Tensor, idx: Tensor, argsort_idx: Tensor, unique_indices: bool)` — [`L1976`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1976)
- `_drop_tokens(all_tokens_per_expert: Tensor, max_size: int)` — [`L2026`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L2026) — Reduce the number of tokens in some experts in some ranks so that after all to all, no rank
- `_get_all_to_all_params(all_sizes: Tensor, ep_shard: Tensor, input_all_sizes_nodrop: Optional[Tensor]=None, output_all_sizes_nodrop: Optional[Tensor]=None)` — [`L2001`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L2001) — Computes sizes and offsets required by `lax.ragged_all_to_all.`
- `_ragged_all_to_all(inputs: Tensor, outputs: Tensor, fwd_params: _RaggedA2AParams, bwd_params: _RaggedA2AParams, axis_name: str)` — [`L2065`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L2065) — Equivalent to `lax.ragged_all_to_all(inputs, outputs, *fwd_params, axis_name=axis_name)`
- `_ragged_all_to_all_bwd(axis_name: str, res: tuple[_RaggedA2AParams, Tensor], g: Tensor)` — [`L2093`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L2093)
- `_ragged_all_to_all_fwd(inputs: Tensor, outputs: Tensor, fwd_params: _RaggedA2AParams, bwd_params: _RaggedA2AParams, axis_name: str)` — [`L2080`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L2080)
- `_router_z_loss(logits: Tensor)` — [`L157`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L157) — Loss that encourages router logits to remain small and improves stability.
- `approx_max_k(*, reduction_dimension=-1, recall_target=0.95, reduction_input_size_override=-1, aggregate_to_topk=True)` — [`L109`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L109) — Creates a configured approximate top-k function.
- `convert_dense_to_moe_parameters(source_parameters: Nested[Tensor], *, target_parameter_specs: Nested[ParameterSpec])` — [`L1827`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L1827) — Converts parameters of a dense BaseTransformerLayer to parameters of a target layer.
- `get_outer_batch_from_mesh(*, mesh_axis_names: Sequence[str], outer_batch_axis_names: Sequence[str], mesh_shape: Optional[Union[MeshShape, HybridMeshShape]])` — [`L260`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L260) — Infer MoE outer batch size from mesh shape.
- `set_interpret_in_moe_config_recursively(cfg: ConfigBase, preferred_element_type: jnp.dtype=jnp.bfloat16)` — [`L2767`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L2767) — Recursively enables `interpret=True` for all `TransformerFeedForwardDropFreeMoE.Config`.
- `sigmoid()` — [`L144`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L144) — Creates a sigmoid scoring function as an alternative to softmax.

## Module values
- `ScoreFn` — [`L106`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L106)
- `TopKFn` — [`L105`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L105)
- `_RaggedA2AParams` — [`L2061`](../../../../../../raw/code/axlearn/axlearn/common/mixture_of_experts.py#L2061)


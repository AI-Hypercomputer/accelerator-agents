---
title: 'Module: src/maxtext/layers/moe.py'
type: catalog
provenance: extracted
module: src/maxtext/layers/moe.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.layers.moe`/
symbols:
  RoutedMoE.config: RoutedMoE#config.
  RoutedMoE.dense_matmul: RoutedMoE#dense_matmul().
  RoutedMoE.route: RoutedMoE#route().
  RoutedMoE.sparse_matmul_route_and_compute: RoutedMoE#sparse_matmul_route_and_compute().
  RoutedMoE.__call__: RoutedMoE#__call__().
  RoutedMoE.permute: RoutedMoE#permute().
  GateLogit.__call__: GateLogit#__call__().
  RoutedMoE.wi: RoutedMoE#wi.
  RoutedMoE.num_experts: RoutedMoE#num_experts.
  RoutedMoE.gate: RoutedMoE#gate.
  RoutedMoE._maybe_shard_with_logical: RoutedMoE#_maybe_shard_with_logical().
  RoutedMoE._logical_to_mesh_axes: RoutedMoE#_logical_to_mesh_axes().
  RoutedMoE.dtype: RoutedMoE#dtype.
  RoutedMoE.sparse_matmul: RoutedMoE#sparse_matmul().
  RoutedMoE.num_experts_per_tok: RoutedMoE#num_experts_per_tok.
  RoutedMoE.get_topk: RoutedMoE#get_topk().
  RoutedMoE._expert_parallelism_name: RoutedMoE#_expert_parallelism_name.
  RoutedMoE.get_einsum: RoutedMoE#get_einsum().
  RoutedMoE.wo_sparsity_module: RoutedMoE#wo_sparsity_module.
  RoutedMoE.unpermute: RoutedMoE#unpermute().
  RoutedMoE.unsort_output_and_ra2a: RoutedMoE#unsort_output_and_ra2a().
  RoutedAndSharedMoE.MoeBlock_0: RoutedAndSharedMoE#MoeBlock_0.
  RoutedMoE.wo_kernel_axes: RoutedMoE#wo_kernel_axes.
  RoutedMoE.get_all_to_all_params: RoutedMoE#get_all_to_all_params().
  RoutedMoE.per_expert_scale: RoutedMoE#per_expert_scale.
  RoutedMoE.wo_bias: RoutedMoE#wo_bias.
  RoutedMoE.gmm: RoutedMoE#gmm().
  RoutedMoE.transform_array: RoutedMoE#transform_array().
  RoutedAndSharedMoE.config: RoutedAndSharedMoE#config.
  RoutedAndSharedMoE.shared_experts: RoutedAndSharedMoE#shared_experts.
  RoutedMoE.moe_expert_input_dim: RoutedMoE#moe_expert_input_dim.
  RoutedMoE.mesh: RoutedMoE#mesh.
  RoutedMoE.reshape_and_update_weights: RoutedMoE#reshape_and_update_weights().
  RoutedMoE.get_expert_parallelism_size: RoutedMoE#get_expert_parallelism_size().
  GateLogit.kernel: GateLogit#kernel.
  RoutedMoE.get_routed_moe_shardings: RoutedMoE#get_routed_moe_shardings().
  RoutedMoE.generate_masks_subgroup: RoutedMoE#generate_masks_subgroup().
  RoutedMoE.fused_moe_matmul: RoutedMoE#fused_moe_matmul().
  RoutedMoE.wi_0_sparsity_module: RoutedMoE#wi_0_sparsity_module.
  RoutedMoE.tid2eid: RoutedMoE#tid2eid.
  RoutedMoE.wi_kernel_axes: RoutedMoE#wi_kernel_axes.
  RoutedMoE.wi_1_sparsity_module: RoutedMoE#wi_1_sparsity_module.
  RoutedMoE._maybe_shard_with_pspec: RoutedMoE#_maybe_shard_with_pspec().
  RoutedMoE.get_all_to_all_params.TransformStrategy: RoutedMoE#get_all_to_all_params().TransformStrategy#
  GateLogit._quant_dot_general_name: GateLogit#_quant_dot_general_name.
  GateLogit.matmul_precision: GateLogit#matmul_precision.
  RoutedMoE.rngs: RoutedMoE#rngs.
  RoutedMoE.jax_ragged_dot_gmm: RoutedMoE#jax_ragged_dot_gmm().
  RoutedMoE.gmm_up: RoutedMoE#gmm_up().
  RoutedMoE.generate_masks: RoutedMoE#generate_masks().
  RoutedMoE.is_hash_routing: RoutedMoE#is_hash_routing.
  RoutedMoE.activation_fn: RoutedMoE#activation_fn.
  RoutedMoE._maybe_shard_moe_dispatch: RoutedMoE#_maybe_shard_moe_dispatch().
  RoutedMoE.apply_ffn_activation: RoutedMoE#apply_ffn_activation().
  RoutedMoE._tensor_parallelism_name: RoutedMoE#_tensor_parallelism_name.
  RoutedMoE.wo: RoutedMoE#wo.
  RoutedMoE.deepseek_routing: RoutedMoE#deepseek_routing().
  RoutedMoE.get_gmm_for_local_experts: RoutedMoE#get_gmm_for_local_experts().
  RoutedMoE.quant: RoutedMoE#quant.
  RoutedMoE.get_tensor_parallelism_size: RoutedMoE#get_tensor_parallelism_size().
  RoutedMoE.maybe_all_gather_kernel_weight_in_expert_parallelism: RoutedMoE#maybe_all_gather_kernel_weight_in_expert_parallelism().
  RoutedMoE.get_tensor_transpose_parallelism_size: RoutedMoE#get_tensor_transpose_parallelism_size().
  GateLogit.bias: GateLogit#bias.
  RoutedMoE: RoutedMoE#
  RoutedMoE.intermediate_dim: RoutedMoE#intermediate_dim.
  RoutedMoE.weight_dtype: RoutedMoE#weight_dtype.
  RoutedMoE.wi_0_bias: RoutedMoE#wi_0_bias.
  RoutedMoE.wi_1_bias: RoutedMoE#wi_1_bias.
  RoutedMoE.should_update_load_balance: RoutedMoE#should_update_load_balance().
  RoutedMoE.local_permute: RoutedMoE#local_permute().
  RoutedMoE.load_balance_loss: RoutedMoE#load_balance_loss().
  _sort_activations: _sort_activations().
  RoutedMoE.get_active_sharding_axes: RoutedMoE#get_active_sharding_axes().
  RoutedMoE.kernel_init: RoutedMoE#kernel_init.
  RoutedMoE.expert_group_mask: RoutedMoE#expert_group_mask().
  RoutedMoE.get_wi_gmm_params: RoutedMoE#get_wi_gmm_params().
  RoutedMoE.get_wo_gmm_params: RoutedMoE#get_wo_gmm_params().
  RoutedMoE.aqt_einsum: RoutedMoE#aqt_einsum().
  GateLogit.out_features_shape: GateLogit#out_features_shape.
  GateLogit.dtype: GateLogit#dtype.
  RoutedAndSharedMoE.__call__: RoutedAndSharedMoE#__call__().
  RoutedMoE.wi_0: RoutedMoE#wi_0.
  RoutedMoE.wi_1: RoutedMoE#wi_1.
  RoutedMoE.get_context_autoregressive_parallelism_size: RoutedMoE#get_context_autoregressive_parallelism_size().
  RoutedMoE.get_context_partition_and_sub_seq: RoutedMoE#get_context_partition_and_sub_seq().
  RouteOutput.sorted_selected_experts: RouteOutput#sorted_selected_experts.
  _sort_activations_custom: _sort_activations_custom().
  GateLogit.axis: GateLogit#axis.
  _sort_activations_custom_fwd: _sort_activations_custom_fwd().
  _sort_activations_custom_bwd: _sort_activations_custom_bwd().
  GateLogit.quant_dot_general: GateLogit#quant_dot_general().
  RoutedMoE.deepseek_scale_weights: RoutedMoE#deepseek_scale_weights().
  RoutedMoE.get_tokamax_group_sizes: RoutedMoE#get_tokamax_group_sizes().
  RoutedMoE.get_quantization_dtypes: RoutedMoE#get_quantization_dtypes().
  RoutedMoE.retrieve_quantized_weight: RoutedMoE#retrieve_quantized_weight().
  RoutedAndSharedMoE.moe_expert_input_dim: RoutedAndSharedMoE#moe_expert_input_dim.
  RoutedAndSharedMoE.routed_moe: RoutedAndSharedMoE#routed_moe().
  RouteOutput.group_sizes: RouteOutput#group_sizes.
  RouteOutput.weights: RouteOutput#weights.
  RouteOutput.lb_loss: RouteOutput#lb_loss.
  RouteOutput.bias_updates: RouteOutput#bias_updates.
  GateLogit.in_features_shape: GateLogit#in_features_shape.
  RoutedMoE.get_all_to_all_params.TransformStrategy.INPUT_OFFSET: RoutedMoE#get_all_to_all_params().TransformStrategy#INPUT_OFFSET.
  RoutedMoE.get_all_to_all_params.TransformStrategy.SEND_SIZE: RoutedMoE#get_all_to_all_params().TransformStrategy#SEND_SIZE.
  RoutedMoE.get_all_to_all_params.TransformStrategy.OUTPUT_OFFSET: RoutedMoE#get_all_to_all_params().TransformStrategy#OUTPUT_OFFSET.
  RoutedMoE.get_all_to_all_params.TransformStrategy.RECV_SIZE: RoutedMoE#get_all_to_all_params().TransformStrategy#RECV_SIZE.
  get_gate_logit: get_gate_logit().
  get_routed_moe: get_routed_moe().
  get_routed_and_shared_moe: get_routed_and_shared_moe().
  DISPATCH: DISPATCH.
  COMBINE: COMBINE.
  RouteOutput.selected_experts: RouteOutput#selected_experts.
  calculate_load_balance_updates: calculate_load_balance_updates().
  GateLogit: GateLogit#
  GateLogit.weight_dtype: GateLogit#weight_dtype.
  GateLogit.kernel_axes: GateLogit#kernel_axes.
  GateLogit.use_bias: GateLogit#use_bias.
  GateLogit.score_func: GateLogit#score_func.
  GateLogit.quant: GateLogit#quant.
  RoutedMoE.get_ragged_buffer_size: RoutedMoE#get_ragged_buffer_size().
  RoutedMoE.extract_vma: RoutedMoE#extract_vma().
  RoutedAndSharedMoE.mesh: RoutedAndSharedMoE#mesh.
  RoutedAndSharedMoE.kernel_init: RoutedAndSharedMoE#kernel_init.
  RoutedAndSharedMoE.quant: RoutedAndSharedMoE#quant.
  RoutedAndSharedMoE.rngs: RoutedAndSharedMoE#rngs.
  set_xla_metadata: set_xla_metadata.
  RouteMetadata: RouteMetadata#
  RouteMetadata.expert_shard_id: RouteMetadata#expert_shard_id.
  RouteMetadata.local_sorted_indices: RouteMetadata#local_sorted_indices.
  RouteMetadata.reshaped_group_sizes: RouteMetadata#reshaped_group_sizes.
  RouteMetadata.all_shards_group_sizes: RouteMetadata#all_shards_group_sizes.
  RouteOutput: RouteOutput#
  RouteOutput.local_group_sizes: RouteOutput#local_group_sizes.
  get_batchsplit_init_kernel_axes: get_batchsplit_init_kernel_axes().
  random_routing: random_routing().
  Tid2EidVar: Tid2EidVar#
  GateLogit.model_name: GateLogit#model_name.
  GateLogit.mesh: GateLogit#mesh.
  GateLogit.kernel_init: GateLogit#kernel_init.
  GateLogit.shard_mode: GateLogit#shard_mode.
  RoutedMoE.kernel_axes: RoutedMoE#kernel_axes.
  RoutedMoE.transform_bias: RoutedMoE#transform_bias().
  RoutedMoE.is_batch_sharded_by_ep: RoutedMoE#is_batch_sharded_by_ep().
  RoutedMoE.explicitly_weight_ag: RoutedMoE#explicitly_weight_ag().
  RoutedMoE.maybe_aqt_partition: RoutedMoE#maybe_aqt_partition().
  RoutedAndSharedMoE: RoutedAndSharedMoE#
  RoutedAndSharedMoE.is_hash_routing: RoutedAndSharedMoE#is_hash_routing.
  GateLogit.__init__: GateLogit#__init__().
  RoutedMoE.__init__: RoutedMoE#__init__().
  RoutedAndSharedMoE.__init__: RoutedAndSharedMoE#__init__().
  RoutedAndSharedMoE.kernel_axes: RoutedAndSharedMoE#kernel_axes.
  RoutedAndSharedMoE.weight_dtype: RoutedAndSharedMoE#weight_dtype.
  RoutedAndSharedMoE.dtype: RoutedAndSharedMoE#dtype.
---
# Module: [`src/maxtext/layers/moe.py`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py)

## Classes
### `GateLogit`  ·  implements/extends Module
- def: [`src/maxtext/layers/moe.py:215`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L215)
- doc: A layer used to compute gate logits, allowing to return the pre bias values for DeepSeek routing.
- signature: `class GateLogit(nnx.Module):`
- members:
  - `__init__(self, in_features_shape: Union[Iterable[int], int], out_features_shape: Union[Iterable[int], int], model_name: str, mesh: Mesh, rngs: nnx.Rngs, axis: Union[Iterable[int], int] = -1, weight_dtype: ctypes.DType = jnp.float32, dtype: ctypes.DType = jnp.float32, kernel_init: NdInitializer = nd_dense_init(1, "fan_in", "truncated_normal"), kernel_axes: Tuple[Optional[str], ...] = (), use_bias: bool = False, score_func: str = "", quant: Optional[quantizations.AqtQuantization] = None, shard_mode: ShardMode = ShardMode.AUTO, matmul_precision: str = "default")` — [`L218`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L218) — Initializes the GateLogit module.
  - `quant_dot_general(self)` — [`L309`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L309)
  - `axis` — [`L259`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L259)
  - `bias` — [`L290`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L290)
  - `dtype` — [`L261`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L261)
  - `in_features_shape` — [`L255`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L255)
  - `kernel` — [`L276`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L276)
  - `kernel_axes` — [`L263`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L263)
  - `kernel_init` — [`L262`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L262)
  - `matmul_precision` — [`L268`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L268)
  - `mesh` — [`L258`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L258)
  - `model_name` — [`L257`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L257)
  - `out_features_shape` — [`L256`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L256)
  - `quant` — [`L266`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L266)
  - `score_func` — [`L265`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L265)
  - `shard_mode` — [`L267`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L267)
  - `use_bias` — [`L264`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L264)
  - `weight_dtype` — [`L260`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L260)
- protocol/private: `__call__`[`L314`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L314), `_quant_dot_general_name`[`L301`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L301)
- used by: [`gate`](moe.md#RoutedMoE.gate), [`get_gate_logit`](moe.md#get_gate_logit)

### `RouteMetadata`
- def: [`src/maxtext/layers/moe.py:61`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L61)
- doc: EP communication state needed to undo the forward all-to-all after expert computation.
- signature: `class RouteMetadata:`
- members:
  - `all_shards_group_sizes` — [`L72`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L72)
  - `expert_shard_id` — [`L65`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L65)
  - `local_sorted_indices` — [`L67`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L67)
  - `reshaped_group_sizes` — [`L69`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L69)
- used by: [`route`](moe.md#RoutedMoE.route)

### `RouteOutput`
- def: [`src/maxtext/layers/moe.py:76`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L76)
- doc: Holds state of routing output
- signature: `class RouteOutput:`
- members:
  - `bias_updates` — [`L90`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L90)
  - `group_sizes` — [`L80`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L80)
  - `lb_loss` — [`L88`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L88)
  - `local_group_sizes` — [`L92`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L92)
  - `selected_experts` — [`L82`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L82)
  - `sorted_selected_experts` — [`L84`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L84)
  - `weights` — [`L86`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L86)
- used by: [`route`](moe.md#RoutedMoE.route), [`sparse_matmul_route_and_compute`](moe.md#RoutedMoE.sparse_matmul_route_and_compute)

### `RoutedAndSharedMoE`  ·  implements/extends Module
- def: [`src/maxtext/layers/moe.py:2669`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L2669)
- doc: Implements a block which combines shared and routed experts.
- signature: `class RoutedAndSharedMoE(nnx.Module):`
- members:
  - `__call__(self, inputs: jax.Array, original_inputs: jax.Array | None = None, gate_inputs: jax.Array | None = None, intermediate_sharding: NamedSharding | None = None, out_sharding: NamedSharding | None = None, input_ids: jax.Array | None = None)` — [`L2746`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L2746) — Executes both the routed experts and the shared expert block.
  - `__init__(self, config: ctypes.Config, mesh: jax.sharding.Mesh, kernel_init: NdInitializer, kernel_axes: Tuple[Optional[str], ...], rngs: nnx.Rngs, weight_dtype: ctypes.DType = jnp.float32, dtype: ctypes.DType = jnp.float32, quant: Optional[quantizations.AqtQuantization] = None, is_hash_routing: bool = False)` — [`L2672`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L2672) — Initializes the RoutedAndSharedMoE module.
  - `routed_moe(self)` — [`L2743`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L2743)
  - `MoeBlock_0` — [`L2712`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L2712)
  - `config` — [`L2697`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L2697)
  - `dtype` — [`L2702`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L2702)
  - `is_hash_routing` — [`L2705`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L2705)
  - `kernel_axes` — [`L2700`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L2700)
  - `kernel_init` — [`L2699`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L2699)
  - `mesh` — [`L2698`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L2698)
  - `moe_expert_input_dim` — [`L2706`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L2706)
  - `quant` — [`L2703`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L2703)
  - `rngs` — [`L2704`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L2704)
  - `shared_experts` — [`L2728`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L2728)
  - `weight_dtype` — [`L2701`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L2701)
- uses (calls/refs, reference-scoped): [`RoutedMoE`](moe.md#RoutedMoE)
- used by: [`get_routed_and_shared_moe`](moe.md#get_routed_and_shared_moe)

### `RoutedMoE`  ·  implements/extends Module
- def: [`src/maxtext/layers/moe.py:356`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L356)
- doc: Implements a routed MoE block.
- signature: `class RoutedMoE(nnx.Module):`
- members:
  - `__call__(self, inputs: jax.Array, input_ids: jax.Array | None = None, gate_inputs: jax.Array | None = None, out_sharding: NamedSharding | None = None)` — [`L2581`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L2581) — Executes the routed MoE block. — documented in [maxtext-layers-moe](../../../../concepts/maxtext-layers-moe.md)
  - `__init__(self, config: ctypes.Config, num_experts: int, num_experts_per_tok: int, mesh: jax.sharding.Mesh, kernel_init: attentions.NdInitializer, kernel_axes: Tuple[Optional[str], ...], rngs: nnx.Rngs, intermediate_dim: int = 2048, weight_dtype: ctypes.DType = jnp.float32, dtype: ctypes.DType = jnp.float32, quant: Optional[quantizations.AqtQuantization] = None, is_hash_routing: bool = False)` — [`L359`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L359) — Initializes the RoutedMoE module.
  - `_maybe_shard_moe_dispatch(self, inputs, logical_axis, peel_expert)` — [`L631`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L631) — Shard a MoE dispatch/MLP activation. When `peel_expert` is set, drop the 'expert' — documented in [maxtext-layers-moe](../../../../concepts/maxtext-layers-moe.md)
  - `apply_ffn_activation(self, layer_w0, layer_w1)` — [`L787`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L787) — Applies FFN activation function. — documented in [maxtext-layers-moe](../../../../concepts/maxtext-layers-moe.md)
  - `aqt_einsum(*args, **kwargs)` — [`L2165`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L2165)
  - `deepseek_routing(self, gate_logits: jax.Array, pre_bias_logits: jax.Array)` — [`L759`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L759) — DeepSeek routing logit.
  - `deepseek_scale_weights(self, weights)` — [`L713`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L713) — Scales weights according to DeepSeek's v3 reference implementation.
  - `dense_matmul(self, inputs, gate_logits, pre_bias_logits, w0_kernel, w1_kernel, wo_kernel, w0_bias, w1_bias, wo_bias, input_ids=None)` — [`L2190`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L2190) — Dense matrix multiplication. — documented in [maxtext-layers-moe](../../../../concepts/maxtext-layers-moe.md)
  - `expert_group_mask(self, gate_logits: jax.Array)` — [`L721`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L721) — Returns a mask that selects only the top-k groups of experts.
  - `explicitly_weight_ag(shard_exp_on_fsdp)` — [`L1388`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L1388)
  - `extract_vma(tensor)` — [`L1304`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L1304)
  - `fused_moe_matmul(self, inputs, gate_logits, wo_kernel, w0_kernel=None, w1_kernel=None, fused_kernel=None)` — [`L2488`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L2488) — Fused MoE via tpu_inference fused_moe_func (vllm_rpa path only). — documented in [maxtext-layers-moe](../../../../concepts/maxtext-layers-moe.md)
  - `generate_masks(self, top_k_indices, softmax_probs)` — [`L2062`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L2062) — Generate masks. — documented in [maxtext-layers-moe](../../../../concepts/maxtext-layers-moe.md)
  - `generate_masks_subgroup(self, top_k_indices, softmax_probs)` — [`L1969`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L1969) — Subgroup mask generation for inference only. — documented in [maxtext-layers-moe](../../../../concepts/maxtext-layers-moe.md)
  - `get_active_sharding_axes(pspec_dim_axes, tensor_dim_index)` — [`L1615`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L1615)
  - `get_all_to_all_params(all_shards_group_sizes, shard_id, num_expert_parallelism, is_batch_sharded=True)` — [`L1096`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L1096) — Generates input offsets, send sizes, output offsets, and receive sizes used for ragged_all_to_all. — documented in [maxtext-layers-moe](../../../../concepts/maxtext-layers-moe.md)
  - `get_context_autoregressive_parallelism_size(self)` — [`L660`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L660)
  - `get_context_partition_and_sub_seq(self, seq_len)` — [`L1962`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L1962)
  - `get_einsum(self, rhs_mesh_axes: Tuple[Optional[str], ...] = (), einsum_name: str | None = None)` — [`L2147`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L2147) — Get the Einstein summation.
  - `get_expert_parallelism_size(self)` — [`L643`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L643) — documented in [maxtext-layers-moe](../../../../concepts/maxtext-layers-moe.md)
  - `get_gmm_for_local_experts(x, routing, route_metadata)` — [`L1706`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L1706) — Return a partial GMM function with preconfigured routing params. — documented in [maxtext-layers-moe](../../../../concepts/maxtext-layers-moe.md)
  - `get_quantization_dtypes()` — [`L1295`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L1295)
  - `get_ragged_buffer_size(local_batch, ep_degree, global_experts, top_k, ragged_buffer_factor)` — [`L1190`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L1190) — Calculates the token batch size of the ragged buffer.
  - `get_routed_moe_shardings(is_batch_sharded_by_expert, has_input_ids)` — [`L1404`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L1404) — documented in [maxtext-layers-moe](../../../../concepts/maxtext-layers-moe.md)
  - `get_tensor_parallelism_size(self)` — [`L649`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L649) — documented in [maxtext-layers-moe](../../../../concepts/maxtext-layers-moe.md)
  - `get_tensor_transpose_parallelism_size(self)` — [`L657`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L657)
  - `get_tokamax_group_sizes(group_sizes, inputs, kernel)` — [`L1283`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L1283)
  - `get_topk(self, gate_logits, pre_bias_logits, rngs=None, input_ids=None)` — [`L671`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L671) — get topk. — documented in [maxtext-layers-moe](../../../../concepts/maxtext-layers-moe.md)
  - `get_wi_gmm_params()` — [`L1625`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L1625)
  - `get_wo_gmm_params()` — [`L1644`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L1644)
  - `gmm(inputs, kernel, tiling, group_sizes, expert_assignments, weight_gather_axes, group_offset)` — [`L1303`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L1303) — documented in [maxtext-layers-moe](../../../../concepts/maxtext-layers-moe.md)
  - `gmm_up(x, w0, w1, w0_bias, w1_bias, gmm_fn, weight_gather)` — [`L1663`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L1663) — Run the two up-projections (gate + up) and apply the FFN activation. — documented in [maxtext-layers-moe](../../../../concepts/maxtext-layers-moe.md)
  - `is_batch_sharded_by_ep(input_activation)` — [`L1383`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L1383)
  - `jax_ragged_dot_gmm(inputs, kernel, tiling, group_sizes, expert_assignments, padding_amount)` — [`L1240`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L1240) — Execute jax.lax.ragged_dot, with potential quantization
  - `load_balance_loss(self, top_k_indices, logits)` — [`L2136`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L2136) — Compute the load balance loss. — documented in [maxtext-layers-moe](../../../../concepts/maxtext-layers-moe.md)
  - `local_permute(inputs, global_group_sizes, local_expert_size, shard_index, is_offset=False, global_sorted_experts=None, use_custom_sort_vjp=True, use_ragged_sort=False)` — [`L994`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L994) — Permutes tokens locally within an expert shard. — documented in [maxtext-layers-moe](../../../../concepts/maxtext-layers-moe.md)
  - `maybe_all_gather_kernel_weight_in_expert_parallelism(self, kernel: jax.Array, kernel_axes: Tuple[Optional[str], ...])` — [`L2177`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L2177) — All-gather kernel weight in expert parallelism if needed. — documented in [maxtext-layers-moe](../../../../concepts/maxtext-layers-moe.md)
  - `maybe_aqt_partition(w0_kernel, w0_pspec, w1_kernel, w1_pspec, wo_kernel, wo_pspec)` — [`L1395`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L1395)
  - `permute(self, inputs, gate_logits, pre_bias_logits, use_custom_sort_vjp=True, rngs=None, roll_to_expert_id=None, input_ids=None)` — [`L807`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L807) — Permute tokens to group by expert to fit gmm call. — documented in [maxtext-layers-moe](../../../../concepts/maxtext-layers-moe.md)
  - `reshape_and_update_weights(self, weights, indices)` — [`L1944`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L1944) — reshape and update weights. — documented in [maxtext-layers-moe](../../../../concepts/maxtext-layers-moe.md)
  - `retrieve_quantized_weight(self, inputs, gate_logits, pre_bias_logits, w0_kernel, w1_kernel, wo_kernel, w0_bias, w1_bias, wo_bias)` — [`L2552`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L2552) — Retrieve quantized weights.
  - `route(x, logits, pre_bias_logits, rngs, input_ids=None)` — [`L1490`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L1490) — Performs both across device and within device token routing/sorting — documented in [maxtext-layers-moe](../../../../concepts/maxtext-layers-moe.md)
  - `should_update_load_balance(self)` — [`L663`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L663) — Determines if loss-free load balancing updates should be applied. — documented in [maxtext-layers-moe](../../../../concepts/maxtext-layers-moe.md)
  - `sparse_matmul(self, inputs, gate_logits, pre_bias_logits, w0_kernel, w1_kernel, wo_kernel, w0_bias, w1_bias, wo_bias, input_ids=None)` — [`L1225`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L1225) — Perform sparse matrix multiplication of inputs and Experts. — documented in [maxtext-layers-moe](../../../../concepts/maxtext-layers-moe.md)
  - `sparse_matmul_route_and_compute(x, logits, pre_bias_logits, w0, w1, wo, w0_bias, w1_bias, wo_bias, sharded_input_ids, rngs)` — [`L1807`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L1807) — documented in [maxtext-layers-moe](../../../../concepts/maxtext-layers-moe.md)
  - `transform_array(input_array, shard_id, strategy, is_batch_sharded)` — [`L1110`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L1110) — Transforms the input array based on the specified strategy. — documented in [maxtext-layers-moe](../../../../concepts/maxtext-layers-moe.md)
  - `transform_bias(self, experts_index, *biases)` — [`L1185`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L1185) — Selects bias values for a variable number of bias tensors based on chosen experts.
  - `unpermute(self, intermediate, sorted_selected_experts, weights, batch_size, sequence_length, use_custom_sort_vjp=True, group_sizes=None)` — [`L933`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L933) — Unpermute tokens to original order and combine weights. — documented in [maxtext-layers-moe](../../../../concepts/maxtext-layers-moe.md)
  - `unsort_output_and_ra2a(intermediate_output, routing, route_metadata, output_shape, is_batch_sharded_by_expert)` — [`L1729`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L1729) — Unsort tokens and return them to original shards using ragged all-to-all. — documented in [maxtext-layers-moe](../../../../concepts/maxtext-layers-moe.md)
  - `activation_fn` — [`L498`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L498)
  - `config` — [`L390`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L390)
  - `dtype` — [`L398`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L398) — documented in [maxtext-layers-moe](../../../../concepts/maxtext-layers-moe.md)
  - `gate` — [`L447`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L447) — documented in [maxtext-layers-moe](../../../../concepts/maxtext-layers-moe.md)
  - `intermediate_dim` — [`L396`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L396)
  - `is_hash_routing` — [`L401`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L401) — documented in [maxtext-layers-moe](../../../../concepts/maxtext-layers-moe.md)
  - `kernel_axes` — [`L395`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L395)
  - `kernel_init` — [`L394`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L394)
  - `mesh` — [`L393`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L393)
  - `moe_expert_input_dim` — [`L416`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L416)
  - `num_experts` — [`L391`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L391) — documented in [maxtext-layers-moe](../../../../concepts/maxtext-layers-moe.md)
  - `num_experts_per_tok` — [`L392`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L392) — documented in [maxtext-layers-moe](../../../../concepts/maxtext-layers-moe.md)
  - `per_expert_scale` — [`L592`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L592) — documented in [maxtext-layers-moe](../../../../concepts/maxtext-layers-moe.md)
  - `quant` — [`L399`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L399)
  - `rngs` — [`L400`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L400)
  - `tid2eid` — [`L409`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L409) — documented in [maxtext-layers-moe](../../../../concepts/maxtext-layers-moe.md)
  - `weight_dtype` — [`L397`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L397)
  - `wi` — [`L517`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L517) — documented in [maxtext-layers-moe](../../../../concepts/maxtext-layers-moe.md)
  - `wi_0` — [`L513`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L513)
  - `wi_0_bias` — [`L574`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L574)
  - `wi_0_sparsity_module` — [`L477`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L477) — documented in [maxtext-layers-moe](../../../../concepts/maxtext-layers-moe.md)
  - `wi_1` — [`L514`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L514)
  - `wi_1_bias` — [`L578`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L578)
  - `wi_1_sparsity_module` — [`L482`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L482) — documented in [maxtext-layers-moe](../../../../concepts/maxtext-layers-moe.md)
  - `wi_kernel_axes` — [`L422`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L422) — documented in [maxtext-layers-moe](../../../../concepts/maxtext-layers-moe.md)
  - `wo` — [`L515`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L515) — documented in [maxtext-layers-moe](../../../../concepts/maxtext-layers-moe.md)
  - `wo_bias` — [`L582`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L582)
  - `wo_kernel_axes` — [`L423`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L423) — documented in [maxtext-layers-moe](../../../../concepts/maxtext-layers-moe.md)
  - `wo_sparsity_module` — [`L487`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L487) — documented in [maxtext-layers-moe](../../../../concepts/maxtext-layers-moe.md)
- protocol/private: `_expert_parallelism_name`[`L440`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L440), `_logical_to_mesh_axes`[`L617`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L617), `_maybe_shard_with_logical`[`L607`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L607), `_maybe_shard_with_pspec`[`L621`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L621), `_tensor_parallelism_name`[`L435`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L435)
- uses (calls/refs, reference-scoped): [`TransformStrategy`](moe.md#RoutedMoE.get_all_to_all_params.TransformStrategy), [`a2a_ragged_sort`](../kernels/ragged/ragged_sort.md#a2a_ragged_sort), [`a2a_ragged_unsort`](../kernels/ragged/ragged_sort.md#a2a_ragged_unsort), [`ring_ragged_sort`](../kernels/ragged/ragged_sort.md#ring_ragged_sort), [`ring_ragged_unsort`](../kernels/ragged/ragged_sort.md#ring_ragged_unsort), [`_sort_activations`](moe.md#_sort_activations), [`sorted_selected_experts`](moe.md#RouteOutput.sorted_selected_experts), [`INPUT_OFFSET`](moe.md#RoutedMoE.get_all_to_all_params.TransformStrategy.INPUT_OFFSET), [`OUTPUT_OFFSET`](moe.md#RoutedMoE.get_all_to_all_params.TransformStrategy.OUTPUT_OFFSET), [`RECV_SIZE`](moe.md#RoutedMoE.get_all_to_all_params.TransformStrategy.RECV_SIZE), [`SEND_SIZE`](moe.md#RoutedMoE.get_all_to_all_params.TransformStrategy.SEND_SIZE), [`bias_updates`](moe.md#RouteOutput.bias_updates), [`group_sizes`](moe.md#RouteOutput.group_sizes), [`lb_loss`](moe.md#RouteOutput.lb_loss), [`weights`](moe.md#RouteOutput.weights), [`GateLogit`](moe.md#GateLogit), [`COMBINE`](moe.md#COMBINE), [`DISPATCH`](moe.md#DISPATCH), [`calculate_load_balance_updates`](moe.md#calculate_load_balance_updates), [`selected_experts`](moe.md#RouteOutput.selected_experts), [`RouteMetadata`](moe.md#RouteMetadata), [`RouteOutput`](moe.md#RouteOutput), [`Tid2EidVar`](moe.md#Tid2EidVar), [`all_shards_group_sizes`](moe.md#RouteMetadata.all_shards_group_sizes), [`expert_shard_id`](moe.md#RouteMetadata.expert_shard_id), [`get_batchsplit_init_kernel_axes`](moe.md#get_batchsplit_init_kernel_axes), [`local_group_sizes`](moe.md#RouteOutput.local_group_sizes), [`local_sorted_indices`](moe.md#RouteMetadata.local_sorted_indices), [`random_routing`](moe.md#random_routing), [`reshaped_group_sizes`](moe.md#RouteMetadata.reshaped_group_sizes), [`set_xla_metadata`](moe.md#set_xla_metadata)
- used by: [`MoeBlock_0`](moe.md#RoutedAndSharedMoE.MoeBlock_0), [`get_routed_moe`](moe.md#get_routed_moe)

### `Tid2EidVar`  ·  implements/extends Variable
- def: [`src/maxtext/layers/moe.py:211`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L211)
- doc: Custom variable to hold tid2eid without trainable param overhead.
- signature: `class Tid2EidVar(nnx.Variable):`
- used by: [`tid2eid`](moe.md#RoutedMoE.tid2eid)

### `TransformStrategy`  ·  implements/extends Enum
- def: [`src/maxtext/layers/moe.py:1104`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L1104) — documented in [maxtext-layers-moe](../../../../concepts/maxtext-layers-moe.md)
- signature: `class TransformStrategy(enum.Enum):`
- members:
  - `INPUT_OFFSET` — [`L1105`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L1105)
  - `OUTPUT_OFFSET` — [`L1107`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L1107)
  - `RECV_SIZE` — [`L1108`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L1108)
  - `SEND_SIZE` — [`L1106`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L1106)
- used by: [`get_all_to_all_params`](moe.md#RoutedMoE.get_all_to_all_params), [`transform_array`](moe.md#RoutedMoE.transform_array)

## Functions
- `_sort_activations(inputs: jax.Array, sort_indices: jax.Array, use_custom_vjp: bool)` — [`L95`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L95) — Sort activations by `sort_indices`.
- `_sort_activations_custom(inputs: jax.Array, sort_indices: jax.Array)` — [`L127`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L127) — Sort functions with custom vjp.
- `_sort_activations_custom_bwd(residuals: jax.Array, grads: jax.Array)` — [`L137`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L137) — Backward pass of the custom vjp for `_sort_activations()`.
- `_sort_activations_custom_fwd(inputs: jax.Array, sort_indices: jax.Array)` — [`L132`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L132) — Forward pass of the custom vjp for `_sort_activations()`.
- `calculate_load_balance_updates(top_k_indices, num_experts, rate)` — [`L187`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L187) — Computes a bias adjustment update based on expert load.
- `get_batchsplit_init_kernel_axes()` — [`L146`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L146)
- `get_gate_logit(inputs_shape: tuple[int, ...], out_features_shape: Union[Iterable[int], int], model_name: str, axis: Union[Iterable[int], int] = -1, weight_dtype: ctypes.DType = jnp.float32, dtype: ctypes.DType = jnp.float32, kernel_init: NdInitializer = nd_dense_init(1, "fan_in", "truncated_normal"), kernel_axes: Tuple[Optional[str], ...] = (), use_bias: bool = False, score_func: str = "", quant: Optional[quantizations.AqtQuantization] = None, matmul_precision: str = "default", name: Optional[str] = None)` — [`L2777`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L2777) — Creates a GateLogit Linen module.
- `get_routed_and_shared_moe(config: ctypes.Config, mesh: jax.sharding.Mesh, kernel_init: NdInitializer, kernel_axes: Tuple[Optional[str], ...], weight_dtype: ctypes.DType = jnp.float32, dtype: ctypes.DType = jnp.float32, quant: Optional[quantizations.AqtQuantization] = None, name: Optional[str] = None, is_hash_routing: bool = False)` — [`L2852`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L2852) — Creates a RoutedAndSharedMoE Linen module.
- `get_routed_moe(config: ctypes.Config, num_experts: int, num_experts_per_tok: int, mesh: jax.sharding.Mesh, kernel_init: NdInitializer, kernel_axes: Tuple[Optional[str], ...], intermediate_dim: int = 2048, weight_dtype: ctypes.DType = jnp.float32, dtype: ctypes.DType = jnp.float32, quant: Optional[quantizations.AqtQuantization] = None, name: Optional[str] = None)` — [`L2818`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L2818) — Creates a RoutedMoE Linen module.
- `random_routing(rng_key, gate_logits, num_experts_per_tok)` — [`L153`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L153) — Performs random routing of tokens to experts.

## Module values
- `COMBINE` — [`L57`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L57)
- `DISPATCH` — [`L56`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L56)
- `set_xla_metadata` — [`L53`](../../../../../../../raw/code/maxtext/src/maxtext/layers/moe.py#L53)


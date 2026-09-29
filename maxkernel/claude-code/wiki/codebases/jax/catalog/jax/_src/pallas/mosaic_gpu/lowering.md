---
title: 'Module: jax/_src/pallas/mosaic_gpu/lowering.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/mosaic_gpu/lowering.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.mosaic_gpu.lowering`/
symbols:
  register_lowering_rule: register_lowering_rule().
  LoweringRuleContext.module_ctx: LoweringRuleContext#module_ctx.
  LoweringRuleContext.avals_in: LoweringRuleContext#avals_in.
  _handle_transforms: _handle_transforms().
  ShapedAbstractValue.dtype: ShapedAbstractValue#dtype.
  LoweringRuleContext: LoweringRuleContext#
  lower_jaxpr_to_module: lower_jaxpr_to_module().
  lower_jaxpr_to_mosaic_gpu: lower_jaxpr_to_mosaic_gpu().
  _run_scoped_lowering_rule: _run_scoped_lowering_rule().
  lower_jaxpr_to_module.body: lower_jaxpr_to_module().body().
  _swap_lowering_rule: _swap_lowering_rule().
  _get_lowering_rule: _get_lowering_rule().
  lower_pipelined_jaxpr_to_module: lower_pipelined_jaxpr_to_module().
  _run_scoped_resource_estimator: _run_scoped_resource_estimator().
  ShapedAbstractValue.shape: ShapedAbstractValue#shape.
  _broadcast_in_dim_lowering_rule: _broadcast_in_dim_lowering_rule().
  _ensure_ir_value: _ensure_ir_value().
  ModuleContext.lowering_semantics: ModuleContext#lowering_semantics.
  _axis_index_rule: _axis_index_rule().
  _run_state_lowering_rule: _run_state_lowering_rule().
  _core_map_lowering_rule: _core_map_lowering_rule().
  _mpmd_map_lowering_rule: _mpmd_map_lowering_rule().
  semantics: semantics.
  f_pred: f_pred.
  _debug_print_lowering_rule: _debug_print_lowering_rule().
  _cond_lowering_rule: _cond_lowering_rule().
  _jaxpr_call_lowering_rule: _jaxpr_call_lowering_rule().
  _select_n_lowering_rule: _select_n_lowering_rule().
  _convert_element_type_lowering_rule_wg: _convert_element_type_lowering_rule_wg().
  _while_lowering_rule: _while_lowering_rule().
  _ensure_fa: _ensure_fa().
  _commute_transform: _commute_transform().
  _scan_lowering_rule: _scan_lowering_rule().
  lower_pipelined_jaxpr_to_module.ref_for_aval: lower_pipelined_jaxpr_to_module().ref_for_aval().
  ModuleContext.scratch_view: ModuleContext#scratch_view().
  _estimate_resources: _estimate_resources().
  _reduce_lowering_rule: _reduce_lowering_rule().
  _swap_lowering_rule_wg: _swap_lowering_rule_wg().
  _convert_element_type_lowering_rule: _convert_element_type_lowering_rule().
  _bitcast_convert_type_lowering_rule: _bitcast_convert_type_lowering_rule().
  _binary_op_lowering_rule: _binary_op_lowering_rule().
  ModuleContext.alloc_tmem: ModuleContext#alloc_tmem().
  LoweringRuleContext.launch_ctx: LoweringRuleContext#launch_ctx.
  _get_lowering_rule_wg: _get_lowering_rule_wg().
  _binary_op_lowering_rule_wg: _binary_op_lowering_rule_wg().
  _comparison_lowering_rule_wg: _comparison_lowering_rule_wg().
  _semaphore_read_lowering_rule: _semaphore_read_lowering_rule().
  _reduce_lowering_rule_wg: _reduce_lowering_rule_wg().
  ModuleContext.primitive_semantics: ModuleContext#primitive_semantics.
  LoweringRuleContext.avals_out: LoweringRuleContext#avals_out.
  _square_lowering_rule: _square_lowering_rule().
  _rsqrt_lowering_rule: _rsqrt_lowering_rule().
  _tanh_lowering_rule: _tanh_lowering_rule().
  _exp_lowering_rule: _exp_lowering_rule().
  _exp2_lowering_rule: _exp2_lowering_rule().
  _sin_lowering_rule: _sin_lowering_rule().
  _cos_lowering_rule: _cos_lowering_rule().
  _log_lowering_rule: _log_lowering_rule().
  _get_cluster_barrier: _get_cluster_barrier().
  ModuleContext.single_lane_predicate: ModuleContext#single_lane_predicate().
  _block_spec_from_block_mapping: _block_spec_from_block_mapping().
  _ndindexer_indices: _ndindexer_indices().
  rule: rule.
  _clz_lowering_rule: _clz_lowering_rule().
  _round_lowering_rule: _round_lowering_rule().
  _round_lowering_rule_wg: _round_lowering_rule_wg().
  _reduce_max_lowering_rule_wg: _reduce_max_lowering_rule_wg().
  _reduce_min_lowering_rule_wg: _reduce_min_lowering_rule_wg().
  _get_global_lowering_rule: _get_global_lowering_rule().
  _bubble_up_transforms_for_lowering: _bubble_up_transforms_for_lowering().
  _lower_jaxpr_to_for_loop: _lower_jaxpr_to_for_loop().
  _lower_fun.f_lowered: _lower_fun().f_lowered().
  impl: impl.
  Resources: Resources#
  _eval_index_map: _eval_index_map().
  _integer_pow_lowering_rule: _integer_pow_lowering_rule().
  _abs_lowering_rule_wg: _abs_lowering_rule_wg().
  _squeeze_lowering_rule: _squeeze_lowering_rule().
  _squeeze_lowering_rule_wg: _squeeze_lowering_rule_wg().
  _reduce_prod_lowering_rule_wg: _reduce_prod_lowering_rule_wg().
  _check_lowering_rule: _check_lowering_rule().
  lower_jaxpr_to_mosaic_gpu.write_env: lower_jaxpr_to_mosaic_gpu().write_env().
  _check_block_mappings: _check_block_mappings().
  _as_index: _as_index().
  Resources.or_: Resources#or_().
  _check_block_mappings.err_details: _check_block_mappings().err_details().
  merge_indexers: merge_indexers().
  mosaic_lowering_rules: mosaic_lowering_rules.
  _pjit_lowering_rule: _pjit_lowering_rule().
  _iota_lowering: _iota_lowering().
  _wrap_in_custom_primitive_if_wg: _wrap_in_custom_primitive_if_wg().
  _binary_boolean_op_lowering_rule_wg: _binary_boolean_op_lowering_rule_wg().
  _copysign_lowering_rule: _copysign_lowering_rule().
  _while_resource_estimator: _while_resource_estimator().
  _reduce_resource_estimator: _reduce_resource_estimator().
  _bubble_up_transform: _bubble_up_transform().
  _concatenate_lowering_rule: _concatenate_lowering_rule().
  _broadcast_in_dim_lowering_rule_wg: _broadcast_in_dim_lowering_rule_wg().
  _clamp_lowering_rule: _clamp_lowering_rule().
  _abs_lowering_rule: _abs_lowering_rule().
  _erf_lowering_rule: _erf_lowering_rule().
  _atan2_lowering_rule: _atan2_lowering_rule().
  _reshape_lowering_rule: _reshape_lowering_rule().
  _reshape_lowering_rule_wg: _reshape_lowering_rule_wg().
  _get_global_resource_estimator: _get_global_resource_estimator().
  _optimization_barrier_lowering: _optimization_barrier_lowering().
  _layout_cast_lowering_wg: _layout_cast_lowering_wg().
  partial: partial.
  _lower_jaxpr_to_for_loop.as_values: _lower_jaxpr_to_for_loop().as_values().
  LoweringRuleContext.estimator_ctx: LoweringRuleContext#estimator_ctx().
  _lower_while_via_fori: _lower_while_via_fori().
  ModuleContext.axis_names: ModuleContext#axis_names.
  _bcast: _bcast().
  _program_id_lowering_rule: _program_id_lowering_rule().
  _concatenate_lowering_rule_wg: _concatenate_lowering_rule_wg().
  _sign_lowering_rule: _sign_lowering_rule().
  _erf_lowering_rule_wg: _erf_lowering_rule_wg().
  _atan2_lowering_rule_wg: _atan2_lowering_rule_wg().
  _optimization_barrier_lowering_wg: _optimization_barrier_lowering_wg().
  _delay_lowering: _delay_lowering().
  _get_barrier: _get_barrier().
  block_id_to_grid_id: block_id_to_grid_id().
  _register_resource_estimator: _register_resource_estimator().
  _bcast_wg: _bcast_wg().
  lower_pipelined_jaxpr_to_module.pipeline_fn: lower_pipelined_jaxpr_to_module().pipeline_fn().
  _sign_lowering_rule.sign: _sign_lowering_rule().sign().
  f_impl: f_impl.
  _ir_constant: _ir_constant().
  _cond_resource_estimator: _cond_resource_estimator().
  _scan_resource_estimator: _scan_resource_estimator().
  _pjit_resource_estimator: _pjit_resource_estimator().
  _multiple_of_wg_lowering_rule: _multiple_of_wg_lowering_rule().
  _slice_lowering_rule_wg: _slice_lowering_rule_wg().
  ModuleContext.reserve_barrier: ModuleContext#reserve_barrier().
  ModuleContext.reserve_semaphores: ModuleContext#reserve_semaphores().
  ShapedAbstractValue: ShapedAbstractValue#
  ResourceEstimatorContext: ResourceEstimatorContext#
  Resources.scoped_gmem_semaphores: Resources#scoped_gmem_semaphores.
  LoweringRuleContext.out_layout_hint: LoweringRuleContext#out_layout_hint.
  _lower_jaxpr_to_for_loop.loop: _lower_jaxpr_to_for_loop().loop().
  Resources.__add__: Resources#__add__().
  _core_map_resource_estimator: _core_map_resource_estimator().
  _mpmd_map_resource_estimator: _mpmd_map_resource_estimator().
  _run_state_resource_estimator: _run_state_resource_estimator().
  _reduce_sum_lowering_rule_wg: _reduce_sum_lowering_rule_wg().
  _iota_lowering_wg: _iota_lowering_wg().
  _reinterpret_cast: _reinterpret_cast().
  ModuleContext.approx_math: ModuleContext#approx_math.
  _device_id_to_logical: _device_id_to_logical().
  ModuleContext.auto_barriers: ModuleContext#auto_barriers.
  merge_indexers._ensure_idx_fa: merge_indexers()._ensure_idx_fa().
  Resources.__post_init__: Resources#__post_init__().
  _jaxpr_call_resource_estimator: _jaxpr_call_resource_estimator().
  _multiple_of_lane_lowering_rule: _multiple_of_lane_lowering_rule().
  _slice_lowering_rule: _slice_lowering_rule().
  _layout_cast_lowering: _layout_cast_lowering().
  _reshard_lowering_rule: _reshard_lowering_rule().
  ResourceEstimatorContext.arrival_multiplier: ResourceEstimatorContext#arrival_multiplier().
  CmpIPred: CmpIPred.
  _resolve_cluster_axis: _resolve_cluster_axis().
  _AxisNames: _AxisNames#
  Resources.barrier_counts: Resources#barrier_counts.
  _is_block_local_scope: _is_block_local_scope().
  ShapedAbstractValue.ndim: ShapedAbstractValue#ndim().
  Resources.smem_scratch_bytes: Resources#smem_scratch_bytes.
  Resources.tmem_collective_scratch_cols: Resources#tmem_collective_scratch_cols.
  _AxisNames.reverse: _AxisNames#reverse().
  ModuleContext.next_smem_allocation_id: ModuleContext#next_smem_allocation_id().
  ModuleContext.next_tmem_allocation_id: ModuleContext#next_tmem_allocation_id().
  lower_pipelined_jaxpr_to_module.scoped_pipeline_fn: lower_pipelined_jaxpr_to_module().scoped_pipeline_fn().
  _handle_dtype_bitcast: _handle_dtype_bitcast().
  _lower_fn_with_avals.inner: _lower_fn_with_avals().inner().
  _convert_element_type_lowering_rule_wg.convert: _convert_element_type_lowering_rule_wg().convert().
  _copysign: _copysign().
  _block_id: _block_id().
  _device_id_to_logical.jax_fn: _device_id_to_logical().jax_fn().
  _i32_constant: _i32_constant().
  ResourceEstimatorContext.axis_names: ResourceEstimatorContext#axis_names.
  ModuleContext.mesh_info: ModuleContext#mesh_info.
  _lower_fun: _lower_fun().
  _extract_aliased_ref: _extract_aliased_ref().
  Resources.tmem_scratch_cols: Resources#tmem_scratch_cols.
  _AxisNames.cluster: _AxisNames#cluster.
  _AxisNames.wg: _AxisNames#wg.
  AnyBarrierRef: AnyBarrierRef.
  AnyBarrier: AnyBarrier.
  _isolate_from_above: _isolate_from_above().
  LoweringResult.gmem_scratch_shapes: LoweringResult#gmem_scratch_shapes.
  lower_pipelined_jaxpr_to_module.scoped_pipeline_fn.body_fn: lower_pipelined_jaxpr_to_module().scoped_pipeline_fn().body_fn().
  _unary_warp_lowering_rule._lowering_rule: _unary_warp_lowering_rule()._lowering_rule().
  _is_global_scope: _is_global_scope().
  op: op.
  _AxisNames.__iter__: _AxisNames#__iter__().
  ModuleContext.warp_axis_name: ModuleContext#warp_axis_name.
  _copysign_abstract_eval: _copysign_abstract_eval().
  Resources.barriers: Resources#barriers().
  ModuleContext.runtime_barriers: ModuleContext#runtime_barriers.
  lower_jaxpr_to_mosaic_gpu.read_env: lower_jaxpr_to_mosaic_gpu().read_env().
  _unravel_program_id: _unravel_program_id().
  _bcast_wg.bcast: _bcast_wg().bcast().
  _resource_estimators._resource_estimators: _resource_estimators._resource_estimators.
  ModuleContext.scoped_gmem_used_semaphores: ModuleContext#scoped_gmem_used_semaphores.
  _copysign_p: _copysign_p.
  CollectiveAxesType: CollectiveAxesType.
  _AxisNames.grid: _AxisNames#grid.
  LoweringError: LoweringError#
  _block_spec_from_block_mapping.index_map: _block_spec_from_block_mapping().index_map().
  CmpFPred: CmpFPred.
  ResourceEstimatorContext.lowering_semantics: ResourceEstimatorContext#lowering_semantics.
  LoweringRuleContext.prim: LoweringRuleContext#prim.
  ResourceEstimator.__call__: ResourceEstimator#__call__().
  ModuleContext.smem_used_bytes: ModuleContext#smem_used_bytes.
  ModuleContext.tmem_used_cols: ModuleContext#tmem_used_cols.
  _i64_constant: _i64_constant().
  ModuleContext.scoped_gmem_semaphore_base_ptr: ModuleContext#scoped_gmem_semaphore_base_ptr.
  ModuleContext.name_stack: ModuleContext#name_stack.
  LoweringResult.new_out_shapes: LoweringResult#new_out_shapes.
  _program_id: _program_id().
  _unary_warp_lowering_rule: _unary_warp_lowering_rule().
  _logistic: _logistic().
  _closed_over_values._closed_over_values_inner: _closed_over_values()._closed_over_values_inner().
  ShapedAbstractValue.weak_type: ShapedAbstractValue#weak_type.
  ModuleContext: ModuleContext#
  ModuleContext.tmem_base: ModuleContext#tmem_base.
  ModuleContext.reduction_scratch_bytes: ModuleContext#reduction_scratch_bytes.
  _register_resource_estimator.deco: _register_resource_estimator().deco().
  register_lowering_rule.deco: register_lowering_rule().deco().
  _compute_name_stack_updates: _compute_name_stack_updates().
  _lower_fn_with_avals: _lower_fn_with_avals().
  _closed_over_values: _closed_over_values().
  _replace_uses_in_block: _replace_uses_in_block().
  unsafe_zip: unsafe_zip.
  ShapedAbstractValue.size: ShapedAbstractValue#size().
  ResourceEstimatorContext.reduction_scratch_bytes: ResourceEstimatorContext#reduction_scratch_bytes.
  ModuleContext.squashed_dims: ModuleContext#squashed_dims.
  LoweringResult: LoweringResult#
  SMEM: SMEM.
  unsafe_map: unsafe_map.
  ModuleContext.program_ids: ModuleContext#program_ids.
  ModuleContext.single_wg_lane_predicate: ModuleContext#single_wg_lane_predicate.
  ModuleContext.smem_requested_bytes: ModuleContext#smem_requested_bytes.
  ModuleContext.outer_traceback: ModuleContext#outer_traceback.
  ModuleContext._smem_allocation_counter: ModuleContext#_smem_allocation_counter.
  ModuleContext._tmem_allocation_counter: ModuleContext#_tmem_allocation_counter.
  LoweringResult.module: LoweringResult#module.
  _reduce_lowering_rule_wg.i32_attr: _reduce_lowering_rule_wg().i32_attr().
  _cond_lowering_rule._yielded_values: _cond_lowering_rule()._yielded_values().
  WARPGROUP_SIZE: WARPGROUP_SIZE.
  ResourceEstimator: ResourceEstimator#
  ModuleContext.single_warp_lane_predicate: ModuleContext#single_warp_lane_predicate.
  ModuleContext.tmem_requested_cols: ModuleContext#tmem_requested_cols.
  ModuleContext.traceback_caches: ModuleContext#traceback_caches.
  LoweringRuleContext.replace: LoweringRuleContext#replace.
  LoweringResult.grid: LoweringResult#grid.
  LoweringResult.block: LoweringResult#block.
  LoweringResult.profiler_spec: LoweringResult#profiler_spec.
  _div: _div().
  _replace_uses_in_block.is_contained_within_block: _replace_uses_in_block().is_contained_within_block().
  si_impl: si_impl.
  ui_impl: ui_impl.
  si_pred: si_pred.
  ui_pred: ui_pred.
  ShapedAbstractValue.update: ShapedAbstractValue#update().
  ModuleContext.name: ModuleContext#name.
---
# Module: [`jax/_src/pallas/mosaic_gpu/lowering.py`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py)

## Classes
### `LoweringError`  ·  implements/extends Exception
- def: [`jax/_src/pallas/mosaic_gpu/lowering.py:677`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L677)
- signature: `class LoweringError(Exception):`
- used by: [`lower_jaxpr_to_mosaic_gpu`](lowering.md#lower_jaxpr_to_mosaic_gpu), [`_core_map_lowering_rule`](lowering.md#_core_map_lowering_rule), [`_mpmd_map_lowering_rule`](lowering.md#_mpmd_map_lowering_rule)

### `LoweringResult`
- def: [`jax/_src/pallas/mosaic_gpu/lowering.py:668`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L668)
- signature: `class LoweringResult:`
- members:
  - `block` — [`L671`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L671)
  - `gmem_scratch_shapes` — [`L674`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L674)
  - `grid` — [`L670`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L670)
  - `module` — [`L669`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L669)
  - `new_out_shapes` — [`L672`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L672)
  - `profiler_spec` — [`L673`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L673)
- uses (calls/refs, reference-scoped): [`ShapeDtypeStruct`](../../core.md#ShapeDtypeStruct)
- used by: [`lower_jaxpr_to_module`](lowering.md#lower_jaxpr_to_module), [`lower_pipelined_jaxpr_to_module`](lowering.md#lower_pipelined_jaxpr_to_module), [`pallas_call_lowering`](pallas_call_registration.md#pallas_call_lowering), [`dump_profile`](pallas_call_registration.md#pallas_call_lowering.dump_profile), [`zero_init_gmem_scratch`](pallas_call_registration.md#pallas_call_lowering.zero_init_gmem_scratch)

### `LoweringRuleContext`
- def: [`jax/_src/pallas/mosaic_gpu/lowering.py:648`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L648) — documented in [jax-_src-pallas-mosaic_gpu-primitives](../../../../../concepts/jax-_src-pallas-mosaic_gpu-primitives.md)
- signature: `class LoweringRuleContext:`
- members:
  - `estimator_ctx(self)` — [`L659`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L659)
  - `avals_in` — [`L652`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L652)
  - `avals_out` — [`L653`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L653)
  - `launch_ctx` — [`L650`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L650)
  - `module_ctx` — [`L649`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L649)
  - `out_layout_hint` — [`L654`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L654)
  - `prim` — [`L651`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L651)
  - `replace` — [`L656`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L656)
- uses (calls/refs, reference-scoped): [`Primitive`](../../core.md#Primitive), [`lowering_semantics`](lowering.md#ModuleContext.lowering_semantics), [`FragmentedLayout`](../../../experimental/mosaic/gpu/fragmented_array.md#FragmentedLayout.FragmentedLayout), [`axis_names`](lowering.md#ModuleContext.axis_names), [`ResourceEstimatorContext`](lowering.md#ResourceEstimatorContext), [`ShapedAbstractValue`](lowering.md#ShapedAbstractValue), [`axis_names`](lowering.md#ResourceEstimatorContext.axis_names), [`LaunchContext`](../../../experimental/mosaic/gpu/launch_context.md#LaunchContext), [`lowering_semantics`](lowering.md#ResourceEstimatorContext.lowering_semantics), [`ModuleContext`](lowering.md#ModuleContext), [`reduction_scratch_bytes`](lowering.md#ModuleContext.reduction_scratch_bytes), [`reduction_scratch_bytes`](lowering.md#ResourceEstimatorContext.reduction_scratch_bytes)
- used by: [`_handle_transforms`](lowering.md#_handle_transforms), [`lower_jaxpr_to_mosaic_gpu`](lowering.md#lower_jaxpr_to_mosaic_gpu), [`_run_scoped_lowering_rule`](lowering.md#_run_scoped_lowering_rule), [`_copy_gmem_to_smem_lowering`](primitives.md#_copy_gmem_to_smem_lowering), [`_swap_lowering_rule`](lowering.md#_swap_lowering_rule), [`_get_lowering_rule`](lowering.md#_get_lowering_rule), [`_semaphore_signal_multicast_lowering`](primitives.md#_semaphore_signal_multicast_lowering), [`_copy_smem_to_gmem_lowering`](primitives.md#_copy_smem_to_gmem_lowering), [`_async_store_smem_lowering`](primitives.md#_async_store_smem_lowering), [`_semaphore_signal_lowering_rule`](primitives.md#_semaphore_signal_lowering_rule), [`_broadcast_in_dim_lowering_rule`](lowering.md#_broadcast_in_dim_lowering_rule), [`_semaphore_wait_lowering_rule`](primitives.md#_semaphore_wait_lowering_rule), [`_axis_index_rule`](lowering.md#_axis_index_rule), [`_core_map_lowering_rule`](lowering.md#_core_map_lowering_rule), [`_mpmd_map_lowering_rule`](lowering.md#_mpmd_map_lowering_rule), [`_multimem_store_lowering_rule`](primitives.md#_multimem_store_lowering_rule), [`_run_state_lowering_rule`](lowering.md#_run_state_lowering_rule), [`_tcgen05_mma_lowering`](primitives.md#_tcgen05_mma_lowering), [`_barrier_arrive_lowering`](primitives.md#_barrier_arrive_lowering), [`_debug_print_lowering_rule`](lowering.md#_debug_print_lowering_rule), [`_cond_lowering_rule`](lowering.md#_cond_lowering_rule), [`_jaxpr_call_lowering_rule`](lowering.md#_jaxpr_call_lowering_rule), [`_inline_mgpu_flat_transformed_args`](primitives.md#_inline_mgpu_flat_transformed_args), [`_async_prefetch_lowering`](primitives.md#_async_prefetch_lowering), [`_multimem_load_reduce_lowering_rule`](primitives.md#_multimem_load_reduce_lowering_rule), [`_convert_element_type_lowering_rule_wg`](lowering.md#_convert_element_type_lowering_rule_wg), [`_select_n_lowering_rule`](lowering.md#_select_n_lowering_rule), [`_while_lowering_rule`](lowering.md#_while_lowering_rule), [`query_cluster_cancel_lowering`](primitives.md#query_cluster_cancel_lowering), [`_scan_lowering_rule`](lowering.md#_scan_lowering_rule), [`try_cluster_cancel_lowering`](primitives.md#try_cluster_cancel_lowering), [`_populate_custom_primitive_op_block`](primitives.md#_populate_custom_primitive_op_block), [`_wgmma_lowering`](primitives.md#_wgmma_lowering), [`_reduce_lowering_rule`](lowering.md#_reduce_lowering_rule), [`_async_copy_smem_to_tmem_lowering_rule`](primitives.md#_async_copy_smem_to_tmem_lowering_rule), [`_bitcast_convert_type_lowering_rule`](lowering.md#_bitcast_convert_type_lowering_rule), [`_convert_element_type_lowering_rule`](lowering.md#_convert_element_type_lowering_rule), [`_inline_mgpu_lowering_rule_wg_semantics`](primitives.md#_inline_mgpu_lowering_rule_wg_semantics), [`_swap_lowering_rule_wg`](lowering.md#_swap_lowering_rule_wg), [`_wait_smem_to_gmem_lowering`](primitives.md#_wait_smem_to_gmem_lowering)  (+95 more)

### `ModuleContext`
- def: [`jax/_src/pallas/mosaic_gpu/lowering.py:457`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L457)
- signature: `class ModuleContext:`
- members:
  - `alloc_tmem(self, struct: jax.ShapeDtypeStruct, *, layout: tcgen05.TMEMLayout)` — [`L547`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L547)
  - `next_smem_allocation_id(self)` — [`L487`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L487)
  - `next_tmem_allocation_id(self)` — [`L492`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L492)
  - `reserve_barrier(self, barrier: mgpu.Barrier | mgpu.ClusterBarrier)` — [`L512`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L512) — Reserves a barrier.
  - `reserve_semaphores(self, shape: tuple[int, ...], collective_axes: CollectiveAxesType)` — [`L530`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L530)
  - `scratch_view(self, struct: jax.ShapeDtypeStruct)` — [`L592`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L592) — Creates a view into the runtime scratch buffer for the given struct.
  - `single_lane_predicate(self)` — [`L498`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L498) — Returns a predicate that is True for a single lane within the current
  - `approx_math` — [`L461`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L461)
  - `auto_barriers` — [`L479`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L479)
  - `axis_names` — [`L459`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L459)
  - `lowering_semantics` — [`L475`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L475) — documented in [jax-_src-pallas-mosaic_gpu-lowering](../../../../../concepts/jax-_src-pallas-mosaic_gpu-lowering.md)
  - `mesh_info` — [`L477`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L477)
  - `name` — [`L458`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L458)
  - `name_stack` — [`L472`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L472)
  - `outer_traceback` — [`L483`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L483)
  - `primitive_semantics` — [`L476`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L476)
  - `program_ids` — [`L460`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L460)
  - `reduction_scratch_bytes` — [`L481`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L481)
  - `runtime_barriers` — [`L471`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L471)
  - `scoped_gmem_semaphore_base_ptr` — [`L470`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L470)
  - `scoped_gmem_used_semaphores` — [`L469`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L469)
  - `single_warp_lane_predicate` — [`L463`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L463)
  - `single_wg_lane_predicate` — [`L462`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L462)
  - `smem_requested_bytes` — [`L464`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L464)
  - `smem_used_bytes` — [`L465`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L465)
  - `squashed_dims` — [`L474`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L474)
  - `tmem_base` — [`L468`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L468)
  - `tmem_requested_cols` — [`L466`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L466)
  - `tmem_used_cols` — [`L467`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L467)
  - `traceback_caches` — [`L473`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L473)
  - `warp_axis_name` — [`L482`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L482)
- protocol/private: `_smem_allocation_counter`[`L484`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L484), `_tmem_allocation_counter`[`L485`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L485)
- uses (calls/refs, reference-scoped): [`LoweringSemantics`](../../../experimental/mosaic/gpu/core.md#LoweringSemantics), [`Warpgroup`](../../../experimental/mosaic/gpu/core.md#LoweringSemantics.Warpgroup), [`Lane`](../../../experimental/mosaic/gpu/core.md#LoweringSemantics.Lane), [`ShapeDtypeStruct`](../../core.md#ShapeDtypeStruct), [`dtype`](../../../numpy/__init__.pyi.md#dtype), [`itemsize_bits`](../../dtypes.md#itemsize_bits), [`dtype_to_ir_type`](../../../experimental/mosaic/gpu/utils.md#dtype_to_ir_type), [`PrimitiveSemantics`](core.md#PrimitiveSemantics), [`to_layout_attr`](../../../experimental/mosaic/gpu/layouts.md#to_layout_attr), [`AxisName`](../../core.md#AxisName), [`Warp`](core.md#PrimitiveSemantics.Warp), [`NameStack`](../../source_info_util.md#NameStack), [`_as_index`](lowering.md#_as_index), [`shape`](../../core.md#ShapeDtypeStruct.shape), [`BarrierRef`](../../../experimental/mosaic/gpu/utils.md#BarrierRef), [`DialectBarrierRef`](../../../experimental/mosaic/gpu/utils.md#DialectBarrierRef), [`dtype`](../../core.md#ShapeDtypeStruct.dtype), [`smem`](../../../experimental/mosaic/gpu/utils.md#smem), [`memref_slice`](../../../experimental/mosaic/gpu/utils.md#memref_slice), [`_AxisNames`](lowering.md#_AxisNames), [`_i32_constant`](lowering.md#_i32_constant), [`MeshInfo`](../utils.md#MeshInfo), [`memref_reshape`](../../../experimental/mosaic/gpu/utils.md#memref_reshape), [`AnyBarrierRef`](lowering.md#AnyBarrierRef), [`CollectiveBarrierRef`](../../../experimental/mosaic/gpu/utils.md#CollectiveBarrierRef), [`AnyBarrier`](lowering.md#AnyBarrier), [`align_to`](core.md#align_to), [`ds`](../../../experimental/mosaic/gpu/utils.md#ds), [`Barrier`](../../../experimental/mosaic/gpu/core.md#Barrier), [`ClusterBarrier`](../../../experimental/mosaic/gpu/core.md#ClusterBarrier), [`SMEM_ALIGNMENT`](core.md#SMEM_ALIGNMENT), [`tmem`](../../../experimental/mosaic/gpu/utils.md#tmem), [`CollectiveAxesType`](lowering.md#CollectiveAxesType), [`Warpgroup`](core.md#PrimitiveSemantics.Warpgroup), [`DYNAMIC`](../../../experimental/mosaic/gpu/utils.md#DYNAMIC), [`TMEM_COL_ALIGNMENT`](core.md#TMEM_COL_ALIGNMENT)
- used by: [`module_ctx`](lowering.md#LoweringRuleContext.module_ctx), [`_handle_transforms`](lowering.md#_handle_transforms), [`lower_jaxpr_to_mosaic_gpu`](lowering.md#lower_jaxpr_to_mosaic_gpu), [`_run_scoped_lowering_rule`](lowering.md#_run_scoped_lowering_rule), [`body`](lowering.md#lower_jaxpr_to_module.body), [`_copy_gmem_to_smem_lowering`](primitives.md#_copy_gmem_to_smem_lowering), [`_swap_lowering_rule`](lowering.md#_swap_lowering_rule), [`_get_lowering_rule`](lowering.md#_get_lowering_rule), [`_semaphore_signal_multicast_lowering`](primitives.md#_semaphore_signal_multicast_lowering), [`_copy_smem_to_gmem_lowering`](primitives.md#_copy_smem_to_gmem_lowering), [`_async_store_smem_lowering`](primitives.md#_async_store_smem_lowering), [`_semaphore_signal_lowering_rule`](primitives.md#_semaphore_signal_lowering_rule), [`_semaphore_wait_lowering_rule`](primitives.md#_semaphore_wait_lowering_rule), [`_axis_index_rule`](lowering.md#_axis_index_rule), [`_core_map_lowering_rule`](lowering.md#_core_map_lowering_rule), [`_mpmd_map_lowering_rule`](lowering.md#_mpmd_map_lowering_rule), [`_multimem_store_lowering_rule`](primitives.md#_multimem_store_lowering_rule), [`_run_state_lowering_rule`](lowering.md#_run_state_lowering_rule), [`_tcgen05_mma_lowering`](primitives.md#_tcgen05_mma_lowering), [`_barrier_arrive_lowering`](primitives.md#_barrier_arrive_lowering), [`_debug_print_lowering_rule`](lowering.md#_debug_print_lowering_rule), [`_cond_lowering_rule`](lowering.md#_cond_lowering_rule), [`_jaxpr_call_lowering_rule`](lowering.md#_jaxpr_call_lowering_rule), [`_inline_mgpu_flat_transformed_args`](primitives.md#_inline_mgpu_flat_transformed_args), [`_async_prefetch_lowering`](primitives.md#_async_prefetch_lowering), [`_multimem_load_reduce_lowering_rule`](primitives.md#_multimem_load_reduce_lowering_rule), [`_convert_element_type_lowering_rule_wg`](lowering.md#_convert_element_type_lowering_rule_wg), [`_select_n_lowering_rule`](lowering.md#_select_n_lowering_rule), [`_while_lowering_rule`](lowering.md#_while_lowering_rule), [`query_cluster_cancel_lowering`](primitives.md#query_cluster_cancel_lowering), [`try_cluster_cancel_lowering`](primitives.md#try_cluster_cancel_lowering), [`_reduce_lowering_rule`](lowering.md#_reduce_lowering_rule), [`_async_copy_smem_to_tmem_lowering_rule`](primitives.md#_async_copy_smem_to_tmem_lowering_rule), [`_bitcast_convert_type_lowering_rule`](lowering.md#_bitcast_convert_type_lowering_rule), [`_convert_element_type_lowering_rule`](lowering.md#_convert_element_type_lowering_rule), [`_inline_mgpu_lowering_rule_wg_semantics`](primitives.md#_inline_mgpu_lowering_rule_wg_semantics), [`_swap_lowering_rule_wg`](lowering.md#_swap_lowering_rule_wg), [`_wait_smem_to_gmem_lowering`](primitives.md#_wait_smem_to_gmem_lowering), [`_binary_op_lowering_rule`](lowering.md#_binary_op_lowering_rule), [`_barrier_test_lowering`](primitives.md#_barrier_test_lowering)  (+33 more)

### `ResourceEstimator`  ·  implements/extends Protocol
- def: [`jax/_src/pallas/mosaic_gpu/lowering.py:233`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L233)
- signature: `class ResourceEstimator(Protocol):`
- protocol/private: `__call__`[`L235`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L235)
- uses (calls/refs, reference-scoped): [`Resources`](lowering.md#Resources), [`ResourceEstimatorContext`](lowering.md#ResourceEstimatorContext)
- used by: [`_resource_estimators`](lowering.md#_resource_estimators._resource_estimators)

### `ResourceEstimatorContext`
- def: [`jax/_src/pallas/mosaic_gpu/lowering.py:110`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L110)
- signature: `class ResourceEstimatorContext:`
- members:
  - `arrival_multiplier(self)` — [`L116`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L116)
  - `axis_names` — [`L112`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L112)
  - `lowering_semantics` — [`L113`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L113)
  - `reduction_scratch_bytes` — [`L111`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L111)
- uses (calls/refs, reference-scoped): [`LoweringSemantics`](../../../experimental/mosaic/gpu/core.md#LoweringSemantics), [`Lane`](../../../experimental/mosaic/gpu/core.md#LoweringSemantics.Lane), [`_AxisNames`](lowering.md#_AxisNames), [`WARPGROUP_SIZE`](lowering.md#WARPGROUP_SIZE)
- used by: [`lower_jaxpr_to_module`](lowering.md#lower_jaxpr_to_module), [`_run_scoped_lowering_rule`](lowering.md#_run_scoped_lowering_rule), [`_run_scoped_resource_estimator`](lowering.md#_run_scoped_resource_estimator), [`_estimate_resources`](lowering.md#_estimate_resources), [`_get_global_resource_estimator`](lowering.md#_get_global_resource_estimator), [`_reduce_resource_estimator`](lowering.md#_reduce_resource_estimator), [`_while_resource_estimator`](lowering.md#_while_resource_estimator), [`estimator_ctx`](lowering.md#LoweringRuleContext.estimator_ctx), [`_cond_resource_estimator`](lowering.md#_cond_resource_estimator), [`_pjit_resource_estimator`](lowering.md#_pjit_resource_estimator), [`_scan_resource_estimator`](lowering.md#_scan_resource_estimator), [`_core_map_resource_estimator`](lowering.md#_core_map_resource_estimator), [`_mpmd_map_resource_estimator`](lowering.md#_mpmd_map_resource_estimator), [`_run_state_resource_estimator`](lowering.md#_run_state_resource_estimator), [`_jaxpr_call_resource_estimator`](lowering.md#_jaxpr_call_resource_estimator), [`__call__`](lowering.md#ResourceEstimator.__call__)

### `Resources`
- def: [`jax/_src/pallas/mosaic_gpu/lowering.py:154`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L154)
- signature: `class Resources:`
- members:
  - `barriers(self)` — [`L186`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L186)
  - `or_(self, other: Resources, axis_names: _AxisNames)` — [`L207`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L207)
  - `barrier_counts` — [`L158`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L158)
  - `scoped_gmem_semaphores` — [`L162`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L162)
  - `smem_scratch_bytes` — [`L155`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L155)
  - `tmem_collective_scratch_cols` — [`L157`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L157)
  - `tmem_scratch_cols` — [`L156`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L156)
- protocol/private: `__add__`[`L189`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L189), `__post_init__`[`L166`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L166)
- uses (calls/refs, reference-scoped): [`_AxisNames`](lowering.md#_AxisNames), [`_is_block_local_scope`](lowering.md#_is_block_local_scope), [`AnyBarrier`](lowering.md#AnyBarrier), [`align_to`](core.md#align_to), [`_is_global_scope`](lowering.md#_is_global_scope), [`SMEM_ALIGNMENT`](core.md#SMEM_ALIGNMENT), [`CollectiveAxesType`](lowering.md#CollectiveAxesType)
- used by: [`lower_jaxpr_to_module`](lowering.md#lower_jaxpr_to_module), [`body`](lowering.md#lower_jaxpr_to_module.body), [`_run_scoped_resource_estimator`](lowering.md#_run_scoped_resource_estimator), [`_estimate_resources`](lowering.md#_estimate_resources), [`_get_global_resource_estimator`](lowering.md#_get_global_resource_estimator), [`_reduce_resource_estimator`](lowering.md#_reduce_resource_estimator), [`_while_resource_estimator`](lowering.md#_while_resource_estimator), [`_cond_resource_estimator`](lowering.md#_cond_resource_estimator), [`_pjit_resource_estimator`](lowering.md#_pjit_resource_estimator), [`_scan_resource_estimator`](lowering.md#_scan_resource_estimator), [`_core_map_resource_estimator`](lowering.md#_core_map_resource_estimator), [`_mpmd_map_resource_estimator`](lowering.md#_mpmd_map_resource_estimator), [`_run_state_resource_estimator`](lowering.md#_run_state_resource_estimator), [`__call__`](lowering.md#ResourceEstimator.__call__)

### `ShapedAbstractValue`  ·  implements/extends Protocol
- def: [`jax/_src/pallas/mosaic_gpu/lowering.py:88`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L88)
- signature: `class ShapedAbstractValue(Protocol):`
- members:
  - `ndim(self)` — [`L94`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L94)
  - `size(self)` — [`L98`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L98)
  - `update(self, **kwargs: Any)` — [`L101`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L101)
  - `dtype` — [`L90`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L90)
  - `shape` — [`L89`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L89)
  - `weak_type` — [`L91`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L91)
- uses (calls/refs, reference-scoped): [`dtype`](../../../numpy/__init__.pyi.md#dtype), [`DimSize`](../../typing.md#DimSize)
- used by: [`avals_in`](lowering.md#LoweringRuleContext.avals_in), [`lower_jaxpr_to_mosaic_gpu`](lowering.md#lower_jaxpr_to_mosaic_gpu), [`_run_scoped_lowering_rule`](lowering.md#_run_scoped_lowering_rule), [`_swap_lowering_rule`](lowering.md#_swap_lowering_rule), [`_get_lowering_rule`](lowering.md#_get_lowering_rule), [`_run_scoped_resource_estimator`](lowering.md#_run_scoped_resource_estimator), [`_async_store_smem_lowering`](primitives.md#_async_store_smem_lowering), [`_broadcast_in_dim_lowering_rule`](lowering.md#_broadcast_in_dim_lowering_rule), [`_core_map_lowering_rule`](lowering.md#_core_map_lowering_rule), [`_mpmd_map_lowering_rule`](lowering.md#_mpmd_map_lowering_rule), [`_multimem_store_lowering_rule`](primitives.md#_multimem_store_lowering_rule), [`_debug_print_lowering_rule`](lowering.md#_debug_print_lowering_rule), [`_cond_lowering_rule`](lowering.md#_cond_lowering_rule), [`_inline_mgpu_flat_transformed_args`](primitives.md#_inline_mgpu_flat_transformed_args), [`_multimem_load_reduce_lowering_rule`](primitives.md#_multimem_load_reduce_lowering_rule), [`_convert_element_type_lowering_rule_wg`](lowering.md#_convert_element_type_lowering_rule_wg), [`_select_n_lowering_rule`](lowering.md#_select_n_lowering_rule), [`_scan_lowering_rule`](lowering.md#_scan_lowering_rule), [`_populate_custom_primitive_op_block`](primitives.md#_populate_custom_primitive_op_block), [`ref_for_aval`](lowering.md#lower_pipelined_jaxpr_to_module.ref_for_aval), [`_reduce_lowering_rule`](lowering.md#_reduce_lowering_rule), [`_bitcast_convert_type_lowering_rule`](lowering.md#_bitcast_convert_type_lowering_rule), [`_convert_element_type_lowering_rule`](lowering.md#_convert_element_type_lowering_rule), [`_swap_lowering_rule_wg`](lowering.md#_swap_lowering_rule_wg), [`_binary_op_lowering_rule`](lowering.md#_binary_op_lowering_rule), [`_multimem_load_reduce_lowering_rule_wg`](primitives.md#_multimem_load_reduce_lowering_rule_wg), [`_binary_op_lowering_rule_wg`](lowering.md#_binary_op_lowering_rule_wg), [`_comparison_lowering_rule_wg`](lowering.md#_comparison_lowering_rule_wg), [`_get_lowering_rule_wg`](lowering.md#_get_lowering_rule_wg), [`_reduce_lowering_rule_wg`](lowering.md#_reduce_lowering_rule_wg), [`avals_out`](lowering.md#LoweringRuleContext.avals_out), [`_cos_lowering_rule`](lowering.md#_cos_lowering_rule), [`_exp2_lowering_rule`](lowering.md#_exp2_lowering_rule), [`_exp_lowering_rule`](lowering.md#_exp_lowering_rule), [`_log_lowering_rule`](lowering.md#_log_lowering_rule), [`_rsqrt_lowering_rule`](lowering.md#_rsqrt_lowering_rule), [`_sin_lowering_rule`](lowering.md#_sin_lowering_rule), [`_square_lowering_rule`](lowering.md#_square_lowering_rule), [`_tanh_lowering_rule`](lowering.md#_tanh_lowering_rule), [`_get_cluster_barrier`](lowering.md#_get_cluster_barrier)  (+33 more)

### `_AxisNames`
- def: [`jax/_src/pallas/mosaic_gpu/lowering.py:437`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L437)
- members:
  - `reverse(self)` — [`L447`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L447)
  - `cluster` — [`L439`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L439)
  - `grid` — [`L438`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L438)
  - `wg` — [`L440`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L440)
- protocol/private: `__iter__`[`L442`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L442)
- used by: [`lower_jaxpr_to_module`](lowering.md#lower_jaxpr_to_module), [`_run_scoped_lowering_rule`](lowering.md#_run_scoped_lowering_rule), [`lower_pipelined_jaxpr_to_module`](lowering.md#lower_pipelined_jaxpr_to_module), [`_axis_index_rule`](lowering.md#_axis_index_rule), [`query_cluster_cancel_lowering`](primitives.md#query_cluster_cancel_lowering), [`_reduce_lowering_rule`](lowering.md#_reduce_lowering_rule), [`_get_cluster_barrier`](lowering.md#_get_cluster_barrier), [`_collective_mma_predicate`](primitives.md#_collective_mma_predicate), [`or_`](lowering.md#Resources.or_), [`axis_names`](lowering.md#ModuleContext.axis_names), [`block_id_to_grid_id`](lowering.md#block_id_to_grid_id), [`_resolve_cluster_axis`](lowering.md#_resolve_cluster_axis), [`_is_block_local_scope`](lowering.md#_is_block_local_scope), [`axis_names`](lowering.md#ResourceEstimatorContext.axis_names), [`_is_global_scope`](lowering.md#_is_global_scope)

## Functions
- `_abs_lowering_rule(ctx: LoweringRuleContext, x)` — [`L3039`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L3039)
- `_abs_lowering_rule_wg(ctx: LoweringRuleContext, x)` — [`L3045`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L3045)
- `_as_index(v: object)` — [`L4465`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L4465)
- `_atan2_lowering_rule(ctx: LoweringRuleContext, y, x)` — [`L3140`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L3140)
- `_atan2_lowering_rule_wg(ctx: LoweringRuleContext, y, x)` — [`L3146`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L3146)
- `_axis_index_rule(ctx: LoweringRuleContext, *, axis_name: Hashable)` — [`L3411`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L3411)
- `_bcast(x: Any, y: Any, x_aval: ShapedAbstractValue, y_aval: ShapedAbstractValue, out_aval: ShapedAbstractValue)` — [`L4331`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L4331)
- `_bcast_wg(x: Any, y: Any, x_aval: ShapedAbstractValue, y_aval: ShapedAbstractValue, out_aval: ShapedAbstractValue)` — [`L4364`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L4364) — Ensures that ``x`` and ``y`` have the expected shapes and dtypes.
- `_binary_boolean_op_lowering_rule_wg(ctx: LoweringRuleContext, x, y, *, impl)` — [`L2805`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L2805)
- `_binary_op_lowering_rule(ctx: LoweringRuleContext, x, y, *, impl, out_dtype=None)` — [`L2698`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L2698)
- `_binary_op_lowering_rule_wg(ctx: LoweringRuleContext, x, y, *, ui_impl, si_impl, f_impl=None, **kwargs)` — [`L2744`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L2744)
- `_bitcast_convert_type_lowering_rule(ctx: LoweringRuleContext, x, *, new_dtype)` — [`L4076`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L4076)
- `_block_id(ctx: LoweringRuleContext, dim: gpu_dialect.Dimension)` — [`L3336`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L3336)
- `_block_spec_from_block_mapping(bm: pallas_core.BlockMapping, which_parallel: Sequence[bool])` — [`L743`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L743)
- `_broadcast_in_dim_lowering_rule(ctx: LoweringRuleContext, x: mgpu.FragmentedArray, *, broadcast_dimensions, shape, sharding)` — [`L2471`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L2471)
- `_broadcast_in_dim_lowering_rule_wg(ctx: LoweringRuleContext, x, *, broadcast_dimensions, shape, sharding)` — [`L2535`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L2535)
- `_bubble_up_transform(ctx: LoweringRuleContext, aval: jax_core.AbstractValue, transforms: Sequence[state_types.Transform], transforms_avals: Sequence[state_types.Transform], t: T, t_aval: T)` — [`L1743`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L1743)
- `_bubble_up_transforms_for_lowering(ctx: LoweringRuleContext, aval: jax_core.AbstractValue, transforms: Sequence[state_types.Transform], transform_avals: Sequence[state_types.Transform], *, handle_transposes: bool = True, handle_reshapes: bool = True)` — [`L1800`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L1800) — Bubbles up eligible `transforms` to the head of the sequence.
- `_check_block_mappings(block_mappings: Sequence[pallas_core.BlockMapping], debug_info: jax_core.DebugInfo)` — [`L702`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L702)
- `_check_lowering_rule(ctx: LoweringRuleContext, *err_args, err_tree, debug)` — [`L4613`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L4613)
- `_clamp_lowering_rule(ctx: LoweringRuleContext, l, x, u)` — [`L2899`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L2899)
- `_closed_over_values(block: ir.Block)` — [`L4121`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L4121) — Returns a list of unique values used within `block` that are defined outside of `block`.
- `_closed_over_values_inner(block: ir.Block, vals_in_block: set[ir.Value])` — [`L4123`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L4123)
- `_clz_lowering_rule(ctx: LoweringRuleContext, x)` — [`L2919`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L2919)
- `_commute_transform(aval: jax_core.AbstractValue, t1: state_types.Transform, t2: state_types.Transform)` — [`L1651`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L1651) — Commutes two transforms.
- `_comparison_lowering_rule_wg(ctx: LoweringRuleContext, x, y, *, si_pred, ui_pred, f_pred)` — [`L2830`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L2830)
- `_compute_name_stack_updates(old_name_stack: list[str], new_name_stack: list[str])` — [`L1194`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L1194)
- `_concatenate_lowering_rule(ctx: LoweringRuleContext, *args, dimension)` — [`L2424`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L2424)
- `_concatenate_lowering_rule_wg(ctx: LoweringRuleContext, *args, dimension)` — [`L2432`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L2432)
- `_cond_lowering_rule(ctx: LoweringRuleContext, index, *args, branches, **params)` — [`L4009`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L4009)
- `_cond_resource_estimator(ctx: ResourceEstimatorContext, *args, branches)` — [`L279`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L279)
- `_convert_element_type_lowering_rule(ctx: LoweringRuleContext, x, *, new_dtype, weak_type, sharding)` — [`L2559`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L2559)
- `_convert_element_type_lowering_rule_wg(ctx: LoweringRuleContext, x, *, new_dtype, weak_type, sharding)` — [`L2578`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L2578)
- `_copysign(x1: jax.typing.ArrayLike, x2: jax.typing.ArrayLike)` — [`L3085`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L3085)
- `_copysign_abstract_eval(x1, x2)` — [`L3090`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L3090)
- `_copysign_lowering_rule(ctx: LoweringRuleContext, x1, x2)` — [`L3095`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L3095)
- `_core_map_lowering_rule(ctx: LoweringRuleContext, *args, jaxpr, mesh, **_)` — [`L4207`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L4207)
- `_core_map_resource_estimator(ctx: ResourceEstimatorContext, *args, jaxpr: jax_core.Jaxpr, **params)` — [`L323`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L323)
- `_cos_lowering_rule(ctx: LoweringRuleContext, x, accuracy)` — [`L3013`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L3013)
- `_debug_print_lowering_rule(ctx: LoweringRuleContext, *args, fmt, ordered, partitioned, in_tree, static_args, np_printoptions, has_placeholders, logging_record)` — [`L3474`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L3474)
- `_delay_lowering(ctx: LoweringRuleContext, nanos)` — [`L4686`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L4686)
- `_device_id_to_logical(ctx: LoweringRuleContext, device_id, device_id_type: primitives.DeviceIdType, device_id_aval: Any)` — [`L4422`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L4422)
- `_div(x, y)` — [`L2708`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L2708)
- `_ensure_fa(x: object, dtype: jnp.dtype)` — [`L4355`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L4355)
- `_ensure_idx_fa(x: Any)` — [`L4502`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L4502)
- `_ensure_ir_value(x: Any, dtype: jnp.dtype)` — [`L4406`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L4406)
- `_erf_lowering_rule(ctx: LoweringRuleContext, x)` — [`L3128`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L3128)
- `_erf_lowering_rule_wg(ctx: LoweringRuleContext, x)` — [`L3134`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L3134)
- `_estimate_resources(ctx: ResourceEstimatorContext, jaxpr: jax_core.Jaxpr)` — [`L252`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L252) — Estimates the resources required by the kernel.
- `_eval_index_map(module_ctx: ModuleContext, launch_ctx: mgpu.LaunchContext, idx: Sequence[ir.Value], block_mapping: pallas_core.BlockMapping)` — [`L681`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L681)
- `_exp2_lowering_rule(ctx: LoweringRuleContext, x, accuracy)` — [`L2987`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L2987)
- `_exp_lowering_rule(ctx: LoweringRuleContext, x, accuracy)` — [`L2973`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L2973)
- `_extract_aliased_ref(ref: ir.Value, ref_aval: state_types.AbstractRef, transform_avals: Sequence[state_types.Transform], transforms: Sequence[state_types.Transform], lowering_semantics: mgpu.LoweringSemantics)` — [`L1490`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L1490)
- `_get_barrier(aval: ShapedAbstractValue, arrival_multiplier: int)` — [`L127`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L127)
- `_get_cluster_barrier(aval: ShapedAbstractValue, axis_names: _AxisNames)` — [`L138`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L138)
- `_get_global_lowering_rule(ctx: LoweringRuleContext, *, what)` — [`L3691`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L3691)
- `_get_global_resource_estimator(ctx: ResourceEstimatorContext, *, what)` — [`L3676`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L3676)
- `_get_lowering_rule(ctx: LoweringRuleContext, x_ref, *leaves, tree, optimized=True)` — [`L2071`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L2071)
- `_get_lowering_rule_wg(ctx: LoweringRuleContext, x_ref, *leaves, tree, optimized=True)` — [`L2194`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L2194)
- `_handle_dtype_bitcast(ref: ir.Value, src_dtype: ir.Type, dst_dtype: ir.Type)` — [`L1431`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L1431) — Allows bitcasting a SMEM ref from one element type to another.
- `_handle_transforms(ctx: LoweringRuleContext, ref_aval: state_types.AbstractRef, ref: T, transform_avals: Sequence[state_types.Transform], transforms: Sequence[state_types.Transform], *, handle_transposes=True, handle_reshapes=True, allow_peer_refs=False, allow_multicast_refs=False)` — [`L1882`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L1882) — documented in [jax-_src-pallas-mosaic_gpu-lowering](../../../../../concepts/jax-_src-pallas-mosaic_gpu-lowering.md)
- `_i32_constant(v: int)` — [`L4453`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L4453)
- `_i64_constant(v: int)` — [`L4459`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L4459)
- `_integer_pow_lowering_rule(ctx: LoweringRuleContext, x, y)` — [`L2870`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L2870)
- `_iota_lowering(ctx: LoweringRuleContext, dtype, shape, dimension, sharding)` — [`L4659`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L4659)
- `_iota_lowering_wg(ctx: LoweringRuleContext, dtype, shape, dimension, sharding)` — [`L4676`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L4676)
- `_ir_constant(v: object, t: ir.Type)` — [`L4440`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L4440)
- `_is_block_local_scope(collective_axes: CollectiveAxesType, axis_names: _AxisNames)` — [`L3360`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L3360) — Returns whether the collective axes represents a block scope.
- `_is_global_scope(collective_axes: CollectiveAxesType, axis_names: _AxisNames)` — [`L3369`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L3369) — Returns whether the collective axes represents a GPU global scope.
- `_isolate_from_above(op: ir.Operation | ir.OpView)` — [`L4160`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L4160) — Makes `op` conform to the `IsolatedFromAbove` trait.
- `_jaxpr_call_lowering_rule(ctx: LoweringRuleContext, *flat_args, jaxpr: jax_core.Jaxpr, ref_treedefs, program_ids_treedef)` — [`L4703`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L4703)
- `_jaxpr_call_resource_estimator(ctx: ResourceEstimatorContext, *args, jaxpr: jax_core.Jaxpr, **params)` — [`L426`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L426)
- `_layout_cast_lowering(ctx: LoweringRuleContext, x, *, new_layout)` — [`L4639`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L4639)
- `_layout_cast_lowering_wg(ctx: LoweringRuleContext, x, *, new_layout)` — [`L4645`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L4645)
- `_log_lowering_rule(ctx: LoweringRuleContext, x, accuracy)` — [`L3026`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L3026)
- `_logistic(x, accuracy)` — [`L2957`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L2957)
- `_lower_fn_with_avals(f, avals_in)` — [`L1722`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L1722)
- `_lower_fun(fun: Callable, *, in_avals: Any | None = None)` — [`L1391`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L1391)
- `_lower_jaxpr_to_for_loop(ctx: LoweringRuleContext, jaxpr: jax_core.Jaxpr, start: ir.Value, length: int | ir.Value, consts, *args, has_loop_index: bool, unroll: int | None = None)` — [`L3759`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L3759)
- `_lower_while_via_fori(ctx: LoweringRuleContext, *args, fori_jaxpr, cond_nconsts, body_nconsts)` — [`L3887`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L3887)
- `_lowering_rule(ctx: LoweringRuleContext, x)` — [`L2676`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L2676)
- `_mpmd_map_lowering_rule(ctx: LoweringRuleContext, *args, jaxprs, meshes, **_)` — [`L4267`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L4267)
- `_mpmd_map_resource_estimator(ctx: ResourceEstimatorContext, *args, jaxprs: tuple[jax_core.Jaxpr, ...], **params)` — [`L331`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L331)
- `_multiple_of_lane_lowering_rule(ctx: LoweringRuleContext, val, *, values)` — [`L1337`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L1337)
- `_multiple_of_wg_lowering_rule(ctx: LoweringRuleContext, val, *, values)` — [`L1347`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L1347)
- `_ndindexer_indices(indexer: indexing.NDIndexer, allow_arrays: bool = False)` — [`L2040`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L2040)
- `_optimization_barrier_lowering(ctx: LoweringRuleContext, *args)` — [`L4104`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L4104)
- `_optimization_barrier_lowering_wg(ctx: LoweringRuleContext, *args)` — [`L4114`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L4114)
- `_pjit_lowering_rule(ctx: LoweringRuleContext, *args, jaxpr, **kwargs)` — [`L2387`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L2387)
- `_pjit_resource_estimator(ctx: ResourceEstimatorContext, *args, jaxpr: jax_core.ClosedJaxpr, **params)` — [`L312`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L312)
- `_program_id(parallel_axis: int, squashed_dims: tuple[int, ...], grid_size: int)` — [`L1374`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L1374) — Returns the id of the current kernel instance along the given axis in the original Pallas grid.
- `_program_id_lowering_rule(ctx: LoweringRuleContext, axis)` — [`L1330`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L1330)
- `_reduce_lowering_rule(op, ctx: LoweringRuleContext, x, *, axes, **kwargs)` — [`L3202`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L3202)
- `_reduce_lowering_rule_wg(ctx: LoweringRuleContext, kind: vector_dialect.CombiningKind, acc: int | float, x, axes)` — [`L3248`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L3248)
- `_reduce_max_lowering_rule_wg(ctx: LoweringRuleContext, x, *, axes)` — [`L3290`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L3290)
- `_reduce_min_lowering_rule_wg(ctx: LoweringRuleContext, x, *, axes)` — [`L3307`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L3307)
- `_reduce_prod_lowering_rule_wg(ctx: LoweringRuleContext, x, *, axes)` — [`L3324`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L3324)
- `_reduce_resource_estimator(ctx: ResourceEstimatorContext, x_aval: jax_core.ShapedArray, *, axes, **kwargs)` — [`L415`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L415)
- `_reduce_sum_lowering_rule_wg(ctx: LoweringRuleContext, x, *, axes, out_sharding)` — [`L3283`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L3283)
- `_register_resource_estimator(primitive: jax_core.Primitive)` — [`L244`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L244)
- `_reinterpret_cast(ref: ir.Value, new_ref_aval: state_types.AbstractRef)` — [`L1771`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L1771)
- `_replace_uses_in_block(old: ir.Value, new: ir.Value, block: ir.Block)` — [`L4144`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L4144) — Replaces all uses of the `old` value with the `new` value in `block`.
- `_reshape_lowering_rule(ctx: LoweringRuleContext, x, new_sizes, dimensions, sharding)` — [`L3152`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L3152)
- `_reshape_lowering_rule_wg(ctx: LoweringRuleContext, x, new_sizes, dimensions, sharding)` — [`L3164`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L3164)
- `_reshard_lowering_rule(ctx, x, dst_sharding, concrete_mesh)` — [`L4696`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L4696)
- `_resolve_cluster_axis(axis_names: _AxisNames | None, axis_name: Hashable)` — [`L3346`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L3346)
- `_round_lowering_rule(ctx: LoweringRuleContext, x, rounding_method)` — [`L3056`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L3056)
- `_round_lowering_rule_wg(ctx: LoweringRuleContext, x, rounding_method)` — [`L3069`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L3069)
- `_rsqrt_lowering_rule(ctx: LoweringRuleContext, x, accuracy)` — [`L2929`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L2929)
- `_run_scoped_lowering_rule(ctx: LoweringRuleContext, *consts, jaxpr: jax_core.Jaxpr, collective_axes, **_)` — [`L3527`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L3527)
- `_run_scoped_resource_estimator(ctx: ResourceEstimatorContext, *consts, jaxpr: jax_core.Jaxpr, collective_axes, **_)` — [`L352`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L352)
- `_run_state_lowering_rule(ctx: LoweringRuleContext, *args, jaxpr: jax_core.Jaxpr, which_linear: tuple[bool, ...], is_initialized: tuple[bool, ...])` — [`L3704`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L3704)
- `_run_state_resource_estimator(ctx: ResourceEstimatorContext, *args, jaxpr: jax_core.Jaxpr, **params)` — [`L344`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L344)
- `_scan_lowering_rule(ctx: LoweringRuleContext, *args, jaxpr: jax_core.ClosedJaxpr, length: int, reverse: bool, unroll: int, num_consts: int, num_carry: int)` — [`L3837`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L3837)
- `_scan_resource_estimator(ctx: ResourceEstimatorContext, *args, jaxpr: jax_core.ClosedJaxpr, **params)` — [`L290`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L290)
- `_select_n_lowering_rule(ctx: LoweringRuleContext, pred, *cases)` — [`L2440`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L2440)
- `_semaphore_read_lowering_rule(ctx: LoweringRuleContext, *args, args_tree)` — [`L4558`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L4558)
- `_sign_lowering_rule(ctx: LoweringRuleContext, x)` — [`L3112`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L3112)
- `_sin_lowering_rule(ctx: LoweringRuleContext, x, accuracy)` — [`L3000`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L3000)
- `_slice_lowering_rule(ctx: LoweringRuleContext, x, limit_indices, start_indices, strides)` — [`L2396`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L2396)
- `_slice_lowering_rule_wg(ctx: LoweringRuleContext, x, limit_indices, start_indices, strides)` — [`L2406`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L2406)
- `_square_lowering_rule(ctx: LoweringRuleContext, x)` — [`L2905`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L2905)
- `_squeeze_lowering_rule(ctx: LoweringRuleContext, x, dimensions)` — [`L3182`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L3182)
- `_squeeze_lowering_rule_wg(ctx: LoweringRuleContext, x, dimensions)` — [`L3189`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L3189)
- `_swap_lowering_rule(ctx: LoweringRuleContext, x_ref, value, *leaves, tree)` — [`L2226`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L2226)
- `_swap_lowering_rule_wg(ctx: LoweringRuleContext, x_smem, value, *leaves, tree)` — [`L2350`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L2350)
- `_tanh_lowering_rule(ctx: LoweringRuleContext, x, accuracy)` — [`L2945`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L2945)
- `_unary_warp_lowering_rule(impl)` — [`L2675`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L2675)
- `_unravel_program_id(block_id: ir.Value, axis: int, dimensions: tuple[int, ...], row_major: bool = False)` — [`L1356`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L1356) — Computes the program ID for axes compressed into one block dimension.
- `_while_lowering_rule(ctx: LoweringRuleContext, *args, cond_jaxpr, body_jaxpr, cond_nconsts, body_nconsts)` — [`L3929`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L3929)
- `_while_resource_estimator(ctx: ResourceEstimatorContext, *args, cond_jaxpr: jax_core.ClosedJaxpr, body_jaxpr: jax_core.ClosedJaxpr, **params)` — [`L298`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L298)
- `_wrap_in_custom_primitive_if_wg(ctx: LoweringRuleContext, operands: Sequence[ir.Value])` — [`L4580`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L4580) — Wraps the body in a CustomPrimitiveOp for warpgroup semantics.
- `_yielded_values(outs, avals)` — [`L4020`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L4020)
- `as_values(vals, avals)` — [`L3776`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L3776)
- `bcast(value, dtype)` — [`L4390`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L4390)
- `block_id_to_grid_id(ctx: LoweringRuleContext, block_ids: Sequence[ir.Value], axis_name: Hashable)` — [`L3374`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L3374)
- `body(launch_ctx: mgpu.LaunchContext, *buffers: Any)` — [`L964`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L964)
- `body_fn(indices, *refs)` — [`L861`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L861)
- `convert(ty, x)` — [`L2620`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L2620)
- `deco(fn)` — [`L245`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L245)
- `deco(fn)` — [`L1186`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L1186)
- `err_details(bm: pallas_core.BlockMapping)` — [`L706`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L706)
- `f_lowered(ctx: LoweringRuleContext, *args, **params)` — [`L1397`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L1397)
- `i32_attr(value: int)` — [`L3273`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L3273)
- `index_map(*indices)` — [`L753`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L753)
- `inner(ctx, *args)` — [`L1723`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L1723)
- `is_contained_within_block(operand: ir.OpOperand, block: ir.Block)` — [`L4147`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L4147)
- `jax_fn(device_id_val)` — [`L4426`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L4426)
- `loop(base_loop_index, body_args)` — [`L3790`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L3790)
- `lower_jaxpr_to_module(jax_mesh: mesh_lib.Mesh | None, axis_names: _AxisNames, grid: tuple[int, ...], block: tuple[int, int, int], cluster: tuple[int, ...], in_shapes: Sequence[jax_core.ShapedArray], out_shapes: Sequence[jax_core.ShapedArray], jaxpr: jax_core.Jaxpr, params: gpu_core.CompilerParams, consts=(), outer_traceback: xc.Traceback | None = None)` — [`L917`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L917)
- `lower_jaxpr_to_mosaic_gpu(module_ctx: ModuleContext, launch_ctx: mgpu.LaunchContext, jaxpr: jax_core.Jaxpr, args: Sequence[ir.Value], consts=())` — [`L1207`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L1207)
- `lower_pipelined_jaxpr_to_module(grid_mapping: pallas_core.GridMapping, gpu_mesh: gpu_core.Mesh | None, jax_mesh: mesh_lib.Mesh | None, jaxpr: jax_core.Jaxpr, params: gpu_core.CompilerParams, cost_estimate: pallas_core.CostEstimate | None, outer_traceback: xc.Traceback | None = None)` — [`L777`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L777)
- `merge_indexers(indexers: Sequence[indexing.NDIndexer])` — [`L4483`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L4483) — Merges multiple indexers into a single indexer.
- `pipeline_fn(*refs)` — [`L849`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L849)
- `read_env(atom: jax_core.Atom)` — [`L1216`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L1216)
- `ref_for_aval(aval: ShapedAbstractValue)` — [`L833`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L833)
- `register_lowering_rule(primitive: jax_core.Primitive, lowering_semantics: mgpu.LoweringSemantics, primitive_semantics: gpu_core.PrimitiveSemantics = gpu_core.PrimitiveSemantics.Warpgroup)` — [`L1181`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L1181) — documented in [jax-_src-pallas-mosaic_gpu-lowering](../../../../../concepts/jax-_src-pallas-mosaic_gpu-lowering.md)
- `scoped_pipeline_fn(*refs, scratch_refs)` — [`L860`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L860)
- `sign(x)` — [`L3113`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L3113)
- `write_env(var: jax_core.Var, val, require_value: bool = True)` — [`L1219`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L1219)

## Module values
- `AnyBarrier` — [`L124`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L124)
- `AnyBarrierRef` — [`L451`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L451)
- `CmpFPred` — [`L2828`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L2828)
- `CmpIPred` — [`L2827`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L2827)
- `CollectiveAxesType` — [`L106`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L106)
- `SMEM` — [`L81`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L81)
- `WARPGROUP_SIZE` — [`L82`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L82)
- `_copysign_p` — [`L3082`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L3082)
- `_resource_estimators` — [`L241`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L241)
- `f_impl` — [`L2768`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L2768)
- `f_pred` — [`L2851`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L2851)
- `impl` — [`L2815`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L2815)
- `mosaic_lowering_rules` — [`L1171`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L1171)
- `op` — [`L2768`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L2768)
- `partial` — [`L80`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L80)
- `rule` — [`L2795`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L2795)
- `semantics` — [`L2712`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L2712)
- `si_impl` — [`L2768`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L2768)
- `si_pred` — [`L2851`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L2851)
- `ui_impl` — [`L2768`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L2768)
- `ui_pred` — [`L2851`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L2851)
- `unsafe_map` — [`L77`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L77)
- `unsafe_zip` — [`L78`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/lowering.py#L78)


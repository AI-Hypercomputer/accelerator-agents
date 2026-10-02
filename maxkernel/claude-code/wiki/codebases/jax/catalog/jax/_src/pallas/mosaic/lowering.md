---
title: 'Module: jax/_src/pallas/mosaic/lowering.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/mosaic/lowering.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.mosaic.lowering`/
symbols:
  jaxpr_subcomp: jaxpr_subcomp().
  lower_jaxpr_into_pipelined_module: lower_jaxpr_into_pipelined_module().
  register_lowering_rule: register_lowering_rule().
  LoweringRuleContext: LoweringRuleContext#
  LoweringRuleContext.avals_in: LoweringRuleContext#avals_in.
  ShapedAbstractValue.dtype: ShapedAbstractValue#dtype.
  LoweringRuleContext.lowering_context: LoweringRuleContext#lowering_context.
  ShapedAbstractValue.shape: ShapedAbstractValue#shape.
  _check_block_mappings: _check_block_mappings().
  LoweringRuleContext.avals_out: LoweringRuleContext#avals_out.
  lower_jaxpr_to_transform_func.body_func: lower_jaxpr_to_transform_func().body_func().
  MosaicGridMapping._dimension_semantics: MosaicGridMapping#_dimension_semantics.
  _load_lowering_rule: _load_lowering_rule().
  lower_jaxpr_into_unpipelined_module: lower_jaxpr_into_unpipelined_module().
  _dtype_to_ir_type: _dtype_to_ir_type().
  aval_to_ir_type: aval_to_ir_type().
  _masked_swap_lowering_rule: _masked_swap_lowering_rule().
  _dot_general_lowering_rule: _dot_general_lowering_rule().
  LoweringRuleContext.aval_to_ir_type: LoweringRuleContext#aval_to_ir_type().
  MosaicGridMapping.__init__: MosaicGridMapping#__init__().
  _convert_element_type_lowering_rule: _convert_element_type_lowering_rule().
  _dma_start_lowering_rule: _dma_start_lowering_rule().
  ir_constant: ir_constant().
  _cmp_lowering_rule: _cmp_lowering_rule().
  _device_id_to_logical.jax_fn: _device_id_to_logical().jax_fn().
  _run_scoped_lowering_rule: _run_scoped_lowering_rule().
  _convert_helper: _convert_helper().
  _prng_key_load_lowering_rule: _prng_key_load_lowering_rule().
  _sign_lowering_rule._lower_fun: _sign_lowering_rule()._lower_fun().
  _lower_jaxpr_to_for_loop: _lower_jaxpr_to_for_loop().
  LoweringContext.dynamic_shape_replacement_fn: LoweringContext#dynamic_shape_replacement_fn.
  _scan_lowering_rule: _scan_lowering_rule().
  _semaphore_signal_lowering_rule: _semaphore_signal_lowering_rule().
  _transform_ref: _transform_ref().
  _gather_lowering_rule: _gather_lowering_rule().
  _mul_lowering_rule: _mul_lowering_rule().
  _div_lowering_rule: _div_lowering_rule().
  _rem_lowering_rule: _rem_lowering_rule().
  PipelinedLoweringContext.from_mosaic_grid_mapping: PipelinedLoweringContext#from_mosaic_grid_mapping().
  reduce_lowering_rule._lowering_rule: reduce_lowering_rule()._lowering_rule().
  _bcast: _bcast().
  _sub_lowering_rule: _sub_lowering_rule().
  _cond_lowering_rule: _cond_lowering_rule().
  _alloc_value: _alloc_value().
  _add_lowering_rule: _add_lowering_rule().
  _max_lowering_rule: _max_lowering_rule().
  _min_lowering_rule: _min_lowering_rule().
  _pow_lowering_rule: _pow_lowering_rule().
  _device_id_to_logical: _device_id_to_logical().
  _memory_space_to_mosaic_attribute: _memory_space_to_mosaic_attribute().
  _unstack_lowering_rule: _unstack_lowering_rule().
  _stochastic_round_lowering_rule: _stochastic_round_lowering_rule().
  _semaphore_read_lowering_rule: _semaphore_read_lowering_rule().
  _dma_wait_lowering_rule: _dma_wait_lowering_rule().
  _debug_print_rule: _debug_print_rule().
  LoweringRuleContext.block_shapes: LoweringRuleContext#block_shapes.
  prim: prim.
  lower_jaxpr_to_transform_func: lower_jaxpr_to_transform_func().
  _cmp_boolean_lowering_helper: _cmp_boolean_lowering_helper().
  _broadcast_in_dim_lowering_rule: _broadcast_in_dim_lowering_rule().
  _select_n_lowering_rule: _select_n_lowering_rule().
  _while_lowering_rule: _while_lowering_rule().
  _axis_index_rule: _axis_index_rule().
  _uncacheable_primitives._uncacheable_primitives: _uncacheable_primitives._uncacheable_primitives.
  UnpipelinedLoweringContext.from_mesh: UnpipelinedLoweringContext#from_mesh().
  _emit_pallas_lowering_rule_as_fun: _emit_pallas_lowering_rule_as_fun().
  _get_mesh_shape_and_semantics: _get_mesh_shape_and_semantics().
  lower_fun.f_lowered: lower_fun().f_lowered().
  _bitcast_memref: _bitcast_memref().
  _not_lowering_rule: _not_lowering_rule().
  _get_lowering_rule: _get_lowering_rule().
  _swap_lowering_rule: _swap_lowering_rule().
  _stack_lowering_rule: _stack_lowering_rule().
  _neg_lowering_rule: _neg_lowering_rule().
  _sign_lowering_rule: _sign_lowering_rule().
  _custom_jvp_call_lowering_rule: _custom_jvp_call_lowering_rule().
  _pack_elementwise_lowering_rule: _pack_elementwise_lowering_rule().
  _bitcast_convert_type_lowering_rule: _bitcast_convert_type_lowering_rule().
  _reduce_index_helper: _reduce_index_helper().
  _reduce_sum_lowering_rule: _reduce_sum_lowering_rule.
  lower_fun: lower_fun().
  _squeeze_lowering_rule: _squeeze_lowering_rule().
  _abs_lowering_rule: _abs_lowering_rule().
  _logistic_lowering_rule: _logistic_lowering_rule().
  _or_lowering_rule: _or_lowering_rule().
  _xor_lowering_rule: _xor_lowering_rule().
  _shift_left_lowering_rule: _shift_left_lowering_rule().
  _shift_right_arithmetic_lowering_rule: _shift_right_arithmetic_lowering_rule().
  _shift_right_logical_lowering_rule: _shift_right_logical_lowering_rule().
  _ensure_mlir_value: _ensure_mlir_value().
  _maybe_cast_load_to_bool: _maybe_cast_load_to_bool().
  _reshape_memref: _reshape_memref().
  _reshape_lowering_rule: _reshape_lowering_rule().
  _split_lowering_rule: _split_lowering_rule().
  _atan2_lowering_rule: _atan2_lowering_rule().
  _pjit_lowering_rule: _pjit_lowering_rule().
  _unpack_elementwise_lowering_rule: _unpack_elementwise_lowering_rule().
  random_fold_in_lowering: random_fold_in_lowering().
  ShapedAbstractValue: ShapedAbstractValue#
  LoweringContext.kernel_type: LoweringContext#kernel_type.
  MosaicGridMapping._prepare_mesh_info: MosaicGridMapping#_prepare_mesh_info().
  lower_jaxpr_to_pipelined_module: lower_jaxpr_to_pipelined_module().
  _indexer_to_start_size_stride: _indexer_to_start_size_stride().
  _reduce_max_lowering_rule: _reduce_max_lowering_rule.
  _reduce_min_lowering_rule: _reduce_min_lowering_rule.
  _lower_multiref_transformed_ref: _lower_multiref_transformed_ref().
  _canonicalize_transforms_to_indexer: _canonicalize_transforms_to_indexer().
  _primitives_needing_grid._primitives_needing_grid: _primitives_needing_grid._primitives_needing_grid.
  _iota_lowering_rule: _iota_lowering_rule().
  _square_lowering_rule: _square_lowering_rule().
  _custom_vjp_call_lowering_rule: _custom_vjp_call_lowering_rule().
  _num_programs_lowering_rule: _num_programs_lowering_rule().
  _semaphore_wait_lowering_rule: _semaphore_wait_lowering_rule().
  _lower_transformed_refs: _lower_transformed_refs().
  random_bits_lowering: random_bits_lowering().
  _join_key_lowering_rule: _join_key_lowering_rule().
  _check_lowering_rule: _check_lowering_rule().
  _dim_as_value_lowering: _dim_as_value_lowering().
  _matmul_pop_lowering_rule: _matmul_pop_lowering_rule().
  MosaicGridMapping._get_nonlocal_axis_names: MosaicGridMapping#_get_nonlocal_axis_names().
  _lower_jaxpr_to_func_common: _lower_jaxpr_to_func_common().
  _cmpsi_lowering_types: _cmpsi_lowering_types.
  _check_block_mappings.err_details: _check_block_mappings().err_details().
  _emit_pallas_lowering_rule_as_fun.body_builder: _emit_pallas_lowering_rule_as_fun().body_builder().
  _cmpui_lowering_types: _cmpui_lowering_types.
  _cmpf_lowering_types: _cmpf_lowering_types.
  _program_id_lowering_rule: _program_id_lowering_rule().
  TPUMemorySpace: TPUMemorySpace.
  should_physicalize_dtype: should_physicalize_dtype().
  LoweringContext.replace: LoweringContext#replace.
  _reduce_or_lowering_rule: _reduce_or_lowering_rule().
  _and_lowering_rule: _and_lowering_rule().
  _is_finite_lowering_rule: _is_finite_lowering_rule().
  _clamp_lowering_rule: _clamp_lowering_rule().
  _slice_lowering_rule: _slice_lowering_rule().
  _get_barrier_semaphore_rule: _get_barrier_semaphore_rule().
  _pad_lowering_rule: _pad_lowering_rule().
  LoweringDynamicShapeEnv.to_placeholder: LoweringDynamicShapeEnv#to_placeholder().
  MosaicGridMapping.grid: MosaicGridMapping#grid.
  MosaicGridMapping.has_communication: MosaicGridMapping#has_communication().
  lower_jaxpr_into_pipelined_module.dynamic_shape_replacement_fn: lower_jaxpr_into_pipelined_module().dynamic_shape_replacement_fn().
  _index_to_start_size_stride: _index_to_start_size_stride().
  _slice_memref: _slice_memref().
  _lower_while_via_fori: _lower_while_via_fori().
  _lower_single_transformed_ref: _lower_single_transformed_ref().
  LoweringContext: LoweringContext#
  LoweringContext.grid_name_context: LoweringContext#grid_name_context().
  NDIndexer: NDIndexer.
  _physical_aval: _physical_aval().
  _transpose_lowering_rule: _transpose_lowering_rule().
  _nextafter_lowering_rule: _nextafter_lowering_rule().
  _integer_pow_lowering_rule: _integer_pow_lowering_rule().
  _population_count_lowering_rule: _population_count_lowering_rule().
  _erf_inv_lowering_rule: _erf_inv_lowering_rule().
  _bitcast_lowering_rule: _bitcast_lowering_rule().
  _empty_ref_lowering_rule: _empty_ref_lowering_rule().
  _prng_seed_lowering_rule: _prng_seed_lowering_rule().
  _prng_random_bits_lowering_rule: _prng_random_bits_lowering_rule().
  random_unwrap_lowering: random_unwrap_lowering().
  _split_key_lowering_rule: _split_key_lowering_rule().
  _iota_2x32_shape_lowering: _iota_2x32_shape_lowering().
  _touch_lowering_rule: _touch_lowering_rule().
  lowering_rules.lowering_rules: lowering_rules.lowering_rules.
  AnyMemorySpace: AnyMemorySpace.
  LoweringContext.user_grid_indices: LoweringContext#user_grid_indices.
  _maybe_cast_to_index: _maybe_cast_to_index().
  lower_with_transformed_refs: lower_with_transformed_refs().
  jaxpr_subcomp.read_block_shape: jaxpr_subcomp().read_block_shape().
  _lower_jaxpr_to_for_loop.body_builder: _lower_jaxpr_to_for_loop().body_builder().
  _aval_to_log_format_spec: _aval_to_log_format_spec().
  _maybe_physicalize_block_shape: _maybe_physicalize_block_shape().
  _canonicalize_dimension_semantic: _canonicalize_dimension_semantic().
  SEMAPHORE: SEMAPHORE.
  _multiple_of_lowering_rule: _multiple_of_lowering_rule().
  _stage_lowering_rule: _stage_lowering_rule().
  _concatenate_lowering_rule: _concatenate_lowering_rule().
  _argmax_lowering_rule: _argmax_lowering_rule().
  _argmin_lowering_rule: _argmin_lowering_rule().
  _exp_lowering_rule: _exp_lowering_rule().
  _reshard_lowering_rule: _reshard_lowering_rule().
  _delay_rule: _delay_rule().
  _threefry2x32_lowering: _threefry2x32_lowering().
  _platform_index_lowering: _platform_index_lowering().
  _trace_value_lowering_rule: _trace_value_lowering_rule().
  _dma_unflatten: _dma_unflatten.
  _get_aval_physical_dtype_shape: _get_aval_physical_dtype_shape().
  _maybe_cast_store_to_memref_type: _maybe_cast_store_to_memref_type().
  _select_to_ifop: _select_to_ifop().
  DynamicShapeReplacementFn: DynamicShapeReplacementFn.
  register_lowering_rule.decorator: register_lowering_rule().decorator().
  lower_jaxpr_into_pipelined_module.ctx_factory: lower_jaxpr_into_pipelined_module().ctx_factory().
  lower_jaxpr_into_unpipelined_module.ctx_factory: lower_jaxpr_into_unpipelined_module().ctx_factory().
  REDUCE_MAX_KINDS: REDUCE_MAX_KINDS.
  REDUCE_MIN_KINDS: REDUCE_MIN_KINDS.
  REDUCE_SUM_KINDS: REDUCE_SUM_KINDS.
  _lower_single_transformed_ref.new_f: _lower_single_transformed_ref().new_f().
  random_bits_lowering.new_lowering: random_bits_lowering().new_lowering().
  _iota_2x32_shape_lowering._lower_fun: _iota_2x32_shape_lowering()._lower_fun().
  ANY: ANY.
  MosaicGridMapping.grid_names: MosaicGridMapping#grid_names.
  MosaicGridMapping.vmapped_dims: MosaicGridMapping#vmapped_dims.
  jaxpr_subcomp.write_env: jaxpr_subcomp().write_env().
  _fold_and_get_constant_value: _fold_and_get_constant_value().
  LoweringContext.dynamic_shape_env: LoweringContext#dynamic_shape_env.
  LoweringRuleContext.is_cloud_tpu_older_than: LoweringRuleContext#is_cloud_tpu_older_than().
  _reduce_and_lowering_rule: _reduce_and_lowering_rule().
  _broadcast_to_lowering_rule: _broadcast_to_lowering_rule().
  _stop_gradient_lowering_rule: _stop_gradient_lowering_rule().
  _rsqrt_lowering_rule: _rsqrt_lowering_rule().
  _sqrt_lowering_rule: _sqrt_lowering_rule().
  _exp2_lowering_rule: _exp2_lowering_rule().
  _sin_lowering_rule: _sin_lowering_rule().
  _cos_lowering_rule: _cos_lowering_rule().
  _tan_lowering_rule: _tan_lowering_rule().
  _tanh_lowering_rule: _tanh_lowering_rule().
  _log_lowering_rule: _log_lowering_rule().
  _log1p_lowering_rule: _log1p_lowering_rule().
  _erf_lowering_rule: _erf_lowering_rule().
  _round_lowering_rule: _round_lowering_rule().
  _ceil_lowering_rule: _ceil_lowering_rule().
  _floor_lowering_rule: _floor_lowering_rule().
  _clz_lowering_rule: _clz_lowering_rule().
  _lower_jaxpr_to_for_loop._run_body: _lower_jaxpr_to_for_loop()._run_body().
  _debug_callback_lowering_rule: _debug_callback_lowering_rule().
  _tile_lowering_rule: _tile_lowering_rule().
  _roll_lowering_rule: _roll_lowering_rule().
  _reciprocal_lowering_rule: _reciprocal_lowering_rule().
  random_seed_lowering: random_seed_lowering().
  random_wrap_lowering: random_wrap_lowering().
  _matmul_push_rhs_lowering_rule: _matmul_push_rhs_lowering_rule().
  _matmul_acc_lhs_lowering_rule: _matmul_acc_lhs_lowering_rule().
  LoweringContext.grid_names: LoweringContext#grid_names.
  LoweringRuleContext.forward_compatible: LoweringRuleContext#forward_compatible().
  jaxpr_subcomp.read_env: jaxpr_subcomp().read_env().
  _fold: _fold().
  _pad_lowering_rule._pad: _pad_lowering_rule()._pad().
  KeyScalarBundle: KeyScalarBundle#
  MosaicGridMapping.mesh_info: MosaicGridMapping#mesh_info.
  LoweringRuleContext.replace: LoweringRuleContext#replace.
  MosaicGridMapping.get_dimension_semantics: MosaicGridMapping#get_dimension_semantics().
  reduce_lowering_rule._lowering_rule._proxy_fun: reduce_lowering_rule()._lowering_rule()._proxy_fun().
  REDUCE_MAX_IDENTITY: REDUCE_MAX_IDENTITY.
  REDUCE_MIN_IDENTITY: REDUCE_MIN_IDENTITY.
  REDUCE_SUM_IDENTITY: REDUCE_SUM_IDENTITY.
  _reduce_and_lowering_rule._proxy_reduce: _reduce_and_lowering_rule()._proxy_reduce().
  _reduce_or_lowering_rule._proxy_reduce: _reduce_or_lowering_rule()._proxy_reduce().
  _iota_lowering_rule._1d_iota_helper: _iota_lowering_rule()._1d_iota_helper().
  _clamp: _clamp().
  LoweringException: LoweringException#
  _get_ref_and_transforms: _get_ref_and_transforms.
  LoweringContext.name_stack: LoweringContext#name_stack.
  LoweringContext.jax_mesh_context: LoweringContext#jax_mesh_context.
  LoweringContext.lowering_cache: LoweringContext#lowering_cache.
  LoweringContext.grid_rank: LoweringContext#grid_rank().
  MosaicGridMapping.jaxpr: MosaicGridMapping#jaxpr.
  MosaicGridMapping.block_mappings: MosaicGridMapping#block_mappings.
  reduce_lowering_rule: reduce_lowering_rule().
  VMEM: VMEM.
  SMEM: SMEM.
  HOST: HOST.
  BOOL_MEMREF_TYPE: BOOL_MEMREF_TYPE.
  LoweringDynamicShapeEnv.dim_expr_to_placeholder: LoweringDynamicShapeEnv#dim_expr_to_placeholder.
  LoweringContext.vmapped_dims: LoweringContext#vmapped_dims.
  LoweringContext.get_num_programs: LoweringContext#get_num_programs().
  ShapedAbstractValue.ndim: ShapedAbstractValue#ndim().
  ShapedAbstractValue.update: ShapedAbstractValue#update().
  MosaicGridMapping.grid_types: MosaicGridMapping#grid_types.
  PallasLoweringCacheKey: PallasLoweringCacheKey#
  LoweringDynamicShapeEnv.snapshot: LoweringDynamicShapeEnv#snapshot().
  LoweringContext.block_shapes: LoweringContext#block_shapes.
  MosaicGridMapping.operand_block_shapes: MosaicGridMapping#operand_block_shapes.
  _get_dimension_semantics: _get_dimension_semantics().
  _lower_jaxpr_to_func_common.body_func: _lower_jaxpr_to_func_common().body_func().
  _compute_name_stack_updates: _compute_name_stack_updates().
  LoweringDynamicShapeEnv: LoweringDynamicShapeEnv#
  LoweringContext.grid_sizes: LoweringContext#grid_sizes.
  LoweringContext.needs_layout_passes: LoweringContext#needs_layout_passes.
  T: T.
  MosaicGridMapping.scalar_prefetch_types: MosaicGridMapping#scalar_prefetch_types.
  MosaicGridMapping.get_grid_indices: MosaicGridMapping#get_grid_indices.
  KeyScalarBundle.scalars: KeyScalarBundle#scalars.
  MosaicGridMapping: MosaicGridMapping#
  PHYSICAL_EXTENDED_DTYPES: PHYSICAL_EXTENDED_DTYPES.
  _dma_tree_leaves: _dma_tree_leaves.
  PipelinedLoweringContext: PipelinedLoweringContext#
  UnpipelinedLoweringContext: UnpipelinedLoweringContext#
  PallasLoweringCacheKey.primitive: PallasLoweringCacheKey#primitive.
  PallasLoweringCacheKey.kernel_type: PallasLoweringCacheKey#kernel_type.
  PallasLoweringCacheKey.avals_in: PallasLoweringCacheKey#avals_in.
  PallasLoweringCacheKey.avals_out: PallasLoweringCacheKey#avals_out.
  PallasLoweringCacheKey.block_shapes: PallasLoweringCacheKey#block_shapes.
  lower_jaxpr_into_pipelined_module.get_jaxpr_indices: lower_jaxpr_into_pipelined_module().get_jaxpr_indices().
  _make_index: _make_index().
  jax_dot_dims_to_tpu_dot_dot_dims: jax_dot_dims_to_tpu_dot_dot_dims().
  _poison_memref: _poison_memref().
  _threefry2x32_lowering._lower_fun: _threefry2x32_lowering()._lower_fun().
  unsafe_zip: unsafe_zip.
  LoweringDynamicShapeEnv.placeholder_to_dim_expr: LoweringDynamicShapeEnv#placeholder_to_dim_expr.
  LoweringContext.forward_compatible: LoweringContext#forward_compatible.
  LoweringContext.backend: LoweringContext#backend.
  MosaicGridMapping.scratch_types: MosaicGridMapping#scratch_types.
  MosaicGridMapping.scalar_prefetch_block_shapes: MosaicGridMapping#scalar_prefetch_block_shapes.
  FoldingError: FoldingError#
  LoweringContext.get_program_id: LoweringContext#get_program_id().
  unsafe_map: unsafe_map.
  LoweringContext.fuse_transposed_lhs_in_matmul: LoweringContext#fuse_transposed_lhs_in_matmul.
  ShapedAbstractValue.weak_type: ShapedAbstractValue#weak_type.
  UncacheablePrimitiveError: UncacheablePrimitiveError#
  _emit_detached_func: _emit_detached_func().
  skip_mlir_conversions: skip_mlir_conversions.
  MosaicGridMapping.operand_types: MosaicGridMapping#operand_types.
  MosaicGridMapping.scratch_block_shapes: MosaicGridMapping#scratch_block_shapes.
  KeyScalarBundle.key_shape: KeyScalarBundle#key_shape.
  MLIR_DYNAMIC: MLIR_DYNAMIC.
  DIM_UPPER_BOUND: DIM_UPPER_BOUND.
  DIM_LOWER_BOUND: DIM_LOWER_BOUND.
  LoweringContext.traceback_caches: LoweringContext#traceback_caches.
  PallasLoweringCacheKey.params: PallasLoweringCacheKey#params.
  PallasLoweringCacheKey.grid_arity: PallasLoweringCacheKey#grid_arity.
  PallasLoweringCacheKey.forward_compatible: PallasLoweringCacheKey#forward_compatible.
  PallasLoweringCacheKey.fuse_transposed_lhs_in_matmul: PallasLoweringCacheKey#fuse_transposed_lhs_in_matmul.
  PallasLoweringCacheKey.grid_sizes: PallasLoweringCacheKey#grid_sizes.
  PallasLoweringCacheKey.vmapped_dims: PallasLoweringCacheKey#vmapped_dims.
  PallasLoweringCacheKey.dynamic_shape_env: PallasLoweringCacheKey#dynamic_shape_env.
  MosaicGridMapping._get_grid_indices: MosaicGridMapping#_get_grid_indices().
  MosaicGridMapping.maybe_compress_grid: MosaicGridMapping#maybe_compress_grid().
  _get_dimension_semantics._get_semantics: _get_dimension_semantics()._get_semantics().
  jax_dot_dims_to_tpu_dot_dot_dims.format_dims: jax_dot_dims_to_tpu_dot_dot_dims().format_dims().
  _dma_start_lowering_rule._dma_start: _dma_start_lowering_rule()._dma_start().
  _dma_wait_lowering_rule._dma_wait: _dma_wait_lowering_rule()._dma_wait().
  partial: partial.
  LoweringDynamicShapeEnv.__init__: LoweringDynamicShapeEnv#__init__().
  ShapedAbstractValue.size: ShapedAbstractValue#size().
---
# Module: [`jax/_src/pallas/mosaic/lowering.py`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py)

## Classes
### `FoldingError`  ·  implements/extends Exception
- def: [`jax/_src/pallas/mosaic/lowering.py:3249`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3249)
- signature: `class FoldingError(Exception):`
- used by: [`_fold_and_get_constant_value`](lowering.md#_fold_and_get_constant_value), [`_fold`](lowering.md#_fold)

### `KeyScalarBundle`
- def: [`jax/_src/pallas/mosaic/lowering.py:2171`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L2171)
- members:
  - `key_shape` — [`L2184`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L2184)
  - `scalars` — [`L2185`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L2185)
- used by: [`_prng_key_load_lowering_rule`](lowering.md#_prng_key_load_lowering_rule), [`_ensure_mlir_value`](lowering.md#_ensure_mlir_value), [`_join_key_lowering_rule`](lowering.md#_join_key_lowering_rule), [`_prng_seed_lowering_rule`](lowering.md#_prng_seed_lowering_rule), [`_split_key_lowering_rule`](lowering.md#_split_key_lowering_rule), [`write_env`](lowering.md#jaxpr_subcomp.write_env)

### `LoweringContext`
- def: [`jax/_src/pallas/mosaic/lowering.py:203`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L203)
- signature: `class LoweringContext:`
- members:
  - `get_num_programs(self, axis: int)` — [`L246`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L246)
  - `get_program_id(self, axis: int)` — [`L241`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L241)
  - `grid_name_context(self)` — [`L228`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L228)
  - `grid_rank(self)` — [`L224`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L224)
  - `backend` — [`L214`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L214)
  - `block_shapes` — [`L208`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L208)
  - `dynamic_shape_env` — [`L217`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L217)
  - `dynamic_shape_replacement_fn` — [`L215`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L215)
  - `forward_compatible` — [`L213`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L213)
  - `fuse_transposed_lhs_in_matmul` — [`L219`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L219)
  - `grid_names` — [`L205`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L205)
  - `grid_sizes` — [`L204`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L204)
  - `jax_mesh_context` — [`L210`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L210)
  - `kernel_type` — [`L211`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L211)
  - `lowering_cache` — [`L216`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L216)
  - `name_stack` — [`L209`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L209)
  - `needs_layout_passes` — [`L218`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L218)
  - `replace` — [`L221`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L221)
  - `traceback_caches` — [`L212`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L212)
  - `user_grid_indices` — [`L207`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L207)
  - `vmapped_dims` — [`L206`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L206)
- uses (calls/refs, reference-scoped): [`CoreType`](core.md#CoreType), [`Squeezed`](../core.md#Squeezed), [`NameStack`](../../source_info_util.md#NameStack), [`extend_axis_env_nd`](../../core.md#extend_axis_env_nd), [`MeshInfo`](../utils.md#MeshInfo), [`DynamicShapeReplacementFn`](lowering.md#DynamicShapeReplacementFn), [`PallasLoweringCacheKey`](lowering.md#PallasLoweringCacheKey), [`LoweringDynamicShapeEnv`](lowering.md#LoweringDynamicShapeEnv), [`PipelinedLoweringContext`](lowering.md#PipelinedLoweringContext), [`UnpipelinedLoweringContext`](lowering.md#UnpipelinedLoweringContext)
- used by: [`jaxpr_subcomp`](lowering.md#jaxpr_subcomp), [`lowering_context`](lowering.md#LoweringRuleContext.lowering_context), [`body_func`](lowering.md#lower_jaxpr_to_transform_func.body_func), [`_load_lowering_rule`](lowering.md#_load_lowering_rule), [`_load_lowering_rule`](sc_lowering.md#_load_lowering_rule), [`_store_lowering_rule`](sc_lowering.md#_store_lowering_rule), [`_masked_swap_lowering_rule`](lowering.md#_masked_swap_lowering_rule), [`_dot_general_lowering_rule`](lowering.md#_dot_general_lowering_rule), [`_dma_start_lowering_rule`](sc_lowering.md#_dma_start_lowering_rule), [`aval_to_ir_type`](lowering.md#LoweringRuleContext.aval_to_ir_type), [`_dma_start_lowering_rule`](lowering.md#_dma_start_lowering_rule), [`_cmp_lowering_rule`](lowering.md#_cmp_lowering_rule), [`jax_fn`](lowering.md#_device_id_to_logical.jax_fn), [`_run_scoped_lowering_rule`](lowering.md#_run_scoped_lowering_rule), [`_dma_wait_lowering_rule`](sc_lowering.md#_dma_wait_lowering_rule), [`_lower_jaxpr_to_for_loop`](lowering.md#_lower_jaxpr_to_for_loop), [`_semaphore_signal_lowering_rule`](lowering.md#_semaphore_signal_lowering_rule), [`_div_lowering_rule`](lowering.md#_div_lowering_rule), [`_gather_lowering_rule`](lowering.md#_gather_lowering_rule), [`_mul_lowering_rule`](lowering.md#_mul_lowering_rule), [`_rem_lowering_rule`](lowering.md#_rem_lowering_rule), [`_cond_lowering_rule`](lowering.md#_cond_lowering_rule), [`_sub_lowering_rule`](lowering.md#_sub_lowering_rule), [`_add_lowering_rule`](lowering.md#_add_lowering_rule), [`_device_id_to_logical`](lowering.md#_device_id_to_logical), [`_max_lowering_rule`](lowering.md#_max_lowering_rule), [`_min_lowering_rule`](lowering.md#_min_lowering_rule), [`_pow_lowering_rule`](lowering.md#_pow_lowering_rule), [`_stochastic_round_lowering_rule`](lowering.md#_stochastic_round_lowering_rule), [`_unstack_lowering_rule`](lowering.md#_unstack_lowering_rule), [`_axis_index_rule`](lowering.md#_axis_index_rule), [`_broadcast_in_dim_lowering_rule`](lowering.md#_broadcast_in_dim_lowering_rule), [`_while_lowering_rule`](lowering.md#_while_lowering_rule), [`_emit_pallas_lowering_rule_as_fun`](lowering.md#_emit_pallas_lowering_rule_as_fun), [`f_lowered`](lowering.md#lower_fun.f_lowered), [`_custom_jvp_call_lowering_rule`](lowering.md#_custom_jvp_call_lowering_rule), [`_pack_elementwise_lowering_rule`](lowering.md#_pack_elementwise_lowering_rule), [`_stack_lowering_rule`](lowering.md#_stack_lowering_rule), [`_custom_fusion_mosaic_lowering_rule`](../fuser/custom_fusion_lib.md#_custom_fusion_mosaic_lowering_rule), [`_or_lowering_rule`](lowering.md#_or_lowering_rule)  (+23 more)

### `LoweringDynamicShapeEnv`
- def: [`jax/_src/pallas/mosaic/lowering.py:167`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L167)
- signature: `class LoweringDynamicShapeEnv:`
- members:
  - `snapshot(self)` — [`L173`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L173)
  - `to_placeholder(self, dim_expr: Any)` — [`L176`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L176)
  - `dim_expr_to_placeholder` — [`L170`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L170)
  - `placeholder_to_dim_expr` — [`L171`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L171)
- protocol/private: `__init__`[`L169`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L169)
- uses (calls/refs, reference-scoped): [`is_constant_dim`](../../core.md#is_constant_dim), [`DIM_LOWER_BOUND`](lowering.md#DIM_LOWER_BOUND), [`DIM_UPPER_BOUND`](lowering.md#DIM_UPPER_BOUND)
- used by: [`jaxpr_subcomp`](lowering.md#jaxpr_subcomp), [`lower_jaxpr_into_pipelined_module`](lowering.md#lower_jaxpr_into_pipelined_module), [`from_mosaic_grid_mapping`](lowering.md#PipelinedLoweringContext.from_mosaic_grid_mapping), [`lower_jaxpr_to_transform_func`](lowering.md#lower_jaxpr_to_transform_func), [`dynamic_shape_replacement_fn`](lowering.md#lower_jaxpr_into_pipelined_module.dynamic_shape_replacement_fn), [`dynamic_shape_env`](lowering.md#LoweringContext.dynamic_shape_env)

### `LoweringException`
- def: [`jax/_src/pallas/mosaic/lowering.py:1669`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L1669)
- used by: [`jaxpr_subcomp`](lowering.md#jaxpr_subcomp), [`lower_jaxpr_into_pipelined_module`](lowering.md#lower_jaxpr_into_pipelined_module)

### `LoweringRuleContext`
- def: [`jax/_src/pallas/mosaic/lowering.py:465`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L465) — documented in [jax-_src-pallas-mosaic-lowering](../../../../../concepts/jax-_src-pallas-mosaic-lowering.md)
- members:
  - `aval_to_ir_type(self, aval, *, shape=None, memory_space=None, is_kernel_boundary=False, allow_extended_types=True)` — [`L484`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L484)
  - `forward_compatible(self)` — [`L474`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L474)
  - `is_cloud_tpu_older_than(self, year: int, month: int, day: int)` — [`L477`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L477)
  - `avals_in` — [`L467`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L467)
  - `avals_out` — [`L468`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L468)
  - `block_shapes` — [`L469`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L469)
  - `lowering_context` — [`L466`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L466)
  - `replace` — [`L471`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L471)
- uses (calls/refs, reference-scoped): [`aval_to_ir_type`](lowering.md#aval_to_ir_type), [`Squeezed`](../core.md#Squeezed), [`dynamic_shape_replacement_fn`](lowering.md#LoweringContext.dynamic_shape_replacement_fn), [`ShapedAbstractValue`](lowering.md#ShapedAbstractValue), [`kernel_type`](lowering.md#LoweringContext.kernel_type), [`LoweringContext`](lowering.md#LoweringContext), [`is_cloud_tpu_older_than`](../../cloud_tpu_init.md#is_cloud_tpu_older_than), [`backend`](lowering.md#LoweringContext.backend), [`forward_compatible`](lowering.md#LoweringContext.forward_compatible)
- used by: [`jaxpr_subcomp`](lowering.md#jaxpr_subcomp), [`_load_lowering_rule`](lowering.md#_load_lowering_rule), [`_load_lowering_rule`](sc_lowering.md#_load_lowering_rule), [`_store_lowering_rule`](sc_lowering.md#_store_lowering_rule), [`_masked_swap_lowering_rule`](lowering.md#_masked_swap_lowering_rule), [`_dot_general_lowering_rule`](lowering.md#_dot_general_lowering_rule), [`_dma_start_lowering_rule`](sc_lowering.md#_dma_start_lowering_rule), [`_convert_element_type_lowering_rule`](lowering.md#_convert_element_type_lowering_rule), [`_dma_start_lowering_rule`](lowering.md#_dma_start_lowering_rule), [`_cmp_lowering_rule`](lowering.md#_cmp_lowering_rule), [`jax_fn`](lowering.md#_device_id_to_logical.jax_fn), [`_run_scoped_lowering_rule`](lowering.md#_run_scoped_lowering_rule), [`_prng_key_load_lowering_rule`](lowering.md#_prng_key_load_lowering_rule), [`_dma_wait_lowering_rule`](sc_lowering.md#_dma_wait_lowering_rule), [`_lower_jaxpr_to_for_loop`](lowering.md#_lower_jaxpr_to_for_loop), [`_scan_lowering_rule`](lowering.md#_scan_lowering_rule), [`_semaphore_signal_lowering_rule`](lowering.md#_semaphore_signal_lowering_rule), [`_div_lowering_rule`](lowering.md#_div_lowering_rule), [`_gather_lowering_rule`](lowering.md#_gather_lowering_rule), [`_mul_lowering_rule`](lowering.md#_mul_lowering_rule), [`_rem_lowering_rule`](lowering.md#_rem_lowering_rule), [`_scatter_lowering_rule`](sc_primitives.md#_scatter_lowering_rule), [`_cond_lowering_rule`](lowering.md#_cond_lowering_rule), [`_lowering_rule`](lowering.md#reduce_lowering_rule._lowering_rule), [`_sub_lowering_rule`](lowering.md#_sub_lowering_rule), [`_alloc_value`](lowering.md#_alloc_value), [`_add_lowering_rule`](lowering.md#_add_lowering_rule), [`_device_id_to_logical`](lowering.md#_device_id_to_logical), [`_gather_lowering_rule`](sc_primitives.md#_gather_lowering_rule), [`_max_lowering_rule`](lowering.md#_max_lowering_rule), [`_min_lowering_rule`](lowering.md#_min_lowering_rule), [`_pow_lowering_rule`](lowering.md#_pow_lowering_rule), [`_debug_print_rule`](lowering.md#_debug_print_rule), [`_dma_wait_lowering_rule`](lowering.md#_dma_wait_lowering_rule), [`_semaphore_read_lowering_rule`](lowering.md#_semaphore_read_lowering_rule), [`_stochastic_round_lowering_rule`](lowering.md#_stochastic_round_lowering_rule), [`_unstack_lowering_rule`](lowering.md#_unstack_lowering_rule), [`LoweringRuleContext`](sc_lowering.md#LoweringRuleContext), [`_axis_index_rule`](lowering.md#_axis_index_rule), [`_broadcast_in_dim_lowering_rule`](lowering.md#_broadcast_in_dim_lowering_rule)  (+115 more)

### `MosaicGridMapping`
- def: [`jax/_src/pallas/mosaic/lowering.py:698`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L698)
- members:
  - `get_dimension_semantics(self)` — [`L869`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L869)
  - `has_communication(self)` — [`L844`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L844)
  - `maybe_compress_grid(self)` — [`L837`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L837)
  - `block_mappings` — [`L702`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L702)
  - `get_grid_indices` — [`L712`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L712)
  - `grid` — [`L699`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L699)
  - `grid_names` — [`L700`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L700)
  - `grid_types` — [`L707`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L707)
  - `jaxpr` — [`L701`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L701)
  - `mesh_info` — [`L711`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L711)
  - `operand_block_shapes` — [`L709`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L709)
  - `operand_types` — [`L705`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L705)
  - `scalar_prefetch_block_shapes` — [`L708`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L708)
  - `scalar_prefetch_types` — [`L704`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L704)
  - `scratch_block_shapes` — [`L710`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L710)
  - `scratch_types` — [`L706`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L706)
  - `vmapped_dims` — [`L703`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L703)
- protocol/private: `__init__`[`L714`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L714), `_dimension_semantics`[`L752`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L752), `_get_grid_indices`[`L809`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L809), `_get_nonlocal_axis_names`[`L846`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L846), `_prepare_mesh_info`[`L820`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L820)
- uses (calls/refs, reference-scoped): [`jaxpr`](../../core.md#ClosedJaxpr.jaxpr), [`Jaxpr`](../../core.md#Jaxpr), [`aval`](../../core.md#Var.aval), [`CoreType`](core.md#CoreType), [`invars`](../../core.md#Jaxpr.invars), [`Mesh`](../../mesh.md#Mesh), [`shape`](lowering.md#ShapedAbstractValue.shape), [`aval_to_ir_type`](lowering.md#aval_to_ir_type), [`Squeezed`](../core.md#Squeezed), [`block_shape`](../core.md#BlockMapping.block_shape), [`block_mappings`](../core.md#GridMapping.block_mappings), [`axis_names`](../../mesh.md#Mesh.axis_names), [`effects`](../../core.md#Jaxpr.effects), [`grid`](../core.md#GridMapping.grid), [`NamedAxisEffect`](../../core.md#NamedAxisEffect), [`index_map_jaxpr`](../core.md#BlockMapping.index_map_jaxpr), [`GridMapping`](../core.md#GridMapping), [`_get_block_dim_size`](../core.md#_get_block_dim_size), [`ShapedAbstractValue`](lowering.md#ShapedAbstractValue), [`BlockMapping`](../core.md#BlockMapping), [`index_map_grid_aval`](../core.md#index_map_grid_aval), [`vmapped_dims`](../core.md#GridMapping.vmapped_dims), [`DimensionSemantics`](core.md#DimensionSemantics), [`squeezed`](../core.md#squeezed), [`grid_names`](../core.md#GridMapping.grid_names), [`_get_block_shape`](../core.md#_get_block_shape), [`GridMappingGrid`](../core.md#GridMappingGrid), [`_canonicalize_dimension_semantic`](lowering.md#_canonicalize_dimension_semantic), [`_maybe_physicalize_block_shape`](lowering.md#_maybe_physicalize_block_shape), [`name`](../../core.md#NamedAxisEffect.name), [`MeshInfo`](../utils.md#MeshInfo), [`slice_block_ops`](../core.md#GridMapping.slice_block_ops), [`DynamicShapeReplacementFn`](lowering.md#DynamicShapeReplacementFn), [`AbstractSemaphore`](core.md#AbstractSemaphore), [`CommsEffect`](../core.md#CommsEffect), [`from_mesh`](../utils.md#MeshInfo.from_mesh), [`slice_scratch_ops`](../core.md#GridMapping.slice_scratch_ops), [`slice_index_ops`](../core.md#GridMapping.slice_index_ops), [`_get_dimension_semantics`](lowering.md#_get_dimension_semantics), [`get_grid_indices`](../core.md#GridMapping.get_grid_indices)
- used by: [`lower_jaxpr_into_pipelined_module`](lowering.md#lower_jaxpr_into_pipelined_module), [`body_func`](lowering.md#lower_jaxpr_to_transform_func.body_func), [`from_mosaic_grid_mapping`](lowering.md#PipelinedLoweringContext.from_mosaic_grid_mapping), [`lower_jaxpr_to_transform_func`](lowering.md#lower_jaxpr_to_transform_func), [`get_jaxpr_indices`](lowering.md#lower_jaxpr_into_pipelined_module.get_jaxpr_indices), [`MosaicGridMapping`](sc_lowering.md#MosaicGridMapping)

### `PallasLoweringCacheKey`
- def: [`jax/_src/pallas/mosaic/lowering.py:368`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L368)
- members:
  - `avals_in` — [`L372`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L372)
  - `avals_out` — [`L373`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L373)
  - `block_shapes` — [`L375`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L375)
  - `dynamic_shape_env` — [`L381`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L381)
  - `forward_compatible` — [`L377`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L377)
  - `fuse_transposed_lhs_in_matmul` — [`L378`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L378)
  - `grid_arity` — [`L376`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L376)
  - `grid_sizes` — [`L379`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L379)
  - `kernel_type` — [`L371`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L371)
  - `params` — [`L374`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L374)
  - `primitive` — [`L370`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L370)
  - `vmapped_dims` — [`L380`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L380)
- uses (calls/refs, reference-scoped): [`Primitive`](../../core.md#Primitive), [`CoreType`](core.md#CoreType), [`Squeezed`](../core.md#Squeezed), [`ShapedAbstractValue`](lowering.md#ShapedAbstractValue)
- used by: [`jaxpr_subcomp`](lowering.md#jaxpr_subcomp), [`from_mosaic_grid_mapping`](lowering.md#PipelinedLoweringContext.from_mosaic_grid_mapping), [`lower_jaxpr_to_transform_func`](lowering.md#lower_jaxpr_to_transform_func), [`from_mesh`](lowering.md#UnpipelinedLoweringContext.from_mesh), [`lowering_cache`](lowering.md#LoweringContext.lowering_cache)

### `PipelinedLoweringContext`  ·  implements/extends LoweringContext
- def: [`jax/_src/pallas/mosaic/lowering.py:259`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L259)
- signature: `class PipelinedLoweringContext(LoweringContext):`
- members:
  - `from_mosaic_grid_mapping(cls, mgm: MosaicGridMapping, jaxpr_indices: Sequence[ir.Value], kernel_type: tpu_core.CoreType, forward_compatible: bool, backend: Any | None, dynamic_shape_replacement_fn: DynamicShapeReplacementFn, fuse_transposed_lhs_in_matmul: bool, lowering_cache: dict[PallasLoweringCacheKey, func.FuncOp], dynamic_shape_env: LoweringDynamicShapeEnv | None = None)` — [`L262`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L262)
- uses (calls/refs, reference-scoped): [`CoreType`](core.md#CoreType), [`NameStack`](../../source_info_util.md#NameStack), [`grid`](lowering.md#MosaicGridMapping.grid), [`LoweringContext`](lowering.md#LoweringContext), [`DynamicShapeReplacementFn`](lowering.md#DynamicShapeReplacementFn), [`grid_names`](lowering.md#MosaicGridMapping.grid_names), [`vmapped_dims`](lowering.md#MosaicGridMapping.vmapped_dims), [`mesh_info`](lowering.md#MosaicGridMapping.mesh_info), [`PallasLoweringCacheKey`](lowering.md#PallasLoweringCacheKey), [`operand_block_shapes`](lowering.md#MosaicGridMapping.operand_block_shapes), [`LoweringDynamicShapeEnv`](lowering.md#LoweringDynamicShapeEnv), [`MosaicGridMapping`](lowering.md#MosaicGridMapping), [`scalar_prefetch_block_shapes`](lowering.md#MosaicGridMapping.scalar_prefetch_block_shapes), [`scratch_block_shapes`](lowering.md#MosaicGridMapping.scratch_block_shapes)
- used by: [`LoweringContext`](lowering.md#LoweringContext), [`ctx_factory`](lowering.md#lower_jaxpr_into_pipelined_module.ctx_factory)

### `ShapedAbstractValue`  ·  implements/extends Protocol
- def: [`jax/_src/pallas/mosaic/lowering.py:345`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L345)
- signature: `class ShapedAbstractValue(Protocol):`
- members:
  - `ndim(self)` — [`L351`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L351)
  - `size(self)` — [`L355`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L355)
  - `update(self, **kwargs: Any)` — [`L358`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L358)
  - `dtype` — [`L347`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L347)
  - `shape` — [`L346`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L346)
  - `weak_type` — [`L348`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L348)
- uses (calls/refs, reference-scoped): [`dtype`](../../../numpy/__init__.pyi.md#dtype), [`DimSize`](../../typing.md#DimSize)
- used by: [`jaxpr_subcomp`](lowering.md#jaxpr_subcomp), [`avals_in`](lowering.md#LoweringRuleContext.avals_in), [`avals_out`](lowering.md#LoweringRuleContext.avals_out), [`_dimension_semantics`](lowering.md#MosaicGridMapping._dimension_semantics), [`_load_lowering_rule`](lowering.md#_load_lowering_rule), [`_load_lowering_rule`](sc_lowering.md#_load_lowering_rule), [`_store_lowering_rule`](sc_lowering.md#_store_lowering_rule), [`_masked_swap_lowering_rule`](lowering.md#_masked_swap_lowering_rule), [`_dot_general_lowering_rule`](lowering.md#_dot_general_lowering_rule), [`_convert_element_type_lowering_rule`](lowering.md#_convert_element_type_lowering_rule), [`_cmp_lowering_rule`](lowering.md#_cmp_lowering_rule), [`_prng_key_load_lowering_rule`](lowering.md#_prng_key_load_lowering_rule), [`_div_lowering_rule`](lowering.md#_div_lowering_rule), [`_gather_lowering_rule`](lowering.md#_gather_lowering_rule), [`_mul_lowering_rule`](lowering.md#_mul_lowering_rule), [`_rem_lowering_rule`](lowering.md#_rem_lowering_rule), [`_bcast`](lowering.md#_bcast), [`_lowering_rule`](lowering.md#reduce_lowering_rule._lowering_rule), [`_sub_lowering_rule`](lowering.md#_sub_lowering_rule), [`_alloc_value`](lowering.md#_alloc_value), [`_add_lowering_rule`](lowering.md#_add_lowering_rule), [`_gather_lowering_rule`](sc_primitives.md#_gather_lowering_rule), [`_max_lowering_rule`](lowering.md#_max_lowering_rule), [`_min_lowering_rule`](lowering.md#_min_lowering_rule), [`_pow_lowering_rule`](lowering.md#_pow_lowering_rule), [`_debug_print_rule`](lowering.md#_debug_print_rule), [`_stochastic_round_lowering_rule`](lowering.md#_stochastic_round_lowering_rule), [`_unstack_lowering_rule`](lowering.md#_unstack_lowering_rule), [`_broadcast_in_dim_lowering_rule`](lowering.md#_broadcast_in_dim_lowering_rule), [`_select_n_lowering_rule`](lowering.md#_select_n_lowering_rule), [`_not_lowering_rule`](lowering.md#_not_lowering_rule), [`_bitcast_convert_type_lowering_rule`](lowering.md#_bitcast_convert_type_lowering_rule), [`_iota_lowering_rule_sc`](sc_lowering.md#_iota_lowering_rule_sc), [`_neg_lowering_rule`](lowering.md#_neg_lowering_rule), [`_pack_elementwise_lowering_rule`](lowering.md#_pack_elementwise_lowering_rule), [`_sign_lowering_rule`](lowering.md#_sign_lowering_rule), [`_stack_lowering_rule`](lowering.md#_stack_lowering_rule), [`_reduce_index_helper`](lowering.md#_reduce_index_helper), [`_abs_lowering_rule`](lowering.md#_abs_lowering_rule), [`_logistic_lowering_rule`](lowering.md#_logistic_lowering_rule)  (+22 more)

### `UncacheablePrimitiveError`  ·  implements/extends Exception
- def: [`jax/_src/pallas/mosaic/lowering.py:384`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L384)
- signature: `class UncacheablePrimitiveError(Exception):`
- used by: [`jaxpr_subcomp`](lowering.md#jaxpr_subcomp), [`body_builder`](lowering.md#_emit_pallas_lowering_rule_as_fun.body_builder)

### `UnpipelinedLoweringContext`  ·  implements/extends LoweringContext
- def: [`jax/_src/pallas/mosaic/lowering.py:299`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L299)
- signature: `class UnpipelinedLoweringContext(LoweringContext):`
- members:
  - `from_mesh(cls, mesh_shape: Sequence[tuple[jax_core.AxisName, int]], jaxpr: jax_core.Jaxpr, jax_mesh: mesh_lib.Mesh | None, core_type: tpu_core.CoreType, forward_compatible: bool, backend: Any | None, lowering_cache: dict[PallasLoweringCacheKey, func.FuncOp], needs_layout_passes: bool = False, mesh_indices: Sequence[ir.Value] = (), fuse_transposed_lhs_in_matmul: bool = False)` — [`L302`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L302)
- uses (calls/refs, reference-scoped): [`Jaxpr`](../../core.md#Jaxpr), [`aval`](../../core.md#Var.aval), [`CoreType`](core.md#CoreType), [`invars`](../../core.md#Jaxpr.invars), [`Mesh`](../../mesh.md#Mesh), [`AxisName`](../../core.md#AxisName), [`NameStack`](../../source_info_util.md#NameStack), [`LoweringContext`](lowering.md#LoweringContext), [`MeshInfo`](../utils.md#MeshInfo), [`from_mesh`](../utils.md#MeshInfo.from_mesh), [`PallasLoweringCacheKey`](lowering.md#PallasLoweringCacheKey)
- used by: [`LoweringContext`](lowering.md#LoweringContext), [`ctx_factory`](lowering.md#lower_jaxpr_into_unpipelined_module.ctx_factory)

## Functions
- `_1d_iota_helper()` — [`L3106`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3106)
- `_abs_lowering_rule(ctx: LoweringRuleContext, x)` — [`L3457`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3457)
- `_add_lowering_rule(ctx: LoweringRuleContext, x, y)` — [`L3237`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3237)
- `_alloc_value(aval: jax_core.AbstractValue | ShapedAbstractValue, *, ctx: LoweringRuleContext)` — [`L4561`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L4561)
- `_and_lowering_rule(ctx: LoweringRuleContext, x, y)` — [`L3829`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3829)
- `_argmax_lowering_rule(ctx: LoweringRuleContext, x, axes, index_dtype)` — [`L3355`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3355)
- `_argmin_lowering_rule(ctx: LoweringRuleContext, x, axes, index_dtype)` — [`L3362`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3362)
- `_atan2_lowering_rule(ctx: LoweringRuleContext, x, y, accuracy=None)` — [`L3634`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3634)
- `_aval_to_log_format_spec(aval)` — [`L4990`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L4990)
- `_axis_index_rule(ctx: LoweringRuleContext, *, axis_name: Hashable)` — [`L4958`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L4958)
- `_bcast(x: ir.Value | object, y: ir.Value | object, x_aval: ShapedAbstractValue, y_aval: ShapedAbstractValue, out_aval: ShapedAbstractValue, dynamic_shape_replacement_fn: DynamicShapeReplacementFn)` — [`L3194`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3194)
- `_bitcast_convert_type_lowering_rule(ctx: LoweringRuleContext, x, *, new_dtype)` — [`L4515`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L4515)
- `_bitcast_lowering_rule(ctx: LoweringRuleContext, x, *, ty)` — [`L4506`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L4506)
- `_bitcast_memref(ref: ir.Value, bitcaster: state_types.BitcastTransform, ref_aval: state.AbstractRef, ref_block_shape: tuple[int | pallas_core.Squeezed, ...])` — [`L2062`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L2062)
- `_broadcast_in_dim_lowering_rule(ctx: LoweringRuleContext, val, *, shape, broadcast_dimensions, sharding)` — [`L2645`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L2645)
- `_broadcast_to_lowering_rule(ctx: LoweringRuleContext, x, shape: Sequence[int])` — [`L2633`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L2633)
- `_canonicalize_dimension_semantic(dimension_semantic: tpu_core.DimensionSemantics)` — [`L689`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L689)
- `_canonicalize_transforms_to_indexer(ref_aval, transforms, transforms_avals)` — [`L2187`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L2187)
- `_ceil_lowering_rule(ctx: LoweringRuleContext, x)` — [`L3688`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3688)
- `_check_block_mappings(block_mappings: tuple[pallas_core.BlockMapping, ...], lowering_context: mlir.LoweringRuleContext, debug_info: jax_core.DebugInfo, kernel_type: tpu_core.CoreType)` — [`L890`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L890)
- `_check_lowering_rule(ctx: LoweringRuleContext, *err_args, err_tree, debug)` — [`L5198`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L5198)
- `_clamp(min, operand, max)` — [`L3906`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3906)
- `_clamp_lowering_rule(ctx: LoweringRuleContext, min, operand, max)` — [`L3912`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3912) — Compute minimum_p(maximum_p(min, operand), max).
- `_clz_lowering_rule(ctx: LoweringRuleContext, x)` — [`L3698`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3698)
- `_cmp_boolean_lowering_helper(primitive, x: Array, y: Array)` — [`L3749`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3749) — A helper function for lowering comparison operations for boolean inputs.
- `_cmp_lowering_rule(primitive, ctx: LoweringRuleContext, x, y)` — [`L3783`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3783)
- `_compute_name_stack_updates(old_name_stack: list[str], new_name_stack: list[str])` — [`L1673`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L1673) — Computes the popped/pushed items to the name stack after an update.
- `_concatenate_lowering_rule(ctx: LoweringRuleContext, *xs, dimension)` — [`L3020`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3020)
- `_cond_lowering_rule(ctx: LoweringRuleContext, *args, branches, **params)` — [`L4207`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L4207)
- `_convert_element_type_lowering_rule(ctx: LoweringRuleContext, x, *, new_dtype, weak_type, sharding)` — [`L2928`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L2928)
- `_convert_helper(x: Array, *, to_dtype: jnp.dtype)` — [`L2892`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L2892)
- `_cos_lowering_rule(ctx: LoweringRuleContext, x, accuracy=None)` — [`L3620`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3620)
- `_custom_jvp_call_lowering_rule(ctx: LoweringRuleContext, *args, call_jaxpr: jax_core.ClosedJaxpr, jvp_jaxpr_fun: lu.WrappedFun, num_consts: int, symbolic_zeros: bool)` — [`L4256`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L4256)
- `_custom_vjp_call_lowering_rule(ctx: LoweringRuleContext, *args, call_jaxpr, fwd_jaxpr_thunk, out_trees, symbolic_zeros, bwd, num_consts)` — [`L4273`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L4273)
- `_debug_callback_lowering_rule(ctx: LoweringRuleContext, *args, **kwargs)` — [`L4289`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L4289)
- `_debug_print_rule(ctx: LoweringRuleContext, *dyn_args, fmt: str, ordered, partitioned, in_tree, static_args, np_printoptions, has_placeholders, logging_record)` — [`L5000`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L5000)
- `_delay_rule(ctx: LoweringRuleContext, nanos: ir.Value)` — [`L4985`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L4985)
- `_device_id_to_logical(ctx: LoweringRuleContext, device_id, device_id_type: primitives.DeviceIdType, device_id_aval: Any, dest_mesh: pallas_core.Mesh | None = None)` — [`L4617`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L4617)
- `_dim_as_value_lowering(ctx: LoweringRuleContext, *, dim)` — [`L5321`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L5321)
- `_div_lowering_rule(ctx: LoweringRuleContext, x, y)` — [`L3415`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3415)
- `_dma_start(src_ref, dst_ref, sem, src_sem)` — [`L4830`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L4830)
- `_dma_start_lowering_rule(ctx: LoweringRuleContext, *args, tree, device_id_type: primitives.DeviceIdType, priority: int, add: bool)` — [`L4798`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L4798)
- `_dma_wait(src_ref, dst_ref, sem)` — [`L4874`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L4874)
- `_dma_wait_lowering_rule(ctx: LoweringRuleContext, *args, tree, device_id_type: primitives.DeviceIdType, insert_dummy_device: bool)` — [`L4851`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L4851)
- `_dot_general_lowering_rule(ctx: LoweringRuleContext, x, y, dimension_numbers, precision, preferred_element_type, **_)` — [`L2759`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L2759)
- `_dtype_to_ir_type(dtype: DTypeLike, is_kernel_boundary: bool = False)` — [`L526`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L526)
- `_emit_detached_func(name: str, input_types: Sequence[ir.Type], output_types: Sequence[ir.Type], body_builder: Callable[[list[ir.Value]], Sequence[ir.Value]])` — [`L388`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L388) — Helper to emit a detached FuncOp.
- `_emit_pallas_lowering_rule_as_fun(ctx: LoweringContext, primitive: jax_core.Primitive, rule: Callable, rule_context: LoweringRuleContext, invals: Sequence[ir.Value], **params)` — [`L408`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L408) — Emits the contents of a Pallas lowering rule as a detached function.
- `_empty_ref_lowering_rule(ctx: LoweringRuleContext, ty, memory_space)` — [`L4611`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L4611)
- `_ensure_mlir_value(val: object, aval: ShapedAbstractValue)` — [`L1871`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L1871)
- `_erf_inv_lowering_rule(ctx: LoweringRuleContext, x)` — [`L4434`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L4434)
- `_erf_lowering_rule(ctx: LoweringRuleContext, x, accuracy=None)` — [`L3671`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3671)
- `_exp2_lowering_rule(ctx: LoweringRuleContext, x, accuracy=None)` — [`L3587`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3587)
- `_exp_lowering_rule(ctx: LoweringRuleContext, x, accuracy=None)` — [`L3553`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3553)
- `_floor_lowering_rule(ctx: LoweringRuleContext, x)` — [`L3693`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3693)
- `_fold(x, fuel)` — [`L3253`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3253)
- `_fold_and_get_constant_value(x)` — [`L3273`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3273)
- `_gather_lowering_rule(ctx: LoweringRuleContext, x, indices, *, dimension_numbers, slice_sizes, unique_indices, indices_are_sorted, mode, fill_value)` — [`L3118`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3118)
- `_get_aval_physical_dtype_shape(aval)` — [`L681`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L681)
- `_get_barrier_semaphore_rule(ctx: LoweringRuleContext)` — [`L4979`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L4979)
- `_get_dimension_semantics(dimension_semantics: Sequence[str])` — [`L873`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L873)
- `_get_lowering_rule(ctx: LoweringRuleContext, ref, *idx, tree)` — [`L1886`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L1886)
- `_get_mesh_shape_and_semantics(mesh: pallas_core.Mesh)` — [`L1368`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L1368)
- `_get_semantics(s: str | None)` — [`L877`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L877)
- `_index_to_start_size_stride(idx: Any, cast_to_index: bool)` — [`L1939`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L1939)
- `_indexer_to_start_size_stride(indexer: NDIndexer, ref_block_shape: tuple[int | pallas_core.Squeezed, ...], *, cast_to_index: bool)` — [`L1963`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L1963)
- `_integer_pow_lowering_rule(ctx: LoweringRuleContext, x, *, y)` — [`L3582`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3582)
- `_iota_2x32_shape_lowering(ctx: LoweringRuleContext, *, shape)` — [`L5239`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L5239)
- `_iota_lowering_rule(ctx: LoweringRuleContext, dtype, shape, dimension, sharding)` — [`L3101`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3101)
- `_is_finite_lowering_rule(ctx: LoweringRuleContext, x)` — [`L3842`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3842)
- `_join_key_lowering_rule(ctx: LoweringRuleContext, *scalars, impl)` — [`L5191`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L5191)
- `_load_lowering_rule(ctx: LoweringRuleContext, *args_flat, args_tree, **_)` — [`L2212`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L2212)
- `_log1p_lowering_rule(ctx: LoweringRuleContext, x, accuracy=None)` — [`L3664`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3664)
- `_log_lowering_rule(ctx: LoweringRuleContext, x, accuracy=None)` — [`L3657`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3657)
- `_logistic_lowering_rule(ctx: LoweringRuleContext, x, accuracy=None)` — [`L3595`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3595)
- `_lower_fun(x)` — [`L3485`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3485)
- `_lower_fun(k1, k2, m1, m2)` — [`L5229`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L5229)
- `_lower_fun(shape)` — [`L5244`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L5244)
- `_lower_jaxpr_to_for_loop(ctx: LoweringRuleContext, jaxpr: jax_core.Jaxpr, start: int | ir.Value, num_steps: int | ir.Value, consts, *args, has_loop_index: bool, unroll: int)` — [`L3917`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3917)
- `_lower_jaxpr_to_func_common(jaxpr: jax_core.Jaxpr, *, name: str, arg_types: list[ir.Type], num_grid: int, get_jaxpr_indices: Callable[[list[ir.Value]], list[ir.Value]], ctx_factory: Callable[[list[ir.Value]], LoweringContext], dynamic_shape_replacement_enabled: bool = False, core_type: tpu_core.CoreType | None = None)` — [`L1577`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L1577)
- `_lower_multiref_transformed_ref(f, ref, ref_ty, ref_block_shape, args, rest_refs)` — [`L4925`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L4925) — Lower f with args as a multiref TransformedRef.
- `_lower_single_transformed_ref(f, ref, ref_ty, ref_block_shape, prev_args, rest_args)` — [`L4908`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L4908) — Let the lowering callback f run the single-ref transforms for `ref`.
- `_lower_transformed_refs(f, args, rest_args)` — [`L4892`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L4892) — Recursively iterate through TransformedRefs and lower them in the call to f.
- `_lower_while_via_fori(ctx: LoweringRuleContext, *args, fori_jaxpr, cond_nconsts, cond_jaxpr, body_nconsts, body_jaxpr)` — [`L4116`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L4116)
- `_lowering_rule(ctx: LoweringRuleContext, x, *, axes, **kwargs)` — [`L2525`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L2525)
- `_make_index(s)` — [`L1925`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L1925)
- `_masked_swap_lowering_rule(ctx: LoweringRuleContext, *args_flat, args_tree, **_)` — [`L2398`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L2398)
- `_matmul_acc_lhs_lowering_rule(ctx: LoweringRuleContext, lhs: ir.Value, *, acc_addr: int, mxu_index: int, load_staged_rhs: int | None)` — [`L5361`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L5361)
- `_matmul_pop_lowering_rule(ctx: LoweringRuleContext, *, acc_addr: int, mxu_index: int, shape: tuple[int, int], dtype: jax.typing.DTypeLike)` — [`L5378`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L5378)
- `_matmul_push_rhs_lowering_rule(ctx: LoweringRuleContext, rhs: ir.Value, *, staging_register: int, mxu_index: int, transpose: bool = False)` — [`L5345`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L5345)
- `_max_lowering_rule(ctx: LoweringRuleContext, x, y)` — [`L3287`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3287)
- `_maybe_cast_load_to_bool(ctx: LoweringRuleContext, out_aval, val: ir.Value)` — [`L2353`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L2353) — Casts a memref load value to bool if the requested value is a bool.
- `_maybe_cast_store_to_memref_type(ctx: LoweringRuleContext, expected_aval, val: ir.Value)` — [`L2387`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L2387) — Casts a boolean value back to an integer for storing in a memref.
- `_maybe_cast_to_index(cast_to_index, x)` — [`L1933`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L1933)
- `_maybe_physicalize_block_shape(aval, block_shape)` — [`L135`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L135)
- `_memory_space_to_mosaic_attribute(memory_space: AnyMemorySpace | None, kernel_type: tpu_core.CoreType)` — [`L504`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L504)
- `_min_lowering_rule(ctx: LoweringRuleContext, x, y)` — [`L3309`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3309)
- `_mul_lowering_rule(ctx: LoweringRuleContext, x, y, *, out_dtype)` — [`L3391`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3391)
- `_multiple_of_lowering_rule(ctx: LoweringRuleContext, val, *, values)` — [`L2517`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L2517)
- `_neg_lowering_rule(ctx: LoweringRuleContext, x)` — [`L3469`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3469)
- `_nextafter_lowering_rule(ctx: LoweringRuleContext, x, y)` — [`L3525`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3525)
- `_not_lowering_rule(ctx: LoweringRuleContext, x)` — [`L3864`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3864)
- `_num_programs_lowering_rule(ctx: LoweringRuleContext, *, axis: int)` — [`L4315`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L4315)
- `_or_lowering_rule(ctx: LoweringRuleContext, x, y)` — [`L3851`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3851)
- `_pack_elementwise_lowering_rule(ctx: LoweringRuleContext, *xs, packed_dtype)` — [`L4480`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L4480)
- `_pad(val, axis=axis)` — [`L5273`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L5273)
- `_pad_lowering_rule(ctx: LoweringRuleContext, *args, **kwargs)` — [`L5261`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L5261)
- `_physical_aval(aval: ShapedAbstractValue)` — [`L362`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L362)
- `_pjit_lowering_rule(ctx: LoweringRuleContext, *args, jaxpr, **_)` — [`L4244`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L4244)
- `_platform_index_lowering(ctx: mlir.LoweringRuleContext, *, platforms: BranchesPlatforms)` — [`L5305`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L5305)
- `_poison_memref(ref: ir.Value)` — [`L4530`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L4530)
- `_population_count_lowering_rule(ctx: LoweringRuleContext, x)` — [`L3703`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3703)
- `_pow_lowering_rule(ctx: LoweringRuleContext, x, y)` — [`L3560`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3560)
- `_prng_key_load_lowering_rule(ctx: LoweringRuleContext, *args_flat, args_tree)` — [`L2304`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L2304) — Lowering rule for loading PRNG keys from SMEM.
- `_prng_random_bits_lowering_rule(ctx: LoweringRuleContext, *, shape)` — [`L5113`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L5113)
- `_prng_seed_lowering_rule(ctx: LoweringRuleContext, *seeds)` — [`L5095`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L5095)
- `_program_id_lowering_rule(ctx: LoweringRuleContext, *, axis: int)` — [`L4298`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L4298)
- `_proxy_fun(val, *, axes)` — [`L2532`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L2532)
- `_proxy_reduce(arg, *, axes)` — [`L2608`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L2608)
- `_proxy_reduce(arg, *, axes)` — [`L2621`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L2621)
- `_reciprocal_lowering_rule(ctx: LoweringRuleContext, x, *, approx, full_range)` — [`L4441`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L4441)
- `_reduce_and_lowering_rule(ctx: LoweringRuleContext, x, *, axes)` — [`L2607`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L2607)
- `_reduce_index_helper(ctx: LoweringRuleContext, x, axes, index_dtype, reduction_kind)` — [`L3327`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3327)
- `_reduce_or_lowering_rule(ctx: LoweringRuleContext, x, *, axes)` — [`L2620`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L2620)
- `_rem_lowering_rule(ctx: LoweringRuleContext, x, y)` — [`L3437`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3437)
- `_reshape_lowering_rule(ctx: LoweringRuleContext, x, new_sizes, dimensions, sharding)` — [`L2990`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L2990)
- `_reshape_memref(ref: ir.Value, reshaper: state_types.ReshapeTransform, ref_aval: state.AbstractRef, ref_block_shape: tuple[int | pallas_core.Squeezed, ...])` — [`L2102`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L2102)
- `_reshard_lowering_rule(ctx: LoweringRuleContext, x, *, dst_sharding, concrete_mesh)` — [`L4250`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L4250)
- `_roll_lowering_rule(ctx: LoweringRuleContext, x, shift, *, axis, stride, stride_axis)` — [`L4340`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L4340)
- `_round_lowering_rule(ctx: LoweringRuleContext, x, *, rounding_method)` — [`L3678`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3678)
- `_rsqrt_lowering_rule(ctx: LoweringRuleContext, x, accuracy=None)` — [`L3532`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3532)
- `_run_body(i, args)` — [`L3969`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3969)
- `_run_scoped_lowering_rule(ctx: LoweringRuleContext, *consts, jaxpr, collective_axes, **_)` — [`L4587`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L4587)
- `_scan_lowering_rule(ctx: LoweringRuleContext, *args, jaxpr: jax_core.ClosedJaxpr, length: int, reverse: bool, unroll: int, num_consts: int, num_carry: int)` — [`L4076`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L4076)
- `_select_n_lowering_rule(ctx: LoweringRuleContext, pred, x, *args)` — [`L3887`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3887)
- `_select_to_ifop(f, prev_refs, rest_refs, idx, options)` — [`L4938`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L4938)
- `_semaphore_read_lowering_rule(ctx: LoweringRuleContext, *args, args_tree)` — [`L4722`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L4722)
- `_semaphore_signal_lowering_rule(ctx: LoweringRuleContext, *args, args_tree, device_id_type: primitives.DeviceIdType)` — [`L4747`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L4747)
- `_semaphore_wait_lowering_rule(ctx: LoweringRuleContext, *args, args_tree)` — [`L4787`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L4787)
- `_shift_left_lowering_rule(ctx: LoweringRuleContext, x, d)` — [`L4387`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L4387)
- `_shift_right_arithmetic_lowering_rule(ctx: LoweringRuleContext, x, d)` — [`L4404`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L4404)
- `_shift_right_logical_lowering_rule(ctx: LoweringRuleContext, x, d)` — [`L4421`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L4421)
- `_sign_lowering_rule(ctx: LoweringRuleContext, x)` — [`L3479`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3479)
- `_sin_lowering_rule(ctx: LoweringRuleContext, x, accuracy=None)` — [`L3613`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3613)
- `_slice_lowering_rule(ctx: LoweringRuleContext, x, limit_indices, start_indices, strides)` — [`L4353`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L4353) — Lowers a slice to vector dialect.
- `_slice_memref(ref: ir.Value, indexer: NDIndexer, ref_aval: state.AbstractRef, ref_block_shape: tuple[int | pallas_core.Squeezed, ...])` — [`L2006`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L2006)
- `_split_key_lowering_rule(ctx: LoweringRuleContext, key_data: KeyScalarBundle)` — [`L5184`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L5184)
- `_split_lowering_rule(ctx: LoweringRuleContext, x, *, sizes, axis)` — [`L3080`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3080)
- `_sqrt_lowering_rule(ctx: LoweringRuleContext, x, accuracy=None)` — [`L3539`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3539)
- `_square_lowering_rule(ctx: LoweringRuleContext, x)` — [`L3546`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3546)
- `_squeeze_lowering_rule(ctx: LoweringRuleContext, x, dimensions)` — [`L3004`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3004)
- `_stack_lowering_rule(ctx: LoweringRuleContext, *xs, axis)` — [`L3025`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3025)
- `_stage_lowering_rule(ctx: LoweringRuleContext, x)` — [`L2985`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L2985)
- `_stochastic_round_lowering_rule(ctx: LoweringRuleContext, x, random_bits, *, target_dtype)` — [`L4456`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L4456)
- `_stop_gradient_lowering_rule(_: LoweringRuleContext, x)` — [`L3281`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3281)
- `_sub_lowering_rule(ctx: LoweringRuleContext, x, y)` — [`L3371`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3371)
- `_swap_lowering_rule(ctx: LoweringRuleContext, ref, val, *idx, tree)` — [`L1903`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L1903)
- `_tan_lowering_rule(ctx: LoweringRuleContext, x, accuracy=None)` — [`L3627`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3627)
- `_tanh_lowering_rule(ctx: LoweringRuleContext, x, accuracy=None)` — [`L3650`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3650)
- `_threefry2x32_lowering(ctx: LoweringRuleContext, k1, k2, m1, m2)` — [`L5228`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L5228)
- `_tile_lowering_rule(ctx: LoweringRuleContext, x, *, reps)` — [`L4331`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L4331)
- `_touch_lowering_rule(ctx: LoweringRuleContext, x: jax.Array)` — [`L5329`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L5329)
- `_trace_value_lowering_rule(ctx: LoweringRuleContext, value, *, label: str)` — [`L5337`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L5337) — Lower trace_value to tpu.trace_value.
- `_transform_ref(ref, ref_ty, ref_block_shape, transforms=())` — [`L2135`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L2135)
- `_transpose_lowering_rule(ctx: LoweringRuleContext, x, *, permutation)` — [`L3189`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3189)
- `_unpack_elementwise_lowering_rule(ctx: LoweringRuleContext, x, index, packed_dtype, unpacked_dtype)` — [`L4493`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L4493)
- `_unstack_lowering_rule(ctx: LoweringRuleContext, x, *, axis)` — [`L3046`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3046)
- `_while_lowering_rule(ctx: LoweringRuleContext, *args, cond_nconsts, cond_jaxpr, body_nconsts, body_jaxpr)` — [`L4146`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L4146)
- `_xor_lowering_rule(ctx: LoweringRuleContext, x, y)` — [`L4370`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L4370)
- `aval_to_ir_type(dynamic_shape_replacement_fn: DynamicShapeReplacementFn, aval, *, shape=None, memory_space: AnyMemorySpace | None = None, is_kernel_boundary: bool = False, allow_extended_types: bool = True, kernel_type: tpu_core.CoreType)` — [`L550`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L550)
- `body_builder(block_args: list[ir.Value])` — [`L432`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L432)
- `body_builder(block_args: list[ir.Value])` — [`L3948`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3948)
- `body_func(*args)` — [`L1532`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L1532)
- `body_func(*args)` — [`L1588`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L1588)
- `ctx_factory(jaxpr_indices)` — [`L1102`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L1102)
- `ctx_factory(mesh_indices)` — [`L1436`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L1436)
- `decorator(rule: T)` — [`L671`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L671)
- `dynamic_shape_replacement_fn(shape: jax_core.Shape)` — [`L1058`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L1058)
- `err_details()` — [`L925`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L925)
- `f_lowered(ctx: LoweringRuleContext, *args, **params)` — [`L1634`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L1634)
- `format_dims(dims)` — [`L2739`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L2739)
- `get_jaxpr_indices(grid_indices)` — [`L1094`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L1094)
- `ir_constant(x: Any, mlir_type: ir.Type | None = None)` — [`L612`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L612)
- `jax_dot_dims_to_tpu_dot_dot_dims(dimension_numbers, lhs_shape, rhs_shape, excluding_lhs_dims=frozenset(), excluding_rhs_dims=frozenset())` — [`L2673`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L2673) — Converts a jax dot dimension numbers to a tpu dot dimension numbers.
- `jax_fn(device_id_val)` — [`L4643`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L4643)
- `jaxpr_subcomp(ctx: LoweringContext, jaxpr: jax_core.Jaxpr, *args: ir.Value)` — [`L1696`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L1696) — documented in [jax-_src-pallas-mosaic-lowering](../../../../../concepts/jax-_src-pallas-mosaic-lowering.md)
- `lower_fun(fun: Callable, *, in_avals: Any | None = None)` — [`L1623`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L1623) — Converts a traceable JAX function `fun` into a lowering rule.
- `lower_jaxpr_into_pipelined_module(lowering_context: mlir.LoweringRuleContext, module: ir.Module, grid_mapping: pallas_core.GridMapping, jaxpr: jax_core.Jaxpr, *, name: str, dimension_semantics: Sequence[tpu_core.DimensionSemantics] | None, kernel_type: tpu_core.CoreType, mesh: mesh_lib.Mesh | None = None, dynamic_shape_replacement_enabled: bool = False, fuse_transposed_lhs_in_matmul: bool = False)` — [`L1032`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L1032) — documented in [jax-_src-pallas-mosaic-lowering](../../../../../concepts/jax-_src-pallas-mosaic-lowering.md)
- `lower_jaxpr_into_unpipelined_module(lowering_context: mlir.LoweringRuleContext, module: ir.Module, jaxpr: jax_core.Jaxpr, *, name: str, pallas_mesh: pallas_core.Mesh, jax_mesh: mesh_lib.Mesh | None, dynamic_shape_replacement_enabled: bool = False, num_scratch: int, needs_layout_passes: bool = False, fuse_transposed_lhs_in_matmul: bool = False)` — [`L1398`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L1398)
- `lower_jaxpr_to_pipelined_module(lowering_context: mlir.LoweringRuleContext, grid_mapping: pallas_core.GridMapping, jaxpr: jax_core.Jaxpr, *, dimension_semantics: Sequence[tpu_core.DimensionSemantics] | None, kernel_type: tpu_core.CoreType, mesh: mesh_lib.Mesh | None = None, dynamic_shape_replacement_enabled: bool = False, fuse_transposed_lhs_in_matmul: bool = False)` — [`L1005`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L1005)
- `lower_jaxpr_to_transform_func(jaxpr: jax_core.Jaxpr, aval: jax_core.AbstractValue, *, name: str, mosaic_grid_mapping: MosaicGridMapping, kernel_type: tpu_core.CoreType, forward_compatible: bool, backend: Any | None, dynamic_shape_replacement_fn: DynamicShapeReplacementFn, lowering_cache: dict[PallasLoweringCacheKey, func.FuncOp], dynamic_shape_env: LoweringDynamicShapeEnv | None = None)` — [`L1514`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L1514)
- `lower_with_transformed_refs(f, args, avals, block_shapes=None)` — [`L4881`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L4881) — Lower f with args as potentially nested TransformedRefs.
- `new_f(*newf_args)` — [`L4916`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L4916)
- `new_lowering(key, bit_width, shape)` — [`L5136`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L5136)
- `random_bits_lowering(ctx: LoweringRuleContext, keys, *, bit_width, shape)` — [`L5129`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L5129)
- `random_fold_in_lowering(ctx: LoweringRuleContext, keys, msgs)` — [`L5145`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L5145)
- `random_seed_lowering(ctx: LoweringRuleContext, seeds, *, impl)` — [`L5123`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L5123)
- `random_unwrap_lowering(ctx: LoweringRuleContext, key)` — [`L5160`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L5160)
- `random_wrap_lowering(ctx: LoweringRuleContext, key_data, *, impl)` — [`L5173`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L5173)
- `read_block_shape(atom: jax_core.Atom)` — [`L1703`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L1703)
- `read_env(atom: jax_core.Atom)` — [`L1713`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L1713)
- `reduce_lowering_rule(reduce_fn, type_to_kind, type_to_identity)` — [`L2524`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L2524)
- `register_lowering_rule(prim: jax_core.Primitive, *, kernel_types: Collection[tpu_core.CoreType] = (tpu_core.CoreType.TC,), ensure_mlir_values: bool = True)` — [`L665`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L665) — documented in [jax-_src-pallas-mosaic-lowering](../../../../../concepts/jax-_src-pallas-mosaic-lowering.md)
- `should_physicalize_dtype(dtype: DTypeLike)` — [`L127`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L127) — Returns whether a dtype should be lowered to a physical type.
- `write_env(var: jax_core.Var, val)` — [`L1716`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L1716)

## Module values
- `ANY` — [`L102`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L102)
- `AnyMemorySpace` — [`L86`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L86)
- `BOOL_MEMREF_TYPE` — [`L104`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L104)
- `DIM_LOWER_BOUND` — [`L112`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L112)
- `DIM_UPPER_BOUND` — [`L111`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L111)
- `DynamicShapeReplacementFn` — [`L197`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L197)
- `HOST` — [`L100`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L100)
- `MLIR_DYNAMIC` — [`L107`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L107)
- `NDIndexer` — [`L85`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L85)
- `PHYSICAL_EXTENDED_DTYPES` — [`L119`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L119)
- `REDUCE_MAX_IDENTITY` — [`L2569`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L2569)
- `REDUCE_MAX_KINDS` — [`L2564`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L2564)
- `REDUCE_MIN_IDENTITY` — [`L2583`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L2583)
- `REDUCE_MIN_KINDS` — [`L2578`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L2578)
- `REDUCE_SUM_IDENTITY` — [`L2597`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L2597)
- `REDUCE_SUM_KINDS` — [`L2592`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L2592)
- `SEMAPHORE` — [`L101`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L101)
- `SMEM` — [`L99`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L99)
- `T` — [`L662`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L662)
- `TPUMemorySpace` — [`L92`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L92)
- `VMEM` — [`L98`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L98)
- `_cmpf_lowering_types` — [`L3731`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3731)
- `_cmpsi_lowering_types` — [`L3711`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3711)
- `_cmpui_lowering_types` — [`L3721`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3721)
- `_dma_tree_leaves` — [`L123`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L123)
- `_dma_unflatten` — [`L121`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L121)
- `_get_ref_and_transforms` — [`L122`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L122)
- `_primitives_needing_grid` — [`L653`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L653)
- `_reduce_max_lowering_rule` — [`L2573`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L2573)
- `_reduce_min_lowering_rule` — [`L2587`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L2587)
- `_reduce_sum_lowering_rule` — [`L2601`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L2601)
- `_uncacheable_primitives` — [`L641`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L641)
- `lowering_rules` — [`L635`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L635)
- `partial` — [`L114`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L114)
- `prim` — [`L3820`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L3820)
- `skip_mlir_conversions` — [`L637`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L637)
- `unsafe_map` — [`L115`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L115)
- `unsafe_zip` — [`L116`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/lowering.py#L116)


---
title: 'Module: jax/experimental/mosaic/gpu/layout_inference.py'
type: catalog
provenance: extracted
module: jax/experimental/mosaic/gpu/layout_inference.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax.experimental.mosaic.gpu.layout_inference`/
symbols:
  VariableType: VariableType#
  ValueSite: ValueSite#
  VariableType.OPERAND: VariableType#OPERAND.
  _add_constraint_system_derivation_rule: _add_constraint_system_derivation_rule().
  ConstraintSystemDerivationRuleResult: ConstraintSystemDerivationRuleResult.
  DerivationContext: DerivationContext#
  DerivationContext.producer_ref: DerivationContext#producer_ref().
  VariableType.RESULT: VariableType#RESULT.
  _wgmma_constraint_system: _wgmma_constraint_system().
  _for_constraint_system: _for_constraint_system().
  _tcgen05_mma_constraint_system: _tcgen05_mma_constraint_system().
  infer_layout: infer_layout().
  _while_constraint_system: _while_constraint_system().
  _mma_constraint_system: _mma_constraint_system().
  _warp_map_constraint_system: _warp_map_constraint_system().
  _custom_primitive_constraint_system: _custom_primitive_constraint_system().
  assign_layouts: assign_layouts().
  derive_relayout_constraints: derive_relayout_constraints().
  _vector_load_constraint_system: _vector_load_constraint_system().
  _memref_collapse_shape_op_constraint_system: _memref_collapse_shape_op_constraint_system().
  extract_assignment_candidates_from_reduce_equation: extract_assignment_candidates_from_reduce_equation().
  _index_switch_constraint_system: _index_switch_constraint_system().
  _layout_cast_constraint_system: _layout_cast_constraint_system().
  _vector_concat_constraint_system: _vector_concat_constraint_system().
  _tmem_layout_cast_constraint_system: _tmem_layout_cast_constraint_system().
  _with_transforms_constraint_system: _with_transforms_constraint_system().
  _async_load_store_constraint_system: _async_load_store_constraint_system().
  _vector_store_constraint_system: _vector_store_constraint_system().
  ValueSite.value: ValueSite#value().
  ValueSite.shape: ValueSite#shape().
  _async_store_smem_constraint_system: _async_store_smem_constraint_system().
  _optimization_barrier_constraint_system: _optimization_barrier_constraint_system().
  _multi_dim_reduction_constraint_system: _multi_dim_reduction_constraint_system().
  _extract_strided_slice_constraint_system: _extract_strided_slice_constraint_system().
  _vector_extract_constraint_system: _vector_extract_constraint_system().
  _insert_strided_slice_constraint_system: _insert_strided_slice_constraint_system().
  _async_load_tmem_constraint_system: _async_load_tmem_constraint_system().
  _async_store_smem_to_tmem_constraint_system: _async_store_smem_to_tmem_constraint_system().
  _slice_tmem_constraint_system: _slice_tmem_constraint_system().
  _memref_subview_constraint_system: _memref_subview_constraint_system().
  _memref_expand_shape_op_equation_system: _memref_expand_shape_op_equation_system().
  _register_layouts_for_optimized_transfer_to_smem: _register_layouts_for_optimized_transfer_to_smem().
  _print_layout_constraint_system: _print_layout_constraint_system().
  _constant_constraint_system: _constant_constraint_system().
  _async_store_scales_smem_to_tmem_constraint_system: _async_store_scales_smem_to_tmem_constraint_system().
  _async_store_tmem_constraint_system: _async_store_tmem_constraint_system().
  _get_cluster_ref_constraint_system: _get_cluster_ref_constraint_system().
  _memref_cast_op_constraint_system: _memref_cast_op_constraint_system().
  _memref_transpose_op_constraint_system: _memref_transpose_op_constraint_system().
  _extract_variable_assignments_from_constraints: _extract_variable_assignments_from_constraints().
  ValueSitesForVariable: ValueSitesForVariable.
  _multimem_load_reduce_constraint_system: _multimem_load_reduce_constraint_system().
  _broadcasted_iota_constraint_system: _broadcasted_iota_constraint_system().
  _vector_broadcast_constraint_system: _vector_broadcast_constraint_system().
  _broadcast_in_dim_constraint_system: _broadcast_in_dim_constraint_system().
  _shape_cast_constraint_system: _shape_cast_constraint_system().
  _tmem_alloc_constraint_system: _tmem_alloc_constraint_system().
  _tmem_dealloc_constraint_system: _tmem_dealloc_constraint_system().
  _async_store_sparse_metadata_smem_to_tmem_constraint_system: _async_store_sparse_metadata_smem_to_tmem_constraint_system().
  _slice_smem_constraint_system: _slice_smem_constraint_system().
  _cluster_launch_control_ops_constraint_system: _cluster_launch_control_ops_constraint_system().
  _vector_value_sites_and_assignments_for_async_ops: _vector_value_sites_and_assignments_for_async_ops().
  producer_result: producer_result().
  _tmem_layout_from_layout_attr: _tmem_layout_from_layout_attr().
  consumer_operands: consumer_operands().
  infer_layout.gather_constraints: infer_layout().gather_constraints().
  vector_value_sites: vector_value_sites().
  ValueSite.__str__: ValueSite#__str__().
  _debug_print_constraint_system: _debug_print_constraint_system().
  _vector_reduction_constraint_system: _vector_reduction_constraint_system().
  _memref_load_store_op_constraint_system: _memref_load_store_op_constraint_system().
  _tcgen05_mma_constraint_system.assign_scaled_layout: _tcgen05_mma_constraint_system().assign_scaled_layout().
  ValueSite.memory_space: ValueSite#memory_space().
  _TypeAndLayout.layout: _TypeAndLayout#layout.
  _is_smem_ref: _is_smem_ref().
  _extract_layout_candidates_from_tmem_registers_transfer: _extract_layout_candidates_from_tmem_registers_transfer().
  _extract_layout_candidates_from_smem_registers_transfer: _extract_layout_candidates_from_smem_registers_transfer().
  conjure_assignment: conjure_assignment().
  _async_prefetch_constraint_system: _async_prefetch_constraint_system().
  _ensure_all_layouts_are_set: _ensure_all_layouts_are_set().
  assign_layouts._to_transform_attrs: assign_layouts()._to_transform_attrs().
  ValueSite.operation: ValueSite#operation.
  _extract_layout_candidates_from_broadcast: _extract_layout_candidates_from_broadcast().
  DerivationContext.value_sites_for_variable: DerivationContext#value_sites_for_variable.
  DerivationContext.update: DerivationContext#update().
  _pointwise_op_constraint_system: _pointwise_op_constraint_system().
  _op: _op.
  VariableType.ARGUMENT: VariableType#ARGUMENT.
  is_vector: is_vector().
  ValueSite.__post_init__: ValueSite#__post_init__().
  _compute_swizzle: _compute_swizzle().
  ValueSite.index: ValueSite#index.
  find_assignments_for: find_assignments_for().
  ConstraintSystemDerivationRule: ConstraintSystemDerivationRule.
  dynamic_gcd: dynamic_gcd().
  _extract_smem_transforms_from_custom_transform_attrs: _extract_smem_transforms_from_custom_transform_attrs().
  ValueSite.region_index: ValueSite#region_index.
  _strided_layout_for_variable: _strided_layout_for_variable().
  _add_constraint_system_derivation_rule.wrapper: _add_constraint_system_derivation_rule().wrapper().
  DerivationContext.variable_for_value_site: DerivationContext#variable_for_value_site.
  _is_tmem_ref: _is_tmem_ref().
  traverse_op: traverse_op().
  _constraint_system_derivation_rules._constraint_system_derivation_rules: _constraint_system_derivation_rules._constraint_system_derivation_rules.
  _terminator: _terminator().
  _conjure_transforms_for_smem_ref: _conjure_transforms_for_smem_ref().
  _extract_layout_candidates_from_mma_tiling: _extract_layout_candidates_from_mma_tiling().
  _ensure_right_number_of_layouts: _ensure_right_number_of_layouts().
  _TypeAndLayout: _TypeAndLayout#
  DerivationContext.slice_smem_aliases: DerivationContext#slice_smem_aliases.
  DerivationContext.slice_tmem_aliases: DerivationContext#slice_tmem_aliases.
  logger: logger.
  _DEFAULT_LAYOUT_INFERENCE_FUEL: _DEFAULT_LAYOUT_INFERENCE_FUEL.
  _op_name_regex: _op_name_regex.
  is_hopper: is_hopper().
  _default_tmem_layout_for_variable: _default_tmem_layout_for_variable().
  _divides_per_var: _divides_per_var().
  _extract_variable_assignments_from_constraints.priority: _extract_variable_assignments_from_constraints().priority().
  prime_decomposition: prime_decomposition().
  is_terminator: is_terminator().
  check_layout_assignment: check_layout_assignment().
---
# Module: [`jax/experimental/mosaic/gpu/layout_inference.py`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py)

## Classes
### `DerivationContext`
- def: [`jax/experimental/mosaic/gpu/layout_inference.py:602`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L602)
- doc: Holds context information used for deriving an constraint system.
- signature: `class DerivationContext:`
- members:
  - `producer_ref(self, operand: ValueSite)` — [`L640`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L640) — Returns the producer reference variable for the given operand. — documented in [jax-experimental-mosaic-gpu-layout_inference](../../../../../concepts/jax-experimental-mosaic-gpu-layout_inference.md)
  - `update(self, mapping: ValueSitesForVariable)` — [`L630`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L630)
  - `slice_smem_aliases` — [`L623`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L623)
  - `slice_tmem_aliases` — [`L626`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L626)
  - `value_sites_for_variable` — [`L609`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L609)
  - `variable_for_value_site` — [`L605`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L605)
- uses (calls/refs, reference-scoped): [`ValueSite`](layout_inference.md#ValueSite), [`ValueSitesForVariable`](layout_inference.md#ValueSitesForVariable), [`producer_result`](layout_inference.md#producer_result)
- used by: [`_wgmma_constraint_system`](layout_inference.md#_wgmma_constraint_system), [`_for_constraint_system`](layout_inference.md#_for_constraint_system), [`_tcgen05_mma_constraint_system`](layout_inference.md#_tcgen05_mma_constraint_system), [`infer_layout`](layout_inference.md#infer_layout), [`_mma_constraint_system`](layout_inference.md#_mma_constraint_system), [`_warp_map_constraint_system`](layout_inference.md#_warp_map_constraint_system), [`_while_constraint_system`](layout_inference.md#_while_constraint_system), [`_custom_primitive_constraint_system`](layout_inference.md#_custom_primitive_constraint_system), [`_memref_collapse_shape_op_constraint_system`](layout_inference.md#_memref_collapse_shape_op_constraint_system), [`_vector_load_constraint_system`](layout_inference.md#_vector_load_constraint_system), [`_async_load_store_constraint_system`](layout_inference.md#_async_load_store_constraint_system), [`_index_switch_constraint_system`](layout_inference.md#_index_switch_constraint_system), [`_layout_cast_constraint_system`](layout_inference.md#_layout_cast_constraint_system), [`_tmem_layout_cast_constraint_system`](layout_inference.md#_tmem_layout_cast_constraint_system), [`_vector_concat_constraint_system`](layout_inference.md#_vector_concat_constraint_system), [`_with_transforms_constraint_system`](layout_inference.md#_with_transforms_constraint_system), [`_vector_store_constraint_system`](layout_inference.md#_vector_store_constraint_system), [`_async_load_tmem_constraint_system`](layout_inference.md#_async_load_tmem_constraint_system), [`_async_store_smem_constraint_system`](layout_inference.md#_async_store_smem_constraint_system), [`_async_store_smem_to_tmem_constraint_system`](layout_inference.md#_async_store_smem_to_tmem_constraint_system), [`_extract_strided_slice_constraint_system`](layout_inference.md#_extract_strided_slice_constraint_system), [`_insert_strided_slice_constraint_system`](layout_inference.md#_insert_strided_slice_constraint_system), [`_memref_expand_shape_op_equation_system`](layout_inference.md#_memref_expand_shape_op_equation_system), [`_memref_subview_constraint_system`](layout_inference.md#_memref_subview_constraint_system), [`_multi_dim_reduction_constraint_system`](layout_inference.md#_multi_dim_reduction_constraint_system), [`_optimization_barrier_constraint_system`](layout_inference.md#_optimization_barrier_constraint_system), [`_slice_tmem_constraint_system`](layout_inference.md#_slice_tmem_constraint_system), [`_vector_extract_constraint_system`](layout_inference.md#_vector_extract_constraint_system), [`_async_store_scales_smem_to_tmem_constraint_system`](layout_inference.md#_async_store_scales_smem_to_tmem_constraint_system), [`_async_store_tmem_constraint_system`](layout_inference.md#_async_store_tmem_constraint_system), [`_constant_constraint_system`](layout_inference.md#_constant_constraint_system), [`_get_cluster_ref_constraint_system`](layout_inference.md#_get_cluster_ref_constraint_system), [`_memref_cast_op_constraint_system`](layout_inference.md#_memref_cast_op_constraint_system), [`_memref_transpose_op_constraint_system`](layout_inference.md#_memref_transpose_op_constraint_system), [`_print_layout_constraint_system`](layout_inference.md#_print_layout_constraint_system), [`_async_store_sparse_metadata_smem_to_tmem_constraint_system`](layout_inference.md#_async_store_sparse_metadata_smem_to_tmem_constraint_system), [`_broadcast_in_dim_constraint_system`](layout_inference.md#_broadcast_in_dim_constraint_system), [`_broadcasted_iota_constraint_system`](layout_inference.md#_broadcasted_iota_constraint_system), [`_cluster_launch_control_ops_constraint_system`](layout_inference.md#_cluster_launch_control_ops_constraint_system), [`_multimem_load_reduce_constraint_system`](layout_inference.md#_multimem_load_reduce_constraint_system)  (+13 more)

### `ValueSite`
- def: [`jax/experimental/mosaic/gpu/layout_inference.py:78`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L78) — documented in [jax-experimental-mosaic-gpu-layout_inference](../../../../../concepts/jax-experimental-mosaic-gpu-layout_inference.md)
- doc: A unique identifier for a variable.
- signature: `class ValueSite:`
- members:
  - `memory_space(self)` — [`L115`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L115) — Returns the memory space associated with this value.
  - `shape(self)` — [`L110`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L110) — Returns the shape of the underlying value.
  - `value(self)` — [`L100`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L100) — Returns the IR value corresponding to this value site. — documented in [jax-experimental-mosaic-gpu-layout_inference](../../../../../concepts/jax-experimental-mosaic-gpu-layout_inference.md)
  - `index` — [`L92`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L92)
  - `operation` — [`L87`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L87)
  - `region_index` — [`L94`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L94)
- protocol/private: `__post_init__`[`L96`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L96), `__str__`[`L127`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L127)
- uses (calls/refs, reference-scoped): [`VariableType`](layout_inference.md#VariableType), [`OPERAND`](layout_inference.md#VariableType.OPERAND), [`RESULT`](layout_inference.md#VariableType.RESULT), [`is_smem_ref`](utils.md#is_smem_ref), [`ARGUMENT`](layout_inference.md#VariableType.ARGUMENT), [`is_cluster_smem_ref`](utils.md#is_cluster_smem_ref), [`is_tmem_ref`](utils.md#is_tmem_ref), [`_op_name_regex`](layout_inference.md#_op_name_regex)
- used by: [`producer_ref`](layout_inference.md#DerivationContext.producer_ref), [`_wgmma_constraint_system`](layout_inference.md#_wgmma_constraint_system), [`_for_constraint_system`](layout_inference.md#_for_constraint_system), [`_tcgen05_mma_constraint_system`](layout_inference.md#_tcgen05_mma_constraint_system), [`infer_layout`](layout_inference.md#infer_layout), [`_mma_constraint_system`](layout_inference.md#_mma_constraint_system), [`_warp_map_constraint_system`](layout_inference.md#_warp_map_constraint_system), [`_while_constraint_system`](layout_inference.md#_while_constraint_system), [`_custom_primitive_constraint_system`](layout_inference.md#_custom_primitive_constraint_system), [`assign_layouts`](layout_inference.md#assign_layouts), [`derive_relayout_constraints`](layout_inference.md#derive_relayout_constraints), [`_memref_collapse_shape_op_constraint_system`](layout_inference.md#_memref_collapse_shape_op_constraint_system), [`_vector_load_constraint_system`](layout_inference.md#_vector_load_constraint_system), [`_async_load_store_constraint_system`](layout_inference.md#_async_load_store_constraint_system), [`_index_switch_constraint_system`](layout_inference.md#_index_switch_constraint_system), [`_layout_cast_constraint_system`](layout_inference.md#_layout_cast_constraint_system), [`_tmem_layout_cast_constraint_system`](layout_inference.md#_tmem_layout_cast_constraint_system), [`_vector_concat_constraint_system`](layout_inference.md#_vector_concat_constraint_system), [`_with_transforms_constraint_system`](layout_inference.md#_with_transforms_constraint_system), [`_vector_store_constraint_system`](layout_inference.md#_vector_store_constraint_system), [`_async_load_tmem_constraint_system`](layout_inference.md#_async_load_tmem_constraint_system), [`_async_store_smem_constraint_system`](layout_inference.md#_async_store_smem_constraint_system), [`_async_store_smem_to_tmem_constraint_system`](layout_inference.md#_async_store_smem_to_tmem_constraint_system), [`_extract_strided_slice_constraint_system`](layout_inference.md#_extract_strided_slice_constraint_system), [`_insert_strided_slice_constraint_system`](layout_inference.md#_insert_strided_slice_constraint_system), [`_memref_expand_shape_op_equation_system`](layout_inference.md#_memref_expand_shape_op_equation_system), [`_memref_subview_constraint_system`](layout_inference.md#_memref_subview_constraint_system), [`_multi_dim_reduction_constraint_system`](layout_inference.md#_multi_dim_reduction_constraint_system), [`_optimization_barrier_constraint_system`](layout_inference.md#_optimization_barrier_constraint_system), [`_slice_tmem_constraint_system`](layout_inference.md#_slice_tmem_constraint_system), [`_vector_extract_constraint_system`](layout_inference.md#_vector_extract_constraint_system), [`_async_store_scales_smem_to_tmem_constraint_system`](layout_inference.md#_async_store_scales_smem_to_tmem_constraint_system), [`_async_store_tmem_constraint_system`](layout_inference.md#_async_store_tmem_constraint_system), [`_constant_constraint_system`](layout_inference.md#_constant_constraint_system), [`_get_cluster_ref_constraint_system`](layout_inference.md#_get_cluster_ref_constraint_system), [`_memref_cast_op_constraint_system`](layout_inference.md#_memref_cast_op_constraint_system), [`_memref_transpose_op_constraint_system`](layout_inference.md#_memref_transpose_op_constraint_system), [`_print_layout_constraint_system`](layout_inference.md#_print_layout_constraint_system), [`ValueSitesForVariable`](layout_inference.md#ValueSitesForVariable), [`_async_store_sparse_metadata_smem_to_tmem_constraint_system`](layout_inference.md#_async_store_sparse_metadata_smem_to_tmem_constraint_system)  (+19 more)

### `VariableType`  ·  implements/extends IntEnum
- def: [`jax/experimental/mosaic/gpu/layout_inference.py:64`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L64) — documented in [jax-experimental-mosaic-gpu-layout_inference](../../../../../concepts/jax-experimental-mosaic-gpu-layout_inference.md)
- doc: The type of a variable.
- signature: `class VariableType(enum.IntEnum):`
- members:
  - `ARGUMENT` — [`L71`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L71)
  - `OPERAND` — [`L69`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L69) — documented in [jax-experimental-mosaic-gpu-layout_inference](../../../../../concepts/jax-experimental-mosaic-gpu-layout_inference.md)
  - `RESULT` — [`L70`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L70)
- used by: [`_wgmma_constraint_system`](layout_inference.md#_wgmma_constraint_system), [`_for_constraint_system`](layout_inference.md#_for_constraint_system), [`_tcgen05_mma_constraint_system`](layout_inference.md#_tcgen05_mma_constraint_system), [`_mma_constraint_system`](layout_inference.md#_mma_constraint_system), [`_warp_map_constraint_system`](layout_inference.md#_warp_map_constraint_system), [`_while_constraint_system`](layout_inference.md#_while_constraint_system), [`_custom_primitive_constraint_system`](layout_inference.md#_custom_primitive_constraint_system), [`assign_layouts`](layout_inference.md#assign_layouts), [`derive_relayout_constraints`](layout_inference.md#derive_relayout_constraints), [`_memref_collapse_shape_op_constraint_system`](layout_inference.md#_memref_collapse_shape_op_constraint_system), [`_vector_load_constraint_system`](layout_inference.md#_vector_load_constraint_system), [`_async_load_store_constraint_system`](layout_inference.md#_async_load_store_constraint_system), [`_index_switch_constraint_system`](layout_inference.md#_index_switch_constraint_system), [`_layout_cast_constraint_system`](layout_inference.md#_layout_cast_constraint_system), [`_tmem_layout_cast_constraint_system`](layout_inference.md#_tmem_layout_cast_constraint_system), [`_vector_concat_constraint_system`](layout_inference.md#_vector_concat_constraint_system), [`_with_transforms_constraint_system`](layout_inference.md#_with_transforms_constraint_system), [`_vector_store_constraint_system`](layout_inference.md#_vector_store_constraint_system), [`value`](layout_inference.md#ValueSite.value), [`_async_load_tmem_constraint_system`](layout_inference.md#_async_load_tmem_constraint_system), [`_async_store_smem_constraint_system`](layout_inference.md#_async_store_smem_constraint_system), [`_async_store_smem_to_tmem_constraint_system`](layout_inference.md#_async_store_smem_to_tmem_constraint_system), [`_extract_strided_slice_constraint_system`](layout_inference.md#_extract_strided_slice_constraint_system), [`_insert_strided_slice_constraint_system`](layout_inference.md#_insert_strided_slice_constraint_system), [`_memref_expand_shape_op_equation_system`](layout_inference.md#_memref_expand_shape_op_equation_system), [`_memref_subview_constraint_system`](layout_inference.md#_memref_subview_constraint_system), [`_multi_dim_reduction_constraint_system`](layout_inference.md#_multi_dim_reduction_constraint_system), [`_optimization_barrier_constraint_system`](layout_inference.md#_optimization_barrier_constraint_system), [`_slice_tmem_constraint_system`](layout_inference.md#_slice_tmem_constraint_system), [`_vector_extract_constraint_system`](layout_inference.md#_vector_extract_constraint_system), [`_async_store_scales_smem_to_tmem_constraint_system`](layout_inference.md#_async_store_scales_smem_to_tmem_constraint_system), [`_async_store_tmem_constraint_system`](layout_inference.md#_async_store_tmem_constraint_system), [`_constant_constraint_system`](layout_inference.md#_constant_constraint_system), [`_get_cluster_ref_constraint_system`](layout_inference.md#_get_cluster_ref_constraint_system), [`_memref_cast_op_constraint_system`](layout_inference.md#_memref_cast_op_constraint_system), [`_memref_transpose_op_constraint_system`](layout_inference.md#_memref_transpose_op_constraint_system), [`_print_layout_constraint_system`](layout_inference.md#_print_layout_constraint_system), [`_async_store_sparse_metadata_smem_to_tmem_constraint_system`](layout_inference.md#_async_store_sparse_metadata_smem_to_tmem_constraint_system), [`_broadcast_in_dim_constraint_system`](layout_inference.md#_broadcast_in_dim_constraint_system), [`_broadcasted_iota_constraint_system`](layout_inference.md#_broadcasted_iota_constraint_system)  (+18 more)

### `_TypeAndLayout`
- def: [`jax/experimental/mosaic/gpu/layout_inference.py:2334`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L2334)
- signature: `class _TypeAndLayout:`
- members:
  - `layout` — [`L2336`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L2336)
- used by: [`assign_layouts`](layout_inference.md#assign_layouts), [`_to_transform_attrs`](layout_inference.md#assign_layouts._to_transform_attrs)

## Functions
- `_add_constraint_system_derivation_rule(op: type[ir.OpView])` — [`L672`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L672) — documented in [jax-experimental-mosaic-gpu-layout_inference](../../../../../concepts/jax-experimental-mosaic-gpu-layout_inference.md)
- `_async_load_store_constraint_system(ctx: DerivationContext, op: mgpu.AsyncLoadOp | mgpu.AsyncStoreOp)` — [`L2182`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L2182)
- `_async_load_tmem_constraint_system(ctx: DerivationContext, op: mgpu.AsyncLoadTmemOp)` — [`L1726`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L1726)
- `_async_prefetch_constraint_system(ctx: DerivationContext, op: mgpu.AsyncPrefetchOp)` — [`L2236`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L2236)
- `_async_store_scales_smem_to_tmem_constraint_system(ctx: DerivationContext, op: mgpu.AsyncStoreScalesSmemToTmemOp)` — [`L1798`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L1798)
- `_async_store_smem_constraint_system(ctx: DerivationContext, op: mgpu.AsyncStoreSmemOp)` — [`L853`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L853)
- `_async_store_smem_to_tmem_constraint_system(ctx: DerivationContext, op: mgpu.AsyncStoreSmemToTmemOp)` — [`L1747`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L1747)
- `_async_store_sparse_metadata_smem_to_tmem_constraint_system(ctx: DerivationContext, op: mgpu.AsyncStoreSparseMetadataSmemToTmemOp)` — [`L1777`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L1777)
- `_async_store_tmem_constraint_system(ctx: DerivationContext, op: mgpu.AsyncStoreTmemOp)` — [`L1850`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L1850)
- `_broadcast_in_dim_constraint_system(ctx: DerivationContext, op: mgpu.BroadcastInDimOp)` — [`L1281`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L1281)
- `_broadcasted_iota_constraint_system(ctx: DerivationContext, op: mgpu.BroadcastedIotaOp)` — [`L889`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L889)
- `_cluster_launch_control_ops_constraint_system(ctx: DerivationContext, op: mgpu.TryClusterCancelOp | mgpu.QueryClusterCancelOp)` — [`L2080`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L2080)
- `_compute_swizzle(ty: ir.Type, tile_transform: lc.TileTransform | None)` — [`L2299`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L2299) — Computes the swizzle mode given a tiling transform and a data type.
- `_conjure_transforms_for_smem_ref(variable: cs.Variable, divide_constraints_per_var: dict[cs.Variable, cs.Divides])` — [`L253`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L253)
- `_constant_constraint_system(ctx: DerivationContext, constant_op: arith.ConstantOp)` — [`L921`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L921)
- `_custom_primitive_constraint_system(ctx: DerivationContext, op: mgpu.CustomPrimitiveOp)` — [`L1491`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L1491)
- `_debug_print_constraint_system(ctx: DerivationContext, op: mgpu.DebugPrintOp)` — [`L869`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L869)
- `_default_tmem_layout_for_variable(variable: cs.Variable)` — [`L205`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L205) — Returns a default TMEM layout for the given variable, if one is defined.
- `_divides_per_var(constraints: Sequence[cs.Constraint])` — [`L402`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L402)
- `_ensure_all_layouts_are_set(op: ir.OpView)` — [`L2260`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L2260)
- `_ensure_right_number_of_layouts(filter_fn: Callable[[ir.Value], bool], attr_suffix: str, value_type: str, op: ir.OpView)` — [`L2271`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L2271) — Ensures that the right number of in/out layouts are provided for an op.
- `_extract_layout_candidates_from_broadcast(src: cs.RegisterLayout, dst: cs.Variable, dims: tuple[int, ...])` — [`L415`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L415) — Yields layout candidates for a broadcast equation.
- `_extract_layout_candidates_from_mma_tiling(mma_tiling: cs.IsValidMmaTiling)` — [`L376`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L376)
- `_extract_layout_candidates_from_smem_registers_transfer(constraint: cs.IsTransferableSmemRegisters, division_constraint_per_var: dict[cs.Variable, cs.Divides], arch: tuple[int, int])` — [`L336`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L336)
- `_extract_layout_candidates_from_tmem_registers_transfer(constraint: cs.IsTransferableTmemRegisters)` — [`L295`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L295)
- `_extract_smem_transforms_from_custom_transform_attrs(transform_attrs: ir.ArrayAttr)` — [`L2090`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L2090)
- `_extract_strided_slice_constraint_system(ctx: DerivationContext, op: vector.ExtractStridedSliceOp)` — [`L1408`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L1408)
- `_extract_variable_assignments_from_constraints(constraints: Sequence[cs.Constraint], arch: tuple[int, int])` — [`L431`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L431) — Attempts to extract variable assignments from all constraints.
- `_for_constraint_system(ctx: DerivationContext, op: scf.ForOp)` — [`L958`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L958)
- `_get_cluster_ref_constraint_system(ctx: DerivationContext, op: mgpu.GetClusterRefOp)` — [`L1889`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L1889)
- `_index_switch_constraint_system(ctx: DerivationContext, op: scf.IndexSwitchOp)` — [`L1068`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L1068)
- `_insert_strided_slice_constraint_system(ctx: DerivationContext, op: vector.InsertStridedSliceOp)` — [`L1464`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L1464)
- `_is_smem_ref(v: ir.Value)` — [`L686`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L686)
- `_is_tmem_ref(v: ir.Value)` — [`L692`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L692)
- `_layout_cast_constraint_system(ctx: DerivationContext, op: mgpu.LayoutCastOp)` — [`L1088`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L1088)
- `_memref_cast_op_constraint_system(ctx: DerivationContext, op: memref.CastOp)` — [`L1949`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L1949)
- `_memref_collapse_shape_op_constraint_system(ctx: DerivationContext, op: memref.CollapseShapeOp)` — [`L2020`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L2020)
- `_memref_expand_shape_op_equation_system(ctx: DerivationContext, op: memref.ExpandShapeOp)` — [`L1984`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L1984)
- `_memref_load_store_op_constraint_system(ctx: DerivationContext, op: memref.LoadOp | memref.StoreOp)` — [`L2059`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L2059)
- `_memref_subview_constraint_system(ctx: DerivationContext, op: memref.SubViewOp)` — [`L1900`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L1900)
- `_memref_transpose_op_constraint_system(ctx: DerivationContext, op: memref.TransposeOp)` — [`L1960`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L1960)
- `_mma_constraint_system(ctx: DerivationContext, op: mgpu.MMAOp)` — [`L1178`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L1178)
- `_multi_dim_reduction_constraint_system(ctx: DerivationContext, op: vector.MultiDimReductionOp)` — [`L1255`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L1255)
- `_multimem_load_reduce_constraint_system(_: DerivationContext, op: mgpu.MultimemLoadReduceOp)` — [`L800`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L800)
- `_optimization_barrier_constraint_system(ctx: DerivationContext, op: mgpu.OptimizationBarrierOp)` — [`L901`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L901)
- `_pointwise_op_constraint_system(ctx: DerivationContext, op: ir.OpView)` — [`L696`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L696)
- `_print_layout_constraint_system(ctx: DerivationContext, op: mgpu.PrintLayoutOp)` — [`L879`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L879)
- `_register_layouts_for_optimized_transfer_to_smem(shaped_type: ir.ShapedType, smem_layout: cs.SMEMTransforms, arch: tuple[int, int])` — [`L218`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L218) — Yields register layout candidates for optimized transfers to SMEM.
- `_shape_cast_constraint_system(ctx: DerivationContext, op: vector.ShapeCastOp)` — [`L1365`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L1365)
- `_slice_smem_constraint_system(ctx: DerivationContext, op: mgpu.SliceSMEMOp)` — [`L1871`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L1871)
- `_slice_tmem_constraint_system(ctx: DerivationContext, op: mgpu.SliceTmemOp)` — [`L1824`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L1824)
- `_strided_layout_for_variable(variable: cs.Variable)` — [`L193`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L193) — Returns a strided layout for the given variable.
- `_tcgen05_mma_constraint_system(ctx: DerivationContext, op: mgpu.TcGen05MMAOp)` — [`L1599`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L1599)
- `_terminator(block: ir.Block, expected_terminator: type[ir.OpView])` — [`L945`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L945) — Returns the terminator of the given block.
- `_tmem_alloc_constraint_system(ctx: DerivationContext, op: mgpu.TmemAllocOp)` — [`L1572`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L1572)
- `_tmem_dealloc_constraint_system(ctx: DerivationContext, op: mgpu.TmemDeallocOp)` — [`L1589`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L1589)
- `_tmem_layout_cast_constraint_system(ctx: DerivationContext, op: mgpu.TmemLayoutCastOp)` — [`L1552`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L1552)
- `_tmem_layout_from_layout_attr(layout_attr: ir.Attribute)` — [`L1541`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L1541)
- `_to_transform_attrs(transforms: list[_TypeAndLayout])` — [`L2413`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L2413)
- `_vector_broadcast_constraint_system(ctx: DerivationContext, op: vector.BroadcastOp)` — [`L1227`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L1227)
- `_vector_concat_constraint_system(ctx: DerivationContext, op: mgpu.VectorConcatOp)` — [`L1344`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L1344)
- `_vector_extract_constraint_system(ctx: DerivationContext, op: vector.ExtractOp)` — [`L1432`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L1432)
- `_vector_load_constraint_system(ctx: DerivationContext, op: mgpu.VectorLoadOp)` — [`L763`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L763)
- `_vector_reduction_constraint_system(ctx: DerivationContext, op: vector.ReductionOp)` — [`L1245`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L1245)
- `_vector_store_constraint_system(ctx: DerivationContext, op: Any)` — [`L813`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L813)
- `_vector_value_sites_and_assignments_for_async_ops(op: mgpu.AsyncLoadOp | mgpu.AsyncStoreOp | mgpu.AsyncPrefetchOp)` — [`L2146`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L2146)
- `_warp_map_constraint_system(ctx: DerivationContext, op: mgpu.WarpMapOp)` — [`L2246`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L2246)
- `_wgmma_constraint_system(ctx: DerivationContext, op: mgpu.WGMMAOp)` — [`L1116`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L1116)
- `_while_constraint_system(ctx: DerivationContext, op: scf.WhileOp)` — [`L1029`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L1029)
- `_with_transforms_constraint_system(ctx: DerivationContext, op: mgpu.WithTransformsOp)` — [`L2116`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L2116)
- `assign_layouts(solution: dict[ValueSite, cs.Constant])` — [`L2339`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L2339) — Assigns the layouts in `solution` to the MLIR ops they belong to.
- `assign_scaled_layout(scale_operand)` — [`L1701`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L1701)
- `check_layout_assignment(var: cs.Variable, layout: cs.Constant)` — [`L2565`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L2565) — Raises if the given layout can not be assigned to the given `Variable`.
- `conjure_assignment(unknowns: Sequence[cs.Variable], constraint_system: cs.ConstraintSystem, arch: tuple[int, int])` — [`L466`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L466) — Attempts to conjure an assignment for an unknown variable.
- `consumer_operands(result: ValueSite)` — [`L2478`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L2478) — Given a result or an argument, returns the corresponding operands in its consumers.
- `derive_relayout_constraints(value_sites_for_variable: ValueSitesForVariable)` — [`L2491`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L2491) — Derives relayout constraints from the given variable mapping.
- `dynamic_gcd(a: int, b: ir.Value)` — [`L1007`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L1007)
- `extract_assignment_candidates_from_reduce_equation(small: cs.RegisterLayout, large: cs.Variable, reduction_dims: tuple[int, ...], keep_dims: bool)` — [`L142`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L142) — Yields layout candidates for the reduce equation `small = reduce(large, reduction_dims).
- `find_assignments_for(unknowns: Sequence[cs.Variable], constraint_system: cs.ConstraintSystem, *, fuel: int, arch: tuple[int, int])` — [`L522`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L522) — Attempts to find assignments that satisfy `constraint_system` for `unknowns`.
- `gather_constraints(op: Any)` — [`L2599`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L2599)
- `infer_layout(module: ir.Module, *, fuel: int = _DEFAULT_LAYOUT_INFERENCE_FUEL, arch: tuple[int, int] = (9, 0))` — [`L2574`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L2574) — Infers layouts for the given module.
- `is_hopper(arch: tuple[int, int])` — [`L138`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L138)
- `is_terminator(op: ir.OpView)` — [`L2542`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L2542)
- `is_vector(v: ir.Value)` — [`L682`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L682)
- `prime_decomposition(n: int)` — [`L986`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L986) — Returns the prime decomposition of the given number `n` as a list of ints.
- `priority(constraint: cs.Constraint)` — [`L436`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L436)
- `producer_result(operand: ValueSite)` — [`L2455`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L2455) — Given an operand, returns the corresponding result in its producer.
- `traverse_op(op: ir.OpView, callback: Callable[[ir.OpView], None])` — [`L2546`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L2546) — Traverses the operation and applies the callback in pre-order fashion.
- `vector_value_sites(op: ir.OpView)` — [`L2440`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L2440) — Returns all the vector operands and results for the given op.
- `wrapper(rule: ConstraintSystemDerivationRule)` — [`L673`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L673)

## Module values
- `ConstraintSystemDerivationRule` — [`L663`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L663)
- `ConstraintSystemDerivationRuleResult` — [`L660`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L660)
- `ValueSitesForVariable` — [`L645`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L645)
- `_DEFAULT_LAYOUT_INFERENCE_FUEL` — [`L61`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L61)
- `_constraint_system_derivation_rules` — [`L667`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L667)
- `_op` — [`L706`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L706)
- `_op_name_regex` — [`L74`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L74)
- `logger` — [`L46`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/layout_inference.py#L46)


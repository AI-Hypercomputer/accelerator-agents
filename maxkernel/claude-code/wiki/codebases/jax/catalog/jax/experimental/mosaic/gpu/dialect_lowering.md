---
title: 'Module: jax/experimental/mosaic/gpu/dialect_lowering.py'
type: catalog
provenance: extracted
module: jax/experimental/mosaic/gpu/dialect_lowering.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax.experimental.mosaic.gpu.dialect_lowering`/
symbols:
  LoweringContext: LoweringContext#
  _register_lowering: _register_lowering().
  _mgpu_async_load_op_lowering_rule: _mgpu_async_load_op_lowering_rule().
  _fragmented_array_from_ir: _fragmented_array_from_ir().
  _vector_load_op_lowering_rule: _vector_load_op_lowering_rule().
  _async_store_smem_op_lowering_rule: _async_store_smem_op_lowering_rule().
  _vector_multi_dim_reduction_op_lowering_rule: _vector_multi_dim_reduction_op_lowering_rule().
  _is_signed: _is_signed.
  _memref_subview_op_lowering_rule: _memref_subview_op_lowering_rule().
  _vector_store_op_lowering_rule: _vector_store_op_lowering_rule().
  _mgpu_async_store_op_lowering_rule: _mgpu_async_store_op_lowering_rule().
  _mgpu_arrive_expect_tx_op_lowering_rule: _mgpu_arrive_expect_tx_op_lowering_rule().
  fragmented_array_to_ir: fragmented_array_to_ir().
  _tmem_ref_from_ir: _tmem_ref_from_ir().
  transform_type: transform_type().
  _vector_reduction_op_lowering_rule: _vector_reduction_op_lowering_rule().
  unwrap_transformed_memref: unwrap_transformed_memref().
  _mgpu_wgmma_op_lowering_rule: _mgpu_wgmma_op_lowering_rule().
  _lowering_context: _lowering_context().
  _binary_impl: _binary_impl.
  _mgpu_arrive_op_lowering_rule: _mgpu_arrive_op_lowering_rule().
  _memref_transpose_op_lowering_rule: _memref_transpose_op_lowering_rule().
  _memref_collapse_shape_op_lowering_rule: _memref_collapse_shape_op_lowering_rule().
  _tcgen05_mma_op_lowering_rule: _tcgen05_mma_op_lowering_rule().
  LoweringContext.single_lane_predicate: LoweringContext#single_lane_predicate().
  pprint_layout: pprint_layout().
  _gmem_slice_and_predicate: _gmem_slice_and_predicate().
  _optimization_barrier_op_lowering_rule: _optimization_barrier_op_lowering_rule().
  _memref_cast_op_lowering_rule: _memref_cast_op_lowering_rule().
  _tcgen05_commit_arrive_op_lowering_rule: _tcgen05_commit_arrive_op_lowering_rule().
  _mgpu_warp_map_op_lowering_rule: _mgpu_warp_map_op_lowering_rule().
  LoweringContext.lower_op: LoweringContext#lower_op().
  _multimem_load_reduce_op_lowering_rule: _multimem_load_reduce_op_lowering_rule().
  _vector_extract_op_lowering_rule: _vector_extract_op_lowering_rule().
  _async_store_smem_to_tmem_lowering_rule: _async_store_smem_to_tmem_lowering_rule().
  _mgpu_mma_op_lowering_rule: _mgpu_mma_op_lowering_rule().
  _while_op_lowering_rule: _while_op_lowering_rule().
  _index_switch_op_lowering_rule: _index_switch_op_lowering_rule().
  _initialize_barrier_op_lowering_rule: _initialize_barrier_op_lowering_rule().
  _arith_constant_op_lowering_rule: _arith_constant_op_lowering_rule().
  _broadcasted_iota_op_lowering_rule: _broadcasted_iota_op_lowering_rule().
  _vector_broadcast_op_lowering_rule: _vector_broadcast_op_lowering_rule().
  _vector_extract_strided_slice_op_lowering_rule: _vector_extract_strided_slice_op_lowering_rule().
  _vector_insert_strided_slice_op_lowering_rule: _vector_insert_strided_slice_op_lowering_rule().
  _mgpu_vector_concat_op_lowering_rule: _mgpu_vector_concat_op_lowering_rule().
  _async_copy_scales_smem_to_tmem_lowering_rule: _async_copy_scales_smem_to_tmem_lowering_rule().
  _async_copy_sparse_metadata_smem_to_tmem_lowering_rule: _async_copy_sparse_metadata_smem_to_tmem_lowering_rule().
  _async_load_tmem_op_lowering_rule: _async_load_tmem_op_lowering_rule().
  _for_op_lowering_rule: _for_op_lowering_rule().
  _try_cluster_cancel_op_lowering_rule: _try_cluster_cancel_op_lowering_rule().
  swizzle_from_transforms_attr: swizzle_from_transforms_attr().
  memref_transforms_from_transforms_attr: memref_transforms_from_transforms_attr().
  _move_scf_block_to_block_with_flattened_arguments: _move_scf_block_to_block_with_flattened_arguments().
  _traverse_op_lowering_rule: _traverse_op_lowering_rule().
  _divf_lowering_rule: _divf_lowering_rule().
  LoweringContext.thread_semantics: LoweringContext#thread_semantics.
  _mgpu_layout_cast_op_lowering_rule: _mgpu_layout_cast_op_lowering_rule().
  _mgpu_broadcast_in_dim_op_lowering_rule: _mgpu_broadcast_in_dim_op_lowering_rule().
  _bitcast_op_lowering_rule: _bitcast_op_lowering_rule().
  _mgpu_slice_smem_op_lowering_rule: _mgpu_slice_smem_op_lowering_rule().
  _slice_smem: _slice_smem().
  _flatten_ir_values: _flatten_ir_values().
  _unflatten_ir_values: _unflatten_ir_values().
  _target_is_signed: _target_is_signed.
  _get_cluster_ref_op_lowering_rule: _get_cluster_ref_op_lowering_rule().
  _print_layout_op_lowering_rule: _print_layout_op_lowering_rule().
  _vector_shape_cast_op_lowering_rule: _vector_shape_cast_op_lowering_rule().
  _mgpu_async_prefetch_op_lowering_rule: _mgpu_async_prefetch_op_lowering_rule().
  _cmpi_op_lowering_rule: _cmpi_op_lowering_rule().
  _select_op_lowering_rule: _select_op_lowering_rule().
  _mgpu_wait_op_lowering_rule: _mgpu_wait_op_lowering_rule().
  _memref_expand_shape_op_lowering_rule: _memref_expand_shape_op_lowering_rule().
  _tmem_alloc_op_lowering_rule: _tmem_alloc_op_lowering_rule().
  _tmem_relinquish_alloc_permit_op_lowering_rule: _tmem_relinquish_alloc_permit_op_lowering_rule().
  _tmem_dealloc_op_lowering_rule: _tmem_dealloc_op_lowering_rule().
  _binary_op_lowering_rule: _binary_op_lowering_rule().
  _conversion_op_lowering_rule: _conversion_op_lowering_rule().
  _unary_op_lowering_rule: _unary_op_lowering_rule().
  _check_collapse_shape: _check_collapse_shape().
  LoweringContext.launch_context: LoweringContext#launch_context.
  lower_mgpu_dialect: lower_mgpu_dialect().
  _retry_on_failure: _retry_on_failure().
  _debug_print_op_lowering_rule: _debug_print_op_lowering_rule().
  _tmem_layout_cast_lowering_rule: _tmem_layout_cast_lowering_rule().
  _cmpf_op_lowering_rule: _cmpf_op_lowering_rule().
  _mgpu_with_transforms_op_lowering_rule: _mgpu_with_transforms_op_lowering_rule().
  _async_store_tmem_op_lowering_rule: _async_store_tmem_op_lowering_rule().
  _if_op_lowering_rule: _if_op_lowering_rule().
  LoweringContext.check_collective: LoweringContext#check_collective().
  _infer_flat_result_types: _infer_flat_result_types().
  _lowerings._lowerings: _lowerings._lowerings.
  MlirLoweringRuleResult: MlirLoweringRuleResult.
  _vector_load_op_lowering_rule.load_tiled: _vector_load_op_lowering_rule().load_tiled().
  _VectorTemplate: _VectorTemplate.
  MlirLoweringRule: MlirLoweringRule.
  _default_is_signed: _default_is_signed().
  wrap_transformed_memref: wrap_transformed_memref().
  _vector_load_op_lowering_rule._fragmented_array_to_ir: _vector_load_op_lowering_rule()._fragmented_array_to_ir().
  _slice_tmem_lowering_rule: _slice_tmem_lowering_rule().
  _memref_load_op_lowering_rule: _memref_load_op_lowering_rule().
  _memref_store_op_lowering_rule: _memref_store_op_lowering_rule().
  _query_cluster_cancel_op_lowering_rule: _query_cluster_cancel_op_lowering_rule().
  _register_lowering.wrapper: _register_lowering().wrapper().
  _vector_load_op_lowering_rule.load_untiled: _vector_load_op_lowering_rule().load_untiled().
  _vector_store_op_lowering_rule.store_tiled: _vector_store_op_lowering_rule().store_tiled().
  _async_store_smem_op_lowering_rule.store_tiled_async: _async_store_smem_op_lowering_rule().store_tiled_async().
  LoweringContext.auto_barriers: LoweringContext#auto_barriers.
  _assume_multiple_op_lowering_rule: _assume_multiple_op_lowering_rule().
  _tcgen05_mma_op_lowering_rule.tmem_layout: _tcgen05_mma_op_lowering_rule().tmem_layout().
  _mgpu_custom_primitive_op_lowering_rule: _mgpu_custom_primitive_op_lowering_rule().
  _op: _op.
  _tmem_ref_to_ir: _tmem_ref_to_ir().
  LoweringContext.smem_requested_bytes: LoweringContext#smem_requested_bytes.
  _undo_conversion_cast: _undo_conversion_cast().
  RECURSED: RECURSED.
  _vector_store_op_lowering_rule.store_untiled: _vector_store_op_lowering_rule().store_untiled().
  CMPF_IMPLS: CMPF_IMPLS.
  _tile_transform_offsets: _tile_transform_offsets().
  LoweringContext.single_warp_per_block_predicate: LoweringContext#single_warp_per_block_predicate.
  LoweringContext.is_collective_kernel: LoweringContext#is_collective_kernel.
  Recursed: Recursed#
  LoweringContext._single_thread_per_warp_predicate: LoweringContext#_single_thread_per_warp_predicate.
  LoweringContext._single_thread_per_warpgroup_predicate: LoweringContext#_single_thread_per_warpgroup_predicate.
  _lowered_barrier_type: _lowered_barrier_type().
  _combining_kind: _combining_kind().
  _is_reduction_signed: _is_reduction_signed().
  _supported_warp_lowerings._supported_warp_lowerings: _supported_warp_lowerings._supported_warp_lowerings.
  LoweringContext.single_thread_per_block_predicate: LoweringContext#single_thread_per_block_predicate.
  _Transfer: _Transfer#
  tile_offset: tile_offset().
  tile_strides: tile_strides().
  CMPI_IMPLS: CMPI_IMPLS.
  _permutation_to_affine_map_attr: _permutation_to_affine_map_attr().
  _should_lower: _should_lower().
  _gpu_launch_op: _gpu_launch_op().
  _source_is_signed: _source_is_signed.
  _unary_impl: _unary_impl.
  _Transfer.__call__: _Transfer#__call__().
---
# Module: [`jax/experimental/mosaic/gpu/dialect_lowering.py`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py)

## Classes
### `LoweringContext`
- def: [`jax/experimental/mosaic/gpu/dialect_lowering.py:53`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L53)
- signature: `class LoweringContext:`
- members:
  - `check_collective(self, op: ir.OpView)` — [`L80`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L80) — Checks that the collective attribute is consistent across operations.
  - `lower_op(self, op: ir.OpView)` — [`L96`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L96)
  - `single_lane_predicate(self)` — [`L69`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L69)
  - `auto_barriers` — [`L59`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L59)
  - `is_collective_kernel` — [`L61`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L61)
  - `launch_context` — [`L54`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L54)
  - `single_thread_per_block_predicate` — [`L57`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L57)
  - `single_warp_per_block_predicate` — [`L58`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L58)
  - `smem_requested_bytes` — [`L60`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L60)
  - `thread_semantics` — [`L64`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L64)
- protocol/private: `_single_thread_per_warp_predicate`[`L55`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L55), `_single_thread_per_warpgroup_predicate`[`L56`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L56)
- uses (calls/refs, reference-scoped): [`ThreadSubset`](utils.md#ThreadSubset), [`WARPGROUP`](utils.md#ThreadSubset.WARPGROUP), [`WARP`](utils.md#ThreadSubset.WARP), [`_lowerings`](dialect_lowering.md#_lowerings._lowerings), [`LaunchContext`](launch_context.md#LaunchContext), [`Recursed`](dialect_lowering.md#Recursed), [`_supported_warp_lowerings`](dialect_lowering.md#_supported_warp_lowerings._supported_warp_lowerings), [`_should_lower`](dialect_lowering.md#_should_lower)
- used by: [`_mgpu_async_load_op_lowering_rule`](dialect_lowering.md#_mgpu_async_load_op_lowering_rule), [`_async_store_smem_op_lowering_rule`](dialect_lowering.md#_async_store_smem_op_lowering_rule), [`_vector_load_op_lowering_rule`](dialect_lowering.md#_vector_load_op_lowering_rule), [`_vector_multi_dim_reduction_op_lowering_rule`](dialect_lowering.md#_vector_multi_dim_reduction_op_lowering_rule), [`_memref_subview_op_lowering_rule`](dialect_lowering.md#_memref_subview_op_lowering_rule), [`_mgpu_arrive_expect_tx_op_lowering_rule`](dialect_lowering.md#_mgpu_arrive_expect_tx_op_lowering_rule), [`_mgpu_async_store_op_lowering_rule`](dialect_lowering.md#_mgpu_async_store_op_lowering_rule), [`_vector_store_op_lowering_rule`](dialect_lowering.md#_vector_store_op_lowering_rule), [`_vector_reduction_op_lowering_rule`](dialect_lowering.md#_vector_reduction_op_lowering_rule), [`_mgpu_wgmma_op_lowering_rule`](dialect_lowering.md#_mgpu_wgmma_op_lowering_rule), [`_lowering_context`](dialect_lowering.md#_lowering_context), [`_memref_collapse_shape_op_lowering_rule`](dialect_lowering.md#_memref_collapse_shape_op_lowering_rule), [`_memref_transpose_op_lowering_rule`](dialect_lowering.md#_memref_transpose_op_lowering_rule), [`_mgpu_arrive_op_lowering_rule`](dialect_lowering.md#_mgpu_arrive_op_lowering_rule), [`_tcgen05_mma_op_lowering_rule`](dialect_lowering.md#_tcgen05_mma_op_lowering_rule), [`_gmem_slice_and_predicate`](dialect_lowering.md#_gmem_slice_and_predicate), [`_memref_cast_op_lowering_rule`](dialect_lowering.md#_memref_cast_op_lowering_rule), [`_mgpu_warp_map_op_lowering_rule`](dialect_lowering.md#_mgpu_warp_map_op_lowering_rule), [`_optimization_barrier_op_lowering_rule`](dialect_lowering.md#_optimization_barrier_op_lowering_rule), [`_tcgen05_commit_arrive_op_lowering_rule`](dialect_lowering.md#_tcgen05_commit_arrive_op_lowering_rule), [`_async_store_smem_to_tmem_lowering_rule`](dialect_lowering.md#_async_store_smem_to_tmem_lowering_rule), [`_index_switch_op_lowering_rule`](dialect_lowering.md#_index_switch_op_lowering_rule), [`_mgpu_mma_op_lowering_rule`](dialect_lowering.md#_mgpu_mma_op_lowering_rule), [`_multimem_load_reduce_op_lowering_rule`](dialect_lowering.md#_multimem_load_reduce_op_lowering_rule), [`_vector_extract_op_lowering_rule`](dialect_lowering.md#_vector_extract_op_lowering_rule), [`_while_op_lowering_rule`](dialect_lowering.md#_while_op_lowering_rule), [`_arith_constant_op_lowering_rule`](dialect_lowering.md#_arith_constant_op_lowering_rule), [`_async_copy_scales_smem_to_tmem_lowering_rule`](dialect_lowering.md#_async_copy_scales_smem_to_tmem_lowering_rule), [`_async_copy_sparse_metadata_smem_to_tmem_lowering_rule`](dialect_lowering.md#_async_copy_sparse_metadata_smem_to_tmem_lowering_rule), [`_async_load_tmem_op_lowering_rule`](dialect_lowering.md#_async_load_tmem_op_lowering_rule), [`_broadcasted_iota_op_lowering_rule`](dialect_lowering.md#_broadcasted_iota_op_lowering_rule), [`_for_op_lowering_rule`](dialect_lowering.md#_for_op_lowering_rule), [`_initialize_barrier_op_lowering_rule`](dialect_lowering.md#_initialize_barrier_op_lowering_rule), [`_mgpu_vector_concat_op_lowering_rule`](dialect_lowering.md#_mgpu_vector_concat_op_lowering_rule), [`_try_cluster_cancel_op_lowering_rule`](dialect_lowering.md#_try_cluster_cancel_op_lowering_rule), [`_vector_broadcast_op_lowering_rule`](dialect_lowering.md#_vector_broadcast_op_lowering_rule), [`_vector_extract_strided_slice_op_lowering_rule`](dialect_lowering.md#_vector_extract_strided_slice_op_lowering_rule), [`_vector_insert_strided_slice_op_lowering_rule`](dialect_lowering.md#_vector_insert_strided_slice_op_lowering_rule), [`_move_scf_block_to_block_with_flattened_arguments`](dialect_lowering.md#_move_scf_block_to_block_with_flattened_arguments), [`_traverse_op_lowering_rule`](dialect_lowering.md#_traverse_op_lowering_rule)  (+33 more)

### `Recursed`
- def: [`jax/experimental/mosaic/gpu/dialect_lowering.py:126`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L126)
- used by: [`lower_op`](dialect_lowering.md#LoweringContext.lower_op), [`MlirLoweringRuleResult`](dialect_lowering.md#MlirLoweringRuleResult), [`RECURSED`](dialect_lowering.md#RECURSED)

### `_Transfer`  ·  implements/extends Protocol
- def: [`jax/experimental/mosaic/gpu/dialect_lowering.py:415`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L415)
- signature: `class _Transfer(Protocol):`
- protocol/private: `__call__`[`L416`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L416)
- used by: [`_retry_on_failure`](dialect_lowering.md#_retry_on_failure)

## Functions
- `_arith_constant_op_lowering_rule(_: LoweringContext, op: arith.ConstantOp)` — [`L389`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L389)
- `_assume_multiple_op_lowering_rule(_: LoweringContext, op: mgpu.AssumeMultipleOp)` — [`L313`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L313)
- `_async_copy_scales_smem_to_tmem_lowering_rule(ctx: LoweringContext, op: mgpu.AsyncStoreScalesSmemToTmemOp)` — [`L1311`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L1311)
- `_async_copy_sparse_metadata_smem_to_tmem_lowering_rule(ctx: LoweringContext, op: mgpu.AsyncStoreSparseMetadataSmemToTmemOp)` — [`L1328`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L1328)
- `_async_load_tmem_op_lowering_rule(ctx: LoweringContext, op: mgpu.AsyncLoadTmemOp)` — [`L2522`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L2522) — Lowering rule for mgpu.AsyncLoadTmemOp.
- `_async_store_smem_op_lowering_rule(ctx: LoweringContext, op: mgpu.AsyncStoreSmemOp)` — [`L612`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L612)
- `_async_store_smem_to_tmem_lowering_rule(ctx: LoweringContext, op: mgpu.AsyncStoreSmemToTmemOp)` — [`L1343`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L1343)
- `_async_store_tmem_op_lowering_rule(ctx: LoweringContext, op: mgpu.AsyncStoreTmemOp)` — [`L2538`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L2538) — Lowering rule for mgpu.AsyncStoreTmemOp.
- `_binary_op_lowering_rule(_: LoweringContext, op: Any, is_signed: bool | None, impl: Callable[[fa.FragmentedArray, fa.FragmentedArray], fa.FragmentedArray])` — [`L1472`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L1472)
- `_bitcast_op_lowering_rule(_: LoweringContext, op: arith.BitcastOp)` — [`L1601`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L1601)
- `_broadcasted_iota_op_lowering_rule(ctx: LoweringContext, op: mgpu.BroadcastedIotaOp)` — [`L719`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L719)
- `_check_collapse_shape(op: memref.CollapseShapeOp, in_transforms: Sequence[lc.MemRefTransform], out_transforms: Sequence[lc.MemRefTransform])` — [`L2216`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L2216)
- `_cmpf_op_lowering_rule(_: LoweringContext, op: arith.CmpFOp)` — [`L1586`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L1586)
- `_cmpi_op_lowering_rule(_: LoweringContext, op: arith.CmpIOp)` — [`L1561`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L1561)
- `_combining_kind(attr: ir.Attribute)` — [`L832`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L832)
- `_conversion_op_lowering_rule(_: LoweringContext, op: ir.OpView, source_is_signed: bool | None, target_is_signed: bool | None)` — [`L1391`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L1391)
- `_debug_print_op_lowering_rule(ctx: LoweringContext, op: mgpu.DebugPrintOp)` — [`L668`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L668)
- `_default_is_signed(dtype: ir.Type)` — [`L196`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L196) — Returns `False` for Integer types, `None` otherwise.
- `_divf_lowering_rule(ctx: LoweringContext, op: Any)` — [`L1522`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L1522)
- `_flatten_ir_values(values: Sequence[ir.Value], fa_layouts: Iterable[ir.Attribute])` — [`L2621`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L2621) — Flattens a sequence of values.
- `_for_op_lowering_rule(ctx: LoweringContext, for_op: scf.ForOp)` — [`L2725`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L2725)
- `_fragmented_array_from_ir(fragmented_array_as_ir: ir.Value, layout: ir.Attribute, is_signed: bool | None = None)` — [`L205`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L205)
- `_fragmented_array_to_ir(fragmented_array: fa.FragmentedArray)` — [`L452`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L452)
- `_get_cluster_ref_op_lowering_rule(_: LoweringContext, op: mgpu.GetClusterRefOp)` — [`L363`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L363)
- `_gmem_slice_and_predicate(ctx: LoweringContext, op: mgpu.AsyncLoadOp | mgpu.AsyncPrefetchOp | mgpu.AsyncStoreOp)` — [`L1128`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L1128) — Returns the GMEM slice and predicate for the given async op.
- `_gpu_launch_op(module: ir.Module)` — [`L2929`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L2929)
- `_if_op_lowering_rule(ctx: LoweringContext, if_op: scf.IfOp)` — [`L2845`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L2845)
- `_index_switch_op_lowering_rule(ctx: LoweringContext, switch_op: scf.IndexSwitchOp)` — [`L2855`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L2855)
- `_infer_flat_result_types(op: ir.OpView, out_layouts: Sequence[ir.Attribute])` — [`L2826`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L2826)
- `_initialize_barrier_op_lowering_rule(ctx: LoweringContext, op: mgpu.InitializeBarrierOp)` — [`L295`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L295)
- `_is_reduction_signed(kind: vector.CombiningKind)` — [`L838`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L838)
- `_lowered_barrier_type()` — [`L290`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L290)
- `_lowering_context(module: ir.Module, launch_context: lc.LaunchContext | None, auto_barriers: bool)` — [`L2939`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L2939) — Returns a `LoweringContext` for the given `LaunchContext`.
- `_memref_cast_op_lowering_rule(ctx: LoweringContext, op: memref.CastOp)` — [`L2023`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L2023)
- `_memref_collapse_shape_op_lowering_rule(ctx: LoweringContext, op: memref.CollapseShapeOp)` — [`L2267`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L2267)
- `_memref_expand_shape_op_lowering_rule(ctx: LoweringContext, op: memref.ExpandShapeOp)` — [`L2171`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L2171)
- `_memref_load_op_lowering_rule(ctx: LoweringContext, op: memref.LoadOp)` — [`L2320`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L2320) — Lowering rule for memref.LoadOp.
- `_memref_store_op_lowering_rule(ctx: LoweringContext, op: memref.StoreOp)` — [`L2342`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L2342) — Lowering rule for memref.StoreOp.
- `_memref_subview_op_lowering_rule(ctx: LoweringContext, op: memref.SubViewOp)` — [`L1897`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L1897)
- `_memref_transpose_op_lowering_rule(ctx: LoweringContext, op: memref.TransposeOp)` — [`L2089`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L2089)
- `_mgpu_arrive_expect_tx_op_lowering_rule(ctx: LoweringContext, arrive_expect_tx_op: mgpu.ArriveExpectTxOp)` — [`L1748`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L1748)
- `_mgpu_arrive_op_lowering_rule(ctx: LoweringContext, arrive_op: mgpu.ArriveOp)` — [`L1718`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L1718)
- `_mgpu_async_load_op_lowering_rule(ctx: LoweringContext, load_op: mgpu.AsyncLoadOp)` — [`L1159`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L1159)
- `_mgpu_async_prefetch_op_lowering_rule(ctx: LoweringContext, load_op: mgpu.AsyncPrefetchOp)` — [`L1227`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L1227)
- `_mgpu_async_store_op_lowering_rule(ctx: LoweringContext, store_op: mgpu.AsyncStoreOp)` — [`L1248`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L1248)
- `_mgpu_broadcast_in_dim_op_lowering_rule(_: LoweringContext, op: mgpu.BroadcastInDimOp)` — [`L937`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L937)
- `_mgpu_custom_primitive_op_lowering_rule(ctx: LoweringContext, op: mgpu.CustomPrimitiveOp)` — [`L2567`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L2567) — Lowering rule for mgpu.CustomPrimitiveOp.
- `_mgpu_layout_cast_op_lowering_rule(_: LoweringContext, op: mgpu.LayoutCastOp)` — [`L926`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L926)
- `_mgpu_mma_op_lowering_rule(ctx: LoweringContext, mma_op: mgpu.MMAOp)` — [`L1693`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L1693)
- `_mgpu_slice_smem_op_lowering_rule(ctx: LoweringContext, op: mgpu.SliceSMEMOp)` — [`L1795`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L1795)
- `_mgpu_vector_concat_op_lowering_rule(_: LoweringContext, op: mgpu.VectorConcatOp)` — [`L953`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L953)
- `_mgpu_wait_op_lowering_rule(_: LoweringContext, wait_op: mgpu.WaitOp)` — [`L1784`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L1784)
- `_mgpu_warp_map_op_lowering_rule(ctx: LoweringContext, op: mgpu.WarpMapOp)` — [`L2593`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L2593) — Lowering rule for mgpu.WarpMapOp.
- `_mgpu_wgmma_op_lowering_rule(ctx: LoweringContext, wgmma_op: mgpu.WGMMAOp)` — [`L1634`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L1634)
- `_mgpu_with_transforms_op_lowering_rule(ctx: LoweringContext, op: mgpu.WithTransformsOp)` — [`L1833`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L1833) — Lowering rule for mgpu.WithTransformsOp.
- `_move_scf_block_to_block_with_flattened_arguments(ctx: LoweringContext, old_block: ir.Block, new_block: ir.Block, last_op_type: type[ir.OpView], args_template: Sequence[_VectorTemplate | None], *new_leading_args: ir.Value)` — [`L2675`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L2675) — Moves the operations from `old_block` to `new_block`.
- `_multimem_load_reduce_op_lowering_rule(ctx: LoweringContext, op: mgpu.MultimemLoadReduceOp)` — [`L524`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L524)
- `_optimization_barrier_op_lowering_rule(_: LoweringContext, op: mgpu.OptimizationBarrierOp)` — [`L321`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L321)
- `_permutation_to_affine_map_attr(permutation: Sequence[int])` — [`L2082`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L2082)
- `_print_layout_op_lowering_rule(ctx: LoweringContext, op: mgpu.PrintLayoutOp)` — [`L703`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L703)
- `_query_cluster_cancel_op_lowering_rule(ctx: LoweringContext, op: mgpu.QueryClusterCancelOp)` — [`L2892`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L2892)
- `_register_lowering(op: str | type[ir.OpView] | None, support_warp_semantics: bool = False)` — [`L275`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L275)
- `_retry_on_failure(transfer: _Transfer, optimized: bool | None)` — [`L420`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L420) — If `optimized` is `None`, retry `transfer` with `optimized=False` on failure.
- `_select_op_lowering_rule(ctx: LoweringContext, op: arith.SelectOp)` — [`L1618`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L1618)
- `_should_lower(op: ir.OpView)` — [`L2916`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L2916) — Returns 'true' if the operation should be lowered.
- `_slice_smem(result: ir.MemRefType, offset: int, smem_size: int)` — [`L1811`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L1811)
- `_slice_tmem_lowering_rule(ctx: LoweringContext, op: mgpu.SliceTmemOp)` — [`L1376`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L1376)
- `_tcgen05_commit_arrive_op_lowering_rule(ctx: LoweringContext, op: mgpu.TcGen05CommitArriveOp)` — [`L2553`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L2553) — Lowering rule for mgpu.TcGen05CommitArriveOp.
- `_tcgen05_mma_op_lowering_rule(ctx: LoweringContext, op: mgpu.TcGen05MMAOp)` — [`L2463`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L2463)
- `_tile_transform_offsets(tiling: Sequence[int], static_offsets: Sequence[int], dynamic_offsets: Sequence[ir.Value])` — [`L1850`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L1850) — Computes the static and dynamic offsets after the given tiling is applied.
- `_tmem_alloc_op_lowering_rule(ctx: LoweringContext, op: mgpu.TmemAllocOp)` — [`L2365`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L2365) — Lowering rule for mgpu.TmemAllocOp.
- `_tmem_dealloc_op_lowering_rule(ctx: LoweringContext, op: mgpu.TmemDeallocOp)` — [`L2401`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L2401) — Lowering rule for mgpu.TmemDeallocOp.
- `_tmem_layout_cast_lowering_rule(ctx: LoweringContext, op: mgpu.TmemLayoutCastOp)` — [`L1363`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L1363)
- `_tmem_ref_from_ir(ref: ir.Value, expected_layout: ir.Attribute)` — [`L2419`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L2419) — Returns a TMEMRef from an IR value.
- `_tmem_ref_to_ir(ref: tcgen05.TMEMRef, ty: ir.MemRefType)` — [`L2455`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L2455) — Returns an IR value from a TMEMRef.
- `_tmem_relinquish_alloc_permit_op_lowering_rule(ctx: LoweringContext, op: mgpu.TmemRelinquishAllocPermitOp)` — [`L2390`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L2390) — Lowering rule for mgpu.TmemRelinquishAllocPermitOp.
- `_traverse_op_lowering_rule(ctx: LoweringContext, op: ir.OpView)` — [`L2901`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L2901)
- `_try_cluster_cancel_op_lowering_rule(ctx: LoweringContext, op: mgpu.TryClusterCancelOp)` — [`L2880`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L2880)
- `_unary_op_lowering_rule(_: LoweringContext, op: Any, impl: Callable[..., fa.FragmentedArray], is_signed: bool | None = None)` — [`L1426`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L1426)
- `_undo_conversion_cast(ir_value: ir.Value, expected_types: Sequence[ir.Type])` — [`L139`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L139) — Undoes the provided unrealized conversion cast.
- `_unflatten_ir_values(flat_values: Sequence[ir.Value], templates: Sequence[_VectorTemplate | None])` — [`L2651`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L2651) — The inverse of ``_flatten_ir_values``.
- `_vector_broadcast_op_lowering_rule(_: LoweringContext, op: vector.BroadcastOp)` — [`L736`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L736)
- `_vector_extract_op_lowering_rule(ctx: LoweringContext, op: vector.ExtractOp)` — [`L787`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L787)
- `_vector_extract_strided_slice_op_lowering_rule(ctx: LoweringContext, op: vector.ExtractStridedSliceOp)` — [`L763`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L763)
- `_vector_insert_strided_slice_op_lowering_rule(ctx: LoweringContext, op: vector.InsertStridedSliceOp)` — [`L812`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L812)
- `_vector_load_op_lowering_rule(_: LoweringContext, op: mgpu.VectorLoadOp)` — [`L435`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L435)
- `_vector_multi_dim_reduction_op_lowering_rule(ctx: LoweringContext, op: vector.MultiDimReductionOp)` — [`L875`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L875)
- `_vector_reduction_op_lowering_rule(ctx: LoweringContext, op: vector.ReductionOp)` — [`L847`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L847)
- `_vector_shape_cast_op_lowering_rule(_: LoweringContext, op: vector.ShapeCastOp)` — [`L750`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L750)
- `_vector_store_op_lowering_rule(ctx: LoweringContext, op: mgpu.VectorStoreOp)` — [`L554`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L554)
- `_while_op_lowering_rule(ctx: LoweringContext, while_op: scf.WhileOp)` — [`L2760`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L2760)
- `fragmented_array_to_ir(fragmented_array: fa.FragmentedArray, ty: ir.Type)` — [`L173`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L173) — Converts a FragmentedArray to an IR value.
- `load_tiled(optimized: bool)` — [`L496`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L496)
- `load_untiled(optimized: bool)` — [`L509`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L509)
- `lower_mgpu_dialect(module: ir.Module, launch_context: lc.LaunchContext | None, auto_barriers: bool = True)` — [`L2971`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L2971)
- `memref_transforms_from_transforms_attr(attr: ir.ArrayAttr)` — [`L977`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L977)
- `pprint_layout(v: fa.FragmentedArray | tcgen05.TMEMRef)` — [`L678`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L678)
- `store_tiled(optimized: bool)` — [`L587`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L587)
- `store_tiled_async(optimized: bool)` — [`L649`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L649)
- `store_untiled(optimized: bool)` — [`L597`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L597)
- `swizzle_from_transforms_attr(attr: ir.ArrayAttr)` — [`L967`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L967)
- `tile_offset(offsets: tuple[int, ...], tiling: tuple[int, ...])` — [`L990`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L990) — Tiles the trailing offsets in `offsets` according to `tiling`.
- `tile_strides(strides: tuple[int, ...], tiling: tuple[int, ...])` — [`L1013`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L1013) — Tiles the trailing strides in `strides` according to `tiling`.
- `tmem_layout(operand)` — [`L2468`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L2468)
- `transform_type(ref_ty: ir.MemRefType, transforms: tuple[lc.MemRefTransform, ...] | ir.ArrayAttr)` — [`L1074`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L1074)
- `unwrap_transformed_memref(ref: ir.Value[ir.MemRefType], expected_transforms: ir.ArrayAttr)` — [`L256`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L256) — Uwraps a memref from an unrealized cast and verifies its transforms.
- `wrap_transformed_memref(transformed_memref: ir.Value, logical_type: ir.Type, transforms: ir.ArrayAttr)` — [`L239`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L239) — Wraps a transformed memref to an unrealized cast with transforms.
- `wrapper(f)` — [`L279`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L279)

## Module values
- `CMPF_IMPLS` — [`L1575`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L1575)
- `CMPI_IMPLS` — [`L1546`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L1546)
- `MlirLoweringRule` — [`L131`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L131)
- `MlirLoweringRuleResult` — [`L130`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L130)
- `RECURSED` — [`L128`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L128)
- `_VectorTemplate` — [`L2618`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L2618)
- `_binary_impl` — [`L1489`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L1489)
- `_is_signed` — [`L1448`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L1448)
- `_lowerings` — [`L136`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L136)
- `_op` — [`L1408`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L1408)
- `_source_is_signed` — [`L1408`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L1408)
- `_supported_warp_lowerings` — [`L272`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L272)
- `_target_is_signed` — [`L1408`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L1408)
- `_unary_impl` — [`L1448`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/dialect_lowering.py#L1448)


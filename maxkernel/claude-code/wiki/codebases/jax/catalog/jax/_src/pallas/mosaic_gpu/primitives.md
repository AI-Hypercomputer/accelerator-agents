---
title: 'Module: jax/_src/pallas/mosaic_gpu/primitives.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/mosaic_gpu/primitives.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.mosaic_gpu.primitives`/
symbols:
  _copy_gmem_to_smem_lowering: _copy_gmem_to_smem_lowering().
  _semaphore_signal_multicast_lowering: _semaphore_signal_multicast_lowering().
  _copy_smem_to_gmem_lowering: _copy_smem_to_gmem_lowering().
  _async_store_smem_lowering: _async_store_smem_lowering().
  _semaphore_signal_lowering_rule: _semaphore_signal_lowering_rule().
  _semaphore_wait_lowering_rule: _semaphore_wait_lowering_rule().
  _multimem_store_lowering_rule: _multimem_store_lowering_rule().
  _tcgen05_mma_lowering: _tcgen05_mma_lowering().
  _barrier_arrive_lowering: _barrier_arrive_lowering().
  _inline_mgpu_flat_transformed_args: _inline_mgpu_flat_transformed_args().
  _async_prefetch_lowering: _async_prefetch_lowering().
  _multimem_load_reduce_lowering_rule: _multimem_load_reduce_lowering_rule().
  query_cluster_cancel_lowering: query_cluster_cancel_lowering().
  try_cluster_cancel_lowering: try_cluster_cancel_lowering().
  _populate_custom_primitive_op_block: _populate_custom_primitive_op_block().
  _wgmma_lowering: _wgmma_lowering().
  barrier_wait: barrier_wait().
  _Ref: _Ref.
  _wait_smem_to_gmem_lowering: _wait_smem_to_gmem_lowering().
  _inline_mgpu_lowering_rule_wg_semantics: _inline_mgpu_lowering_rule_wg_semantics().
  _async_copy_smem_to_tmem_lowering_rule: _async_copy_smem_to_tmem_lowering_rule().
  _barrier_test_lowering: _barrier_test_lowering().
  _inline_mgpu_lowering_rule: _inline_mgpu_lowering_rule().
  _multimem_load_reduce_lowering_rule_wg: _multimem_load_reduce_lowering_rule_wg().
  copy_gmem_to_smem: copy_gmem_to_smem().
  _tcgen05_mma_lowering_wg: _tcgen05_mma_lowering_wg().
  _atomic_store_discharge_rule: _atomic_store_discharge_rule().
  _atomic_store_lowering_rule: _atomic_store_lowering_rule().
  _async_copy_to_tmem_lowering_rule: _async_copy_to_tmem_lowering_rule().
  copy_smem_to_gmem: copy_smem_to_gmem().
  barrier_arrive: barrier_arrive().
  wgmma: wgmma().
  _extract_gmem_copy_params: _extract_gmem_copy_params().
  _print_layout_lowering: _print_layout_lowering().
  _wgmma_warpgroup_lowering: _wgmma_warpgroup_lowering().
  _tcgen05_commit_arrive_lowering: _tcgen05_commit_arrive_lowering().
  _async_load_tmem_lowering_rule: _async_load_tmem_lowering_rule().
  _get_barrier_base_index: _get_barrier_base_index().
  _barrier_wait_lowering: _barrier_wait_lowering().
  _tcgen05_commit_arrive_lowering_wg: _tcgen05_commit_arrive_lowering_wg().
  load: load().
  _copy_smem_to_gmem_pp_eqn: _copy_smem_to_gmem_pp_eqn().
  semaphore_signal_parallel: semaphore_signal_parallel().
  wait_smem_to_gmem: wait_smem_to_gmem().
  _async_store_tmem_lowering_rule_wg: _async_store_tmem_lowering_rule_wg().
  _async_copy_smem_to_tmem_lowering_rule_wg: _async_copy_smem_to_tmem_lowering_rule_wg().
  _atomic_store_lowering_rule_wg: _atomic_store_lowering_rule_wg().
  _copy_gmem_to_smem_pp_eqn: _copy_gmem_to_smem_pp_eqn().
  inline_mgpu.inner.wrapper: inline_mgpu().inner().wrapper().
  _custom_primitive_in_specs: _custom_primitive_in_specs().
  semaphore_signal: semaphore_signal().
  _check_ref: _check_ref().
  _collective_mma_predicate: _collective_mma_predicate().
  _async_store_tmem_lowering_rule: _async_store_tmem_lowering_rule().
  _multimem_store_abstract_eval: _multimem_store_abstract_eval().
  _wgmma_ref_pp_eqn: _wgmma_ref_pp_eqn().
  tcgen05_mma: tcgen05_mma().
  async_load_tmem: async_load_tmem().
  _async_store_smem_abstract_eval: _async_store_smem_abstract_eval().
  _wgmma_accumulator_store_abstract_eval: _wgmma_accumulator_store_abstract_eval().
  _closed_call_lowering_rule: _closed_call_lowering_rule().
  commit_smem: commit_smem().
  _barrier_arrive_pp_eqn: _barrier_arrive_pp_eqn().
  _barrier_test_pp_eqn: _barrier_test_pp_eqn().
  _barrier_wait_pp_eqn: _barrier_wait_pp_eqn().
  AtomicOpType: AtomicOpType#
  semaphore_signal_multicast: semaphore_signal_multicast().
  copy_smem_to_gmem_p: copy_smem_to_gmem_p.
  copy_gmem_to_smem_p: copy_gmem_to_smem_p.
  _commit_group_lowering: _commit_group_lowering().
  _wgmma_accumulator_deref_lowering: _wgmma_accumulator_deref_lowering().
  _inline_mgpu_abstract_eval: _inline_mgpu_abstract_eval().
  _async_load_tmem_lowering_rule_wg: _async_load_tmem_lowering_rule_wg().
  _async_copy_to_tmem_abstract_eval: _async_copy_to_tmem_abstract_eval().
  _async_copy_smem_to_tmem_abstract_eval: _async_copy_smem_to_tmem_abstract_eval().
  query_cluster_cancel: query_cluster_cancel().
  _atomic_op_type_to_int: _atomic_op_type_to_int().
  multimem_store: multimem_store().
  multimem_load_reduce: multimem_load_reduce().
  print_layout: print_layout().
  _custom_primitive_op_results: _custom_primitive_op_results().
  _atomic_store: _atomic_store().
  semaphore_wait: semaphore_wait().
  tcgen05_commit_arrive: tcgen05_commit_arrive().
  barrier_arrive_p: barrier_arrive_p.
  barrier_wait_p: barrier_wait_p.
  _wgmma_ref_effectful_abstract_eval: _wgmma_ref_effectful_abstract_eval().
  _wgmma_accumulator_store_lowering: _wgmma_accumulator_store_lowering().
  _commit_smem_lowering: _commit_smem_lowering().
  _griddepcontrol_wait_lowering: _griddepcontrol_wait_lowering().
  _griddepcontrol_launch_dependents_lowering: _griddepcontrol_launch_dependents_lowering().
  _semaphore_signal_parallel_abstract_eval: _semaphore_signal_parallel_abstract_eval().
  async_prefetch_p: async_prefetch_p.
  try_cluster_cancel: try_cluster_cancel().
  wait_smem_to_gmem_p: wait_smem_to_gmem_p.
  wgmma_p: wgmma_p.
  tcgen05_mma_p: tcgen05_mma_p.
  async_store_smem: async_store_smem().
  async_prefetch: async_prefetch().
  barrier_test: barrier_test().
  async_copy_scales_to_tmem: async_copy_scales_to_tmem().
  async_copy_sparse_metadata_to_tmem: async_copy_sparse_metadata_to_tmem().
  async_copy_smem_to_tmem: async_copy_smem_to_tmem().
  atomic_store_p: atomic_store_p.
  commit_group_p: commit_group_p.
  broadcasted_iota: broadcasted_iota().
  inline_mgpu_p: inline_mgpu_p.
  async_store_smem_p: async_store_smem_p.
  _barrier_test_abstract_eval: _barrier_test_abstract_eval().
  _wgmma_wait_lowering: _wgmma_wait_lowering().
  _tcgen05_mma_abstract_eval: _tcgen05_mma_abstract_eval().
  _set_max_registers_lowering: _set_max_registers_lowering().
  _load_abstract_eval: _load_abstract_eval().
  _async_copy_scales_to_tmem_lowering_rule: _async_copy_scales_to_tmem_lowering_rule().
  _async_copy_scales_to_tmem_lowering_rule_wg: _async_copy_scales_to_tmem_lowering_rule_wg().
  _async_copy_sparse_metadata_to_tmem_lowering_rule: _async_copy_sparse_metadata_to_tmem_lowering_rule().
  _async_copy_sparse_metadata_to_tmem_lowering_rule_wg: _async_copy_sparse_metadata_to_tmem_lowering_rule_wg().
  try_cluster_cancel_p: try_cluster_cancel_p.
  query_cluster_cancel_p: query_cluster_cancel_p.
  wgmma_ref_p: wgmma_ref_p.
  tcgen05_commit_arrive_p: tcgen05_commit_arrive_p.
  commit_smem_p: commit_smem_p.
  _type_check_mgpu_lane_semantics: _type_check_mgpu_lane_semantics().
  async_copy_scales_to_tmem_p: async_copy_scales_to_tmem_p.
  async_copy_sparse_metadata_to_tmem_p: async_copy_sparse_metadata_to_tmem_p.
  async_copy_smem_to_tmem_p: async_copy_smem_to_tmem_p.
  semaphore_signal_multicast_p: semaphore_signal_multicast_p.
  semaphore_signal_p: semaphore_signal_p.
  semaphore_wait_p: semaphore_wait_p.
  barrier_test_p: barrier_test_p.
  wgmma_wait_p: wgmma_wait_p.
  set_max_registers_p: set_max_registers_p.
  async_store_tmem: async_store_tmem().
  _split_gmem_slice: _split_gmem_slice().
  print_layout_p: print_layout_p.
  _copy_smem_to_gmem_abstract_eval: _copy_smem_to_gmem_abstract_eval().
  _copy_gmem_to_smem_abstract_eval: _copy_gmem_to_smem_abstract_eval().
  _barrier_arrive_abstract_eval: _barrier_arrive_abstract_eval().
  _barrier_wait_abstract_eval: _barrier_wait_abstract_eval().
  _wgmma_effectful_abstract_eval: _wgmma_effectful_abstract_eval().
  _wgmma_accumulator_deref_abstract_eval: _wgmma_accumulator_deref_abstract_eval().
  _wgmma_accumulator_store_warpgroup_lowering: _wgmma_accumulator_store_warpgroup_lowering().
  commit_tmem_p: commit_tmem_p.
  _commit_tmem_lowering: _commit_tmem_lowering().
  griddepcontrol_wait_p: griddepcontrol_wait_p.
  griddepcontrol_launch_dependents_p: griddepcontrol_launch_dependents_p.
  RefType: RefType#
  load_p: load_p.
  _async_load_tmem_abstract_eval: _async_load_tmem_abstract_eval().
  wait_load_tmem_p: wait_load_tmem_p.
  _wait_load_tmem_lowering: _wait_load_tmem_lowering().
  async_store_tmem_p: async_store_tmem_p.
  _async_store_tmem_abstract_eval: _async_store_tmem_abstract_eval().
  semaphore_signal_parallel_p: semaphore_signal_parallel_p.
  multimem_store_p: multimem_store_p.
  _multimem_load_reduce_abstract_eval: _multimem_load_reduce_abstract_eval().
  _semaphore_signal_multicast_abstract_eval: _semaphore_signal_multicast_abstract_eval().
  inline_mgpu: inline_mgpu().
  multimem_load_reduce_p: multimem_load_reduce_p.
  commit_smem_to_gmem_group: commit_smem_to_gmem_group().
  atomic_add: atomic_add().
  atomic_max: atomic_max().
  atomic_min: atomic_min().
  atomic_and: atomic_and().
  atomic_or: atomic_or().
  atomic_xor: atomic_xor().
  SomeLayout: SomeLayout.
  wgmma_accumulator_deref_p: wgmma_accumulator_deref_p.
  wgmma_accumulator_load: wgmma_accumulator_load().
  _wgmma_ref_discharge: _wgmma_ref_discharge().
  set_max_registers: set_max_registers().
  _griddepcontrol_wait_abstract_eval: _griddepcontrol_wait_abstract_eval().
  _query_cluster_cancel_abstract_eval: _query_cluster_cancel_abstract_eval().
  _atomic_store_abstract_eval: _atomic_store_abstract_eval().
  _semaphore_signal_abstract_eval: _semaphore_signal_abstract_eval().
  _semaphore_wait_abstract_eval: _semaphore_wait_abstract_eval().
  _extract_smem_copy_params: _extract_smem_copy_params().
  wgmma_accumulator_store: wgmma_accumulator_store().
  wgmma_wait: wgmma_wait().
  wgmma_accumulator_store_p: wgmma_accumulator_store_p.
  _tcgen05_mma_lowering_wg.handle_transforms_and_get_ref: _tcgen05_mma_lowering_wg().handle_transforms_and_get_ref().
  _ref_type_to_transforms: _ref_type_to_transforms().
  async_load_tmem_p: async_load_tmem_p.
  wait_load_tmem: wait_load_tmem().
  _print_layout_abstract_eval: _print_layout_abstract_eval().
  _async_prefetch_abstract_eval: _async_prefetch_abstract_eval().
  _wait_smem_to_gmem_abstract_eval: _wait_smem_to_gmem_abstract_eval().
  _commit_group_abstract_eval: _commit_group_abstract_eval().
  wgmma_wait_effectful_abstract_eval: wgmma_wait_effectful_abstract_eval().
  _wgmma_accumulator_deref_discharge: _wgmma_accumulator_deref_discharge().
  _wgmma_accumulator_store_discharge: _wgmma_accumulator_store_discharge().
  _tcgen05_commit_arrive_abstract_eval: _tcgen05_commit_arrive_abstract_eval().
  _commit_tmem_abstract_eval: _commit_tmem_abstract_eval().
  _set_max_registers_abstract_eval: _set_max_registers_abstract_eval().
  _commit_smem_abstract_eval: _commit_smem_abstract_eval().
  _closed_call_resource_estimator: _closed_call_resource_estimator().
  _griddepcontrol_launch_dependents_abstract_eval: _griddepcontrol_launch_dependents_abstract_eval().
  _wait_load_tmem_abstract_eval: _wait_load_tmem_abstract_eval().
  WARPGROUP_SIZE: WARPGROUP_SIZE.
  commit_tmem: commit_tmem().
  ShapeDtypeStruct: ShapeDtypeStruct#
  ShapeDtypeStruct.shape: ShapeDtypeStruct#shape.
  griddepcontrol_wait: griddepcontrol_wait().
  griddepcontrol_launch_dependents: griddepcontrol_launch_dependents().
  AxisName: AxisName.
  _inline_mgpu_discharge: _inline_mgpu_discharge().
  ShapeDtypeStruct.dtype: ShapeDtypeStruct#dtype.
  ShapeDtypeStruct.layout: ShapeDtypeStruct#layout.
  inline_mgpu.inner: inline_mgpu().inner().
  SemaphoreSignal.ref: SemaphoreSignal#ref.
  SemaphoreSignal.device_id: SemaphoreSignal#device_id.
  SemaphoreSignal.inc: SemaphoreSignal#inc.
  AtomicOpType.ADD: AtomicOpType#ADD.
  AtomicOpType.MAX: AtomicOpType#MAX.
  AtomicOpType.MIN: AtomicOpType#MIN.
  _try_cluster_cancel_abstract_eval: _try_cluster_cancel_abstract_eval().
  RefType.transforms: RefType#transforms.
  SemaphoreSignal: SemaphoreSignal#
  AtomicOpType.AND: AtomicOpType#AND.
  AtomicOpType.OR: AtomicOpType#OR.
  AtomicOpType.XOR: AtomicOpType#XOR.
  WARP_SIZE: WARP_SIZE.
---
# Module: [`jax/_src/pallas/mosaic_gpu/primitives.py`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py)

## Classes
### `AtomicOpType`
- def: [`jax/_src/pallas/mosaic_gpu/primitives.py:4575`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L4575)
- members:
  - `ADD` — [`L4576`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L4576)
  - `AND` — [`L4579`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L4579)
  - `MAX` — [`L4577`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L4577)
  - `MIN` — [`L4578`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L4578)
  - `OR` — [`L4580`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L4580)
  - `XOR` — [`L4581`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L4581)
- used by: [`_async_store_smem_lowering`](primitives.md#_async_store_smem_lowering), [`_atomic_store_discharge_rule`](primitives.md#_atomic_store_discharge_rule), [`_atomic_store_lowering_rule`](primitives.md#_atomic_store_lowering_rule), [`_atomic_store_lowering_rule_wg`](primitives.md#_atomic_store_lowering_rule_wg), [`_atomic_op_type_to_int`](primitives.md#_atomic_op_type_to_int), [`_atomic_store`](primitives.md#_atomic_store), [`atomic_add`](primitives.md#atomic_add), [`atomic_and`](primitives.md#atomic_and), [`atomic_max`](primitives.md#atomic_max), [`atomic_min`](primitives.md#atomic_min), [`atomic_or`](primitives.md#atomic_or), [`atomic_xor`](primitives.md#atomic_xor)

### `RefType`
- def: [`jax/_src/pallas/mosaic_gpu/primitives.py:2988`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L2988)
- signature: `class RefType:`
- members:
  - `transforms` — [`L2989`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L2989)
- uses (calls/refs, reference-scoped): [`Transform`](../../state/types.md#Transform)
- used by: [`_inline_mgpu_flat_transformed_args`](primitives.md#_inline_mgpu_flat_transformed_args), [`wrapper`](primitives.md#inline_mgpu.inner.wrapper), [`_type_check_mgpu_lane_semantics`](primitives.md#_type_check_mgpu_lane_semantics), [`inline_mgpu`](primitives.md#inline_mgpu), [`_ref_type_to_transforms`](primitives.md#_ref_type_to_transforms)

### `SemaphoreSignal`
- def: [`jax/_src/pallas/mosaic_gpu/primitives.py:4219`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L4219)
- signature: `class SemaphoreSignal:`
- members:
  - `device_id` — [`L4222`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L4222)
  - `inc` — [`L4223`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L4223)
  - `ref` — [`L4220`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L4220)
- uses (calls/refs, reference-scoped): [`Array`](../../basearray.md#Array), [`_Ref`](primitives.md#_Ref), [`DeviceId`](../primitives.md#DeviceId)
- used by: [`semaphore_signal_parallel`](primitives.md#semaphore_signal_parallel)

### `ShapeDtypeStruct`
- def: [`jax/_src/pallas/mosaic_gpu/primitives.py:2915`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L2915)
- signature: `class ShapeDtypeStruct:`
- members:
  - `dtype` — [`L2917`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L2917)
  - `layout` — [`L2918`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L2918)
  - `shape` — [`L2916`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L2916)
- uses (calls/refs, reference-scoped): [`dtype`](../../../numpy/__init__.pyi.md#dtype), [`SomeLayout`](primitives.md#SomeLayout)
- used by: [`_inline_mgpu_lowering_rule_wg_semantics`](primitives.md#_inline_mgpu_lowering_rule_wg_semantics), [`_inline_mgpu_lowering_rule`](primitives.md#_inline_mgpu_lowering_rule), [`_custom_primitive_op_results`](primitives.md#_custom_primitive_op_results), [`_type_check_mgpu_lane_semantics`](primitives.md#_type_check_mgpu_lane_semantics), [`inline_mgpu`](primitives.md#inline_mgpu)

## Functions
- `_async_copy_scales_to_tmem_lowering_rule(*args, **kwargs)` — [`L3969`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L3969)
- `_async_copy_scales_to_tmem_lowering_rule_wg(*args, **kwargs)` — [`L3981`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L3981)
- `_async_copy_smem_to_tmem_abstract_eval(smem_ref, tmem_ref, *args, smem_tree, tmem_tree, **_kwargs)` — [`L4062`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L4062)
- `_async_copy_smem_to_tmem_lowering_rule(ctx: lowering.LoweringRuleContext, smem_ref, tmem_ref, *leaves, smem_tree, tmem_tree, collective_axis)` — [`L4097`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L4097)
- `_async_copy_smem_to_tmem_lowering_rule_wg(ctx: lowering.LoweringRuleContext, smem_ref, tmem_ref, *leaves, smem_tree, tmem_tree, collective_axis)` — [`L4166`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L4166)
- `_async_copy_sparse_metadata_to_tmem_lowering_rule(*args, **kwargs)` — [`L3995`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L3995)
- `_async_copy_sparse_metadata_to_tmem_lowering_rule_wg(*args, **kwargs)` — [`L4007`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L4007)
- `_async_copy_to_tmem_abstract_eval(smem_ref, tmem_ref, *_args, **_kwargs)` — [`L3903`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L3903)
- `_async_copy_to_tmem_lowering_rule(impl, ctx: lowering.LoweringRuleContext, smem_ref, tmem_ref, *leaves, smem_tree, tmem_tree, collective_axis)` — [`L3910`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L3910)
- `_async_load_tmem_abstract_eval(src, *avals_flat, tree)` — [`L3620`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L3620)
- `_async_load_tmem_lowering_rule(ctx: lowering.LoweringRuleContext, x_ref, *leaves, tree)` — [`L3626`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L3626)
- `_async_load_tmem_lowering_rule_wg(ctx: lowering.LoweringRuleContext, x_ref: ir.Value, *leaves, tree)` — [`L3653`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L3653)
- `_async_prefetch_abstract_eval(ref, *args, **params)` — [`L1046`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1046)
- `_async_prefetch_lowering(ctx: lowering.LoweringRuleContext, ref, *flat_ref_transforms, ref_transforms_treedef, collective_axes, leader_tracked)` — [`L1056`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1056)
- `_async_store_smem_abstract_eval(src, ref, barrier, cluster_idx, *flat_transforms_avals, ref_transforms_treedef, barrier_transforms_treedef, **_)` — [`L481`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L481)
- `_async_store_smem_lowering(ctx: lowering.LoweringRuleContext, src, ref, barrier, cluster_idx, *flat_transforms, ref_transforms_treedef, barrier_transforms_treedef, cluster_dim, optimized, atomic)` — [`L528`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L528) — documented in [jax-_src-pallas-mosaic_gpu-primitives](../../../../../concepts/jax-_src-pallas-mosaic_gpu-primitives.md)
- `_async_store_tmem_abstract_eval(ref, val, *avals_flat, tree)` — [`L3731`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L3731)
- `_async_store_tmem_lowering_rule(ctx: lowering.LoweringRuleContext, x_ref, value, *leaves, tree)` — [`L3740`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L3740)
- `_async_store_tmem_lowering_rule_wg(ctx: lowering.LoweringRuleContext, x_ref: ir.Value, value: ir.Value, *leaves, tree)` — [`L3779`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L3779)
- `_atomic_op_type_to_int(atomic_type: AtomicOpType)` — [`L4584`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L4584)
- `_atomic_store(x_ref_or_view, val, *, atomic_type: AtomicOpType)` — [`L4674`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L4674)
- `_atomic_store_abstract_eval(*avals_flat, args_tree, atomic_type)` — [`L4607`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L4607)
- `_atomic_store_discharge_rule(in_avals, out_avals, *args_flat, args_tree, atomic_type: AtomicOpType)` — [`L4626`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L4626)
- `_atomic_store_lowering_rule(ctx: lowering.LoweringRuleContext, *args_flat, args_tree, atomic_type: AtomicOpType)` — [`L4716`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L4716)
- `_atomic_store_lowering_rule_wg(ctx: lowering.LoweringRuleContext, *args_flat, args_tree, atomic_type: AtomicOpType)` — [`L4692`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L4692)
- `_barrier_arrive_abstract_eval(barrier, *args, **params)` — [`L1199`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1199)
- `_barrier_arrive_lowering(ctx: lowering.LoweringRuleContext, barrier, *flat_transforms, transforms_treedef)` — [`L1228`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1228)
- `_barrier_arrive_pp_eqn(eqn: jax_core.JaxprEqn, context: jax_core.JaxprPpContext, settings: jax_core.JaxprPpSettings)` — [`L1205`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1205)
- `_barrier_test_abstract_eval(barrier, *args, **params)` — [`L1288`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1288)
- `_barrier_test_lowering(ctx: lowering.LoweringRuleContext, barrier, *flat_transforms, transforms_treedef)` — [`L1315`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1315)
- `_barrier_test_pp_eqn(eqn: jax_core.JaxprEqn, context: jax_core.JaxprPpContext, settings: jax_core.JaxprPpSettings)` — [`L1294`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1294)
- `_barrier_wait_abstract_eval(barrier, *args, **params)` — [`L1361`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1361)
- `_barrier_wait_lowering(ctx: lowering.LoweringRuleContext, barrier, *flat_transforms, transforms_treedef)` — [`L1392`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1392)
- `_barrier_wait_pp_eqn(eqn: jax_core.JaxprEqn, context: jax_core.JaxprPpContext, settings: jax_core.JaxprPpSettings)` — [`L1367`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1367)
- `_check_ref(aval: object, name: str, memory_space: gpu_core.MemorySpace)` — [`L73`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L73)
- `_closed_call_lowering_rule(ctx, *args, call_jaxpr: jax_core.ClosedJaxpr)` — [`L2901`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L2901)
- `_closed_call_resource_estimator(ctx, *args, call_jaxpr)` — [`L2908`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L2908)
- `_collective_mma_predicate(ctx: lowering.LoweringRuleContext, collective_axis: str)` — [`L2782`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L2782) — Computes a predicate to run only on the leader block.
- `_commit_group_abstract_eval()` — [`L1471`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1471)
- `_commit_group_lowering(ctx: lowering.LoweringRuleContext)` — [`L1481`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1481)
- `_commit_smem_abstract_eval()` — [`L2868`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L2868)
- `_commit_smem_lowering(ctx: lowering.LoweringRuleContext)` — [`L2875`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L2875)
- `_commit_tmem_abstract_eval()` — [`L2808`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L2808)
- `_commit_tmem_lowering(_)` — [`L2816`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L2816)
- `_copy_gmem_to_smem_abstract_eval(src, dst, barrier, *args, **params)` — [`L688`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L688)
- `_copy_gmem_to_smem_lowering(ctx: lowering.LoweringRuleContext, src, dst, barrier, *flat_transforms, src_transforms_treedef, dst_transforms_treedef, barrier_transforms_treedef, collective_axes, leader_tracked, oob_mode)` — [`L748`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L748) — documented in [jax-_src-pallas-mosaic_gpu-primitives](../../../../../concepts/jax-_src-pallas-mosaic_gpu-primitives.md)
- `_copy_gmem_to_smem_pp_eqn(eqn: jax_core.JaxprEqn, context: jax_core.JaxprPpContext, settings: jax_core.JaxprPpSettings)` — [`L696`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L696)
- `_copy_smem_to_gmem_abstract_eval(src, dst, *args, **params)` — [`L158`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L158)
- `_copy_smem_to_gmem_lowering(ctx: lowering.LoweringRuleContext, src, dst, *flat_args, src_transforms_treedef, dst_transforms_treedef, has_user_predicate, commit_group, reduction_op)` — [`L210`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L210) — documented in [jax-_src-pallas-mosaic_gpu-primitives](../../../../../concepts/jax-_src-pallas-mosaic_gpu-primitives.md)
- `_copy_smem_to_gmem_pp_eqn(eqn: jax_core.JaxprEqn, context: jax_core.JaxprPpContext, settings: jax_core.JaxprPpSettings)` — [`L165`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L165)
- `_custom_primitive_in_specs(ctx: lowering.LoweringRuleContext, flat_arg_types, flat_transformed_args, pytree_args)` — [`L3274`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L3274) — Returns a tuple containing the list of MLIR input types, layouts, and
- `_custom_primitive_op_results(flat_ret_ty)` — [`L3310`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L3310) — Returns a tuple containing the list of output MLIR types, and layouts for
- `_extract_gmem_copy_params(ctx, transforms, transform_avals, supports_multicast=False)` — [`L351`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L351)
- `_extract_smem_copy_params(aval, transforms)` — [`L404`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L404)
- `_get_barrier_base_index(aval, transforms)` — [`L1152`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1152)
- `_griddepcontrol_launch_dependents_abstract_eval()` — [`L2957`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L2957)
- `_griddepcontrol_launch_dependents_lowering(ctx: lowering.LoweringRuleContext)` — [`L2967`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L2967)
- `_griddepcontrol_wait_abstract_eval()` — [`L2926`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L2926)
- `_griddepcontrol_wait_lowering(ctx: lowering.LoweringRuleContext)` — [`L2936`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L2936)
- `_inline_mgpu_abstract_eval(*flat_args_and_transforms, flat_arg_types, flat_ret_ty, pytree_args, pytree_ref_transforms, pytree_ret_ty, mgpu_fn)` — [`L3076`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L3076)
- `_inline_mgpu_discharge(*args, **kwargs)` — [`L3103`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L3103)
- `_inline_mgpu_flat_transformed_args(ctx: lowering.LoweringRuleContext, flat_args_and_transforms, flat_arg_types, pytree_args, pytree_ref_transforms)` — [`L3133`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L3133)
- `_inline_mgpu_lowering_rule(ctx: lowering.LoweringRuleContext, *flat_args_and_transforms, mgpu_fn: Callable[..., Any], flat_arg_types, flat_ret_ty, pytree_args, pytree_ref_transforms, pytree_ret_ty)` — [`L3221`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L3221)
- `_inline_mgpu_lowering_rule_wg_semantics(ctx: lowering.LoweringRuleContext, *flat_args_and_transforms, mgpu_fn: Callable[..., Any], flat_arg_types, flat_ret_ty, pytree_args, pytree_ref_transforms, pytree_ret_ty)` — [`L3445`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L3445)
- `_load_abstract_eval(src, *avals_flat, tree, optimized)` — [`L3510`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L3510)
- `_multimem_load_reduce_abstract_eval(ref, *avals_flat, tree, collective_axes, reduction_op)` — [`L4931`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L4931)
- `_multimem_load_reduce_lowering_rule(ctx: lowering.LoweringRuleContext, ref, *transforms_leaves, tree, collective_axes, reduction_op)` — [`L4942`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L4942)
- `_multimem_load_reduce_lowering_rule_wg(ctx: lowering.LoweringRuleContext, ref, *transforms_leaves, tree, collective_axes, reduction_op)` — [`L4986`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L4986)
- `_multimem_store_abstract_eval(source, ref, *transforms_leaves, transforms_tree, **_)` — [`L4863`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L4863)
- `_multimem_store_lowering_rule(ctx: lowering.LoweringRuleContext, value, local_ref, *transforms_leaves, transforms_tree, collective_axes)` — [`L4881`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L4881)
- `_populate_custom_primitive_op_block(ctx: lowering.LoweringRuleContext, block: ir.Block, mgpu_fn: Callable[..., Any], pytree_args, in_layouts: Sequence[ir.Attribute], in_transforms: Sequence[ir.ArrayAttr], results_ty: Sequence[ir.Type], out_layouts: Sequence[ir.Attribute | None])` — [`L3332`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L3332) — Calls the given mgpu_fn to populate the block, handling inputs and outputs.
- `_print_layout_abstract_eval(aval_in, fmt, *_, **params)` — [`L95`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L95)
- `_print_layout_lowering(ctx: lowering.LoweringRuleContext, x: mgpu.FragmentedArray | tcgen05.TMEMRef | ir.Value, fmt: str, *transforms_leaves, transforms_tree)` — [`L104`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L104)
- `_query_cluster_cancel_abstract_eval(try_cancel_buffer, *transforms_leaves, grid_names, transforms_tree)` — [`L4479`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L4479)
- `_ref_type_to_transforms(ref_type: RefType)` — [`L3267`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L3267) — Returns the Mosaic GPU transforms for the given ref type.
- `_semaphore_signal_abstract_eval(*avals, args_tree, device_id_type, memory_scope)` — [`L5208`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L5208)
- `_semaphore_signal_lowering_rule(ctx: lowering.LoweringRuleContext, *args, args_tree)` — [`L4288`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L4288)
- `_semaphore_signal_multicast_abstract_eval(*avals, args_tree, collective_axes)` — [`L5109`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L5109)
- `_semaphore_signal_multicast_lowering(ctx: lowering.LoweringRuleContext, *args, args_tree, collective_axes)` — [`L5132`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L5132) — documented in [jax-_src-pallas-mosaic_gpu-primitives](../../../../../concepts/jax-_src-pallas-mosaic_gpu-primitives.md)
- `_semaphore_signal_parallel_abstract_eval(*avals, args_tree)` — [`L4259`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L4259)
- `_semaphore_wait_abstract_eval(*avals, args_tree, memory_scope)` — [`L5320`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L5320)
- `_semaphore_wait_lowering_rule(ctx: lowering.LoweringRuleContext, *args, args_tree, memory_scope: Literal["sys", "gpu"] = "sys")` — [`L5335`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L5335)
- `_set_max_registers_abstract_eval(n, *, action)` — [`L2836`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L2836)
- `_set_max_registers_lowering(ctx: lowering.LoweringRuleContext, n, *, action)` — [`L2845`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L2845)
- `_split_gmem_slice(gmem_slice)` — [`L324`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L324)
- `_tcgen05_commit_arrive_abstract_eval(barrier, *barrier_transforms_leaves, barrier_transforms_tree, collective_axis)` — [`L2695`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L2695)
- `_tcgen05_commit_arrive_lowering(ctx: lowering.LoweringRuleContext, barrier_ref: mgpu.BarrierRef, *barrier_transforms_leaves, barrier_transforms_tree, collective_axis)` — [`L2711`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L2711)
- `_tcgen05_commit_arrive_lowering_wg(ctx: lowering.LoweringRuleContext, barrier_ref: mgpu.DialectBarrierRef, *barrier_transforms_leaves, barrier_transforms_tree, collective_axis)` — [`L2750`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L2750)
- `_tcgen05_mma_abstract_eval(acc, a, b, accumulate, *barrier_scales_and_transforms_leaves, acc_transforms_tree, a_transforms_tree, b_transforms_tree, barrier_transforms_tree, a_scale_transforms_tree, b_scale_transforms_tree, a_sparse_metadata_transforms_tree, collective_axis, arrive, scaled, sparse)` — [`L2151`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L2151)
- `_tcgen05_mma_lowering(ctx: lowering.LoweringRuleContext, acc: tcgen05.TMEMRef, a_ref, b_ref, accumulate: bool | ir.Value, *barrier_scales_and_transforms_leaves, acc_transforms_tree, a_transforms_tree, b_transforms_tree, barrier_transforms_tree, a_scale_transforms_tree, b_scale_transforms_tree, a_sparse_metadata_transforms_tree, collective_axis, arrive, scaled: bool, sparse: bool)` — [`L2202`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L2202)
- `_tcgen05_mma_lowering_wg(ctx: lowering.LoweringRuleContext, acc_ref, a_ref, b_ref, accumulate: bool | ir.Value, *barrier_scales_and_transforms_leaves, acc_transforms_tree, a_transforms_tree, b_transforms_tree, barrier_transforms_tree, a_scale_transforms_tree, b_scale_transforms_tree, a_sparse_metadata_transforms_tree, collective_axis, arrive, scaled: bool, sparse: bool)` — [`L2471`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L2471)
- `_try_cluster_cancel_abstract_eval(*args, **params)` — [`L4337`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L4337)
- `_type_check_mgpu_lane_semantics(v, ty)` — [`L3108`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L3108)
- `_wait_load_tmem_abstract_eval()` — [`L3696`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L3696)
- `_wait_load_tmem_lowering(_)` — [`L3702`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L3702)
- `_wait_smem_to_gmem_abstract_eval(n, *, wait_read_only)` — [`L1427`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1427)
- `_wait_smem_to_gmem_lowering(ctx: lowering.LoweringRuleContext, n, *, wait_read_only)` — [`L1441`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1441)
- `_wgmma_accumulator_deref_abstract_eval(acc, **_)` — [`L1892`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1892)
- `_wgmma_accumulator_deref_discharge(in_avals, out_avals, acc, *, wait_n)` — [`L1900`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1900)
- `_wgmma_accumulator_deref_lowering(ctx: lowering.LoweringRuleContext, acc, *, wait_n: int | None)` — [`L1911`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1911)
- `_wgmma_accumulator_store_abstract_eval(acc, val)` — [`L1938`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1938)
- `_wgmma_accumulator_store_discharge(in_avals, out_avals, acc, val)` — [`L1963`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1963)
- `_wgmma_accumulator_store_lowering(ctx: lowering.LoweringRuleContext, acc, val)` — [`L1971`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1971)
- `_wgmma_accumulator_store_warpgroup_lowering(ctx: lowering.LoweringRuleContext, acc, val)` — [`L1981`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1981)
- `_wgmma_effectful_abstract_eval(acc, lhs_ref, *args, **kwargs)` — [`L1842`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1842)
- `_wgmma_lowering(ctx: lowering.LoweringRuleContext, acc, a, b, *transforms_leaves, acc_transforms_tree, a_transforms_tree, b_transforms_tree)` — [`L1616`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1616)
- `_wgmma_ref_discharge(in_avals, out_avals, *args, **kwargs)` — [`L1606`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1606)
- `_wgmma_ref_effectful_abstract_eval(acc_aval, a_aval, b_aval, *_, **params)` — [`L1559`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1559)
- `_wgmma_ref_pp_eqn(eqn: jax_core.JaxprEqn, context: jax_core.JaxprPpContext, settings: jax_core.JaxprPpSettings)` — [`L1572`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1572)
- `_wgmma_wait_lowering(ctx: lowering.LoweringRuleContext, allow_groups)` — [`L1866`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1866)
- `_wgmma_warpgroup_lowering(ctx: lowering.LoweringRuleContext, acc, a, b, *transforms_leaves, acc_transforms_tree, a_transforms_tree, b_transforms_tree)` — [`L1744`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1744)
- `async_copy_scales_to_tmem(smem_ref: _Ref, tmem_ref: _Ref, collective_axis: AxisName | None = None)` — [`L3837`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L3837) — Copies the MMA scales from SMEM to TMEM.
- `async_copy_smem_to_tmem(smem_ref: _Ref, tmem_ref: _Ref, collective_axis: AxisName | None = None)` — [`L4019`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L4019) — Copies data from SMEM to TMEM using the tcgen05.cp instruction.
- `async_copy_sparse_metadata_to_tmem(smem_ref: _Ref, tmem_ref: _Ref, collective_axis: AxisName | None = None)` — [`L3871`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L3871) — Copies the MMA sparse metadata from SMEM to TMEM.
- `async_load_tmem(src: _Ref, *, layout: SomeLayout | None = None)` — [`L3581`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L3581) — Performs an asynchronous load from the TMEM array.
- `async_prefetch(ref: _Ref, *, collective_axes: str | tuple[str, ...] | None = None, leader_tracked: CopyPartition | None = None)` — [`L1113`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1113) — Asynchronously prefetches a GMEM reference to the L2 cache.
- `async_store_smem(src: jax.Array, ref: _Ref, barrier: _Ref, *, cluster_idx: jax.Array, cluster_dim: Hashable | int, optimized: bool = True, atomic: Literal["add", "max", "min", "and", "or", "xor"] | None = None)` — [`L634`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L634) — Asynchronously stores an array to a SMEM reference within the cluster.
- `async_store_tmem(ref: _Ref, value)` — [`L3710`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L3710) — Stores the value to TMEM.
- `atomic_add(ref: _Ref, val)` — [`L4756`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L4756) — Performs an atomic store-add of the value to the reference.
- `atomic_and(ref: _Ref, val)` — [`L4794`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L4794) — Performs an atomic store-and of the value to the reference.
- `atomic_max(ref: _Ref, val)` — [`L4770`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L4770) — Performs an atomic store-max of the value to the reference.
- `atomic_min(ref: _Ref, val)` — [`L4782`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L4782) — Performs an atomic store-min of the value to the reference.
- `atomic_or(ref: _Ref, val)` — [`L4806`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L4806) — Performs an atomic store-or of the value to the reference.
- `atomic_xor(ref: _Ref, val)` — [`L4818`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L4818) — Performs an atomic store-xor of the value to the reference.
- `barrier_arrive(barrier: state.AbstractRef)` — [`L1272`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1272) — Arrives at the given barrier.
- `barrier_test(barrier: state.AbstractRef)` — [`L1339`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1339) — Tests the given barrier.
- `barrier_wait(barrier: state.AbstractRef)` — [`L1411`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1411) — Waits on the given barrier.
- `broadcasted_iota(dtype: jax.typing.DTypeLike, shape: Sequence[int], dimension: int, *, layout: SomeLayout | None = None)` — [`L2886`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L2886)
- `commit_smem()` — [`L2881`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L2881) — Commits all reads from/writes to SMEM, making them visible to TMA and MMA operations.
- `commit_smem_to_gmem_group()` — [`L1487`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1487) — Commits all issued but uncommitted SMEM->GMEM copies to a group.
- `commit_tmem()` — [`L2821`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L2821) — Commits all writes to TMEM issued by the current thread.
- `copy_gmem_to_smem(src: _Ref, dst: _Ref, barrier: _Ref, *, collective_axes: str | tuple[str, ...] | None = None, leader_tracked: CopyPartition | None = None, oob_mode: OOBFillMode = OOBFillMode.ZEROS)` — [`L952`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L952) — Asynchronously copies a GMEM reference to a SMEM reference.
- `copy_smem_to_gmem(src: _Ref, dst: _Ref, predicate: jax.Array | None = None, *, commit_group: bool = True, reduction_op: mgpu.TMAReductionOp | None = None)` — [`L423`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L423) — Asynchronously copies a SMEM reference to a GMEM reference.
- `griddepcontrol_launch_dependents()` — [`L2978`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L2978) — Signal that dependents can be launched.
- `griddepcontrol_wait()` — [`L2945`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L2945) — Wait for dependent grids to finish.
- `handle_transforms_and_get_ref(tree, leaves, leaves_avals, ref, ref_aval, handle_transposes=True)` — [`L2558`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L2558)
- `inline_mgpu(*, arg_types=(), return_type=None)` — [`L2992`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L2992) — Returns a decorator that inlines Mosaic GPU code.
- `inner(f)` — [`L3035`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L3035)
- `load(src: _Ref, idx: Any = api.NotSpecified(), *, layout: SomeLayout | None = None, optimized: bool = True)` — [`L3533`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L3533) — Loads from a reference into an array with the specified layout.
- `multimem_load_reduce(ref: _Ref, *, collective_axes: Hashable | tuple[Hashable, ...], reduction_op: mgpu.MultimemReductionOp)` — [`L5042`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L5042) — Loads from a GMEM reference on all devices present in collective_axes and reduces the loaded values.
- `multimem_store(source: jax.Array, ref: _Ref, collective_axes: Hashable | tuple[Hashable, ...])` — [`L4834`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L4834) — Stores the value to ref on all devices present in collective_axes.
- `print_layout(fmt: str, x: jax.typing.ArrayLike | _Ref)` — [`L130`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L130) — Prints the layout chosen by Mosaic GPU for a given array or TMEM reference.
- `query_cluster_cancel(result_ref: _Ref, grid_names: Sequence[Hashable])` — [`L4539`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L4539) — Decodes the result of a ``try_cluster_cancel`` operation.
- `query_cluster_cancel_lowering(ctx: lowering.LoweringRuleContext, result_ref, *transforms_leaves, grid_names, transforms_tree)` — [`L4496`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L4496)
- `semaphore_signal(semaphore, inc: int | jax.Array = 1, *, device_id: pallas_primitives.DeviceId | None = None, memory_scope: Literal["sys", "gpu"] = "sys")` — [`L5175`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L5175) — Signals a semaphore, optionally on a remote device.
- `semaphore_signal_multicast(semaphore, value: int | jax.Array = 1, *, collective_axes: Hashable | tuple[Hashable, ...])` — [`L5079`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L5079) — Signals a semaphore on all devices along collective_axes.
- `semaphore_signal_parallel(*signals: SemaphoreSignal)` — [`L4226`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L4226) — Signals multiple semaphores without any guaranteed ordering of signal arrivals.
- `semaphore_wait(semaphore, value: int | jax.Array = 1, *, decrement: bool = True, memory_scope: Literal["sys", "gpu"] = "sys")` — [`L5288`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L5288) — Waits on a semaphore until it reaches at least ``value``.
- `set_max_registers(n: int, *, action: Literal["increase", "decrease"])` — [`L2858`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L2858) — Sets the maximum number of per-lane registers in the thread.
- `tcgen05_commit_arrive(barrier: _Ref, collective_axis: str | None = None)` — [`L2665`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L2665) — Tracks completion of all preceding ``tcgen05_mma`` and ``async_copy_smem_to_tmem`` calls.
- `tcgen05_mma(acc: _Ref, a: _Ref, b: _Ref, barrier: _Ref | None = None, *, a_scale: _Ref | None = None, b_scale: _Ref | None = None, a_sparse_metadata: _Ref | None = None, accumulate: bool | jax.Array = True, collective_axis: str | None = None)` — [`L1993`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1993) — Asynchronous matrix-multiply accumulate for TensorCore gen 5 (Blackwell).
- `try_cluster_cancel(result_ref: _Ref, barrier: _Ref)` — [`L4433`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L4433) — Initiates an async request to claim a new work unit from the grid.
- `try_cluster_cancel_lowering(ctx: lowering.LoweringRuleContext, result_ref, barrier, *transforms_leaves, result_transforms_tree, barrier_transforms_tree)` — [`L4344`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L4344)
- `wait_load_tmem()` — [`L3684`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L3684) — Awaits all previously asynchronous TMEM loads issued by the calling thread.
- `wait_smem_to_gmem(n: int, wait_read_only: bool = False)` — [`L1455`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1455) — Waits until no more than the most recent ``n`` SMEM->GMEM copies issued by the calling thread are in flight.
- `wgmma(acc: gpu_core.WGMMAAbstractAccumulatorRef, a, b)` — [`L1497`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1497) — Performs an asynchronous warp group matmul-accumulate on the given references.
- `wgmma_accumulator_load(acc, *, wait_n: int | None = 0)` — [`L1875`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1875) — Dereferences an accumulator register.
- `wgmma_accumulator_store(acc_ref, val)` — [`L1931`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1931)
- `wgmma_wait(n: int)` — [`L1854`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1854) — Waits until there is no more than ``n`` WGMMA operations in flight.
- `wgmma_wait_effectful_abstract_eval(_)` — [`L1860`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1860)
- `wrapper(*args)` — [`L3036`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L3036)

## Module values
- `AxisName` — [`L65`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L65)
- `SomeLayout` — [`L71`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L71)
- `WARPGROUP_SIZE` — [`L67`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L67)
- `WARP_SIZE` — [`L66`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L66)
- `_Ref` — [`L70`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L70)
- `async_copy_scales_to_tmem_p` — [`L3833`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L3833)
- `async_copy_smem_to_tmem_p` — [`L4015`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L4015)
- `async_copy_sparse_metadata_to_tmem_p` — [`L3867`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L3867)
- `async_load_tmem_p` — [`L3579`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L3579)
- `async_prefetch_p` — [`L1042`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1042)
- `async_store_smem_p` — [`L476`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L476)
- `async_store_tmem_p` — [`L3707`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L3707)
- `atomic_store_p` — [`L4602`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L4602)
- `barrier_arrive_p` — [`L1194`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1194)
- `barrier_test_p` — [`L1283`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1283)
- `barrier_wait_p` — [`L1356`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1356)
- `commit_group_p` — [`L1466`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1466)
- `commit_smem_p` — [`L2863`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L2863)
- `commit_tmem_p` — [`L2803`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L2803)
- `copy_gmem_to_smem_p` — [`L683`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L683)
- `copy_smem_to_gmem_p` — [`L153`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L153)
- `griddepcontrol_launch_dependents_p` — [`L2950`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L2950)
- `griddepcontrol_wait_p` — [`L2921`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L2921)
- `inline_mgpu_p` — [`L2983`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L2983)
- `load_p` — [`L3506`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L3506)
- `multimem_load_reduce_p` — [`L4928`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L4928)
- `multimem_store_p` — [`L4830`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L4830)
- `print_layout_p` — [`L90`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L90)
- `query_cluster_cancel_p` — [`L4475`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L4475)
- `semaphore_signal_multicast_p` — [`L5076`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L5076)
- `semaphore_signal_p` — [`L5171`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L5171)
- `semaphore_signal_parallel_p` — [`L4214`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L4214)
- `semaphore_wait_p` — [`L5284`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L5284)
- `set_max_registers_p` — [`L2831`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L2831)
- `tcgen05_commit_arrive_p` — [`L2661`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L2661)
- `tcgen05_mma_p` — [`L1990`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1990)
- `try_cluster_cancel_p` — [`L4333`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L4333)
- `wait_load_tmem_p` — [`L3681`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L3681)
- `wait_smem_to_gmem_p` — [`L1422`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1422)
- `wgmma_accumulator_deref_p` — [`L1872`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1872)
- `wgmma_accumulator_store_p` — [`L1928`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1928)
- `wgmma_p` — [`L1612`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1612)
- `wgmma_ref_p` — [`L1493`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1493)
- `wgmma_wait_p` — [`L1850`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/primitives.py#L1850)


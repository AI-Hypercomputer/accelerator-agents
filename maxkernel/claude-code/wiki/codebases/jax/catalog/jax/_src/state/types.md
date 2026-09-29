---
title: 'Module: jax/_src/state/types.py'
type: catalog
provenance: extracted
module: jax/_src/state/types.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.state.types`/
symbols:
  AbstractRef: AbstractRef#
  TransformedRef: TransformedRef#
  Transform: Transform#
  TransformedRef.transforms: TransformedRef#transforms.
  AbstractRef.shape: AbstractRef#shape().
  TransformedRef.ref: TransformedRef#ref.
  Transform.transform_type: Transform#transform_type().
  Transform.undo: Transform#undo().
  WriteEffect: WriteEffect#
  ReadEffect: ReadEffect#
  TransformedRef.shape: TransformedRef#shape().
  AbstractRef.update: AbstractRef#update().
  transform_type: transform_type().
  Transform.pretty_print: Transform#pretty_print().
  TransposeTransform: TransposeTransform#
  TransformedRef.dtype: TransformedRef#dtype().
  RefEffect._pretty_print: RefEffect#_pretty_print().
  TransformedRef.type: TransformedRef#type().
  RefEffect: RefEffect#
  TransposeTransform.transform_type: TransposeTransform#transform_type().
  AbstractRef.dtype: AbstractRef#dtype().
  ReshapeTransform.transform_type: ReshapeTransform#transform_type().
  BitcastTransform.transform_type: BitcastTransform#transform_type().
  ReshapeTransform: ReshapeTransform#
  TransformedRef.reshape: TransformedRef#reshape().
  AbstractRef.__init__: AbstractRef#__init__().
  TransformedRef.bitcast: TransformedRef#bitcast().
  RefIndexer.__getitem__: RefIndexer#__getitem__().
  TransformedRef.multiref: TransformedRef#multiref().
  ReshapeTransform.shape: ReshapeTransform#shape.
  flatten_func: flatten_func.
  TransposeTransform.permutation: TransposeTransform#permutation.
  TransformedRef.transpose: TransformedRef#transpose().
  AbstractRef.at: AbstractRef#at().
  TransposeTransform.undo: TransposeTransform#undo().
  TransformedRef.__getattr__: TransformedRef#__getattr__().
  BitcastTransform: BitcastTransform#
  TransformedRef._type: TransformedRef#_type().
  zeros_like_abstract_ref: zeros_like_abstract_ref().
  AbstractLinVal: AbstractLinVal#
  _ref_type_aval_mappings._ref_type_aval_mappings: _ref_type_aval_mappings._ref_type_aval_mappings.
  MultiRefTransform: MultiRefTransform#
  BitcastTransform.pretty_print: BitcastTransform#pretty_print().
  ReshapeTransform.pretty_print: ReshapeTransform#pretty_print().
  TransposeTransform.pretty_print: TransposeTransform#pretty_print().
  SelectTransform.pretty_print: SelectTransform#pretty_print().
  AbstractRef.raise_val: AbstractRef#raise_val().
  AbstractRef._getitem: AbstractRef#_getitem().
  AbstractRef.str_short: AbstractRef#str_short().
  AccumEffect: AccumEffect#
  BitcastTransform.dtype: BitcastTransform#dtype.
  StateEffect: StateEffect.
  uninitialized: uninitialized.
  _default_value_to_ref_aval: _default_value_to_ref_aval().
  SelectTransform: SelectTransform#
  get_ref_aval_from_value: get_ref_aval_from_value().
  TransformedRef.__post_init__: TransformedRef#__post_init__().
  AbstractRef.lower_val: AbstractRef#lower_val().
  AbstractRef.update_weak_type: AbstractRef#update_weak_type().
  AbstractRef.update_manual_axis_type: AbstractRef#update_manual_axis_type().
  AbstractRef.bitcast: AbstractRef#bitcast().
  AbstractRef.reshape: AbstractRef#reshape().
  AbstractRef.transpose: AbstractRef#transpose().
  AbstractRef.T: AbstractRef#T().
  AbstractRef._setitem: AbstractRef#_setitem().
  AbstractRef.is_high: AbstractRef#is_high().
  MultiRefTransform.transform_types: MultiRefTransform#transform_types().
  MultiRefTransform.getattr: MultiRefTransform#getattr().
  SelectTransform._type: SelectTransform#_type().
  AbstractRef.sharding: AbstractRef#sharding().
  AbstractRef.manual_axis_type: AbstractRef#manual_axis_type().
  AbstractRef.to_ct_aval: AbstractRef#to_ct_aval().
  _map_ref: _map_ref().
  _unmap_ref: _unmap_ref().
  shaped_array_ref: shaped_array_ref().
  _shard_ref: _shard_ref().
  RefEffect.__str__: RefEffect#__str__().
  AccumEffect.name: AccumEffect#name.
  BitcastTransform.undo: BitcastTransform#undo().
  ReshapeTransform.undo: ReshapeTransform#undo().
  SelectTransform.undo: SelectTransform#undo().
  RefIndexer.ref_or_view: RefIndexer#ref_or_view.
  AbstractRef.lo_ty: AbstractRef#lo_ty().
  AbstractRef.weak_type: AbstractRef#weak_type().
  AbstractRef.get: AbstractRef#get().
  AbstractRef.swap: AbstractRef#swap().
  AbstractRef.set: AbstractRef#set().
  AbstractRef.addupdate: AbstractRef#addupdate().
  AbstractRef.to_tangent_aval: AbstractRef#to_tangent_aval().
  unsafe_zip: unsafe_zip.
  TransformedRef.is_dynamic_size: TransformedRef#is_dynamic_size().
  _ref_effect_color: _ref_effect_color.
  ReshapeTransform._validate_shape: ReshapeTransform#_validate_shape().
  SelectTransform.idx: SelectTransform#idx.
  TransformedRef.at: TransformedRef#at().
  disallow_transformed_ref_avals: disallow_transformed_ref_avals().
  unflatten_func: unflatten_func.
  AbstractRef.__repr__: AbstractRef#__repr__().
  _unshard_ref: _unshard_ref().
  AbstractLinVal.inner_aval: AbstractLinVal#inner_aval.
  _perm_inverse: _perm_inverse().
  RefIndexer: RefIndexer#
  RefEffect.__eq__: RefEffect#__eq__().
  RefEffect.__hash__: RefEffect#__hash__().
  SelectTransform.transform_types: SelectTransform#transform_types().
  SelectTransform.getattr: SelectTransform#getattr().
  TransformedRef.set: TransformedRef#set().
  TransformedRef.swap: TransformedRef#swap().
  TransformedRef.get: TransformedRef#get().
  TransformedRef.__getitem__: TransformedRef#__getitem__().
  TransformedRef.__setitem__: TransformedRef#__setitem__().
  AbstractRef._len: AbstractRef#_len().
  AbstractRef.mat: AbstractRef#mat().
  AbstractRef._addupdate: AbstractRef#_addupdate().
  AbstractRef.__str__: AbstractRef#__str__.
  AbstractRef.__eq__: AbstractRef#__eq__().
  AbstractRef.__hash__: AbstractRef#__hash__().
  unsafe_map: unsafe_map.
  RefEffect.name: RefEffect#name.
  AbstractRef.ndim: AbstractRef#ndim.
  Uninitialized: Uninitialized#
  _canonicalize_reshape: _canonicalize_reshape().
  TransformedRef.size: TransformedRef#size.
  TransformedRef.T: TransformedRef#T.
  TransformedRefAvalError: TransformedRefAvalError#
  AbstractLinVal.dtype: AbstractLinVal#dtype.
  ReadEffect.name: ReadEffect#name.
  WriteEffect.name: WriteEffect#name.
  TransformedRef.ndim: TransformedRef#ndim.
  AbstractRef.__slots__: AbstractRef#__slots__.
  AbstractRef.size: AbstractRef#size.
  AbstractLinVal.memory_space: AbstractLinVal#memory_space.
  AbstractLinVal.shape: AbstractLinVal#shape.
  AbstractLinVal.ndim: AbstractLinVal#ndim.
---
# Module: [`jax/_src/state/types.py`](../../../../../../../raw/code/jax/jax/_src/state/types.py)

## Classes
### `AbstractLinVal`  ·  implements/extends AbstractValue
- def: [`jax/_src/state/types.py:694`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L694)
- signature: `class AbstractLinVal(core.AbstractValue):`
- members:
  - `dtype` — [`L699`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L699)
  - `inner_aval` — [`L695`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L695)
  - `memory_space` — [`L696`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L696)
  - `ndim` — [`L700`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L700)
  - `shape` — [`L698`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L698)
- uses (calls/refs, reference-scoped): [`AbstractValue`](../core.md#AbstractValue)
- used by: [`AbstractValue`](../core.md#AbstractValue), [`to_block_mapping`](../pallas/core.md#BlockSpec.to_block_mapping), [`_pallas_call_abstract_eval`](../pallas/pallas_call.md#_pallas_call_abstract_eval), [`_jaxpr_kernel_aval_to_mosaic`](../pallas/mosaic/pallas_call_registration.md#_jaxpr_kernel_aval_to_mosaic), [`_create_linear_abstract_eval`](primitives.md#_create_linear_abstract_eval), [`_pin_abstract_eval`](primitives.md#_pin_abstract_eval), [`_unpin_abstract_eval`](primitives.md#_unpin_abstract_eval)

### `AbstractRef`
- def: [`jax/_src/state/types.py:434`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L434) — documented in [jax-_src-state-types](../../../../concepts/jax-_src-state-types.md)
- members:
  - `T(self)` — [`L567`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L567)
  - `addupdate(tracer, value, idx=())` — [`L590`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L590)
  - `at(self)` — [`L551`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L551)
  - `bitcast(self, dtype)` — [`L555`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L555)
  - `dtype(self)` — [`L520`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L520)
  - `get(tracer, idx=())` — [`L572`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L572)
  - `is_high(self)` — [`L463`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L463)
  - `lo_ty(self)` — [`L466`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L466)
  - `lower_val(self, ref)` — [`L472`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L472)
  - `manual_axis_type(self)` — [`L538`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L538)
  - `mat(self)` — [`L547`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L547)
  - `raise_val(self, *vals)` — [`L477`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L477)
  - `reshape(self, *shape)` — [`L559`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L559)
  - `set(tracer, value, idx=())` — [`L584`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L584)
  - `shape(self)` — [`L511`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L511)
  - `sharding(self)` — [`L529`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L529)
  - `str_short(self, short_dtypes=False, mesh_axis_types=False)` — [`L606`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L606)
  - `swap(tracer, value, idx=())` — [`L578`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L578)
  - `to_ct_aval(self)` — [`L623`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L623)
  - `to_tangent_aval(self)` — [`L619`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L619)
  - `transpose(self, *permutation)` — [`L563`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L563)
  - `update(self, inner_aval=None, memory_space=None, kind=None)` — [`L495`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L495)
  - `update_manual_axis_type(self, mat)` — [`L492`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L492)
  - `update_weak_type(self, weak_type)` — [`L489`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L489)
  - `weak_type(self)` — [`L484`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L484)
  - `ndim` — [`L501`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L501)
  - `size` — [`L502`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L502)
- protocol/private: `__eq__`[`L627`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L627), `__hash__`[`L631`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L631), `__init__`[`L443`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L443), `__repr__`[`L615`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L615), `__slots__`[`L441`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L441), `__str__`[`L617`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L617), `_addupdate`[`L602`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L602), `_getitem`[`L594`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L594), `_len`[`L504`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L504), `_setitem`[`L598`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L598)
- uses (calls/refs, reference-scoped): [`Array`](../basearray.md#Array), [`ShapedArray`](../core.md#ShapedArray), [`AbstractValue`](../core.md#AbstractValue), [`shape`](../core.md#ShapedArray.shape), [`dtype`](../core.md#ShapedArray.dtype), [`TransformedRef`](types.md#TransformedRef), [`sharding`](../core.md#ShapedArray.sharding), [`MemorySpace`](../core.md#MemorySpace), [`update`](../core.md#ShapedArray.update), [`ref_swap`](primitives.md#ref_swap), [`memory_space`](../core.md#ShapedArray.memory_space), [`ref_get`](primitives.md#ref_get), [`str_short`](../core.md#AbstractValue.str_short), [`aval_method`](../core.md#aval_method), [`ref_set`](primitives.md#ref_set), [`ref_addupdate`](primitives.md#ref_addupdate), [`to_ct_aval`](../core.md#AbstractValue.to_ct_aval), [`weak_type`](../core.md#ShapedArray.weak_type), [`Ref`](../core.md#Ref), [`reshape`](types.md#TransformedRef.reshape), [`to_tangent_aval`](../core.md#AbstractValue.to_tangent_aval), [`bitcast`](types.md#TransformedRef.bitcast), [`lo_ty`](../core.md#AbstractValue.lo_ty), [`Device`](../core.md#MemorySpace.Device), [`transpose`](types.md#TransformedRef.transpose), [`is_high`](../core.md#AbstractValue.is_high), [`update`](../pallas/mosaic_gpu/core.md#AbstractRefUnion.update), [`update`](../pallas/mosaic_gpu/core.md#AbstractTMEMRef.update), [`update`](../pallas/mosaic_gpu/core.md#WGMMAAbstractAccumulatorRef.update), [`update_weak_type`](../core.md#AbstractValue.update_weak_type), [`manual_axis_type`](../core.md#ShapedArray.manual_axis_type), [`update_manual_axis_type`](../core.md#AbstractValue.update_manual_axis_type), [`_getitem`](../pallas/mosaic_gpu/core.md#WGMMAAbstractAccumulatorRef._getitem), [`_setitem`](../pallas/mosaic_gpu/core.md#WGMMAAbstractAccumulatorRef._setitem), [`aval_property`](../core.md#aval_property), [`RefIndexer`](types.md#RefIndexer), [`T`](types.md#TransformedRef.T), [`lower_val`](../core.md#ShapedArray.lower_val), [`raise_val`](../core.md#ShapedArray.raise_val), [`_getitem`](../pallas/mosaic_gpu/core.md#AbstractRefUnion._getitem)  (+1 more)
- used by: [`_interpret_jaxpr`](../pallas/mosaic/interpret/interpret_pallas_call.md#_interpret_jaxpr), [`interpret_pallas_call`](../pallas/mosaic/interpret/interpret_pallas_call.md#interpret_pallas_call), [`lower_jaxpr_into_pipelined_module`](../pallas/mosaic/lowering.md#lower_jaxpr_into_pipelined_module), [`_handle_transforms`](../pallas/mosaic_gpu/lowering.md#_handle_transforms), [`to_block_mapping`](../pallas/core.md#BlockSpec.to_block_mapping), [`_run_scoped_lowering_rule`](../pallas/mosaic_gpu/lowering.md#_run_scoped_lowering_rule), [`_check_jaxpr`](../core.md#_check_jaxpr), [`_eval_jaxpr_discharge_state`](discharge.md#_eval_jaxpr_discharge_state), [`_copy_gmem_to_smem_lowering`](../pallas/mosaic_gpu/primitives.md#_copy_gmem_to_smem_lowering), [`_swap_lowering_rule`](../pallas/mosaic_gpu/lowering.md#_swap_lowering_rule), [`_allocate_for_aval`](../pallas/mosaic_gpu/interpret/jaxpr_interpret.md#JaxprInterpreter._allocate_for_aval), [`_get_lowering_rule`](../pallas/mosaic_gpu/lowering.md#_get_lowering_rule), [`_pallas_call_state_discharge_rule`](../pallas/pallas_call.md#_pallas_call_state_discharge_rule), [`cond`](../lax/control_flow/conditionals.md#cond), [`_run_scoped_resource_estimator`](../pallas/mosaic_gpu/lowering.md#_run_scoped_resource_estimator), [`_check_block_mappings`](../pallas/mosaic/lowering.md#_check_block_mappings), [`_semaphore_signal_multicast_lowering`](../pallas/mosaic_gpu/primitives.md#_semaphore_signal_multicast_lowering), [`_masked_load_lowering_rule`](../pallas/triton/lowering.md#_masked_load_lowering_rule), [`_copy_smem_to_gmem_lowering`](../pallas/mosaic_gpu/primitives.md#_copy_smem_to_gmem_lowering), [`body_func`](../pallas/mosaic/lowering.md#lower_jaxpr_to_transform_func.body_func), [`_load_lowering_rule`](../pallas/mosaic/sc_lowering.md#_load_lowering_rule), [`_store_lowering_rule`](../pallas/mosaic/sc_lowering.md#_store_lowering_rule), [`_async_store_smem_lowering`](../pallas/mosaic_gpu/primitives.md#_async_store_smem_lowering), [`_semaphore_signal_lowering_rule`](../pallas/mosaic_gpu/primitives.md#_semaphore_signal_lowering_rule), [`lower_jaxpr_into_unpipelined_module`](../pallas/mosaic/lowering.md#lower_jaxpr_into_unpipelined_module), [`_semaphore_wait_lowering_rule`](../pallas/mosaic_gpu/primitives.md#_semaphore_wait_lowering_rule), [`_scan_partial_eval_custom`](../lax/control_flow/loops.md#_scan_partial_eval_custom), [`_multimem_store_lowering_rule`](../pallas/mosaic_gpu/primitives.md#_multimem_store_lowering_rule), [`_run_state_lowering_rule`](../pallas/mosaic_gpu/lowering.md#_run_state_lowering_rule), [`aval_to_ir_type`](../pallas/mosaic/lowering.md#aval_to_ir_type), [`get_ref_and_transforms`](primitives.md#get_ref_and_transforms), [`_atomic_lowering_rule`](../pallas/triton/primitives.md#_atomic_lowering_rule), [`_tcgen05_mma_lowering`](../pallas/mosaic_gpu/primitives.md#_tcgen05_mma_lowering), [`_barrier_arrive_lowering`](../pallas/mosaic_gpu/primitives.md#_barrier_arrive_lowering), [`_swap_discharge_rule`](../pallas/primitives.md#_swap_discharge_rule), [`_jaxpr_call_lowering_rule`](../pallas/mosaic_gpu/lowering.md#_jaxpr_call_lowering_rule), [`_inline_mgpu_flat_transformed_args`](../pallas/mosaic_gpu/primitives.md#_inline_mgpu_flat_transformed_args), [`_extract_indirect_offsets_from_indices`](../pallas/mosaic/sc_lowering.md#_extract_indirect_offsets_from_indices), [`_multimem_load_reduce_lowering_rule`](../pallas/mosaic_gpu/primitives.md#_multimem_load_reduce_lowering_rule), [`_rewrite_jaxpr_for_lowering`](../pallas/mosaic/pallas_call_registration.md#_rewrite_jaxpr_for_lowering)  (+209 more)

### `AccumEffect`  ·  implements/extends RefEffect
- def: [`jax/_src/state/types.py:76`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L76)
- signature: `class AccumEffect(RefEffect):`
- members:
  - `name` — [`L77`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L77)
- uses (calls/refs, reference-scoped): [`RefEffect`](types.md#RefEffect), [`InternalMutableArrayEffect`](../core.md#InternalMutableArrayEffect)
- used by: [`_construct_output_fusions`](../pallas/fuser/jaxpr_fusion.md#_construct_output_fusions), [`_addupdate_abstract_eval`](primitives.md#_addupdate_abstract_eval), [`_mpmd_map_discharge_rule`](../pallas/mpmd.md#_mpmd_map_discharge_rule), [`RefEffect`](types.md#RefEffect), [`StateEffect`](types.md#StateEffect), [`get_write_indices`](../pallas/fuser/fuser_utils.md#get_write_indices)

### `BitcastTransform`  ·  implements/extends Transform
- def: [`jax/_src/state/types.py:116`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L116)
- signature: `class BitcastTransform(Transform):`
- members:
  - `pretty_print(self, context: core.JaxprPpContext)` — [`L136`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L136)
  - `transform_type(self, x)` — [`L119`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L119)
  - `undo(self, x: core.AbstractValue)` — [`L133`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L133)
  - `dtype` — [`L117`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L117)
- uses (calls/refs, reference-scoped): [`AbstractRef`](types.md#AbstractRef), [`ShapedArray`](../core.md#ShapedArray), [`AbstractValue`](../core.md#AbstractValue), [`text`](../pretty_printer.md#text), [`Transform`](types.md#Transform), [`DType`](../typing.md#DType), [`sharding`](../core.md#ShapedArray.sharding), [`JaxprPpContext`](../core.md#JaxprPpContext), [`Doc`](../pretty_printer.md#Doc), [`update`](../core.md#ShapedArray.update), [`register_dataclass`](../tree_util.md#register_dataclass), [`update`](types.md#AbstractRef.update), [`eval_bitcast_shape`](utils.md#eval_bitcast_shape)
- used by: [`Transform`](types.md#Transform), [`transform_swap_array`](discharge.md#transform_swap_array), [`transform_type`](types.md#Transform.transform_type), [`undo`](types.md#Transform.undo), [`_transform_ref`](../pallas/mosaic/lowering.md#_transform_ref), [`transform_array`](discharge.md#transform_array), [`_bitcast_memref`](../pallas/mosaic/lowering.md#_bitcast_memref), [`pretty_print`](types.md#Transform.pretty_print), [`bitcast`](types.md#TransformedRef.bitcast)

### `MultiRefTransform`  ·  implements/extends Transform
- def: [`jax/_src/state/types.py:103`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L103)
- signature: `class MultiRefTransform(Transform):`
- members:
  - `getattr(self, name: str, xs: Sequence[core.AbstractValue])` — [`L110`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L110)
  - `transform_types(self, xs: Sequence[core.AbstractValue])` — [`L105`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L105)
- uses (calls/refs, reference-scoped): [`AbstractValue`](../core.md#AbstractValue), [`Transform`](types.md#Transform), [`SelectTransform`](types.md#SelectTransform), [`getattr`](types.md#SelectTransform.getattr), [`transform_types`](types.md#SelectTransform.transform_types)
- used by: [`Transform`](types.md#Transform), [`type`](types.md#TransformedRef.type), [`_lower_multiref_transformed_ref`](../pallas/mosaic/lowering.md#_lower_multiref_transformed_ref), [`__getattr__`](types.md#TransformedRef.__getattr__), [`SelectTransform`](types.md#SelectTransform), [`__post_init__`](types.md#TransformedRef.__post_init__)

### `ReadEffect`  ·  implements/extends RefEffect
- def: [`jax/_src/state/types.py:70`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L70)
- signature: `class ReadEffect(RefEffect):`
- members:
  - `name` — [`L71`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L71)
- uses (calls/refs, reference-scoped): [`RefEffect`](types.md#RefEffect)
- used by: [`_emit_pipeline_effectful_abstract_eval`](../pallas/mosaic/pipeline.md#_emit_pipeline_effectful_abstract_eval), [`remove_duplicate_writes_between_async_stages`](../pallas/pipelining/schedule_api.md#remove_duplicate_writes_between_async_stages), [`RefEffect`](types.md#RefEffect), [`_gather_abstract_eval`](../pallas/mosaic/sc_primitives.md#_gather_abstract_eval), [`_async_copy_smem_to_tmem_abstract_eval`](../pallas/mosaic_gpu/primitives.md#_async_copy_smem_to_tmem_abstract_eval), [`_async_copy_to_tmem_abstract_eval`](../pallas/mosaic_gpu/primitives.md#_async_copy_to_tmem_abstract_eval), [`_inline_mgpu_abstract_eval`](../pallas/mosaic_gpu/primitives.md#_inline_mgpu_abstract_eval), [`_pipeline_body_effectful_abstract_eval`](../pallas/mosaic/pipeline.md#_pipeline_body_effectful_abstract_eval), [`_scatter_abstract_eval`](../pallas/mosaic/sc_primitives.md#_scatter_abstract_eval), [`_get_dma_effects`](../pallas/mosaic/primitives.md#_get_dma_effects), [`_wgmma_ref_effectful_abstract_eval`](../pallas/mosaic_gpu/primitives.md#_wgmma_ref_effectful_abstract_eval), [`_get_abstract_eval`](primitives.md#_get_abstract_eval), [`_load_abstract_eval`](../pallas/mosaic_gpu/primitives.md#_load_abstract_eval), [`_load_abstract_eval`](../pallas/primitives.md#_load_abstract_eval), [`_copy_gmem_to_smem_abstract_eval`](../pallas/mosaic_gpu/primitives.md#_copy_gmem_to_smem_abstract_eval), [`_copy_smem_to_gmem_abstract_eval`](../pallas/mosaic_gpu/primitives.md#_copy_smem_to_gmem_abstract_eval), [`_fetch_and_add_abstract_eval`](../pallas/mosaic/sc_primitives.md#_fetch_and_add_abstract_eval), [`_swap_abstract_eval`](../pallas/mosaic/sc_primitives.md#_swap_abstract_eval), [`_touch_abstract_eval`](../pallas/mosaic/primitives.md#_touch_abstract_eval), [`_wgmma_effectful_abstract_eval`](../pallas/mosaic_gpu/primitives.md#_wgmma_effectful_abstract_eval), [`RefEffect`](../pallas/pipelining/internal.md#RefEffect), [`_load_abstract_eval`](../pallas/mosaic/sc_primitives.md#_load_abstract_eval), [`StateEffect`](types.md#StateEffect), [`_async_prefetch_abstract_eval`](../pallas/mosaic_gpu/primitives.md#_async_prefetch_abstract_eval), [`ReadEffect`](../pallas/pipelining/internal.md#ReadEffect)

### `RefEffect`  ·  implements/extends JaxprInputEffect
- def: [`jax/_src/state/types.py:45`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L45)
- signature: `class RefEffect(effects.JaxprInputEffect):`
- members:
  - `name` — [`L46`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L46)
- protocol/private: `__eq__`[`L48`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L48), `__hash__`[`L53`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L53), `__str__`[`L67`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L67), `_pretty_print`[`L56`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L56)
- uses (calls/refs, reference-scoped): [`text`](../pretty_printer.md#text), [`Var`](../core.md#Var), [`JaxprPpContext`](../core.md#JaxprPpContext), [`Doc`](../pretty_printer.md#Doc), [`concat`](../pretty_printer.md#concat), `input`, [`WriteEffect`](types.md#WriteEffect), [`pp_var`](../core.md#pp_var), [`ReadEffect`](types.md#ReadEffect), [`color`](../pretty_printer.md#color), [`AccumEffect`](types.md#AccumEffect), [`_ref_effect_color`](types.md#_ref_effect_color)
- used by: [`WriteEffect`](types.md#WriteEffect), [`_cond_partial_eval`](../lax/control_flow/conditionals.md#_cond_partial_eval), [`_cond_batching_rule`](../lax/control_flow/conditionals.md#_cond_batching_rule), [`ReadEffect`](types.md#ReadEffect), [`_pjit_state_discharge_rule`](discharge.md#_pjit_state_discharge_rule), [`trace_fun`](../pallas/pipelining/schedule_api.md#trace_fun), [`_get_fastpath_data`](../pjit.md#_get_fastpath_data), [`_run_state_abstract_eval`](discharge.md#_run_state_abstract_eval), [`AccumEffect`](types.md#AccumEffect), [`name`](types.md#AccumEffect.name)

### `RefIndexer`
- def: [`jax/_src/state/types.py:270`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L270)
- doc: An object temporarily generated when doing `ref.at`.
- signature: `class RefIndexer:`
- members:
  - `ref_or_view` — [`L272`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L272)
- protocol/private: `__getitem__`[`L274`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L274)
- uses (calls/refs, reference-scoped): [`TransformedRef`](types.md#TransformedRef), [`NDIndexer`](indexing.md#NDIndexer), [`transforms`](types.md#TransformedRef.transforms), [`ref`](types.md#TransformedRef.ref), [`from_indices_shape`](indexing.md#NDIndexer.from_indices_shape), [`multiref`](types.md#TransformedRef.multiref)
- used by: [`at`](types.md#AbstractRef.at), [`at`](types.md#TransformedRef.at)

### `ReshapeTransform`  ·  implements/extends Transform
- def: [`jax/_src/state/types.py:166`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L166)
- signature: `class ReshapeTransform(Transform):`
- members:
  - `pretty_print(self, context: core.JaxprPpContext)` — [`L191`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L191)
  - `transform_type(self, x)` — [`L175`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L175)
  - `undo(self, x: core.AbstractValue)` — [`L188`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L188)
  - `shape` — [`L167`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L167)
- protocol/private: `_validate_shape`[`L169`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L169)
- uses (calls/refs, reference-scoped): [`AbstractRef`](types.md#AbstractRef), [`ShapedArray`](../core.md#ShapedArray), [`AbstractValue`](../core.md#AbstractValue), [`shape`](../core.md#ShapedArray.shape), [`text`](../pretty_printer.md#text), [`Transform`](types.md#Transform), [`sharding`](../core.md#ShapedArray.sharding), [`JaxprPpContext`](../core.md#JaxprPpContext), [`Doc`](../pretty_printer.md#Doc), [`update`](../core.md#ShapedArray.update), [`register_dataclass`](../tree_util.md#register_dataclass), [`update`](types.md#AbstractRef.update)
- used by: [`_handle_transforms`](../pallas/mosaic_gpu/lowering.md#_handle_transforms), [`Transform`](types.md#Transform), [`transform_swap_array`](discharge.md#transform_swap_array), [`_commute_transform`](../pallas/mosaic_gpu/lowering.md#_commute_transform), [`transform_type`](types.md#Transform.transform_type), [`undo`](types.md#Transform.undo), [`_transform_ref`](../pallas/mosaic/lowering.md#_transform_ref), [`transform_array`](discharge.md#transform_array), [`_bubble_up_transforms_for_lowering`](../pallas/mosaic_gpu/lowering.md#_bubble_up_transforms_for_lowering), [`pretty_print`](types.md#Transform.pretty_print), [`_reshape_memref`](../pallas/mosaic/lowering.md#_reshape_memref), [`reshape`](types.md#TransformedRef.reshape), [`commute_reshape`](../pallas/mosaic_gpu/core.md#UntilingTransform.commute_reshape), [`commute_reshape`](../pallas/mosaic_gpu/core.md#UnswizzleRef.commute_reshape)

### `SelectTransform`  ·  implements/extends MultiRefTransform
- def: [`jax/_src/state/types.py:236`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L236)
- signature: `class SelectTransform(MultiRefTransform):`
- members:
  - `getattr(self, name: str, xs: Sequence[core.AbstractValue])` — [`L262`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L262)
  - `pretty_print(self, context: core.JaxprPpContext)` — [`L258`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L258)
  - `transform_types(self, xs)` — [`L239`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L239)
  - `undo(self, x: core.AbstractValue)` — [`L255`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L255)
  - `idx` — [`L237`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L237)
- protocol/private: `_type`[`L240`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L240)
- uses (calls/refs, reference-scoped): [`Array`](../basearray.md#Array), [`AbstractRef`](types.md#AbstractRef), [`ShapedArray`](../core.md#ShapedArray), [`AbstractValue`](../core.md#AbstractValue), [`text`](../pretty_printer.md#text), [`Transform`](types.md#Transform), [`JaxprPpContext`](../core.md#JaxprPpContext), [`Doc`](../pretty_printer.md#Doc), [`register_dataclass`](../tree_util.md#register_dataclass), [`MultiRefTransform`](types.md#MultiRefTransform)
- used by: [`undo`](types.md#Transform.undo), [`_transform_ref`](../pallas/mosaic/lowering.md#_transform_ref), [`pretty_print`](types.md#Transform.pretty_print), [`_lower_multiref_transformed_ref`](../pallas/mosaic/lowering.md#_lower_multiref_transformed_ref), [`select_ref`](../pallas/helpers.md#select_ref), [`MultiRefTransform`](types.md#MultiRefTransform), [`getattr`](types.md#MultiRefTransform.getattr), [`transform_types`](types.md#MultiRefTransform.transform_types)

### `Transform`  ·  implements/extends Protocol
- def: [`jax/_src/state/types.py:91`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L91) — documented in [jax-_src-state-types](../../../../concepts/jax-_src-state-types.md)
- signature: `class Transform(Protocol):`
- members:
  - `pretty_print(self, context: core.JaxprPpContext)` — [`L99`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L99)
  - `transform_type(self, x: core.AbstractValue)` — [`L93`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L93)
  - `undo(self, x: core.AbstractValue)` — [`L96`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L96)
- uses (calls/refs, reference-scoped): [`AbstractValue`](../core.md#AbstractValue), [`text`](../pretty_printer.md#text), [`JaxprPpContext`](../core.md#JaxprPpContext), [`Doc`](../pretty_printer.md#Doc), [`transform_type`](indexing.md#NDIndexer.transform_type), [`TransposeTransform`](types.md#TransposeTransform), [`pretty_print`](indexing.md#NDIndexer.pretty_print), [`transform_type`](types.md#TransposeTransform.transform_type), [`transform_type`](../pallas/mosaic_gpu/core.md#ExpandLeadingBatchDimensionsTransform.transform_type), [`transform_type`](../pallas/mosaic_gpu/core.md#SwizzleTransform.transform_type), [`transform_type`](types.md#ReshapeTransform.transform_type), [`ReshapeTransform`](types.md#ReshapeTransform), [`transform_type`](types.md#BitcastTransform.transform_type), [`transform_type`](../pallas/mosaic_gpu/core.md#CollapseLeadingBatchDimensionsTransform.transform_type), [`transform_type`](../pallas/mosaic_gpu/core.md#UntilingTransform.transform_type), [`transform_type`](../pallas/mosaic_gpu/core.md#ExtractAliasedRef.transform_type), [`transform_type`](../pallas/mosaic_gpu/core.md#TilingTransform.transform_type), [`TilingTransform`](../pallas/mosaic_gpu/core.md#TilingTransform), [`PeerMemRef`](../pallas/mosaic_gpu/core.md#PeerMemRef), [`undo`](types.md#TransposeTransform.undo), [`BitcastTransform`](types.md#BitcastTransform), [`MulticastRef`](../pallas/mosaic_gpu/core.md#MulticastRef), [`undo`](../pallas/mosaic_gpu/core.md#UntilingTransform.undo), [`ClusterRefTransform`](../pallas/mosaic_gpu/core.md#ClusterRefTransform), [`MultiRefTransform`](types.md#MultiRefTransform), [`pretty_print`](../pallas/mosaic_gpu/core.md#UnswizzleRef.pretty_print), [`pretty_print`](../pallas/mosaic_gpu/core.md#UntilingTransform.pretty_print), [`pretty_print`](types.md#BitcastTransform.pretty_print), [`pretty_print`](types.md#ReshapeTransform.pretty_print), [`pretty_print`](types.md#SelectTransform.pretty_print), [`pretty_print`](types.md#TransposeTransform.pretty_print), [`undo`](../pallas/mosaic_gpu/core.md#SwizzleTransform.undo), [`undo`](../pallas/mosaic_gpu/core.md#TilingTransform.undo), [`undo`](../pallas/mosaic_gpu/core.md#UnswizzleRef.undo), [`undo`](../pallas/mosaic_gpu/core.md#CollapseLeadingBatchDimensionsTransform.undo), [`undo`](../pallas/mosaic_gpu/core.md#ClusterRefTransform.undo), [`undo`](../pallas/mosaic_gpu/core.md#IdentityTransform.undo), [`undo`](../pallas/mosaic_gpu/core.md#MulticastRef.undo), [`undo`](../pallas/mosaic_gpu/core.md#PeerMemRef.undo), [`undo`](types.md#BitcastTransform.undo)  (+8 more)
- used by: [`_handle_transforms`](../pallas/mosaic_gpu/lowering.md#_handle_transforms), [`_allocate_for_aval`](../pallas/mosaic_gpu/interpret/jaxpr_interpret.md#JaxprInterpreter._allocate_for_aval), [`interpret_pallas_call`](../pallas/mosaic_gpu/interpret/interpret_pallas_call.md#interpret_pallas_call), [`NDIndexer`](indexing.md#NDIndexer), [`transforms`](types.md#TransformedRef.transforms), [`get_ref_and_transforms`](primitives.md#get_ref_and_transforms), [`_jaxpr_call_lowering_rule`](../pallas/mosaic_gpu/lowering.md#_jaxpr_call_lowering_rule), [`_commute_transform`](../pallas/mosaic_gpu/lowering.md#_commute_transform), [`get_ref_aval`](../pallas/mosaic_gpu/core.md#GPUMemoryRef.get_ref_aval), [`_trace_kernel_to_jaxpr`](../pallas/pallas_call.md#_trace_kernel_to_jaxpr), [`__call__`](../pallas/mosaic_gpu/core.md#MemorySpace.__call__), [`_get_kernel_buffers`](../pallas/mosaic_gpu/interpret/interpret_pallas_call.md#_get_kernel_buffers), [`to_block_mapping`](../pallas/mosaic_gpu/core.md#BlockSpec.to_block_mapping), [`UntilingTransform`](../pallas/mosaic_gpu/core.md#UntilingTransform), [`commute_ndindexer`](../pallas/mosaic_gpu/core.md#ExpandLeadingBatchDimensionsTransform.commute_ndindexer), [`UnswizzleRef`](../pallas/mosaic_gpu/core.md#UnswizzleRef), [`_bubble_up_transforms_for_lowering`](../pallas/mosaic_gpu/lowering.md#_bubble_up_transforms_for_lowering), [`transform_type`](types.md#transform_type), [`TransposeTransform`](types.md#TransposeTransform), [`type`](types.md#TransformedRef.type), [`_extract_indirect_offsets`](../pallas/mosaic/sc_lowering.md#_extract_indirect_offsets), [`_bubble_up_transform`](../pallas/mosaic_gpu/lowering.md#_bubble_up_transform), [`ReshapeTransform`](types.md#ReshapeTransform), [`_has_indirect_offsets`](../pallas/mosaic/sc_lowering.md#_has_indirect_offsets), [`SwizzleTransform`](../pallas/mosaic_gpu/core.md#SwizzleTransform), [`TilingTransform`](../pallas/mosaic_gpu/core.md#TilingTransform), [`to_gpu_transform`](../pallas/mosaic_gpu/core.md#to_gpu_transform), [`transform_ref`](../pallas/mosaic_gpu/core.md#transform_ref), [`transforms`](../pallas/mosaic_gpu/core.md#BlockSpec.transforms), [`undo_transforms`](../pallas/core.md#undo_transforms), [`apply_unswizzle_and_untile`](../pallas/mosaic_gpu/interpret/jaxpr_interpret.md#apply_unswizzle_and_untile), [`PeerMemRef`](../pallas/mosaic_gpu/core.md#PeerMemRef), [`_pp_transforms`](primitives.md#_pp_transforms), [`batch_transform`](../pallas/mosaic_gpu/core.md#batch_transform), [`RefType`](../pallas/mosaic_gpu/primitives.md#RefType), [`undo`](types.md#TransposeTransform.undo), [`BitcastTransform`](types.md#BitcastTransform), [`MulticastRef`](../pallas/mosaic_gpu/core.md#MulticastRef), [`to_transform_attr`](../pallas/mosaic_gpu/core.md#to_transform_attr), [`undo`](../pallas/mosaic_gpu/core.md#UntilingTransform.undo)  (+21 more)

### `TransformedRef`
- def: [`jax/_src/state/types.py:289`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L289) — documented in [jax-_src-state-types](../../../../concepts/jax-_src-state-types.md)
- members:
  - `at(self)` — [`L348`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L348)
  - `bitcast(self, dtype)` — [`L351`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L351)
  - `dtype(self)` — [`L338`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L338)
  - `get(self, idx=())` — [`L390`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L390)
  - `is_dynamic_size(self)` — [`L310`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L310)
  - `multiref(self)` — [`L302`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L302)
  - `reshape(self, *shape)` — [`L361`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L361)
  - `set(self, value, idx=())` — [`L382`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L382)
  - `shape(self)` — [`L332`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L332)
  - `swap(self, value, idx=())` — [`L386`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L386)
  - `transpose(self, permutation: Sequence[int])` — [`L374`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L374)
  - `type(self)` — [`L314`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L314)
  - `T` — [`L345`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L345)
  - `ndim` — [`L343`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L343)
  - `ref` — [`L290`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L290)
  - `size` — [`L344`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L344)
  - `transforms` — [`L291`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L291) — documented in [jax-_src-state-types](../../../../concepts/jax-_src-state-types.md)
- protocol/private: `__getattr__`[`L394`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L394), `__getitem__`[`L399`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L399), `__post_init__`[`L293`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L293), `__setitem__`[`L403`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L403), `_type`[`L315`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L315)
- uses (calls/refs, reference-scoped): [`Array`](../basearray.md#Array), [`typeof`](../core.md#typeof), [`AbstractValue`](../core.md#AbstractValue), [`Transform`](types.md#Transform), [`dtype`](../dtypes.md#dtype), [`transform_type`](types.md#Transform.transform_type), [`ref_swap`](primitives.md#ref_swap), [`ref_get`](primitives.md#ref_get), [`ref_set`](primitives.md#ref_set), [`pytype_aval_mappings`](../core.md#pytype_aval_mappings.pytype_aval_mappings), [`TransposeTransform`](types.md#TransposeTransform), [`ReshapeTransform`](types.md#ReshapeTransform), [`BitcastTransform`](types.md#BitcastTransform), [`MultiRefTransform`](types.md#MultiRefTransform), [`getattr`](types.md#MultiRefTransform.getattr), [`transform_types`](types.md#MultiRefTransform.transform_types), [`RefIndexer`](types.md#RefIndexer), [`_canonicalize_reshape`](types.md#_canonicalize_reshape)
- used by: [`wrapped`](../pallas/mosaic/pipeline.md#emit_pipeline.wrapped), [`get_ref_and_transforms`](primitives.md#get_ref_and_transforms), [`wrapped`](../pallas/pallas_call.md#_pallas_call.wrapped), [`pp_ref_transforms`](primitives.md#pp_ref_transforms), [`_extract_indirect_offsets_from_indices`](../pallas/mosaic/sc_lowering.md#_extract_indirect_offsets_from_indices), [`run_scoped`](../pallas/primitives.md#run_scoped), [`get_ref_aval`](../pallas/mosaic_gpu/core.md#GPUMemoryRef.get_ref_aval), [`flatten_ref_union`](../pallas/mosaic_gpu/core.md#flatten_ref_union), [`_Ref`](../pallas/mosaic_gpu/primitives.md#_Ref), [`ref_swap`](primitives.md#ref_swap), [`from_indices_shape`](indexing.md#NDIndexer.from_indices_shape), [`ref_get`](primitives.md#ref_get), [`unflatten`](../pallas/mosaic_gpu/core.md#flatten_ref_union.unflatten), [`_transform_ref`](../pallas/mosaic/lowering.md#_transform_ref), [`fetch_and_add`](../pallas/mosaic/sc_primitives.md#fetch_and_add), [`wgmma`](../pallas/mosaic_gpu/primitives.md#wgmma), [`__post_init__`](indexing.md#NDIndexer.__post_init__), [`_emit_pipeline_effectful_abstract_eval`](../pallas/mosaic/pipeline.md#_emit_pipeline_effectful_abstract_eval), [`ref_set`](primitives.md#ref_set), [`remote_ref`](../pallas/mosaic_gpu/core.md#remote_ref), [`_jaxpr_call`](../pallas/primitives.md#_jaxpr_call), [`ref_addupdate`](primitives.md#ref_addupdate), [`ref_aval`](../pallas/core.md#BlockMapping.ref_aval), [`wrapper`](../pallas/mosaic_gpu/primitives.md#inline_mgpu.inner.wrapper), [`_dma_wait_abstract_eval`](../pallas/mosaic/primitives.md#_dma_wait_abstract_eval), [`tcgen05_mma`](../pallas/mosaic_gpu/primitives.md#tcgen05_mma), [`_async_store_smem_abstract_eval`](../pallas/mosaic_gpu/primitives.md#_async_store_smem_abstract_eval), [`_gather_abstract_eval`](../pallas/mosaic/sc_primitives.md#_gather_abstract_eval), [`_get_ref_and_transforms`](../pallas/mosaic/primitives.md#_get_ref_and_transforms), [`_ref_to_value_aval`](../pallas/mosaic/pipeline.md#_ref_to_value_aval), [`addupdate`](../pallas/mosaic/sc_primitives.md#addupdate), [`addupdate_compressed`](../pallas/mosaic/sc_primitives.md#addupdate_compressed), [`cluster_ref`](../pallas/mosaic_gpu/core.md#cluster_ref), [`load_expanded`](../pallas/mosaic/sc_primitives.md#load_expanded), [`multicast_ref`](../pallas/mosaic_gpu/core.md#multicast_ref), [`store_compressed`](../pallas/mosaic/sc_primitives.md#store_compressed), [`_pipeline_body_effectful_abstract_eval`](../pallas/mosaic/pipeline.md#_pipeline_body_effectful_abstract_eval), [`_scatter_abstract_eval`](../pallas/mosaic/sc_primitives.md#_scatter_abstract_eval), [`multimem_store`](../pallas/mosaic_gpu/primitives.md#multimem_store), [`query_cluster_cancel`](../pallas/mosaic_gpu/primitives.md#query_cluster_cancel)  (+44 more)

### `TransformedRefAvalError`  ·  implements/extends Exception
- def: [`jax/_src/state/types.py:407`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L407)
- signature: `class TransformedRefAvalError(Exception):`
- used by: [`disallow_transformed_ref_avals`](types.md#disallow_transformed_ref_avals)

### `TransposeTransform`  ·  implements/extends Transform
- def: [`jax/_src/state/types.py:205`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L205)
- signature: `class TransposeTransform(Transform):`
- members:
  - `pretty_print(self, context: core.JaxprPpContext)` — [`L229`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L229)
  - `transform_type(self, x)` — [`L211`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L211)
  - `undo(self, x: core.AbstractValue)` — [`L208`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L208)
  - `permutation` — [`L206`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L206)
- uses (calls/refs, reference-scoped): [`AbstractRef`](types.md#AbstractRef), [`ShapedArray`](../core.md#ShapedArray), [`AbstractValue`](../core.md#AbstractValue), [`shape`](../core.md#ShapedArray.shape), [`text`](../pretty_printer.md#text), [`Transform`](types.md#Transform), [`sharding`](../core.md#ShapedArray.sharding), [`JaxprPpContext`](../core.md#JaxprPpContext), [`Doc`](../pretty_printer.md#Doc), [`update`](../core.md#ShapedArray.update), [`ndim`](../core.md#ShapedArray.ndim), [`register_dataclass`](../tree_util.md#register_dataclass), [`update`](types.md#AbstractRef.update), [`_perm_inverse`](types.md#_perm_inverse)
- used by: [`_handle_transforms`](../pallas/mosaic_gpu/lowering.md#_handle_transforms), [`Transform`](types.md#Transform), [`_swap_lowering_rule`](../pallas/mosaic_gpu/lowering.md#_swap_lowering_rule), [`_get_lowering_rule`](../pallas/mosaic_gpu/lowering.md#_get_lowering_rule), [`_tcgen05_mma_lowering`](../pallas/mosaic_gpu/primitives.md#_tcgen05_mma_lowering), [`_commute_transform`](../pallas/mosaic_gpu/lowering.md#_commute_transform), [`transform_type`](types.md#Transform.transform_type), [`_wgmma_lowering`](../pallas/mosaic_gpu/primitives.md#_wgmma_lowering), [`undo`](types.md#Transform.undo), [`_bubble_up_transforms_for_lowering`](../pallas/mosaic_gpu/lowering.md#_bubble_up_transforms_for_lowering), [`pretty_print`](types.md#Transform.pretty_print), [`commute_transpose_indexer`](../pallas/mosaic_gpu/core.md#commute_transpose_indexer), [`transpose`](types.md#TransformedRef.transpose), [`batch_transform`](../pallas/mosaic_gpu/core.md#batch_transform), [`commute_transpose`](../pallas/mosaic_gpu/core.md#UntilingTransform.commute_transpose), [`commute_transpose`](../pallas/mosaic_gpu/core.md#UnswizzleRef.commute_transpose), [`TransposeTransform`](../pallas/mosaic_gpu/core.md#TransposeTransform)

### `Uninitialized`
- def: [`jax/_src/state/types.py:660`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L660)
- signature: `class Uninitialized:`
- used by: [`_ref_type_aval_mappings`](types.md#_ref_type_aval_mappings._ref_type_aval_mappings), [`uninitialized`](types.md#uninitialized)

### `WriteEffect`  ·  implements/extends RefEffect
- def: [`jax/_src/state/types.py:73`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L73)
- signature: `class WriteEffect(RefEffect):`
- members:
  - `name` — [`L74`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L74)
- uses (calls/refs, reference-scoped): [`RefEffect`](types.md#RefEffect)
- used by: [`_construct_output_fusions`](../pallas/fuser/jaxpr_fusion.md#_construct_output_fusions), [`_while_partial_discharge_rule`](../lax/control_flow/loops.md#_while_partial_discharge_rule), [`_emit_pipeline_effectful_abstract_eval`](../pallas/mosaic/pipeline.md#_emit_pipeline_effectful_abstract_eval), [`default_mesh_discharge_rule`](../pallas/core.md#default_mesh_discharge_rule), [`_mpmd_map_discharge_rule`](../pallas/mpmd.md#_mpmd_map_discharge_rule), [`_swap_abstract_eval`](primitives.md#_swap_abstract_eval), [`_multimem_store_abstract_eval`](../pallas/mosaic_gpu/primitives.md#_multimem_store_abstract_eval), [`RefEffect`](types.md#RefEffect), [`_async_store_smem_abstract_eval`](../pallas/mosaic_gpu/primitives.md#_async_store_smem_abstract_eval), [`_wgmma_accumulator_store_abstract_eval`](../pallas/mosaic_gpu/primitives.md#_wgmma_accumulator_store_abstract_eval), [`_async_copy_smem_to_tmem_abstract_eval`](../pallas/mosaic_gpu/primitives.md#_async_copy_smem_to_tmem_abstract_eval), [`_async_copy_to_tmem_abstract_eval`](../pallas/mosaic_gpu/primitives.md#_async_copy_to_tmem_abstract_eval), [`_inline_mgpu_abstract_eval`](../pallas/mosaic_gpu/primitives.md#_inline_mgpu_abstract_eval), [`_pipeline_body_effectful_abstract_eval`](../pallas/mosaic/pipeline.md#_pipeline_body_effectful_abstract_eval), [`_scatter_abstract_eval`](../pallas/mosaic/sc_primitives.md#_scatter_abstract_eval), [`_get_dma_effects`](../pallas/mosaic/primitives.md#_get_dma_effects), [`_wgmma_ref_effectful_abstract_eval`](../pallas/mosaic_gpu/primitives.md#_wgmma_ref_effectful_abstract_eval), [`convert_accum_effects_to_writes`](../pallas/pipelining/schedule_api.md#convert_accum_effects_to_writes), [`_swap_abstract_eval`](../pallas/primitives.md#_swap_abstract_eval), [`_copy_gmem_to_smem_abstract_eval`](../pallas/mosaic_gpu/primitives.md#_copy_gmem_to_smem_abstract_eval), [`_copy_smem_to_gmem_abstract_eval`](../pallas/mosaic_gpu/primitives.md#_copy_smem_to_gmem_abstract_eval), [`_fetch_and_add_abstract_eval`](../pallas/mosaic/sc_primitives.md#_fetch_and_add_abstract_eval), [`_swap_abstract_eval`](../pallas/mosaic/sc_primitives.md#_swap_abstract_eval), [`_touch_abstract_eval`](../pallas/mosaic/primitives.md#_touch_abstract_eval), [`RefEffect`](../pallas/pipelining/internal.md#RefEffect), [`_atomic_store_abstract_eval`](../pallas/mosaic_gpu/primitives.md#_atomic_store_abstract_eval), [`StateEffect`](types.md#StateEffect), [`get_write_indices`](../pallas/fuser/fuser_utils.md#get_write_indices), [`WriteEffect`](../pallas/pipelining/internal.md#WriteEffect), [`_atomic_cas_abstract_eval`](../pallas/triton/primitives.md#_atomic_cas_abstract_eval)

## Functions
- `_canonicalize_reshape(input_shape: tuple[int, ...], shape: tuple[int, ...])` — [`L141`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L141)
- `_default_value_to_ref_aval(x: Any)` — [`L670`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L670)
- `_map_ref(size, axis, ref_aval)` — [`L634`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L634)
- `_perm_inverse(permutation: tuple[int, ...])` — [`L196`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L196)
- `_shard_ref(mesh, auto, check_rep, names, ref_aval: AbstractRef)` — [`L649`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L649)
- `_unmap_ref(size, axis, explicit_mesh_axis, ref_aval)` — [`L638`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L638)
- `_unshard_ref(mesh, check_rep, names, ref_aval: AbstractRef)` — [`L654`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L654)
- `disallow_transformed_ref_avals(_)` — [`L410`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L410)
- `get_ref_aval_from_value(x: Any)` — [`L676`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L676)
- `shaped_array_ref(shape: tuple[int, ...], dtype, weak_type: bool = False)` — [`L645`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L645)
- `transform_type(ts: Sequence[Transform], ty: core.AbstractValue)` — [`L425`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L425)
- `zeros_like_abstract_ref(aval: AbstractRef)` — [`L682`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L682)

## Module values
- `StateEffect` — [`L85`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L85)
- `_ref_effect_color` — [`L43`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L43)
- `_ref_type_aval_mappings` — [`L665`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L665)
- `flatten_func` — [`L419`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L419)
- `unflatten_func` — [`L418`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L418)
- `uninitialized` — [`L662`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L662)
- `unsafe_map` — [`L39`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L39)
- `unsafe_zip` — [`L40`](../../../../../../../raw/code/jax/jax/_src/state/types.py#L40)


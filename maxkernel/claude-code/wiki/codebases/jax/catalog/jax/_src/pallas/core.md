---
title: 'Module: jax/_src/pallas/core.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/core.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.core`/
symbols:
  BlockSpec: BlockSpec#
  BlockSpec.to_block_mapping: BlockSpec#to_block_mapping().
  get_grid_mapping: get_grid_mapping().
  BlockSpec.block_shape: BlockSpec#block_shape.
  Squeezed: Squeezed#
  MemorySpace: MemorySpace#
  Mesh: Mesh#
  BlockMapping.block_shape: BlockMapping#block_shape.
  _get_sds: _get_sds().
  no_block_spec: no_block_spec.
  BlockMapping.array_aval: BlockMapping#array_aval.
  BlockSpec.index_map: BlockSpec#index_map.
  GridMapping.block_mappings: GridMapping#block_mappings.
  GridMapping.check_invariants: GridMapping#check_invariants().
  Blocked: Blocked#
  BlockSpec.memory_space: BlockSpec#memory_space.
  Mesh.core_type: Mesh#core_type().
  BlockDim.BlockDim: BlockDim.BlockDim.
  GridMapping.to_lojax: GridMapping#to_lojax().
  GridMapping.__repr__: GridMapping#__repr__().
  core_map.wrapped: core_map().wrapped().
  Element: Element#
  GridMapping.grid: GridMapping#grid.
  _convert_out_shape_to_aval: _convert_out_shape_to_aval().
  BlockMapping.check_invariants: BlockMapping#check_invariants().
  get_block_size: get_block_size().
  GridMapping.num_index_operands: GridMapping#num_index_operands.
  default_mesh_discharge_rule: default_mesh_discharge_rule().
  BlockMapping.index_map_jaxpr: BlockMapping#index_map_jaxpr.
  GridSpec.__init__: GridSpec#__init__().
  BlockMapping.to_lojax: BlockMapping#to_lojax().
  BlockMapping.ref_aval: BlockMapping#ref_aval().
  GridMapping.num_inputs: GridMapping#num_inputs.
  tracing_grid_env: tracing_grid_env().
  _convert_block_spec_to_block_mapping: _convert_block_spec_to_block_mapping().
  BlockMapping.has_trivial_window: BlockMapping#has_trivial_window().
  BlockMapping.__repr__: BlockMapping#__repr__().
  GridMapping.trace_env: GridMapping#trace_env().
  CostEstimate: CostEstimate#
  GridMapping.in_shapes: GridMapping#in_shapes().
  core_map_p: core_map_p.
  _canonicalize_block_dim: _canonicalize_block_dim().
  BlockMapping.to_block_spec: BlockMapping#to_block_spec().
  GridMapping: GridMapping#
  _core_map_to_lojax: _core_map_to_lojax().
  _core_map_abstract_eval: _core_map_abstract_eval().
  get_memory_space_aval: get_memory_space_aval().
  kernel_local_effects.kernel_local_effects: kernel_local_effects.kernel_local_effects.
  MemoryRef: MemoryRef#
  CoreMemorySpace: CoreMemorySpace#
  _get_block_dim_size: _get_block_dim_size().
  GridMapping.num_outputs: GridMapping#num_outputs.
  _core_map_discharge_rule: _core_map_discharge_rule().
  GridMapping.num_dynamic_grid_bounds: GridMapping#num_dynamic_grid_bounds().
  CoreMemorySpace.mesh: CoreMemorySpace#mesh.
  Mesh.shape: Mesh#shape().
  MemoryRef.get_array_aval: MemoryRef#get_array_aval().
  semaphore_dtype: semaphore_dtype#
  MemorySpace.ANY: MemorySpace#ANY.
  GridMapping.block_mappings_output: GridMapping#block_mappings_output().
  get_grid_mapping._with_default_memory_space: get_grid_mapping()._with_default_memory_space().
  Mesh.check_is_compatible_with: Mesh#check_is_compatible_with().
  BlockMapping.compute_start_indices_interpret: BlockMapping#compute_start_indices_interpret().
  GridMapping.vmapped_dims: GridMapping#vmapped_dims.
  index_map_grid_aval: index_map_grid_aval.
  Mesh.default_memory_space: Mesh#default_memory_space().
  BlockMapping: BlockMapping#
  MemoryRef.get_ref_aval: MemoryRef#get_ref_aval().
  BlockMapping.block_aval: BlockMapping#block_aval().
  MemoryRef.memory_space: MemoryRef#memory_space.
  MemorySpace.DEFAULT: MemorySpace#DEFAULT.
  undo_transforms: undo_transforms().
  CostEstimate.flops: CostEstimate#flops.
  CostEstimate.transcendentals: CostEstimate#transcendentals.
  Mesh.discharges_effect: Mesh#discharges_effect().
  _pallas_tracing_env: _pallas_tracing_env.
  BlockMapping.transformed_block_aval: BlockMapping#transformed_block_aval.
  GridMapping.index_map_avals: GridMapping#index_map_avals.
  CostEstimate.bytes_accessed: CostEstimate#bytes_accessed.
  core_map: core_map().
  SEMAPHORE_INTERPRET_DTYPE: SEMAPHORE_INTERPRET_DTYPE.
  squeezed: squeezed.
  BlockMapping.index_map: BlockMapping#index_map().
  Mesh.supported_memory_spaces: Mesh#supported_memory_spaces().
  Mesh.tracing_context: Mesh#tracing_context().
  AbstractSemaphoreTy: AbstractSemaphoreTy#
  dynamic_shapes_export_enabled: dynamic_shapes_export_enabled().
  GridMapping.grid_names: GridMapping#grid_names.
  get_interpret_effects: get_interpret_effects().
  grid_env: grid_env().
  GridMapping.out_shapes: GridMapping#out_shapes().
  GridSpec: GridSpec#
  GridSpec.grid: GridSpec#grid.
  semaphore: semaphore#
  ScratchShapeTree: ScratchShapeTree.
  _get_block_shape: _get_block_shape().
  _core_map_typecheck_rule: _core_map_typecheck_rule().
  lower_as_mlir: lower_as_mlir().
  GridMappingGrid: GridMappingGrid.
  _get_ref_block_shape: _get_ref_block_shape().
  CompilerParams: CompilerParams#
  CoreMemorySpace.memory_space: CoreMemorySpace#memory_space.
  BoundedSlice: BoundedSlice#
  GridMapping.replace: GridMapping#replace().
  GridMapping.slice_block_ops: GridMapping#slice_block_ops().
  CostEstimate.to_json: CostEstimate#to_json().
  with_memory_space_constraint_abstract_eval: with_memory_space_constraint_abstract_eval().
  TupleGrid: TupleGrid.
  MemoryRef.inner_aval: MemoryRef#inner_aval.
  axis_frame: axis_frame().
  current_grid_env: current_grid_env().
  BlockSpec.pipeline_mode: BlockSpec#pipeline_mode.
  BlockMapping.replace: BlockMapping#replace().
  GridMapping.index_map_tree: GridMapping#index_map_tree.
  unzip_dynamic_grid_bounds: unzip_dynamic_grid_bounds().
  pytreedef_mismatch_err_msg: pytreedef_mismatch_err_msg().
  _canonicalize_block_shape: _canonicalize_block_shape().
  dynamic_grid_dim: dynamic_grid_dim.
  MemorySpace.from_type: MemorySpace#from_type().
  BlockMapping.origin: BlockMapping#origin.
  BlockMapping.pipeline_mode: BlockMapping#pipeline_mode.
  BlockMapping._get_start_index: BlockMapping#_get_start_index().
  GridMapping.scratch_avals: GridMapping#scratch_avals.
  SEMAPHORE_MAX_VALUE: SEMAPHORE_MAX_VALUE.
  BarrierSemaphore: BarrierSemaphore#
  NoBlockSpec: NoBlockSpec#
  BlockSpecTree: BlockSpecTree.
  CommsEffect: CommsEffect#
  debug_checks_enabled: debug_checks_enabled().
  MemoryRefBase.get_ref_aval: MemoryRefBase#get_ref_aval().
  MemoryRef.__lt__: MemoryRef#__lt__().
  CoreMemorySpace.__post_init__: CoreMemorySpace#__post_init__().
  CoreMemorySpace.__str__: CoreMemorySpace#__str__().
  CoreMemorySpace.__repr__: CoreMemorySpace#__repr__().
  CoreMemorySpace.name: CoreMemorySpace#name().
  PallasTracingEnv.grid_context: PallasTracingEnv#grid_context.
  BlockMapping.index_map_out_tree: BlockMapping#index_map_out_tree.
  BlockMapping.transforms: BlockMapping#transforms.
  BlockMapping.__post_init__: BlockMapping#__post_init__().
  is_dynamic_dim: is_dynamic_dim().
  GridMapping.static_grid: GridMapping#static_grid().
  GridMapping.slice_scratch_ops: GridMapping#slice_scratch_ops().
  comms_effect: comms_effect.
  with_memory_space_constraint_p: with_memory_space_constraint_p.
  Grid: Grid.
  PallasGridContext.size: PallasGridContext#size().
  _IndexMapFunc.index_map: _IndexMapFunc#index_map.
  pallas_export_experimental: pallas_export_experimental().
  split_list: split_list.
  barrier_semaphore: barrier_semaphore#
  PallasTracingEnv.grid_env_stack: PallasTracingEnv#grid_env_stack.
  GridMapping.num_scratch_operands: GridMapping#num_scratch_operands().
  GridMapping.slice_index_ops: GridMapping#slice_index_ops().
  GridSpec.scratch_shapes: GridSpec#scratch_shapes.
  _out_shape_to_aval_mapping._out_shape_to_aval_mapping: _out_shape_to_aval_mapping._out_shape_to_aval_mapping.
  DynamicGridDim: DynamicGridDim#
  OriginStr: OriginStr.
  Buffered: Buffered#
  Element.padding: Element#padding.
  Blocked.block_size: Blocked#block_size.
  Indirect: Indirect#
  _is_valid_grid_dim: _is_valid_grid_dim().
  ScratchShape.get_ref_aval: ScratchShape#get_ref_aval().
  default_mesh_discharge_rule.body: default_mesh_discharge_rule().body().
  _core_map_mesh_rules._core_map_mesh_rules: _core_map_mesh_rules._core_map_mesh_rules.
  register_lowering_rule: register_lowering_rule().
  Buffered.revisit: Buffered#revisit.
  MemoryRef.dtype: MemoryRef#dtype().
  MemoryRef.shape: MemoryRef#shape().
  GridEnv: GridEnv.
  GridSpec.in_specs: GridSpec#in_specs.
  GridSpec.out_specs: GridSpec#out_specs.
  AbstractSemaphoreTyRules.pallas_interpret_element_aval: AbstractSemaphoreTyRules#pallas_interpret_element_aval().
  AbstractSemaphoreTyRules.physical_element_aval: AbstractSemaphoreTyRules#physical_element_aval().
  MemoryRefBase.get_array_aval: MemoryRefBase#get_array_aval().
  MemorySpace.__call__: MemorySpace#__call__().
  CoreMemorySpace.__call__: CoreMemorySpace#__call__().
  CoreMemorySpace.memory_kind: CoreMemorySpace#memory_kind().
  Element.__str__: Element#__str__().
  _IndexMapFunc.__eq__: _IndexMapFunc#__eq__().
  BlockSpec.__post_init__: BlockSpec#__post_init__().
  GridMapping.local_grid_env: GridMapping#local_grid_env.
  with_memory_space_constraint_impl: with_memory_space_constraint_impl().
  Semaphore: Semaphore#
  poison_buffers_enabled: poison_buffers_enabled().
  get_lowering_rule: get_lowering_rule().
  PallasGridContext.grid: PallasGridContext#grid.
  PallasTracingEnv.dynamic_shapes: PallasTracingEnv#dynamic_shapes.
  GridAxis: GridAxis#
  Element.block_size: Element#block_size.
  BlockSpec.replace: BlockSpec#replace.
  BlockMapping.debug: BlockMapping#debug.
  GridElement: GridElement.
  NamedGrid: NamedGrid.
  enable_debug_checks: enable_debug_checks.
  enable_poison_buffers: enable_poison_buffers.
  MemoryRefBase: MemoryRefBase#
  GridSpec._make_scalar_ref_aval: GridSpec#_make_scalar_ref_aval().
  RevisitMode: RevisitMode#
  MemorySpace.KEY: MemorySpace#KEY.
  PallasGridContext: PallasGridContext#
  GridMapping.get_grid_indices: GridMapping#get_grid_indices.
  GridMapping.debug: GridMapping#debug.
  ScratchShape: ScratchShape#
  AbstractSemaphoreTy._rules: AbstractSemaphoreTy#_rules.
  AbstractSemaphoreTy.__repr__: AbstractSemaphoreTy#__repr__().
  Semaphore.type: Semaphore#type.
  GridAxis.index: GridAxis#index.
  Blocked.__str__: Blocked#__str__().
  BoundedSlice.__repr__: BoundedSlice#__repr__().
  Indirect.__repr__: Indirect#__repr__().
  _IndexMapFunc.__init__: _IndexMapFunc#__init__().
  _IndexMapFunc.__hash__: _IndexMapFunc#__hash__().
  _IndexMapFunc.__call__: _IndexMapFunc#__call__().
  _IndexMapFunc.__repr__: _IndexMapFunc#__repr__().
  _IndexMapFunc.__str__: _IndexMapFunc#__str__().
  BlockMapping.__str__: BlockMapping#__str__().
  GridMapping.__str__: GridMapping#__str__().
  ScratchShape.get_array_aval: ScratchShape#get_array_aval().
  unsafe_map: unsafe_map.
  unsafe_zip: unsafe_zip.
  partial: partial.
  StaticGrid: StaticGrid.
  _backend_lowering_rules: _backend_lowering_rules.
  default_index_map: default_index_map().
  _IndexMapFunc: _IndexMapFunc#
  GridSpec.grid_names: GridSpec#grid_names.
  GridName: GridName.
  AbstractSemaphoreTyRules: AbstractSemaphoreTyRules#
  AbstractSemaphoreTy.name: AbstractSemaphoreTy#name.
  debug_check: debug_check().
  RevisitMode.ANY: RevisitMode#ANY.
  Buffered.buffer_count: Buffered#buffer_count.
  Buffered.use_lookahead: Buffered#use_lookahead.
  MemorySpace.ERROR: MemorySpace#ERROR.
  MemorySpace.INDEX: MemorySpace#INDEX.
  PallasTracingEnv: PallasTracingEnv#
  BoundedSlice.block_size: BoundedSlice#block_size.
  Indirect.block_size: Indirect#block_size.
  BlockMapping.allow_captured_consts: BlockMapping#allow_captured_consts.
  CostEstimate.remote_bytes_transferred: CostEstimate#remote_bytes_transferred.
  _core_map_is_high: _core_map_is_high().
  core_map_lowering_rule: core_map_lowering_rule().
  CoreType: CoreType.
  with_memory_space_constraint_lowering_rule: with_memory_space_constraint_lowering_rule().
  _core_map_partial_eval_custom: _core_map_partial_eval_custom().
  DynamicGridDim.__repr__: DynamicGridDim#__repr__().
  GridNames: GridNames.
  AbstractSemaphoreTy.__eq__: AbstractSemaphoreTy#__eq__().
  AbstractSemaphoreTy.__hash__: AbstractSemaphoreTy#__hash__().
  Semaphore.name: Semaphore#name.
  BarrierSemaphore.name: BarrierSemaphore#name.
  BarrierSemaphore.type: BarrierSemaphore#type.
  CompilerParams.__dataclass_fields__: CompilerParams#__dataclass_fields__.
  RevisitMode.IMMEDIATE: RevisitMode#IMMEDIATE.
  Buffered.prefetched_count: Buffered#prefetched_count.
  MemorySpace.memory_kind: MemorySpace#memory_kind().
  MemorySpace.__str__: MemorySpace#__str__().
  PallasGridContext.mapped_dims: PallasGridContext#mapped_dims.
  PallasTracingEnv.is_interpret_mode: PallasTracingEnv#is_interpret_mode.
  PallasTracingEnv.module_export_fn: PallasTracingEnv#module_export_fn.
  GridAxis.size: GridAxis#size.
  NoBlockSpec.__repr__: NoBlockSpec#__repr__().
  CostEstimate.__post_init__: CostEstimate#__post_init__().
---
# Module: [`jax/_src/pallas/core.py`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py)

## Classes
### `AbstractSemaphoreTy`  ·  implements/extends ExtendedDType
- def: [`jax/_src/pallas/core.py:102`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L102)
- signature: `class AbstractSemaphoreTy(dtypes.ExtendedDType):`
- members:
  - `name` — [`L103`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L103)
- protocol/private: `__eq__`[`L109`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L109), `__hash__`[`L112`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L112), `__repr__`[`L106`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L106), `_rules`[`L104`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L104)
- uses (calls/refs, reference-scoped): [`ExtendedDType`](../dtypes.md#ExtendedDType), [`BarrierSemaphore`](core.md#BarrierSemaphore), [`Semaphore`](core.md#Semaphore), [`DMASemaphore`](mosaic/core.md#DMASemaphore), [`AbstractSemaphoreTyRules`](core.md#AbstractSemaphoreTyRules)
- used by: [`ExtendedDType`](../dtypes.md#ExtendedDType), [`BarrierSemaphore`](core.md#BarrierSemaphore), [`Semaphore`](core.md#Semaphore), [`DMASemaphore`](mosaic/core.md#DMASemaphore)

### `AbstractSemaphoreTyRules`
- def: [`jax/_src/pallas/core.py:92`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L92)
- signature: `class AbstractSemaphoreTyRules:`
- members:
  - `pallas_interpret_element_aval(_)` — [`L94`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L94)
  - `physical_element_aval(_)` — [`L98`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L98)
- uses (calls/refs, reference-scoped): [`ShapedArray`](../core.md#ShapedArray), [`int32`](../numpy/scalar_types.md#int32), [`SEMAPHORE_INTERPRET_DTYPE`](core.md#SEMAPHORE_INTERPRET_DTYPE)
- used by: [`_rules`](core.md#AbstractSemaphoreTy._rules)

### `BarrierSemaphore`  ·  implements/extends AbstractSemaphoreTy
- def: [`jax/_src/pallas/core.py:138`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L138)
- signature: `class BarrierSemaphore(AbstractSemaphoreTy):`
- members:
  - `name` — [`L139`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L139)
  - `type` — [`L140`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L140)
- uses (calls/refs, reference-scoped): [`AbstractSemaphoreTy`](core.md#AbstractSemaphoreTy), [`barrier_semaphore`](core.md#barrier_semaphore)
- used by: [`__call__`](mosaic_gpu/core.md#SemaphoreType.__call__), [`dtype`](mosaic/core.md#SemaphoreType.dtype), [`AbstractSemaphoreTy`](core.md#AbstractSemaphoreTy), [`_get_barrier_semaphore_abstract_eval`](mosaic/primitives.md#_get_barrier_semaphore_abstract_eval)

### `BlockMapping`
- def: [`jax/_src/pallas/core.py:755`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L755)
- members:
  - `block_aval(self)` — [`L800`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L800)
  - `check_invariants(self)` — [`L778`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L778)
  - `compute_start_indices_interpret(self, loop_idx, *args)` — [`L820`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L820)
  - `has_trivial_window(self)` — [`L839`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L839) — If block shape is same as the array shape and index_map returns 0s.
  - `index_map(*args)` — [`L850`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L850)
  - `ref_aval(self)` — [`L807`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L807) — Returns the abstract value of the Ref after transformations.
  - `replace(self, **kwargs)` — [`L792`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L792)
  - `to_block_spec(self)` — [`L849`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L849)
  - `to_lojax(self, index_map_avals, index_map_tree, grid, vmapped_dims)` — [`L861`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L861)
  - `allow_captured_consts` — [`L771`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L771)
  - `array_aval` — [`L766`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L766)
  - `block_shape` — [`L762`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L762) — documented in [jax-_src-pallas-core](../../../../concepts/jax-_src-pallas-core.md)
  - `debug` — [`L770`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L770)
  - `index_map_jaxpr` — [`L764`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L764)
  - `index_map_out_tree` — [`L765`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L765)
  - `origin` — [`L767`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L767)
  - `pipeline_mode` — [`L769`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L769)
  - `transformed_block_aval` — [`L763`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L763)
  - `transforms` — [`L768`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L768)
- protocol/private: `__post_init__`[`L773`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L773), `__repr__`[`L891`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L891), `__str__`[`L906`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L906), `_get_start_index`[`L827`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L827)
- uses (calls/refs, reference-scoped): [`AbstractRef`](../state/types.md#AbstractRef), [`ShapedArray`](../core.md#ShapedArray), [`shape`](../core.md#ShapedArray.shape), [`tree_unflatten`](../tree_util.md#tree_unflatten), [`BlockSpec`](core.md#BlockSpec), [`jaxpr`](../core.md#ClosedJaxpr.jaxpr), [`ClosedJaxpr`](../core.md#ClosedJaxpr), [`TransformedRef`](../state/types.md#TransformedRef), [`Transform`](../state/types.md#Transform), [`consts`](../core.md#ClosedJaxpr.consts), [`PyTreeDef`](../tree_util.md#PyTreeDef), [`tree_leaves`](../tree_util.md#tree_leaves), [`Squeezed`](core.md#Squeezed), [`shape`](../state/types.md#AbstractRef.shape), [`out_avals`](../core.md#ClosedJaxpr.out_avals), [`jaxpr_as_fun`](../core.md#jaxpr_as_fun), [`update`](../core.md#ShapedArray.update), [`Blocked`](core.md#Blocked), [`memory_space`](core.md#BlockSpec.memory_space), [`BlockDim`](core.md#BlockDim.BlockDim), [`Element`](core.md#Element), [`discharge_state`](../state/discharge.md#discharge_state), [`val`](../core.md#Literal.val), [`_convert_block_spec_to_block_mapping`](core.md#_convert_block_spec_to_block_mapping), [`int32`](../numpy/scalar_types.md#int32), [`enable_checks`](../config.md#enable_checks), [`_get_block_dim_size`](core.md#_get_block_dim_size), [`undo_transforms`](core.md#undo_transforms), [`is_high`](../core.md#AbstractValue.is_high), [`_get_ref_block_shape`](core.md#_get_ref_block_shape), [`int64`](../numpy/scalar_types.md#int64), [`pipeline_mode`](core.md#BlockSpec.pipeline_mode), [`split_list`](core.md#split_list), [`Buffered`](core.md#Buffered), [`OriginStr`](core.md#OriginStr), [`lo_ty`](../core.md#ShapedArray.lo_ty)
- used by: [`interpret_pallas_call`](mosaic/interpret/interpret_pallas_call.md#interpret_pallas_call), [`lower_jaxpr_into_pipelined_module`](mosaic/lowering.md#lower_jaxpr_into_pipelined_module), [`to_block_mapping`](core.md#BlockSpec.to_block_mapping), [`get_grid_mapping`](core.md#get_grid_mapping), [`interpret_pallas_call`](mosaic_gpu/interpret/interpret_pallas_call.md#interpret_pallas_call), [`lower_pipelined_jaxpr_to_module`](mosaic_gpu/lowering.md#lower_pipelined_jaxpr_to_module), [`pallas_call_hlo_interpret`](hlo_interpreter.md#pallas_call_hlo_interpret), [`_check_block_mappings`](mosaic/lowering.md#_check_block_mappings), [`pallas_call_lowering`](triton/pallas_call_registration.md#pallas_call_lowering), [`_dimension_semantics`](mosaic/lowering.md#MosaicGridMapping._dimension_semantics), [`lower_jaxpr_to_triton_module`](triton/lowering.md#lower_jaxpr_to_triton_module), [`wrapped`](mosaic/pipeline.md#emit_pipeline.wrapped), [`_pallas_call_to_lojax`](pallas_call.md#_pallas_call_to_lojax), [`_allocate_buffers_for_outputs`](mosaic_gpu/interpret/interpret_pallas_call.md#_allocate_buffers_for_outputs), [`_batch_with_explicit_loop`](pallas_call.md#_batch_with_explicit_loop), [`_batch_block_mapping`](pallas_call.md#_batch_block_mapping), [`body`](hlo_interpreter.md#pallas_call_hlo_interpret.body), [`_eval_index_map`](triton/lowering.md#_eval_index_map), [`block_mappings`](core.md#GridMapping.block_mappings), [`check_invariants`](core.md#GridMapping.check_invariants), [`to_block_mapping`](mosaic_gpu/core.md#BlockSpec.to_block_mapping), [`_block_spec_from_block_mapping`](mosaic_gpu/lowering.md#_block_spec_from_block_mapping), [`to_lojax`](core.md#GridMapping.to_lojax), [`_eval_index_map`](mosaic_gpu/lowering.md#_eval_index_map), [`_check_block_mappings`](mosaic_gpu/lowering.md#_check_block_mappings), [`_convert_block_spec_to_block_mapping`](core.md#_convert_block_spec_to_block_mapping), [`err_details`](mosaic_gpu/lowering.md#_check_block_mappings.err_details), [`in_shapes`](core.md#GridMapping.in_shapes), [`block_mappings_output`](core.md#GridMapping.block_mappings_output), [`err_details`](mosaic/lowering.md#_check_block_mappings.err_details), [`_block_map_function`](pallas_call.md#_batch_block_mapping._block_map_function), [`_initialize_output_vals`](hlo_interpreter.md#_initialize_output_vals), [`has_communication`](mosaic/lowering.md#MosaicGridMapping.has_communication), [`out_shapes`](core.md#GridMapping.out_shapes), [`_get_index_alignment`](triton/lowering.md#_get_index_alignment), [`block_mappings`](mosaic/lowering.md#MosaicGridMapping.block_mappings), [`BlockMapping`](pallas_call.md#BlockMapping), [`BlockMapping`](triton/lowering.md#BlockMapping), [`BlockMapping`](hlo_interpreter.md#BlockMapping)

### `BlockSpec`
- def: [`jax/_src/pallas/core.py:559`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L559) — documented in [jax-_src-pallas-core](../../../../concepts/jax-_src-pallas-core.md)
- doc: Specifies how an array should be sliced for each invocation of a kernel.
- signature: `class BlockSpec:`
- members:
  - `to_block_mapping(self, origin: OriginStr, array_aval: jax_core.ShapedArray, *, index_map_avals: Sequence[jax_core.AbstractValue], index_map_tree: tree_util.PyTreeDef, grid: GridMappingGrid, vmapped_dims: tuple[int, ...], debug: bool = False, allow_captured_consts: bool = False)` — [`L587`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L587) — documented in [jax-_src-pallas-core](../../../../concepts/jax-_src-pallas-core.md)
  - `block_shape` — [`L574`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L574) — documented in [jax-_src-pallas-core](../../../../concepts/jax-_src-pallas-core.md)
  - `index_map` — [`L575`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L575)
  - `memory_space` — [`L576`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L576)
  - `pipeline_mode` — [`L577`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L577)
  - `replace` — [`L723`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L723)
- protocol/private: `__post_init__`[`L579`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L579)
- uses (calls/refs, reference-scoped): [`AbstractRef`](../state/types.md#AbstractRef), [`ShapedArray`](../core.md#ShapedArray), [`AbstractValue`](../core.md#AbstractValue), [`shape`](../core.md#ShapedArray.shape), [`consts`](../core.md#ClosedJaxpr.consts), [`debug_info`](../api_util.md#debug_info), [`Slice`](../indexing.md#Slice), [`PyTreeDef`](../tree_util.md#PyTreeDef), [`MemorySpace`](../core.md#MemorySpace), [`Squeezed`](core.md#Squeezed), [`shape`](../state/types.md#AbstractRef.shape), [`update`](../core.md#ShapedArray.update), [`MemorySpace`](core.md#MemorySpace), [`block_shape`](core.md#BlockMapping.block_shape), [`array_aval`](core.md#BlockMapping.array_aval), [`Blocked`](core.md#Blocked), [`to_block_mapping`](mosaic_gpu/core.md#BlockSpec.to_block_mapping), [`BlockDim`](core.md#BlockDim.BlockDim), [`Element`](core.md#Element), [`check_invariants`](core.md#BlockMapping.check_invariants), [`index_map_jaxpr`](core.md#BlockMapping.index_map_jaxpr), [`BlockSpec`](mosaic_gpu/core.md#BlockSpec), [`tracing_grid_env`](core.md#tracing_grid_env), [`int32`](../numpy/scalar_types.md#int32), [`is_constant_shape`](../core.md#is_constant_shape), [`BlockMapping`](core.md#BlockMapping), [`Device`](../core.md#MemorySpace.Device), [`DEFAULT`](core.md#MemorySpace.DEFAULT), [`transformed_block_aval`](core.md#BlockMapping.transformed_block_aval), [`dynamic_shapes_export_enabled`](core.md#dynamic_shapes_export_enabled), [`AbstractLinVal`](../state/types.md#AbstractLinVal), [`GridMappingGrid`](core.md#GridMappingGrid), [`_get_ref_block_shape`](core.md#_get_ref_block_shape), [`int64`](../numpy/scalar_types.md#int64), [`BoundedSlice`](core.md#BoundedSlice), [`_canonicalize_block_shape`](core.md#_canonicalize_block_shape), [`origin`](core.md#BlockMapping.origin), [`pipeline_mode`](core.md#BlockMapping.pipeline_mode), [`index_map_out_tree`](core.md#BlockMapping.index_map_out_tree), [`Buffered`](core.md#Buffered)  (+6 more)
- used by: [`get_grid_mapping`](core.md#get_grid_mapping), [`_splash_attention_bwd_dkv`](../../experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.md#_splash_attention_bwd_dkv), [`_splash_attention_bwd_dq`](../../experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.md#_splash_attention_bwd_dq), [`_pallas_call_state_discharge_rule`](pallas_call.md#_pallas_call_state_discharge_rule), [`gmm`](../../experimental/pallas/ops/tpu/megablox/gmm.md#gmm), [`tgmm`](../../experimental/pallas/ops/tpu/megablox/gmm.md#tgmm), [`paged_attention`](../../experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.md#paged_attention), [`_flash_attention_bwd_dkv`](../../experimental/pallas/ops/tpu/flash_attention.md#_flash_attention_bwd_dkv), [`_flash_attention_bwd_dq`](../../experimental/pallas/ops/tpu/flash_attention.md#_flash_attention_bwd_dq), [`ragged_paged_attention`](../../experimental/pallas/ops/tpu/ragged_paged_attention/kernel.md#ragged_paged_attention), [`create`](mosaic/pipeline.md#BufferedRef.create), [`_push_block_spec_jaxpr`](fuser/block_spec.md#_push_block_spec_jaxpr), [`_flash_attention_impl`](../../experimental/pallas/ops/tpu/flash_attention.md#_flash_attention_impl), [`_mha_backward`](../../experimental/pallas/ops/gpu/attention.md#_mha_backward), [`scoped_pipeline`](mosaic_gpu/pipeline.md#emit_pipeline.scoped_pipeline), [`_get_eval_rule`](fuser/block_spec.md#_get_eval_rule), [`compute_block`](mosaic_gpu/pipeline.md#emit_pipeline_warp_specialized.scoped_pipeline.compute_block), [`loop_body`](mosaic_gpu/pipeline.md#emit_pipeline.scoped_pipeline.loop_body), [`matmul`](../../experimental/pallas/ops/tpu/matmul.md#matmul), [`compute_loop_body`](mosaic_gpu/pipeline.md#emit_pipeline_warp_specialized.scoped_pipeline.compute_block.compute_loop_body), [`mha`](../../experimental/pallas/ops/gpu/attention.md#mha), [`paged_attention_unbatched`](../../experimental/pallas/ops/gpu/paged_attention.md#paged_attention_unbatched), [`_concatenate_eval_rule`](fuser/block_spec.md#_concatenate_eval_rule), [`_swap_eval_rule`](fuser/block_spec.md#_swap_eval_rule), [`philox_4x32_kernel`](../../experimental/pallas/ops/tpu/random/philox.md#philox_4x32_kernel), [`_reshape_push_rule`](fuser/block_spec.md#_reshape_push_rule), [`to_block_mapping`](mosaic_gpu/core.md#BlockSpec.to_block_mapping), [`_apply_block_transform`](fuser/block_spec.md#_apply_block_transform), [`_block_spec_from_block_mapping`](mosaic_gpu/lowering.md#_block_spec_from_block_mapping), [`decode_attn_unbatched`](../../experimental/pallas/ops/gpu/decode_attention.md#decode_attn_unbatched), [`_init_block_transforms`](fuser/block_spec.md#_init_block_transforms), [`emit_pipeline`](mosaic_gpu/pipeline.md#emit_pipeline), [`kernel_dq`](../../experimental/pallas/ops/gpu/attention_mgpu.md#_attention_bwd.kernel_dq), [`compute_gmem_slice`](mosaic_gpu/pipeline.md#BufferedRef.compute_gmem_slice), [`get_pipeline`](../../experimental/pallas/ops/gpu/hopper_matmul_mgpu.md#kernel.get_pipeline), [`_mpmd_map_fallback_lowering`](mpmd.md#_mpmd_map_fallback_lowering), [`_preprocess_backward`](../../experimental/pallas/ops/gpu/attention.md#_preprocess_backward), [`BlockSpec`](mosaic_gpu/core.md#BlockSpec), [`_broadcast_in_dim_push_rule`](fuser/block_spec.md#_broadcast_in_dim_push_rule), [`_reduce_sum_push_rule`](fuser/block_spec.md#_reduce_sum_push_rule)  (+69 more)

### `Blocked`
- def: [`jax/_src/pallas/core.py:418`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L418)
- doc: The default BlockShape type.
- signature: `class Blocked:`
- members:
  - `block_size` — [`L420`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L420)
- protocol/private: `__str__`[`L422`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L422)
- used by: [`lower_jaxpr_into_pipelined_module`](mosaic/lowering.md#lower_jaxpr_into_pipelined_module), [`to_block_mapping`](core.md#BlockSpec.to_block_mapping), [`_check_block_mappings`](mosaic/lowering.md#_check_block_mappings), [`_get_pull_rule`](fuser/block_spec.md#_get_pull_rule), [`_reshape_pull_rule`](fuser/block_spec.md#_reshape_pull_rule), [`_make_block_slice`](mosaic/pipeline.md#_make_block_slice), [`_reshape_push_rule`](fuser/block_spec.md#_reshape_push_rule), [`BlockDim`](core.md#BlockDim.BlockDim), [`_offset_indexer`](fuser/block_spec.md#_offset_indexer), [`get_block_size`](core.md#get_block_size), [`_eval_index_map`](mosaic_gpu/lowering.md#_eval_index_map), [`_reduce_sum_pull_rule`](fuser/block_spec.md#_reduce_sum_pull_rule), [`_unstack_push_rule`](fuser/block_spec.md#_unstack_push_rule), [`_block_size`](fuser/block_spec.md#_block_size), [`_concatenate_push_rule`](fuser/block_spec.md#_concatenate_push_rule), [`_unstack_pull_rule`](fuser/block_spec.md#_unstack_pull_rule), [`_canonicalize_block_dim`](core.md#_canonicalize_block_dim), [`_get_block_dim_size`](core.md#_get_block_dim_size), [`_get_start_index`](triton/lowering.md#_eval_index_map._get_start_index), [`_make_block_slice`](mosaic_gpu/pipeline.md#BufferedRef._make_block_slice), [`_get_bdim_alignment`](triton/lowering.md#_get_index_alignment._get_bdim_alignment), [`_get_block_size`](mosaic_gpu/pipeline.md#_get_block_size), [`_get_start_index`](mosaic/interpret/interpret_pallas_call.md#_compute_start_indices._get_start_index), [`_block_dim_equal`](fuser/block_spec.md#_block_dim_equal), [`_get_start_index`](core.md#BlockMapping._get_start_index), [`_slice`](fuser/block_spec.md#_get_eval_rule._slice), [`Blocked`](mosaic/pipeline.md#Blocked), [`Blocked`](triton/lowering.md#Blocked)

### `BoundedSlice`
- def: [`jax/_src/pallas/core.py:426`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L426)
- doc: Allows to specify a bounded slice of a dimension.
- signature: `class BoundedSlice:`
- members:
  - `block_size` — [`L433`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L433)
- protocol/private: `__repr__`[`L435`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L435)
- used by: [`to_block_mapping`](core.md#BlockSpec.to_block_mapping), [`BlockDim`](core.md#BlockDim.BlockDim), [`_offset_indexer`](fuser/block_spec.md#_offset_indexer), [`get_block_size`](core.md#get_block_size), [`_slice_rule`](fuser/block_spec.md#_slice_rule), [`_block_size`](fuser/block_spec.md#_block_size), [`_canonicalize_block_dim`](core.md#_canonicalize_block_dim), [`_get_block_dim_size`](core.md#_get_block_dim_size), [`BoundedSlice`](mosaic/pipeline.md#BoundedSlice)

### `Buffered`
- def: [`jax/_src/pallas/core.py:213`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L213)
- doc: Specifies how a block should be buffered for a pipeline.
- signature: `class Buffered:`
- members:
  - `buffer_count` — [`L234`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L234)
  - `prefetched_count` — [`L237`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L237)
  - `revisit` — [`L236`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L236)
  - `use_lookahead` — [`L235`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L235)
- uses (calls/refs, reference-scoped): [`RevisitMode`](core.md#RevisitMode)
- used by: [`lower_jaxpr_into_pipelined_module`](mosaic/lowering.md#lower_jaxpr_into_pipelined_module), [`pipeline_mode`](core.md#BlockSpec.pipeline_mode), [`pipeline_mode`](core.md#BlockMapping.pipeline_mode), [`pipeline_mode`](fuser/block_spec.md#BlockIndexTransform.pipeline_mode)

### `CommsEffect`  ·  implements/extends Effect
- def: [`jax/_src/pallas/core.py:1605`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1605)
- signature: `class CommsEffect(effects.Effect):`
- used by: [`_core_map_abstract_eval`](core.md#_core_map_abstract_eval), [`kernel_local_effects`](core.md#kernel_local_effects.kernel_local_effects), [`_get_nonlocal_axis_names`](mosaic/lowering.md#MosaicGridMapping._get_nonlocal_axis_names), [`comms_effect`](core.md#comms_effect)

### `CompilerParams`  ·  implements/extends Protocol
- def: [`jax/_src/pallas/core.py:143`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L143)
- doc: Base class for compiler parameters.
- signature: `class CompilerParams(Protocol):`
- protocol/private: `__dataclass_fields__`[`L146`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L146)
- used by: [`interpret_pallas_call`](mosaic/interpret/interpret_pallas_call.md#interpret_pallas_call), [`pallas_call_lowering`](triton/pallas_call_registration.md#pallas_call_lowering), [`pallas_call_lowering`](mosaic_gpu/pallas_call_registration.md#pallas_call_lowering), [`pallas_call`](pallas_call.md#pallas_call), [`pallas_call_tpu_lowering_rule`](mosaic/pallas_call_registration.md#pallas_call_tpu_lowering_rule), [`kernel`](mosaic_gpu/core.md#kernel), [`gpu_lowering`](pallas_call.md#_pallas_call_lowering.gpu_lowering), [`kernel`](helpers.md#kernel), [`CompilerParams`](pallas_call.md#CompilerParams)

### `CoreMemorySpace`
- def: [`jax/_src/pallas/core.py:310`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L310)
- doc: A memory space tied to a Pallas mesh.
- signature: `class CoreMemorySpace:`
- members:
  - `memory_kind(self)` — [`L337`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L337)
  - `name(self)` — [`L333`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L333)
  - `memory_space` — [`L313`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L313)
  - `mesh` — [`L314`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L314)
- protocol/private: `__call__`[`L323`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L323), `__post_init__`[`L316`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L316), `__repr__`[`L329`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L329), `__str__`[`L326`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L326)
- uses (calls/refs, reference-scoped): [`ShapedArray`](../core.md#ShapedArray), [`Mesh`](core.md#Mesh), [`core_type`](core.md#Mesh.core_type), [`MemoryRef`](core.md#MemoryRef), [`supported_memory_spaces`](core.md#Mesh.supported_memory_spaces), [`mem_space_to_kind`](../core.md#mem_space_to_kind)
- used by: [`wrapper`](mpmd.md#_mpmd_map.wrapper), [`_get_memory_space_from_aval`](mosaic/pallas_call_registration.md#_get_memory_space_from_aval), [`memory_space_to_tpu_memory_space`](mosaic/core.md#memory_space_to_tpu_memory_space), [`_extract_indirect_offsets_from_indices`](mosaic/sc_lowering.md#_extract_indirect_offsets_from_indices), [`_dma_start_lowering_rule`](mosaic/sc_lowering.md#_dma_start_lowering_rule), [`_dma_start_lowering_rule`](mosaic/lowering.md#_dma_start_lowering_rule), [`_dma_wait_lowering_rule`](mosaic/sc_lowering.md#_dma_wait_lowering_rule), [`_semaphore_signal_lowering_rule`](mosaic/lowering.md#_semaphore_signal_lowering_rule), [`_scatter_lowering_rule`](mosaic/sc_primitives.md#_scatter_lowering_rule), [`_memory_space_to_mosaic_attribute`](mosaic/lowering.md#_memory_space_to_mosaic_attribute), [`_dma_wait_lowering_rule`](mosaic/lowering.md#_dma_wait_lowering_rule), [`TPUMemorySpace`](mosaic/lowering.md#TPUMemorySpace), [`__matmul__`](mosaic/core.md#SemaphoreType.__matmul__), [`AnyMemorySpace`](mosaic/lowering.md#AnyMemorySpace), [`__matmul__`](mosaic/core.md#MemorySpace.__matmul__), [`CoreMemorySpace`](mosaic/sc_lowering.md#CoreMemorySpace)

### `CostEstimate`
- def: [`jax/_src/pallas/core.py:1443`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1443)
- signature: `class CostEstimate:`
- members:
  - `to_json(self)` — [`L1457`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1457)
  - `bytes_accessed` — [`L1446`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1446)
  - `flops` — [`L1444`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1444)
  - `remote_bytes_transferred` — [`L1447`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1447)
  - `transcendentals` — [`L1445`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1445)
- protocol/private: `__post_init__`[`L1449`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1449)
- used by: [`interpret_pallas_call`](mosaic/interpret/interpret_pallas_call.md#interpret_pallas_call), [`_pallas_call_batching_rule`](pallas_call.md#_pallas_call_batching_rule), [`interpret_pallas_call`](mosaic_gpu/interpret/interpret_pallas_call.md#interpret_pallas_call), [`lower_pipelined_jaxpr_to_module`](mosaic_gpu/lowering.md#lower_pipelined_jaxpr_to_module), [`_lower_to_custom_call`](mosaic/pallas_call_registration.md#_lower_to_custom_call), [`pallas_call_lowering`](triton/pallas_call_registration.md#pallas_call_lowering), [`gmm`](../../experimental/pallas/ops/tpu/megablox/gmm.md#gmm), [`tgmm`](../../experimental/pallas/ops/tpu/megablox/gmm.md#tgmm), [`_pallas_call_jvp_rule`](pallas_call.md#_pallas_call_jvp_rule), [`pallas_call_lowering`](mosaic_gpu/pallas_call_registration.md#pallas_call_lowering), [`pallas_call_tpu_lowering_rule`](mosaic/pallas_call_registration.md#pallas_call_tpu_lowering_rule), [`cost_estimate_jaxpr`](cost_estimate.md#cost_estimate_jaxpr), [`_tensorcore_mesh_discharge_rule`](mosaic/core.md#_tensorcore_mesh_discharge_rule), [`_mpmd_map`](mpmd.md#_mpmd_map), [`_custom_vjp_rule`](cost_estimate.md#_custom_vjp_rule), [`_fwd_cost_estimate`](../../experimental/pallas/ops/tpu/flash_attention.md#_fwd_cost_estimate), [`_pjit_cost_rule`](cost_estimate.md#_pjit_cost_rule), [`_run_state_rule`](cost_estimate.md#_run_state_rule), [`estimate_cost`](cost_estimate.md#estimate_cost), [`kernel`](helpers.md#kernel), [`mpmd_map`](mpmd.md#mpmd_map), [`core_map`](core.md#core_map), [`CostEstimate`](pallas_call.md#CostEstimate), [`CostEstimate`](hlo_interpreter.md#CostEstimate)

### `DynamicGridDim`
- def: [`jax/_src/pallas/core.py:68`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L68)
- signature: `class DynamicGridDim:`
- protocol/private: `__repr__`[`L69`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L69)
- used by: [`interpret_pallas_call`](mosaic/interpret/interpret_pallas_call.md#interpret_pallas_call), [`pallas_call_hlo_interpret`](hlo_interpreter.md#pallas_call_hlo_interpret), [`wrapped`](mosaic/pipeline.md#emit_pipeline.wrapped), [`GridMappingGrid`](core.md#GridMappingGrid), [`dynamic_grid_dim`](core.md#dynamic_grid_dim), [`size`](core.md#PallasGridContext.size)

### `Element`
- def: [`jax/_src/pallas/core.py:401`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L401)
- doc: Use to index an array using an elementwise start index.
- signature: `class Element:`
- members:
  - `block_size` — [`L403`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L403)
  - `padding` — [`L404`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L404)
- protocol/private: `__str__`[`L406`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L406)
- used by: [`lower_jaxpr_into_pipelined_module`](mosaic/lowering.md#lower_jaxpr_into_pipelined_module), [`to_block_mapping`](core.md#BlockSpec.to_block_mapping), [`pallas_call_hlo_interpret`](hlo_interpreter.md#pallas_call_hlo_interpret), [`_concatenate_eval_rule`](fuser/block_spec.md#_concatenate_eval_rule), [`_eval_index_map`](triton/lowering.md#_eval_index_map), [`_make_block_slice`](mosaic/pipeline.md#_make_block_slice), [`BlockDim`](core.md#BlockDim.BlockDim), [`_concatenate_rule`](fuser/block_spec.md#_concatenate_rule), [`_offset_indexer`](fuser/block_spec.md#_offset_indexer), [`get_block_size`](core.md#get_block_size), [`_eval_index_map`](mosaic_gpu/lowering.md#_eval_index_map), [`_slice_rule`](fuser/block_spec.md#_slice_rule), [`_block_size`](fuser/block_spec.md#_block_size), [`_check_block_mappings`](mosaic_gpu/lowering.md#_check_block_mappings), [`_stack_pull_rule`](fuser/block_spec.md#_stack_pull_rule), [`_canonicalize_block_dim`](core.md#_canonicalize_block_dim), [`_get_block_dim_size`](core.md#_get_block_dim_size), [`_pull_bcast_block_spec`](fuser/block_spec.md#_pull_bcast_block_spec), [`_get_start_index`](triton/lowering.md#_eval_index_map._get_start_index), [`_make_block_slice`](mosaic_gpu/pipeline.md#BufferedRef._make_block_slice), [`_get_bdim_alignment`](triton/lowering.md#_get_index_alignment._get_bdim_alignment), [`_push_bcast_block_spec`](fuser/block_spec.md#_push_bcast_block_spec), [`_get_block_size`](mosaic_gpu/pipeline.md#_get_block_size), [`_get_start_index`](mosaic/interpret/interpret_pallas_call.md#_compute_start_indices._get_start_index), [`_get_start_index`](core.md#BlockMapping._get_start_index), [`Element`](mosaic/pipeline.md#Element)

### `GridAxis`
- def: [`jax/_src/pallas/core.py:378`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L378)
- signature: `class GridAxis:`
- members:
  - `index` — [`L379`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L379)
  - `size` — [`L380`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L380)
- uses (calls/refs, reference-scoped): [`Array`](../basearray.md#Array)
- used by: [`body`](hlo_interpreter.md#pallas_call_hlo_interpret.body), [`grid_env`](mosaic/pipeline.md#Scheduler.grid_env), [`_get_local_grid_env`](mosaic/interpret/interpret_pallas_call.md#interpret_pallas_call._get_local_grid_env), [`GridEnv`](core.md#GridEnv)

### `GridMapping`
- def: [`jax/_src/pallas/core.py:943`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L943)
- doc: An internal canonicalized version of GridSpec.
- signature: `class GridMapping:`
- members:
  - `block_mappings_output(self)` — [`L1090`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1090)
  - `check_invariants(self)` — [`L980`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L980)
  - `in_shapes(self)` — [`L1078`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1078) — The shapes of ``*index``, ``*inputs``.
  - `num_dynamic_grid_bounds(self)` — [`L1025`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1025)
  - `num_scratch_operands(self)` — [`L1029`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1029)
  - `out_shapes(self)` — [`L1097`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1097)
  - `replace(self, **kwargs)` — [`L1019`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1019)
  - `slice_block_ops(self)` — [`L1057`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1057) — Returns a slice to select the block operands to a kernel.
  - `slice_index_ops(self)` — [`L1050`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1050) — Returns a slice object to select the index operands to a kernel.
  - `slice_scratch_ops(self)` — [`L1068`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1068) — Returns a slice object to select the scratch operands to a kernel.
  - `static_grid(self)` — [`L1033`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1033)
  - `to_lojax(self)` — [`L1102`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1102)
  - `trace_env(self)` — [`L1039`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1039)
  - `block_mappings` — [`L964`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L964)
  - `debug` — [`L978`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L978)
  - `get_grid_indices` — [`L975`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L975)
  - `grid` — [`L960`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L960)
  - `grid_names` — [`L961`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L961)
  - `index_map_avals` — [`L967`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L967)
  - `index_map_tree` — [`L966`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L966)
  - `local_grid_env` — [`L976`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L976)
  - `num_index_operands` — [`L972`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L972)
  - `num_inputs` — [`L973`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L973)
  - `num_outputs` — [`L974`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L974)
  - `scratch_avals` — [`L970`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L970)
  - `vmapped_dims` — [`L969`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L969)
- protocol/private: `__repr__`[`L1140`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1140), `__str__`[`L1160`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1160)
- uses (calls/refs, reference-scoped): [`AbstractValue`](../core.md#AbstractValue), [`shape`](../core.md#ShapedArray.shape), [`dtype`](../core.md#ShapedArray.dtype), [`ShapeDtypeStruct`](../core.md#ShapeDtypeStruct), [`PyTreeDef`](../tree_util.md#PyTreeDef), [`like`](../core.md#ShapeDtypeStruct.like), [`_check_vma`](../config.md#_check_vma), [`array_aval`](core.md#BlockMapping.array_aval), [`check_invariants`](core.md#BlockMapping.check_invariants), [`in_avals`](../core.md#ClosedJaxpr.in_avals), [`index_map_jaxpr`](core.md#BlockMapping.index_map_jaxpr), [`to_lojax`](core.md#BlockMapping.to_lojax), [`tracing_grid_env`](core.md#tracing_grid_env), [`int32`](../numpy/scalar_types.md#int32), [`enable_checks`](../config.md#enable_checks), [`lo_ty`](../core.md#AbstractValue.lo_ty), [`BlockMapping`](core.md#BlockMapping), [`extend_axis_env_nd`](../core.md#extend_axis_env_nd), [`is_high`](../core.md#AbstractValue.is_high), [`GridMappingGrid`](core.md#GridMappingGrid), [`is_dynamic_dim`](core.md#is_dynamic_dim), [`split_list`](core.md#split_list), [`StaticGrid`](core.md#StaticGrid)
- used by: [`_interpret_jaxpr`](mosaic/interpret/interpret_pallas_call.md#_interpret_jaxpr), [`interpret_pallas_call`](mosaic/interpret/interpret_pallas_call.md#interpret_pallas_call), [`lower_jaxpr_into_pipelined_module`](mosaic/lowering.md#lower_jaxpr_into_pipelined_module), [`_pallas_call_batching_rule`](pallas_call.md#_pallas_call_batching_rule), [`get_grid_mapping`](core.md#get_grid_mapping), [`interpret_pallas_call`](mosaic_gpu/interpret/interpret_pallas_call.md#interpret_pallas_call), [`_pallas_call_state_discharge_rule`](pallas_call.md#_pallas_call_state_discharge_rule), [`lower_pipelined_jaxpr_to_module`](mosaic_gpu/lowering.md#lower_pipelined_jaxpr_to_module), [`pallas_call_hlo_interpret`](hlo_interpreter.md#pallas_call_hlo_interpret), [`_body`](mosaic/interpret/interpret_pallas_call.md#interpret_pallas_call._execute_grid_for_core._body), [`pallas_call_lowering`](triton/pallas_call_registration.md#pallas_call_lowering), [`_dimension_semantics`](mosaic/lowering.md#MosaicGridMapping._dimension_semantics), [`lower_jaxpr_to_triton_module`](triton/lowering.md#lower_jaxpr_to_triton_module), [`wrapped`](mosaic/pipeline.md#emit_pipeline.wrapped), [`_pallas_call_to_lojax`](pallas_call.md#_pallas_call_to_lojax), [`_allocate_buffers_for_outputs`](mosaic_gpu/interpret/interpret_pallas_call.md#_allocate_buffers_for_outputs), [`_pallas_call_jvp_rule`](pallas_call.md#_pallas_call_jvp_rule), [`pallas_call_lowering`](mosaic_gpu/pallas_call_registration.md#pallas_call_lowering), [`wrapped`](pallas_call.md#_pallas_call.wrapped), [`pallas_call_tpu_lowering_rule`](mosaic/pallas_call_registration.md#pallas_call_tpu_lowering_rule), [`__init__`](mosaic/lowering.md#MosaicGridMapping.__init__), [`_execute_grid_for_core`](mosaic/interpret/interpret_pallas_call.md#interpret_pallas_call._execute_grid_for_core), [`_batch_with_explicit_loop`](pallas_call.md#_batch_with_explicit_loop), [`_batch_block_mapping`](pallas_call.md#_batch_block_mapping), [`_trace_kernel_to_jaxpr`](pallas_call.md#_trace_kernel_to_jaxpr), [`body`](hlo_interpreter.md#pallas_call_hlo_interpret.body), [`_get_kernel_buffers`](mosaic_gpu/interpret/interpret_pallas_call.md#_get_kernel_buffers), [`_pallas_call_physicalize_rule`](fuser/fusible_dtype.md#_pallas_call_physicalize_rule), [`kernel_to_hlo_jaxpr`](hlo_interpreter.md#kernel_to_hlo_jaxpr), [`_get_grid_and_cluster_dims_and_num_threads`](mosaic_gpu/interpret/interpret_pallas_call.md#_get_grid_and_cluster_dims_and_num_threads), [`_process_grid_to_3d_grid`](triton/lowering.md#_process_grid_to_3d_grid), [`lower_jaxpr_to_pipelined_module`](mosaic/lowering.md#lower_jaxpr_to_pipelined_module), [`pipeline_fn`](mosaic_gpu/lowering.md#lower_pipelined_jaxpr_to_module.pipeline_fn), [`_axis_index_rule`](triton/lowering.md#_axis_index_rule), [`_rewritten_body`](pallas_call.md#_pallas_call_state_discharge_rule._rewritten_body), [`GridMapping`](pallas_call.md#GridMapping), [`_get_grid_bounds`](mosaic_gpu/interpret/interpret_pallas_call.md#_get_grid_bounds), [`_get_local_grid_env`](mosaic/interpret/interpret_pallas_call.md#interpret_pallas_call._get_local_grid_env), [`grid_mapping`](mosaic/interpret/interpret_pallas_call.md#InterpretContext.grid_mapping), [`GridMapping`](triton/lowering.md#GridMapping)  (+1 more)

### `GridSpec`
- def: [`jax/_src/pallas/core.py:1214`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1214)
- doc: Encodes the grid parameters for `jax.experimental.pallas.pallas_call`.
- signature: `class GridSpec:`
- members:
  - `grid` — [`L1222`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1222)
  - `grid_names` — [`L1223`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1223)
  - `in_specs` — [`L1224`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1224)
  - `out_specs` — [`L1225`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1225)
  - `scratch_shapes` — [`L1226`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1226)
- protocol/private: `__init__`[`L1228`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1228), `_make_scalar_ref_aval`[`L1261`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1261)
- uses (calls/refs, reference-scoped): [`unzip2`](../util.md#unzip2), [`no_block_spec`](core.md#no_block_spec), [`PrefetchScalarGridSpec`](mosaic/core.md#PrefetchScalarGridSpec), [`ScratchShapeTree`](core.md#ScratchShapeTree), [`_make_scalar_ref_aval`](mosaic/core.md#PrefetchScalarGridSpec._make_scalar_ref_aval), [`TupleGrid`](core.md#TupleGrid), [`BlockSpecTree`](core.md#BlockSpecTree), [`Grid`](core.md#Grid), [`_is_valid_grid_dim`](core.md#_is_valid_grid_dim)
- used by: [`get_grid_mapping`](core.md#get_grid_mapping), [`wrapped`](mosaic/pipeline.md#emit_pipeline.wrapped), [`pallas_call`](pallas_call.md#pallas_call), [`_pallas_call`](pallas_call.md#_pallas_call), [`_mpmd_map_fallback_lowering`](mpmd.md#_mpmd_map_fallback_lowering), [`__init__`](mosaic/core.md#PrefetchScalarGridSpec.__init__), [`PrefetchScalarGridSpec`](mosaic/core.md#PrefetchScalarGridSpec), [`unzip_dynamic_grid_bounds`](core.md#unzip_dynamic_grid_bounds)

### `Indirect`
- def: [`jax/_src/pallas/core.py:440`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L440)
- doc: A dimension indexed by an array of indices.
- signature: `class Indirect:`
- members:
  - `block_size` — [`L445`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L445)
- protocol/private: `__repr__`[`L447`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L447)
- used by: [`BlockDim`](core.md#BlockDim.BlockDim), [`get_block_size`](core.md#get_block_size), [`_canonicalize_block_dim`](core.md#_canonicalize_block_dim), [`_get_block_dim_size`](core.md#_get_block_dim_size), [`Indirect`](mosaic/pipeline.md#Indirect)

### `MemoryRef`  ·  implements/extends MemoryRefBase
- def: [`jax/_src/pallas/core.py:251`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L251)
- doc: Like jax.ShapeDtypeStruct but with memory spaces.
- signature: `class MemoryRef(MemoryRefBase):`
- members:
  - `dtype(self)` — [`L270`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L270)
  - `get_array_aval(self)` — [`L257`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L257)
  - `get_ref_aval(self)` — [`L266`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L266)
  - `shape(self)` — [`L274`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L274)
  - `inner_aval` — [`L253`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L253)
  - `memory_space` — [`L255`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L255)
- protocol/private: `__lt__`[`L277`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L277)
- uses (calls/refs, reference-scoped): [`AbstractRef`](../state/types.md#AbstractRef), [`ShapedArray`](../core.md#ShapedArray), [`AbstractValue`](../core.md#AbstractValue), [`dtype`](../core.md#ShapedArray.dtype), [`TransformedRef`](../state/types.md#TransformedRef), [`ExtendedDType`](../dtypes.md#ExtendedDType), [`get_ref_aval`](mosaic_gpu/core.md#GPUMemoryRef.get_ref_aval), [`update`](../core.md#ShapedArray.update), [`MemoryRef`](mosaic/core.md#MemoryRef), [`MemoryRefBase`](core.md#MemoryRefBase)
- used by: [`lower_pipelined_jaxpr_to_module`](mosaic_gpu/lowering.md#lower_pipelined_jaxpr_to_module), [`get_ref_aval`](mosaic_gpu/core.md#GPUMemoryRef.get_ref_aval), [`ref_for_aval`](mosaic_gpu/lowering.md#lower_pipelined_jaxpr_to_module.ref_for_aval), [`__call__`](mosaic_gpu/core.md#MemorySpace.__call__), [`_get_sds`](core.md#_get_sds), [`_convert_out_shape_to_aval`](core.md#_convert_out_shape_to_aval), [`default_mesh_discharge_rule`](core.md#default_mesh_discharge_rule), [`_mpmd_map_fallback_lowering`](mpmd.md#_mpmd_map_fallback_lowering), [`empty_ref_like`](helpers.md#empty_ref_like), [`__call__`](mosaic_gpu/core.md#SemaphoreType.__call__), [`__post_init__`](mosaic_gpu/core.md#GPUMemoryRef.__post_init__), [`get_ref_aval`](mosaic_gpu/core.md#RefUnion.get_ref_aval), [`TryClusterCancelResult`](mosaic_gpu/core.md#TryClusterCancelResult), [`from_type`](core.md#MemorySpace.from_type), [`GPUMemoryRef`](mosaic_gpu/core.md#GPUMemoryRef), [`get_ref_aval`](core.md#MemoryRefBase.get_ref_aval), [`get_array_aval`](mosaic/core.md#SemaphoreType.get_array_aval), [`__call__`](core.md#CoreMemorySpace.__call__), [`__matmul__`](mosaic/core.md#MemoryRef.__matmul__), [`get_array_aval`](core.md#MemoryRefBase.get_array_aval), [`get_array_aval`](mosaic_gpu/core.md#SemaphoreType.get_array_aval), [`get_ref_aval`](mosaic/core.md#SemaphoreType.get_ref_aval), [`get_ref_aval`](mosaic_gpu/core.md#SemaphoreType.get_ref_aval), [`MemoryRef`](mosaic/core.md#MemoryRef), [`MemoryRefBase`](core.md#MemoryRefBase)

### `MemoryRefBase`  ·  implements/extends Protocol
- def: [`jax/_src/pallas/core.py:241`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L241)
- signature: `class MemoryRefBase(Protocol):`
- members:
  - `get_array_aval(self)` — [`L243`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L243)
  - `get_ref_aval(self)` — [`L246`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L246)
- uses (calls/refs, reference-scoped): [`AbstractRef`](../state/types.md#AbstractRef), [`ShapedArray`](../core.md#ShapedArray), [`TransformedRef`](../state/types.md#TransformedRef), [`MemoryRef`](core.md#MemoryRef), [`get_array_aval`](core.md#MemoryRef.get_array_aval), [`get_ref_aval`](core.md#MemoryRef.get_ref_aval)
- used by: [`MemoryRef`](core.md#MemoryRef)

### `MemorySpace`  ·  implements/extends Enum
- def: [`jax/_src/pallas/core.py:282`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L282)
- doc: Logical, device-agnostic memory spaces.
- signature: `class MemorySpace(enum.Enum):`
- members:
  - `from_type(self, type: jax_core.AbstractValue)` — [`L298`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L298)
  - `memory_kind(self)` — [`L295`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L295)
  - `ANY` — [`L288`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L288)
  - `DEFAULT` — [`L289`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L289)
  - `ERROR` — [`L290`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L290)
  - `INDEX` — [`L291`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L291)
  - `KEY` — [`L292`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L292)
- protocol/private: `__call__`[`L301`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L301), `__str__`[`L305`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L305)
- uses (calls/refs, reference-scoped): [`ShapedArray`](../core.md#ShapedArray), [`AbstractValue`](../core.md#AbstractValue), [`MemoryRef`](core.md#MemoryRef), [`memory_space`](core.md#MemoryRef.memory_space)
- used by: [`lower_jaxpr_into_pipelined_module`](mosaic/lowering.md#lower_jaxpr_into_pipelined_module), [`to_block_mapping`](core.md#BlockSpec.to_block_mapping), [`get_grid_mapping`](core.md#get_grid_mapping), [`_pallas_call_state_discharge_rule`](pallas_call.md#_pallas_call_state_discharge_rule), [`_check_block_mappings`](mosaic/lowering.md#_check_block_mappings), [`lower_jaxpr_into_unpipelined_module`](mosaic/lowering.md#lower_jaxpr_into_unpipelined_module), [`_get_memory_space_from_aval`](mosaic/pallas_call_registration.md#_get_memory_space_from_aval), [`memory_space_to_tpu_memory_space`](mosaic/core.md#memory_space_to_tpu_memory_space), [`ref_for_aval`](mosaic_gpu/lowering.md#lower_pipelined_jaxpr_to_module.ref_for_aval), [`_scatter_lowering_rule`](mosaic/sc_primitives.md#_scatter_lowering_rule), [`_gather_lowering_rule`](mosaic/sc_primitives.md#_gather_lowering_rule), [`_memory_space_to_mosaic_attribute`](mosaic/lowering.md#_memory_space_to_mosaic_attribute), [`TPU_MEMORY_SPACE_IDXS`](mosaic/interpret/interpret_pallas_call.md#TPU_MEMORY_SPACE_IDXS.TPU_MEMORY_SPACE_IDXS), [`with_memory_space_constraint`](mosaic/primitives.md#with_memory_space_constraint), [`_check_ref`](mosaic_gpu/primitives.md#_check_ref), [`host_idx`](mosaic/interpret/interpret_pallas_call.md#host_idx), [`_remove_memory_space_abstract_eval`](mosaic/interpret/interpret_pallas_call.md#_remove_memory_space_abstract_eval), [`empty_ref_like`](helpers.md#empty_ref_like), [`_random_fold_in_pull_rule`](fuser/block_spec.md#_random_fold_in_pull_rule), [`_in_smem`](mosaic_gpu/pipeline.md#_in_smem), [`get_memory_space_idx`](mosaic_gpu/interpret/gpu_callbacks.md#get_memory_space_idx), [`_random_bits_pull_rule`](fuser/block_spec.md#_random_bits_pull_rule), [`_with_default_memory_space`](core.md#get_grid_mapping._with_default_memory_space), [`default_memory_space`](core.md#Mesh.default_memory_space), [`TPUMemorySpace`](mosaic/lowering.md#TPUMemorySpace), [`any_idx`](mosaic/interpret/interpret_pallas_call.md#any_idx), [`ANY`](../../experimental/pallas/__init__.md#ANY), [`AnyMemorySpace`](mosaic/lowering.md#AnyMemorySpace), [`ANY`](mosaic/lowering.md#ANY), [`ANY`](mosaic/pipeline.md#ANY), [`_ANY`](mosaic/interpret/interpret_pallas_call.md#_ANY), [`default_memory_space`](mosaic/core.md#TensorCoreMesh.default_memory_space)

### `Mesh`
- def: [`jax/_src/pallas/core.py:1671`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1671)
- members:
  - `check_is_compatible_with(self, other_mesh)` — [`L1689`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1689) — Raise if this mesh (e.g., its axes names) cannot be used with other_mesh.
  - `core_type(self)` — [`L1682`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1682)
  - `default_memory_space(self)` — [`L1674`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1674)
  - `discharges_effect(self, effect: jax_core.Effect)` — [`L1686`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1686)
  - `shape(self)` — [`L1678`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1678)
  - `supported_memory_spaces(self)` — [`L1699`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1699) — Return the memory spaces supported by the mesh.
  - `tracing_context(self)` — [`L1703`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1703)
- uses (calls/refs, reference-scoped): [`MemorySpace`](core.md#MemorySpace), [`check_is_compatible_with`](mosaic/sc_core.md#VectorSubcoreMesh.check_is_compatible_with), [`check_is_compatible_with`](mosaic/sc_core.md#ScalarSubcoreMesh.check_is_compatible_with), [`Effect`](../core.md#Effect), [`shape`](mosaic_gpu/core.md#Mesh.shape), [`TensorCoreMesh`](mosaic/core.md#TensorCoreMesh), [`supported_memory_spaces`](mosaic/core.md#TensorCoreMesh.supported_memory_spaces), [`supported_memory_spaces`](mosaic/sc_core.md#VectorSubcoreMesh.supported_memory_spaces), [`shape`](mosaic/sc_core.md#VectorSubcoreMesh.shape), [`supported_memory_spaces`](mosaic/sc_core.md#ScalarSubcoreMesh.supported_memory_spaces), [`supported_memory_spaces`](mosaic_gpu/core.md#WarpMesh.supported_memory_spaces), [`ScalarSubcoreMesh`](mosaic/sc_core.md#ScalarSubcoreMesh), [`WarpMesh`](mosaic_gpu/core.md#WarpMesh), [`shape`](mosaic/core.md#TensorCoreMesh.shape), [`discharges_effect`](mosaic_gpu/core.md#Mesh.discharges_effect), [`shape`](mosaic/sc_core.md#ScalarSubcoreMesh.shape), [`check_is_compatible_with`](mosaic/core.md#TensorCoreMesh.check_is_compatible_with), [`core_type`](mosaic/core.md#TensorCoreMesh.core_type), [`core_type`](mosaic/sc_core.md#ScalarSubcoreMesh.core_type), [`core_type`](mosaic/sc_core.md#VectorSubcoreMesh.core_type), [`default_memory_space`](mosaic/core.md#TensorCoreMesh.default_memory_space), [`default_memory_space`](mosaic/sc_core.md#ScalarSubcoreMesh.default_memory_space), [`default_memory_space`](mosaic/sc_core.md#VectorSubcoreMesh.default_memory_space), [`default_memory_space`](mosaic_gpu/core.md#Mesh.default_memory_space), [`shape`](mosaic_gpu/core.md#WarpMesh.shape), [`tracing_context`](mosaic_gpu/core.md#Mesh.tracing_context), [`default_memory_space`](mosaic_gpu/core.md#WarpMesh.default_memory_space), [`discharges_effect`](mosaic/core.md#TensorCoreMesh.discharges_effect), [`discharges_effect`](mosaic_gpu/core.md#WarpMesh.discharges_effect), [`supported_memory_spaces`](mosaic_gpu/core.md#Mesh.supported_memory_spaces), [`CoreType`](core.md#CoreType), [`check_is_compatible_with`](mosaic_gpu/core.md#Mesh.check_is_compatible_with), [`check_is_compatible_with`](mosaic_gpu/core.md#WarpMesh.check_is_compatible_with), [`core_type`](mosaic_gpu/core.md#Mesh.core_type), [`core_type`](mosaic_gpu/core.md#WarpMesh.core_type), [`discharges_effect`](mosaic/sc_core.md#ScalarSubcoreMesh.discharges_effect), [`discharges_effect`](mosaic/sc_core.md#VectorSubcoreMesh.discharges_effect), [`tracing_context`](mosaic/core.md#TensorCoreMesh.tracing_context), [`tracing_context`](mosaic/sc_core.md#ScalarSubcoreMesh.tracing_context), [`tracing_context`](mosaic/sc_core.md#VectorSubcoreMesh.tracing_context)  (+1 more)
- used by: [`_interpret_jaxpr`](mosaic/interpret/interpret_pallas_call.md#_interpret_jaxpr), [`interpret_pallas_call`](mosaic/interpret/interpret_pallas_call.md#interpret_pallas_call), [`_pallas_call_batching_rule`](pallas_call.md#_pallas_call_batching_rule), [`_pallas_call_state_discharge_rule`](pallas_call.md#_pallas_call_state_discharge_rule), [`pallas_call_hlo_interpret`](hlo_interpreter.md#pallas_call_hlo_interpret), [`pallas_call_lowering`](triton/pallas_call_registration.md#pallas_call_lowering), [`wrapper`](mpmd.md#_mpmd_map.wrapper), [`lower_jaxpr_into_unpipelined_module`](mosaic/lowering.md#lower_jaxpr_into_unpipelined_module), [`_pallas_call_to_lojax`](pallas_call.md#_pallas_call_to_lojax), [`_get_memory_space_from_aval`](mosaic/pallas_call_registration.md#_get_memory_space_from_aval), [`_pallas_call_jvp_rule`](pallas_call.md#_pallas_call_jvp_rule), [`memory_space_to_tpu_memory_space`](mosaic/core.md#memory_space_to_tpu_memory_space), [`pallas_call_lowering`](mosaic_gpu/pallas_call_registration.md#pallas_call_lowering), [`pallas_call_tpu_lowering_rule`](mosaic/pallas_call_registration.md#pallas_call_tpu_lowering_rule), [`_rewrite_jaxpr_for_lowering`](mosaic/pallas_call_registration.md#_rewrite_jaxpr_for_lowering), [`_dma_start_lowering_rule`](mosaic/lowering.md#_dma_start_lowering_rule), [`_batch_with_explicit_loop`](pallas_call.md#_batch_with_explicit_loop), [`_semaphore_signal_lowering_rule`](mosaic/lowering.md#_semaphore_signal_lowering_rule), [`_dedup_consts_and_unify_jaxpr_signatures`](mpmd.md#_dedup_consts_and_unify_jaxpr_signatures), [`_device_id_to_logical`](mosaic/lowering.md#_device_id_to_logical), [`_memory_space_to_mosaic_attribute`](mosaic/lowering.md#_memory_space_to_mosaic_attribute), [`_pallas_call`](pallas_call.md#_pallas_call), [`_get_mesh_shape_and_semantics`](mosaic/lowering.md#_get_mesh_shape_and_semantics), [`check_is_compatible_with`](mosaic/sc_core.md#VectorSubcoreMesh.check_is_compatible_with), [`_mpmd_map`](mpmd.md#_mpmd_map), [`check_is_compatible_with`](mosaic/sc_core.md#ScalarSubcoreMesh.check_is_compatible_with), [`kernel`](helpers.md#kernel), [`_aval_to_ref_aval`](mpmd.md#_aval_to_ref_aval), [`mesh`](core.md#CoreMemorySpace.mesh), [`mpmd_map`](mpmd.md#mpmd_map), [`mpmd_map_tracing_context`](mpmd.md#mpmd_map_tracing_context), [`Mesh`](mosaic_gpu/core.md#Mesh), [`TensorCoreMesh`](mosaic/core.md#TensorCoreMesh), [`__matmul__`](mosaic/core.md#SemaphoreType.__matmul__), [`VectorSubcoreMesh`](mosaic/sc_core.md#VectorSubcoreMesh), [`ScalarSubcoreMesh`](mosaic/sc_core.md#ScalarSubcoreMesh), [`WarpMesh`](mosaic_gpu/core.md#WarpMesh), [`ctx_factory`](mosaic/lowering.md#lower_jaxpr_into_unpipelined_module.ctx_factory), [`__post_init__`](core.md#CoreMemorySpace.__post_init__), [`__repr__`](core.md#CoreMemorySpace.__repr__)  (+7 more)

### `NoBlockSpec`
- def: [`jax/_src/pallas/core.py:726`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L726)
- signature: `class NoBlockSpec:`
- protocol/private: `__repr__`[`L727`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L727)
- used by: [`_push_block_spec_jaxpr`](fuser/block_spec.md#_push_block_spec_jaxpr), [`no_block_spec`](core.md#no_block_spec), [`_apply_block_transform`](fuser/block_spec.md#_apply_block_transform), [`_pull_block_spec`](fuser/block_spec.md#_pull_block_spec), [`_dot_general_push_rule`](fuser/block_spec.md#_dot_general_push_rule), [`_read_block_spec`](fuser/block_spec.md#_push_block_spec_jaxpr._read_block_spec), [`_write_block_spec`](fuser/block_spec.md#_push_block_spec_jaxpr._write_block_spec)

### `PallasGridContext`
- def: [`jax/_src/pallas/core.py:342`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L342)
- signature: `class PallasGridContext:`
- members:
  - `size(self, axis: int)` — [`L346`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L346)
  - `grid` — [`L343`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L343)
  - `mapped_dims` — [`L344`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L344)
- uses (calls/refs, reference-scoped): [`GridMappingGrid`](core.md#GridMappingGrid), [`DynamicGridDim`](core.md#DynamicGridDim)
- used by: [`tracing_grid_env`](core.md#tracing_grid_env), [`_num_programs_bind_with_trace`](primitives.md#_num_programs_bind_with_trace), [`program_id_bind_with_trace`](primitives.md#program_id_bind_with_trace), [`axis_frame`](core.md#axis_frame), [`grid_context`](core.md#PallasTracingEnv.grid_context)

### `PallasTracingEnv`  ·  implements/extends local
- def: [`jax/_src/pallas/core.py:358`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L358)
- signature: `class PallasTracingEnv(threading.local):`
- members:
  - `dynamic_shapes` — [`L362`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L362)
  - `grid_context` — [`L359`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L359)
  - `grid_env_stack` — [`L360`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L360)
  - `is_interpret_mode` — [`L361`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L361)
  - `module_export_fn` — [`L363`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L363)
- uses (calls/refs, reference-scoped): [`GridEnv`](core.md#GridEnv), [`PallasGridContext`](core.md#PallasGridContext)
- used by: [`tracing_grid_env`](core.md#tracing_grid_env), [`_pallas_tracing_env`](core.md#_pallas_tracing_env), [`dynamic_shapes_export_enabled`](core.md#dynamic_shapes_export_enabled), [`grid_env`](core.md#grid_env), [`axis_frame`](core.md#axis_frame), [`current_grid_env`](core.md#current_grid_env), [`pallas_export_experimental`](core.md#pallas_export_experimental)

### `RevisitMode`  ·  implements/extends Enum
- def: [`jax/_src/pallas/core.py:194`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L194)
- doc: Specifies whether an output buffer supports revisiting.
- signature: `class RevisitMode(enum.Enum):`
- members:
  - `ANY` — [`L209`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L209)
  - `IMMEDIATE` — [`L208`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L208)
- used by: [`lower_jaxpr_into_pipelined_module`](mosaic/lowering.md#lower_jaxpr_into_pipelined_module), [`revisit`](core.md#Buffered.revisit)

### `ScratchShape`  ·  implements/extends Protocol
- def: [`jax/_src/pallas/core.py:1200`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1200)
- signature: `class ScratchShape(Protocol):`
- members:
  - `get_array_aval(self)` — [`L1201`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1201)
  - `get_ref_aval(self)` — [`L1203`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1203)
- uses (calls/refs, reference-scoped): [`AbstractRef`](../state/types.md#AbstractRef), [`AbstractValue`](../core.md#AbstractValue), [`TransformedRef`](../state/types.md#TransformedRef)
- used by: [`get_global`](primitives.md#get_global), [`ScratchShapeTree`](core.md#ScratchShapeTree)

### `Semaphore`  ·  implements/extends AbstractSemaphoreTy
- def: [`jax/_src/pallas/core.py:128`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L128)
- signature: `class Semaphore(AbstractSemaphoreTy):`
- members:
  - `name` — [`L129`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L129)
  - `type` — [`L130`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L130)
- uses (calls/refs, reference-scoped): [`AbstractSemaphoreTy`](core.md#AbstractSemaphoreTy), [`semaphore`](core.md#semaphore)
- used by: [`__call__`](mosaic_gpu/core.md#SemaphoreType.__call__), [`dtype`](mosaic/core.md#SemaphoreType.dtype), [`AbstractSemaphoreTy`](core.md#AbstractSemaphoreTy)

### `Squeezed`
- def: [`jax/_src/pallas/core.py:412`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L412)
- doc: Represents a one-sized block dimension that is squeezed out in the kernel.
- signature: `class Squeezed:`
- used by: [`interpret_pallas_call`](mosaic/interpret/interpret_pallas_call.md#interpret_pallas_call), [`lower_jaxpr_into_pipelined_module`](mosaic/lowering.md#lower_jaxpr_into_pipelined_module), [`to_block_mapping`](core.md#BlockSpec.to_block_mapping), [`pallas_call_hlo_interpret`](hlo_interpreter.md#pallas_call_hlo_interpret), [`_compute_offsets_from_indices`](triton/lowering.md#_compute_offsets_from_indices), [`_dimension_semantics`](mosaic/lowering.md#MosaicGridMapping._dimension_semantics), [`lower_jaxpr_to_triton_module`](triton/lowering.md#lower_jaxpr_to_triton_module), [`_get_pull_rule`](fuser/block_spec.md#_get_pull_rule), [`_get_eval_rule`](fuser/block_spec.md#_get_eval_rule), [`BlockDim`](core.md#BlockDim.BlockDim), [`_concatenate_rule`](fuser/block_spec.md#_concatenate_rule), [`_tile_pull_rule`](fuser/block_spec.md#_tile_pull_rule), [`_offset_indexer`](fuser/block_spec.md#_offset_indexer), [`block_shapes`](mosaic/lowering.md#LoweringRuleContext.block_shapes), [`get_block_size`](core.md#get_block_size), [`_eval_index_map`](mosaic_gpu/lowering.md#_eval_index_map), [`_slice_rule`](fuser/block_spec.md#_slice_rule), [`_block_size`](fuser/block_spec.md#_block_size), [`_bitcast_memref`](mosaic/lowering.md#_bitcast_memref), [`_stack_pull_rule`](fuser/block_spec.md#_stack_pull_rule), [`_canonicalize_block_dim`](core.md#_canonicalize_block_dim), [`_reshape_memref`](mosaic/lowering.md#_reshape_memref), [`_get_block_dim_size`](core.md#_get_block_dim_size), [`_indexer_to_start_size_stride`](mosaic/lowering.md#_indexer_to_start_size_stride), [`_tile_eval_rule`](fuser/block_spec.md#_tile_eval_rule), [`_transpose_eval_rule`](fuser/block_spec.md#_transpose_eval_rule), [`_get_start_index`](triton/lowering.md#_eval_index_map._get_start_index), [`_make_block_slice`](mosaic_gpu/pipeline.md#BufferedRef._make_block_slice), [`_get_bdim_alignment`](triton/lowering.md#_get_index_alignment._get_bdim_alignment), [`_slice_memref`](mosaic/lowering.md#_slice_memref), [`squeezed`](core.md#squeezed), [`_get_block_shape`](mosaic_gpu/pipeline.md#_get_block_shape), [`_get_start_index`](mosaic/interpret/interpret_pallas_call.md#_compute_start_indices._get_start_index), [`_get_ref_block_shape`](core.md#_get_ref_block_shape), [`_get_start_index`](core.md#BlockMapping._get_start_index), [`_slice`](fuser/block_spec.md#_get_eval_rule._slice), [`Squeezed`](mosaic/pipeline.md#Squeezed), [`_slice_eval_rule`](fuser/block_spec.md#_slice_eval_rule), [`new_block_index_transform`](fuser/block_spec.md#_get_pull_rule.new_block_index_transform), [`block_shape`](triton/lowering.md#BlockInfo.block_shape)  (+4 more)

### `_IndexMapFunc`
- def: [`jax/_src/pallas/core.py:528`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L528)
- doc: Helper class that checks for index_map equality.
- signature: `class _IndexMapFunc:`
- members:
  - `index_map` — [`L532`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L532)
- protocol/private: `__call__`[`L543`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L543), `__eq__`[`L535`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L535), `__hash__`[`L540`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L540), `__init__`[`L531`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L531), `__repr__`[`L551`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L551), `__str__`[`L554`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L554)
- used by: [`__post_init__`](core.md#BlockSpec.__post_init__)

### `barrier_semaphore`  ·  implements/extends semaphore_dtype
- def: [`jax/_src/pallas/core.py:132`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L132)
- doc: Barrier semaphore dtype.
- signature: `class barrier_semaphore(semaphore_dtype):`
- uses (calls/refs, reference-scoped): [`semaphore_dtype`](core.md#semaphore_dtype)
- used by: [`_dtype_to_ir_type`](mosaic/lowering.md#_dtype_to_ir_type), [`_semaphore_read_lowering_rule`](mosaic/lowering.md#_semaphore_read_lowering_rule), [`check_sem_avals`](primitives.md#check_sem_avals), [`semaphore_dtype`](core.md#semaphore_dtype), [`BarrierSemaphore`](core.md#BarrierSemaphore)

### `semaphore`  ·  implements/extends semaphore_dtype
- def: [`jax/_src/pallas/core.py:122`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L122)
- doc: Regular semaphore dtype.
- signature: `class semaphore(semaphore_dtype):`
- uses (calls/refs, reference-scoped): [`semaphore_dtype`](core.md#semaphore_dtype)
- used by: [`_run_scoped_lowering_rule`](mosaic_gpu/lowering.md#_run_scoped_lowering_rule), [`_run_scoped_resource_estimator`](mosaic_gpu/lowering.md#_run_scoped_resource_estimator), [`_dtype_to_ir_type`](mosaic/lowering.md#_dtype_to_ir_type), [`_get_global_lowering_rule`](mosaic_gpu/lowering.md#_get_global_lowering_rule), [`_semaphore_read_lowering_rule`](mosaic/lowering.md#_semaphore_read_lowering_rule), [`_get_global_resource_estimator`](mosaic_gpu/lowering.md#_get_global_resource_estimator), [`check_sem_avals`](primitives.md#check_sem_avals), [`semaphore_dtype`](core.md#semaphore_dtype), [`type`](core.md#Semaphore.type)

### `semaphore_dtype`  ·  implements/extends extended
- def: [`jax/_src/pallas/core.py:115`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L115)
- doc: Common dtype for all kinds of semaphore dtypes.
- signature: `class semaphore_dtype(dtypes.extended):`
- uses (calls/refs, reference-scoped): [`extended`](../dtypes.md#extended), [`semaphore`](core.md#semaphore), [`dma_semaphore`](mosaic/core.md#dma_semaphore), [`barrier_semaphore`](core.md#barrier_semaphore)
- used by: [`_dtype_to_ir_type`](mosaic/lowering.md#_dtype_to_ir_type), [`extended`](../dtypes.md#extended), [`_alloc_value`](mosaic/lowering.md#_alloc_value), [`uninitialized_value`](primitives.md#uninitialized_value), [`semaphore`](core.md#semaphore), [`dma_semaphore`](mosaic/core.md#dma_semaphore), [`barrier_semaphore`](core.md#barrier_semaphore), [`PHYSICAL_EXTENDED_DTYPES`](mosaic/lowering.md#PHYSICAL_EXTENDED_DTYPES)

## Functions
- `_canonicalize_block_dim(dim: BlockDim | int | None)` — [`L458`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L458)
- `_canonicalize_block_shape(block_shape: Sequence[BlockDim | int | None])` — [`L479`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L479)
- `_convert_block_spec_to_block_mapping(block_spec: BlockSpec, origin: OriginStr, array_aval: jax_core.ShapedArray, *, index_map_avals: Sequence[jax_core.AbstractValue], index_map_tree: tree_util.PyTreeDef, grid: GridMappingGrid, vmapped_dims: tuple[int, ...], allow_captured_consts: bool, debug: bool = False)` — [`L1170`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1170)
- `_convert_out_shape_to_aval(out_shape: Any)` — [`L1883`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1883)
- `_core_map_abstract_eval(*args, jaxpr, mesh, interpret, **kwargs)` — [`L1630`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1630)
- `_core_map_discharge_rule(in_avals, out_avals, *args_flat, jaxpr, debug_info, mesh, **kwargs)` — [`L1812`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1812)
- `_core_map_is_high(*avals, jaxpr, **params)` — [`L1503`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1503)
- `_core_map_partial_eval_custom(saveable, unks_in, inst_in, eqn)` — [`L1917`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1917)
- `_core_map_to_lojax(*consts, jaxpr, mesh, **params)` — [`L1508`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1508)
- `_core_map_typecheck_rule(_, *in_atoms, jaxpr, mesh, **kwargs)` — [`L1848`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1848)
- `_get_block_dim_size(dim: BlockDim)` — [`L484`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L484)
- `_get_block_shape(block_shape: tuple[BlockDim, ...])` — [`L515`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L515)
- `_get_ref_block_shape(block_shape: tuple[BlockDim, ...])` — [`L518`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L518)
- `_get_sds(aval: jax_core.AbstractValue)` — [`L1479`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1479)
- `_is_valid_grid_dim(dim: int | jax_typing.Array)` — [`L1164`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1164)
- `_with_default_memory_space(bs: BlockSpec)` — [`L1327`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1327)
- `axis_frame()` — [`L368`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L368)
- `body(*args)` — [`L1779`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1779)
- `core_map(mesh, *, compiler_params: Any | None = None, interpret: bool = False, debug: bool = False, cost_estimate: CostEstimate | None = None, name: str | None = None, metadata: dict[str, str] | None = None, scratch_shapes: ScratchShapeTree = ())` — [`L1525`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1525) — Runs a function on a mesh, mapping it over the devices in the mesh.
- `core_map_lowering_rule(ctx: mlir.LoweringRuleContext, *args, jaxpr, **kwargs)` — [`L1654`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1654)
- `current_grid_env()` — [`L394`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L394)
- `debug_check(condition, message)` — [`L165`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L165) — Check the condition if
- `debug_checks_enabled()` — [`L152`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L152) — Returns runtime checks are enabled.
- `default_index_map(ndim: int)` — [`L454`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L454)
- `default_mesh_discharge_rule(in_avals, out_avals, *args, mesh, compiler_params, jaxpr, debug, interpret, cost_estimate, name, metadata)` — [`L1735`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1735) — Discharges a ``core_map`` over a mesh to a ``mpmd_map``.
- `dynamic_shapes_export_enabled()` — [`L934`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L934)
- `get_block_size(dim: BlockDim | int | None)` — [`L498`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L498)
- `get_grid_mapping(grid_spec: GridSpec, in_avals: Sequence[jax_core.AbstractValue], in_tree: tree_util.PyTreeDef, in_origins: Sequence[OriginStr], out_avals: Sequence[jax_core.AbstractValue], out_tree: tree_util.PyTreeDef, out_origins: Sequence[OriginStr], allow_captured_consts: bool = False, debug: bool = False)` — [`L1265`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1265) — documented in [jax-_src-pallas-core](../../../../concepts/jax-_src-pallas-core.md)
- `get_interpret_effects(interpret: Any)` — [`L1619`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1619)
- `get_lowering_rule(params_cls, expected_platform: str)` — [`L180`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L180)
- `get_memory_space_aval(aval: jax_core.AbstractValue)` — [`L1465`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1465) — Queries the memory space of an array.
- `grid_env(env: GridEnv)` — [`L386`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L386)
- `is_dynamic_dim(d)` — [`L938`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L938)
- `lower_as_mlir(f, *args, dynamic_shapes=False, device=None, static_argnames=(), platforms=None, **kwargs)` — [`L1857`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1857) — Lower the function to MLIR.
- `pallas_export_experimental(dynamic_shapes: bool)` — [`L925`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L925)
- `poison_buffers_enabled()` — [`L160`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L160) — Returns whether Pallas buffer poisoning is enabled.
- `pytreedef_mismatch_err_msg(what1: str, tree1: tree_util.PyTreeDef, what2: str, tree2: tree_util.PyTreeDef)` — [`L1426`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1426)
- `register_lowering_rule(params_cls, rule, platform: str)` — [`L176`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L176)
- `tracing_grid_env(grid: GridMappingGrid, mapped_dims: tuple[int, ...])` — [`L911`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L911)
- `undo_transforms(aval: jax_core.AbstractValue, memory_transforms: Sequence[state_types.Transform])` — [`L737`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L737) — Extract the `Transform`s that reverse the `Transforms`s
- `unzip_dynamic_grid_bounds(grid_spec: GridSpec)` — [`L1412`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1412)
- `with_memory_space_constraint_abstract_eval(x, *, memory_space)` — [`L1721`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1721)
- `with_memory_space_constraint_impl(x, *, memory_space)` — [`L1715`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1715)
- `with_memory_space_constraint_lowering_rule(ctx, x, *, memory_space)` — [`L1727`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1727)
- `wrapped(f: Callable)` — [`L1557`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1557)

## Module values
- `BlockDim` — [`L451`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L451)
- `BlockSpecTree` — [`L734`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L734)
- `CoreType` — [`L1667`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1667)
- `Grid` — [`L80`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L80)
- `GridElement` — [`L75`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L75)
- `GridEnv` — [`L383`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L383)
- `GridMappingGrid` — [`L82`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L82)
- `GridName` — [`L76`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L76)
- `GridNames` — [`L77`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L77)
- `NamedGrid` — [`L78`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L78)
- `OriginStr` — [`L83`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L83)
- `SEMAPHORE_INTERPRET_DTYPE` — [`L89`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L89)
- `SEMAPHORE_MAX_VALUE` — [`L90`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L90)
- `ScratchShapeTree` — [`L1207`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1207)
- `StaticGrid` — [`L81`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L81)
- `TupleGrid` — [`L79`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L79)
- `_backend_lowering_rules` — [`L173`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L173)
- `_core_map_mesh_rules` — [`L1708`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1708)
- `_out_shape_to_aval_mapping` — [`L1878`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1878)
- `_pallas_tracing_env` — [`L365`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L365)
- `comms_effect` — [`L1608`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1608)
- `core_map_p` — [`L1500`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1500)
- `dynamic_grid_dim` — [`L71`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L71)
- `enable_debug_checks` — [`L149`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L149)
- `enable_poison_buffers` — [`L157`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L157)
- `index_map_grid_aval` — [`L1197`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1197)
- `kernel_local_effects` — [`L1614`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1614)
- `no_block_spec` — [`L729`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L729)
- `partial` — [`L74`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L74)
- `split_list` — [`L63`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L63)
- `squeezed` — [`L415`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L415)
- `unsafe_map` — [`L64`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L64)
- `unsafe_zip` — [`L65`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L65)
- `with_memory_space_constraint_p` — [`L1711`](../../../../../../../raw/code/jax/jax/_src/pallas/core.py#L1711)


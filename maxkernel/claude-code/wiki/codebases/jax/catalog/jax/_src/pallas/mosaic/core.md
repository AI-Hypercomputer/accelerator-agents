---
title: 'Module: jax/_src/pallas/mosaic/core.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/mosaic/core.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.mosaic.core`/
symbols:
  CoreType: CoreType#
  MemorySpace: MemorySpace#
  memory_space_to_tpu_memory_space: memory_space_to_tpu_memory_space().
  pass_scalars_as_refs: pass_scalars_as_refs().
  _tensorcore_mesh_discharge_rule: _tensorcore_mesh_discharge_rule().
  CompilerParams: CompilerParams#
  MemorySpace.HBM: MemorySpace#HBM.
  MemorySpace.SMEM: MemorySpace#SMEM.
  SemaphoreType: SemaphoreType#
  MemorySpace.VMEM: MemorySpace#VMEM.
  CoreType.SC_VECTOR_SUBCORE: CoreType#SC_VECTOR_SUBCORE.
  PrefetchScalarGridSpec.__init__: PrefetchScalarGridSpec#__init__().
  CoreType.TC: CoreType#TC.
  SemaphoreType.dtype: SemaphoreType#dtype().
  GridDimensionSemantics: GridDimensionSemantics#
  create_tensorcore_mesh: create_tensorcore_mesh().
  PrefetchScalarGridSpec: PrefetchScalarGridSpec#
  DimensionSemantics: DimensionSemantics.
  SemaphoreType.DMA: SemaphoreType#DMA.
  TensorCoreMesh: TensorCoreMesh#
  MemorySpace.SEMAPHORE: MemorySpace#SEMAPHORE.
  SemaphoreType.__call__: SemaphoreType#__call__().
  SemaphoreType.__matmul__: SemaphoreType#__matmul__().
  TensorCoreMesh.supported_memory_spaces: TensorCoreMesh#supported_memory_spaces().
  CoreType.SC_SCALAR_SUBCORE: CoreType#SC_SCALAR_SUBCORE.
  pass_scalars_as_refs.new_body: pass_scalars_as_refs().new_body().
  CompilerParams.__init__: CompilerParams#__init__().
  PrefetchScalarGridSpec._make_scalar_ref_aval: PrefetchScalarGridSpec#_make_scalar_ref_aval().
  CompilerParams.dimension_semantics: CompilerParams#dimension_semantics.
  pass_scalars_as_refs.allowed_aval: pass_scalars_as_refs().allowed_aval().
  _convert_semaphore_type_to_aval: _convert_semaphore_type_to_aval().
  MemorySpace.VMEM_SHARED: MemorySpace#VMEM_SHARED.
  AbstractSemaphore: AbstractSemaphore#
  TensorCoreMesh.shape: TensorCoreMesh#shape().
  MemorySpace.__getattr__: MemorySpace#__getattr__().
  MemorySpace.__call__: MemorySpace#__call__().
  MemorySpace.like: MemorySpace#like().
  dma_semaphore: dma_semaphore#
  TensorCoreMesh.dimension_semantics: TensorCoreMesh#dimension_semantics().
  PARALLEL: PARALLEL.
  SideEffectType: SideEffectType#
  MemorySpace.from_type: MemorySpace#from_type().
  ARBITRARY: ARBITRARY.
  SemaphoreType.get_array_aval: SemaphoreType#get_array_aval().
  TensorCoreMesh.devices: TensorCoreMesh#devices.
  TensorCoreMesh.axis_names: TensorCoreMesh#axis_names.
  CORE_PARALLEL: CORE_PARALLEL.
  SUBCORE_PARALLEL: SUBCORE_PARALLEL.
  MemoryRef.__matmul__: MemoryRef#__matmul__().
  MemorySpace.__matmul__: MemorySpace#__matmul__().
  SemaphoreType.get_ref_aval: SemaphoreType#get_ref_aval().
  TensorCoreMesh.__hash__: TensorCoreMesh#__hash__().
  TensorCoreMesh.core_type: TensorCoreMesh#core_type().
  TensorCoreMesh.default_memory_space: TensorCoreMesh#default_memory_space().
  TensorCoreMesh.check_is_compatible_with: TensorCoreMesh#check_is_compatible_with().
  no_block_spec: no_block_spec.
  MemoryRef: MemoryRef#
  GridDimensionSemantics.CORE_PARALLEL: GridDimensionSemantics#CORE_PARALLEL.
  AbstractSemaphore.sem_type: AbstractSemaphore#sem_type.
  CompilerParams.has_side_effects: CompilerParams#has_side_effects.
  CompilerParams.opt_level: CompilerParams#opt_level.
  DMASemaphore: DMASemaphore#
  GridDimensionSemantics.PARALLEL: GridDimensionSemantics#PARALLEL.
  GridDimensionSemantics.SUBCORE_PARALLEL: GridDimensionSemantics#SUBCORE_PARALLEL.
  LiteralDimensionSemantics: LiteralDimensionSemantics.
  CompilerParams.collective_id: CompilerParams#collective_id.
  CompilerParams.use_tc_tiling_on_sc: CompilerParams#use_tc_tiling_on_sc.
  _out_shape_to_aval_mapping: _out_shape_to_aval_mapping.
  DMASemaphore.type: DMASemaphore#type.
  TensorCoreMesh.discharges_effect: TensorCoreMesh#discharges_effect().
  unsafe_map: unsafe_map.
  unsafe_zip: unsafe_zip.
  GridDimensionSemantics.ARBITRARY: GridDimensionSemantics#ARBITRARY.
  CompilerParams.needs_layout_passes: CompilerParams#needs_layout_passes.
  MemorySpace.CMEM: MemorySpace#CMEM.
  SemaphoreType.BARRIER: SemaphoreType#BARRIER.
  SideEffectType.PURE: SideEffectType#PURE.
  SideEffectType.DATAFLOW_SIDE_EFFECTING: SideEffectType#DATAFLOW_SIDE_EFFECTING.
  SideEffectType.SIDE_EFFECTING: SideEffectType#SIDE_EFFECTING.
  CompilerParams.allow_input_fusion: CompilerParams#allow_input_fusion.
  CompilerParams.vmem_limit_bytes: CompilerParams#vmem_limit_bytes.
  CompilerParams.flags: CompilerParams#flags.
  CompilerParams.internal_scratch_in_bytes: CompilerParams#internal_scratch_in_bytes.
  CompilerParams.serialization_format: CompilerParams#serialization_format.
  CompilerParams.disable_bounds_checks: CompilerParams#disable_bounds_checks.
  CompilerParams.disable_semaphore_checks: CompilerParams#disable_semaphore_checks.
  CompilerParams.skip_device_barrier: CompilerParams#skip_device_barrier.
  CompilerParams.allow_collective_id_without_custom_barrier: CompilerParams#allow_collective_id_without_custom_barrier.
  CompilerParams.shape_invariant_numerics: CompilerParams#shape_invariant_numerics.
  CompilerParams.fuse_transposed_lhs_in_matmul: CompilerParams#fuse_transposed_lhs_in_matmul.
  CompilerParams.replace: CompilerParams#replace.
  SemaphoreType.REGULAR: SemaphoreType#REGULAR.
  PrefetchScalarGridSpec.num_scalar_prefetch: PrefetchScalarGridSpec#num_scalar_prefetch.
  TensorCore: TensorCore#
  CoreType.__str__: CoreType#__str__().
  CoreType.__repr__: CoreType#__repr__().
  MemorySpace.memory_kind: MemorySpace#memory_kind().
  MemorySpace.__str__: MemorySpace#__str__().
  MemorySpace.__repr__: MemorySpace#__repr__().
  DMASemaphore.name: DMASemaphore#name.
  TensorCore.id: TensorCore#id.
  TensorCoreMesh.__init__: TensorCoreMesh#__init__().
  TensorCoreMesh.tracing_context: TensorCoreMesh#tracing_context().
---
# Module: [`jax/_src/pallas/mosaic/core.py`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py)

## Classes
### `AbstractSemaphore`  ·  implements/extends AbstractValue
- def: [`jax/_src/pallas/mosaic/core.py:310`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L310)
- signature: `class AbstractSemaphore(jax_core.AbstractValue):`
- members:
  - `sem_type` — [`L311`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L311)
- uses (calls/refs, reference-scoped): [`AbstractValue`](../../core.md#AbstractValue), [`SemaphoreType`](core.md#SemaphoreType)
- used by: [`AbstractValue`](../../core.md#AbstractValue), [`_dimension_semantics`](lowering.md#MosaicGridMapping._dimension_semantics), [`aval_to_ir_type`](lowering.md#aval_to_ir_type), [`_alloc_value`](lowering.md#_alloc_value)

### `CompilerParams`
- def: [`jax/_src/pallas/mosaic/core.py:88`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L88)
- doc: Mosaic TPU compiler parameters.
- signature: `class CompilerParams:`
- members:
  - `allow_collective_id_without_custom_barrier` — [`L141`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L141)
  - `allow_input_fusion` — [`L131`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L131)
  - `collective_id` — [`L133`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L133)
  - `dimension_semantics` — [`L130`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L130)
  - `disable_bounds_checks` — [`L138`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L138)
  - `disable_semaphore_checks` — [`L139`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L139)
  - `flags` — [`L135`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L135)
  - `fuse_transposed_lhs_in_matmul` — [`L145`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L145)
  - `has_side_effects` — [`L134`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L134)
  - `internal_scratch_in_bytes` — [`L136`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L136)
  - `needs_layout_passes` — [`L144`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L144)
  - `opt_level` — [`L146`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L146)
  - `replace` — [`L211`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L211)
  - `serialization_format` — [`L137`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L137)
  - `shape_invariant_numerics` — [`L142`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L142)
  - `skip_device_barrier` — [`L140`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L140)
  - `use_tc_tiling_on_sc` — [`L143`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L143)
  - `vmem_limit_bytes` — [`L132`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L132)
- protocol/private: `__init__`[`L148`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L148)
- uses (calls/refs, reference-scoped): [`FrozenDict`](../../frozen_dict.md#FrozenDict), [`DimensionSemantics`](core.md#DimensionSemantics), [`OptLevel`](../../tpu_custom_call.md#OptLevel), [`SideEffectType`](core.md#SideEffectType)
- used by: [`_interpret_jaxpr`](interpret/interpret_pallas_call.md#_interpret_jaxpr), [`interpret_pallas_call`](interpret/interpret_pallas_call.md#interpret_pallas_call), [`_splash_attention_bwd_dkv`](../../../experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.md#_splash_attention_bwd_dkv), [`_splash_attention_bwd_dq`](../../../experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.md#_splash_attention_bwd_dq), [`_lower_to_custom_call`](pallas_call_registration.md#_lower_to_custom_call), [`gmm`](../../../experimental/pallas/ops/tpu/megablox/gmm.md#gmm), [`tgmm`](../../../experimental/pallas/ops/tpu/megablox/gmm.md#tgmm), [`paged_attention`](../../../experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.md#paged_attention), [`_flash_attention_bwd_dkv`](../../../experimental/pallas/ops/tpu/flash_attention.md#_flash_attention_bwd_dkv), [`_flash_attention_bwd_dq`](../../../experimental/pallas/ops/tpu/flash_attention.md#_flash_attention_bwd_dq), [`ragged_paged_attention`](../../../experimental/pallas/ops/tpu/ragged_paged_attention/kernel.md#ragged_paged_attention), [`_flash_attention_impl`](../../../experimental/pallas/ops/tpu/flash_attention.md#_flash_attention_impl), [`mpmd_map_tpu_lowering_rule`](pallas_call_registration.md#mpmd_map_tpu_lowering_rule), [`pallas_call_tpu_lowering_rule`](pallas_call_registration.md#pallas_call_tpu_lowering_rule), [`matmul`](../../../experimental/pallas/ops/tpu/matmul.md#matmul), [`_tensorcore_mesh_discharge_rule`](core.md#_tensorcore_mesh_discharge_rule), [`_get_randomized_grid_coordinates`](interpret/interpret_pallas_call.md#_get_randomized_grid_coordinates), [`ag_local`](../../../experimental/pallas/ops/tpu/all_gather.md#all_gather.ag_local), [`_resolve_tiling`](pallas_call_registration.md#_resolve_tiling), [`_scalar_subcore_mesh_discharge_rule`](sc_core.md#_scalar_subcore_mesh_discharge_rule), [`_vector_subcore_mesh_discharge_rule`](sc_core.md#_vector_subcore_mesh_discharge_rule), [`_get_parallel_dim_semantics`](interpret/interpret_pallas_call.md#_get_parallel_dim_semantics), [`mosaic_params`](interpret/interpret_pallas_call.md#InterpretContext.mosaic_params)

### `CoreType`  ·  implements/extends Enum
- def: [`jax/_src/pallas/mosaic/core.py:47`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L47)
- signature: `class CoreType(enum.Enum):`
- members:
  - `SC_SCALAR_SUBCORE` — [`L49`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L49)
  - `SC_VECTOR_SUBCORE` — [`L50`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L50)
  - `TC` — [`L48`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L48)
- protocol/private: `__repr__`[`L55`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L55), `__str__`[`L52`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L52)
- used by: [`lower_jaxpr_into_pipelined_module`](lowering.md#lower_jaxpr_into_pipelined_module), [`register_lowering_rule`](lowering.md#register_lowering_rule), [`_lower_to_custom_call`](pallas_call_registration.md#_lower_to_custom_call), [`_check_block_mappings`](lowering.md#_check_block_mappings), [`lower_jaxpr_into_unpipelined_module`](lowering.md#lower_jaxpr_into_unpipelined_module), [`aval_to_ir_type`](lowering.md#aval_to_ir_type), [`_get_memory_space_from_aval`](pallas_call_registration.md#_get_memory_space_from_aval), [`create`](pipeline.md#BufferedRef.create), [`memory_space_to_tpu_memory_space`](core.md#memory_space_to_tpu_memory_space), [`mpmd_map_tpu_lowering_rule`](pallas_call_registration.md#mpmd_map_tpu_lowering_rule), [`_extract_indirect_offsets_from_indices`](sc_lowering.md#_extract_indirect_offsets_from_indices), [`pallas_call_tpu_lowering_rule`](pallas_call_registration.md#pallas_call_tpu_lowering_rule), [`_rewrite_jaxpr_for_lowering`](pallas_call_registration.md#_rewrite_jaxpr_for_lowering), [`__init__`](lowering.md#MosaicGridMapping.__init__), [`_convert_element_type_lowering_rule`](lowering.md#_convert_element_type_lowering_rule), [`jax_fn`](lowering.md#_device_id_to_logical.jax_fn), [`_run_scoped_lowering_rule`](lowering.md#_run_scoped_lowering_rule), [`register_lowering_rule`](sc_lowering.md#register_lowering_rule), [`_scan_lowering_rule`](lowering.md#_scan_lowering_rule), [`_semaphore_signal_lowering_rule`](lowering.md#_semaphore_signal_lowering_rule), [`new_body`](pallas_call_registration.md#_rewrite_jaxpr_for_lowering.new_body), [`_div_lowering_rule`](lowering.md#_div_lowering_rule), [`_gather_lowering_rule`](lowering.md#_gather_lowering_rule), [`_mul_lowering_rule`](lowering.md#_mul_lowering_rule), [`_rem_lowering_rule`](lowering.md#_rem_lowering_rule), [`_cond_lowering_rule`](lowering.md#_cond_lowering_rule), [`_sub_lowering_rule`](lowering.md#_sub_lowering_rule), [`from_mosaic_grid_mapping`](lowering.md#PipelinedLoweringContext.from_mosaic_grid_mapping), [`_add_lowering_rule`](lowering.md#_add_lowering_rule), [`_max_lowering_rule`](lowering.md#_max_lowering_rule), [`_min_lowering_rule`](lowering.md#_min_lowering_rule), [`_memory_space_to_mosaic_attribute`](lowering.md#_memory_space_to_mosaic_attribute), [`_semaphore_read_lowering_rule`](lowering.md#_semaphore_read_lowering_rule), [`_unstack_lowering_rule`](lowering.md#_unstack_lowering_rule), [`prim`](lowering.md#prim), [`lower_jaxpr_to_transform_func`](lowering.md#lower_jaxpr_to_transform_func), [`_axis_index_rule`](lowering.md#_axis_index_rule), [`_broadcast_in_dim_lowering_rule`](lowering.md#_broadcast_in_dim_lowering_rule), [`_select_n_lowering_rule`](lowering.md#_select_n_lowering_rule), [`_while_lowering_rule`](lowering.md#_while_lowering_rule)  (+52 more)

### `DMASemaphore`  ·  implements/extends AbstractSemaphoreTy
- def: [`jax/_src/pallas/mosaic/core.py:273`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L273)
- signature: `class DMASemaphore(pallas_core.AbstractSemaphoreTy):`
- members:
  - `name` — [`L275`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L275)
  - `type` — [`L274`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L274)
- uses (calls/refs, reference-scoped): [`AbstractSemaphoreTy`](../core.md#AbstractSemaphoreTy), [`dma_semaphore`](core.md#dma_semaphore)
- used by: [`dtype`](core.md#SemaphoreType.dtype), [`AbstractSemaphoreTy`](../core.md#AbstractSemaphoreTy)

### `GridDimensionSemantics`  ·  implements/extends Enum
- def: [`jax/_src/pallas/mosaic/core.py:59`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L59)
- signature: `class GridDimensionSemantics(enum.Enum):`
- members:
  - `ARBITRARY` — [`L63`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L63)
  - `CORE_PARALLEL` — [`L61`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L61)
  - `PARALLEL` — [`L60`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L60)
  - `SUBCORE_PARALLEL` — [`L62`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L62)
- used by: [`DimensionSemantics`](core.md#DimensionSemantics), [`_canonicalize_dimension_semantic`](lowering.md#_canonicalize_dimension_semantic), [`dimension_semantics`](sc_core.md#VectorSubcoreMesh.dimension_semantics), [`dimension_semantics`](sc_core.md#ScalarSubcoreMesh.dimension_semantics), [`dimension_semantics`](core.md#TensorCoreMesh.dimension_semantics), [`PARALLEL`](core.md#PARALLEL), [`ARBITRARY`](core.md#ARBITRARY), [`GridDimensionSemantics`](pipeline.md#GridDimensionSemantics), [`ARBITRARY`](../../../experimental/pallas/tpu.md#ARBITRARY), [`CORE_PARALLEL`](core.md#CORE_PARALLEL), [`CORE_PARALLEL`](../../../experimental/pallas/tpu.md#CORE_PARALLEL), [`PARALLEL`](../../../experimental/pallas/tpu.md#PARALLEL), [`SUBCORE_PARALLEL`](core.md#SUBCORE_PARALLEL), [`SUBCORE_PARALLEL`](../../../experimental/pallas/tpu.md#SUBCORE_PARALLEL)

### `MemoryRef`  ·  implements/extends MemoryRef
- def: [`jax/_src/pallas/mosaic/core.py:214`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L214)
- signature: `class MemoryRef(pallas_core.MemoryRef):`
- protocol/private: `__matmul__`[`L216`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L216)
- uses (calls/refs, reference-scoped): [`Mesh`](../core.md#Mesh), [`MemoryRef`](../core.md#MemoryRef), [`memory_space`](../core.md#MemoryRef.memory_space)
- used by: [`MemoryRef`](../core.md#MemoryRef), [`__call__`](core.md#SemaphoreType.__call__), [`from_type`](core.md#MemorySpace.from_type)

### `MemorySpace`  ·  implements/extends Enum
- def: [`jax/_src/pallas/mosaic/core.py:222`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L222)
- signature: `class MemorySpace(enum.Enum):`
- members:
  - `from_type(self, ty)` — [`L251`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L251)
  - `like(self, shape_dtype_like)` — [`L258`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L258)
  - `memory_kind(self)` — [`L231`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L231)
  - `CMEM` — [`L226`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L226)
  - `HBM` — [`L228`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L228)
  - `SEMAPHORE` — [`L227`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L227)
  - `SMEM` — [`L225`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L225)
  - `VMEM` — [`L223`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L223)
  - `VMEM_SHARED` — [`L224`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L224)
- protocol/private: `__call__`[`L254`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L254), `__getattr__`[`L234`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L234), `__matmul__`[`L263`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L263), `__repr__`[`L248`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L248), `__str__`[`L245`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L245)
- uses (calls/refs, reference-scoped): [`ShapedArray`](../../core.md#ShapedArray), [`typeof`](../../core.md#typeof), [`AbstractValue`](../../core.md#AbstractValue), [`dtype`](../../../numpy/__init__.pyi.md#dtype), [`MemorySpace`](../../core.md#MemorySpace), [`Mesh`](../core.md#Mesh), [`CoreMemorySpace`](../core.md#CoreMemorySpace), [`Host`](../../core.md#MemorySpace.Host), [`warn`](../../deprecations.md#warn), [`MemoryRef`](core.md#MemoryRef)
- used by: [`_interpret_jaxpr`](interpret/interpret_pallas_call.md#_interpret_jaxpr), [`interpret_pallas_call`](interpret/interpret_pallas_call.md#interpret_pallas_call), [`lower_jaxpr_into_pipelined_module`](lowering.md#lower_jaxpr_into_pipelined_module), [`_check_block_mappings`](lowering.md#_check_block_mappings), [`wrapper`](../mpmd.md#_mpmd_map.wrapper), [`_load_lowering_rule`](sc_lowering.md#_load_lowering_rule), [`_store_lowering_rule`](sc_lowering.md#_store_lowering_rule), [`lower_jaxpr_into_unpipelined_module`](lowering.md#lower_jaxpr_into_unpipelined_module), [`_get_memory_space_from_aval`](pallas_call_registration.md#_get_memory_space_from_aval), [`create`](pipeline.md#BufferedRef.create), [`memory_space_to_tpu_memory_space`](core.md#memory_space_to_tpu_memory_space), [`_extract_indirect_offsets_from_indices`](sc_lowering.md#_extract_indirect_offsets_from_indices), [`_rewrite_jaxpr_for_lowering`](pallas_call_registration.md#_rewrite_jaxpr_for_lowering), [`pass_scalars_as_refs`](core.md#pass_scalars_as_refs), [`_allocate_buffer`](interpret/interpret_pallas_call.md#_allocate_buffer), [`_store_slice_to_kernel_input`](interpret/interpret_pallas_call.md#interpret_pallas_call._execute_grid_for_core._body._store_slice_to_kernel_input), [`new_body`](pallas_call_registration.md#_rewrite_jaxpr_for_lowering.new_body), [`_scatter_lowering_rule`](sc_primitives.md#_scatter_lowering_rule), [`_gather_lowering_rule`](sc_primitives.md#_gather_lowering_rule), [`_memory_space_to_mosaic_attribute`](lowering.md#_memory_space_to_mosaic_attribute), [`_tensorcore_mesh_discharge_rule`](core.md#_tensorcore_mesh_discharge_rule), [`TPU_MEMORY_SPACE_IDXS`](interpret/interpret_pallas_call.md#TPU_MEMORY_SPACE_IDXS.TPU_MEMORY_SPACE_IDXS), [`with_memory_space_constraint`](primitives.md#with_memory_space_constraint), [`_prepare_dma_refs`](sc_lowering.md#_prepare_dma_refs), [`host_idx`](interpret/interpret_pallas_call.md#host_idx), [`VMEM`](../../../experimental/pallas/tpu.md#VMEM), [`_remove_memory_space_abstract_eval`](interpret/interpret_pallas_call.md#_remove_memory_space_abstract_eval), [`all_gather`](../../../experimental/pallas/ops/tpu/all_gather.md#all_gather), [`MemorySpace`](sc_lowering.md#MemorySpace), [`_store_to_output_buffer`](interpret/interpret_pallas_call.md#interpret_pallas_call._execute_grid_for_core._body._store_to_output_buffer), [`TPUMemorySpace`](lowering.md#TPUMemorySpace), [`__call__`](core.md#SemaphoreType.__call__), [`__matmul__`](core.md#SemaphoreType.__matmul__), [`_fetch_and_add_abstract_eval`](sc_primitives.md#_fetch_and_add_abstract_eval), [`_get_barrier_semaphore_abstract_eval`](primitives.md#_get_barrier_semaphore_abstract_eval), [`supported_memory_spaces`](core.md#TensorCoreMesh.supported_memory_spaces), [`supported_memory_spaces`](sc_core.md#VectorSubcoreMesh.supported_memory_spaces), [`AnyMemorySpace`](lowering.md#AnyMemorySpace), [`TPU_MEMORY_SPACE_NAMES`](interpret/interpret_pallas_call.md#TPU_MEMORY_SPACE_NAMES), [`SMEM`](../../../experimental/pallas/tpu.md#SMEM)  (+16 more)

### `PrefetchScalarGridSpec`  ·  implements/extends GridSpec
- def: [`jax/_src/pallas/mosaic/core.py:315`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L315)
- signature: `class PrefetchScalarGridSpec(pallas_core.GridSpec):`
- members:
  - `num_scalar_prefetch` — [`L316`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L316)
- protocol/private: `__init__`[`L318`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L318), `_make_scalar_ref_aval`[`L330`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L330)
- uses (calls/refs, reference-scoped): [`AbstractRef`](../../state/types.md#AbstractRef), [`ShapedArray`](../../core.md#ShapedArray), [`MemorySpace`](core.md#MemorySpace), [`__init__`](../core.md#GridSpec.__init__), [`SMEM`](core.md#MemorySpace.SMEM), [`GridSpec`](../core.md#GridSpec), [`ScratchShapeTree`](../core.md#ScratchShapeTree), [`BlockSpecTree`](../core.md#BlockSpecTree), [`Grid`](../core.md#Grid), [`scratch_shapes`](../core.md#GridSpec.scratch_shapes), [`no_block_spec`](core.md#no_block_spec)
- used by: [`_splash_attention_bwd_dkv`](../../../experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.md#_splash_attention_bwd_dkv), [`_splash_attention_bwd_dq`](../../../experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.md#_splash_attention_bwd_dq), [`gmm`](../../../experimental/pallas/ops/tpu/megablox/gmm.md#gmm), [`tgmm`](../../../experimental/pallas/ops/tpu/megablox/gmm.md#tgmm), [`paged_attention`](../../../experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.md#paged_attention), [`_flash_attention_bwd_dkv`](../../../experimental/pallas/ops/tpu/flash_attention.md#_flash_attention_bwd_dkv), [`_flash_attention_bwd_dq`](../../../experimental/pallas/ops/tpu/flash_attention.md#_flash_attention_bwd_dq), [`ragged_paged_attention`](../../../experimental/pallas/ops/tpu/ragged_paged_attention/kernel.md#ragged_paged_attention), [`_flash_attention_impl`](../../../experimental/pallas/ops/tpu/flash_attention.md#_flash_attention_impl), [`matmul`](../../../experimental/pallas/ops/tpu/matmul.md#matmul), [`ag_local`](../../../experimental/pallas/ops/tpu/all_gather.md#all_gather.ag_local), [`GridSpec`](../core.md#GridSpec), [`_make_scalar_ref_aval`](../core.md#GridSpec._make_scalar_ref_aval)

### `SemaphoreType`  ·  implements/extends Enum
- def: [`jax/_src/pallas/mosaic/core.py:278`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L278)
- signature: `class SemaphoreType(enum.Enum):`
- members:
  - `dtype(self)` — [`L284`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L284)
  - `get_array_aval(self)` — [`L302`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L302)
  - `get_ref_aval(self)` — [`L305`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L305)
  - `BARRIER` — [`L281`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L281)
  - `DMA` — [`L280`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L280)
  - `REGULAR` — [`L279`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L279)
- protocol/private: `__call__`[`L292`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L292), `__matmul__`[`L295`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L295)
- uses (calls/refs, reference-scoped): [`AbstractRef`](../../state/types.md#AbstractRef), [`ShapedArray`](../../core.md#ShapedArray), [`MemorySpace`](core.md#MemorySpace), [`Mesh`](../core.md#Mesh), [`CoreMemorySpace`](../core.md#CoreMemorySpace), [`get_array_aval`](../core.md#MemoryRef.get_array_aval), [`get_ref_aval`](../core.md#MemoryRef.get_ref_aval), [`SEMAPHORE`](core.md#MemorySpace.SEMAPHORE), [`BarrierSemaphore`](../core.md#BarrierSemaphore), [`MemoryRef`](core.md#MemoryRef), [`Semaphore`](../core.md#Semaphore), [`DMASemaphore`](core.md#DMASemaphore)
- used by: [`paged_attention`](../../../experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.md#paged_attention), [`aval_to_ir_type`](lowering.md#aval_to_ir_type), [`ragged_paged_attention`](../../../experimental/pallas/ops/tpu/ragged_paged_attention/kernel.md#ragged_paged_attention), [`create`](pipeline.md#BufferedRef.create), [`ag_local`](../../../experimental/pallas/ops/tpu/all_gather.md#all_gather.ag_local), [`sync_copy`](helpers.md#sync_copy), [`_convert_semaphore_type_to_aval`](core.md#_convert_semaphore_type_to_aval), [`AbstractSemaphore`](core.md#AbstractSemaphore), [`SemaphoreType`](pipeline.md#SemaphoreType)

### `SideEffectType`  ·  implements/extends Enum
- def: [`jax/_src/pallas/mosaic/core.py:78`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L78)
- signature: `class SideEffectType(enum.Enum):`
- members:
  - `DATAFLOW_SIDE_EFFECTING` — [`L82`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L82)
  - `PURE` — [`L80`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L80)
  - `SIDE_EFFECTING` — [`L84`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L84)
- used by: [`_resolve_side_effect_type`](pallas_call_registration.md#_resolve_side_effect_type), [`__init__`](core.md#CompilerParams.__init__), [`has_side_effects`](core.md#CompilerParams.has_side_effects)

### `TensorCore`
- def: [`jax/_src/pallas/mosaic/core.py:337`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L337)
- signature: `class TensorCore:`
- members:
  - `id` — [`L338`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L338)
- used by: [`create_tensorcore_mesh`](core.md#create_tensorcore_mesh)

### `TensorCoreMesh`  ·  implements/extends Mesh
- def: [`jax/_src/pallas/mosaic/core.py:342`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L342)
- doc: A mesh of TensorCores.
- signature: `class TensorCoreMesh(pallas_core.Mesh):`
- members:
  - `check_is_compatible_with(self, other_mesh)` — [`L379`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L379)
  - `core_type(self)` — [`L360`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L360)
  - `default_memory_space(self)` — [`L364`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L364)
  - `dimension_semantics(self)` — [`L372`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L372)
  - `discharges_effect(self, effect: jax_core.Effect)` — [`L375`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L375)
  - `shape(self)` — [`L368`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L368)
  - `supported_memory_spaces(self)` — [`L386`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L386)
  - `tracing_context(self)` — [`L395`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L395)
  - `axis_names` — [`L346`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L346)
  - `devices` — [`L345`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L345)
- protocol/private: `__hash__`[`L354`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L354), `__init__`[`L348`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L348)
- uses (calls/refs, reference-scoped): [`CoreType`](core.md#CoreType), [`MemorySpace`](core.md#MemorySpace), [`MemorySpace`](../core.md#MemorySpace), [`Mesh`](../core.md#Mesh), [`SMEM`](core.md#MemorySpace.SMEM), [`Effect`](../../core.md#Effect), [`VMEM`](core.md#MemorySpace.VMEM), [`ANY`](../core.md#MemorySpace.ANY), [`TC`](core.md#CoreType.TC), [`check_is_compatible_with`](../core.md#Mesh.check_is_compatible_with), [`GridDimensionSemantics`](core.md#GridDimensionSemantics), [`DimensionSemantics`](core.md#DimensionSemantics), [`SEMAPHORE`](core.md#MemorySpace.SEMAPHORE), [`PARALLEL`](core.md#GridDimensionSemantics.PARALLEL), [`CMEM`](core.md#MemorySpace.CMEM)
- used by: [`interpret_pallas_call`](interpret/interpret_pallas_call.md#interpret_pallas_call), [`Mesh`](../core.md#Mesh), [`core_type`](../core.md#Mesh.core_type), [`_tensorcore_mesh_discharge_rule`](core.md#_tensorcore_mesh_discharge_rule), [`_get_mesh_shape_and_semantics`](lowering.md#_get_mesh_shape_and_semantics), [`check_is_compatible_with`](sc_core.md#VectorSubcoreMesh.check_is_compatible_with), [`check_is_compatible_with`](sc_core.md#ScalarSubcoreMesh.check_is_compatible_with), [`shape`](../core.md#Mesh.shape), [`check_is_compatible_with`](../core.md#Mesh.check_is_compatible_with), [`create_tensorcore_mesh`](core.md#create_tensorcore_mesh), [`default_memory_space`](../core.md#Mesh.default_memory_space), [`discharges_effect`](../core.md#Mesh.discharges_effect), [`supported_memory_spaces`](../core.md#Mesh.supported_memory_spaces), [`tracing_context`](../core.md#Mesh.tracing_context)

### `dma_semaphore`  ·  implements/extends semaphore_dtype
- def: [`jax/_src/pallas/mosaic/core.py:269`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L269)
- signature: `class dma_semaphore(pallas_core.semaphore_dtype):`
- uses (calls/refs, reference-scoped): [`semaphore_dtype`](../core.md#semaphore_dtype)
- used by: [`_dtype_to_ir_type`](lowering.md#_dtype_to_ir_type), [`_semaphore_read_lowering_rule`](lowering.md#_semaphore_read_lowering_rule), [`_dma_wait_abstract_eval`](primitives.md#_dma_wait_abstract_eval), [`semaphore_dtype`](../core.md#semaphore_dtype), [`type`](core.md#DMASemaphore.type)

## Functions
- `_convert_semaphore_type_to_aval(out_shape: SemaphoreType)` — [`L597`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L597)
- `_tensorcore_mesh_discharge_rule(in_avals, out_avals, *args, mesh, jaxpr, compiler_params: Any | None, interpret: Any, debug: bool, cost_estimate: pallas_core.CostEstimate | None, name: str, metadata: FrozenDict[str, str] | None)` — [`L534`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L534)
- `allowed_aval(aval)` — [`L435`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L435)
- `create_tensorcore_mesh(axis_name: str, devices: Sequence[jax.Device] | None = None, num_cores: int | None = None)` — [`L399`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L399)
- `memory_space_to_tpu_memory_space(memory_space: (MemorySpace | pallas_core.MemorySpace | pallas_core.CoreMemorySpace | jax_core.MemorySpace | None), core_type: CoreType)` — [`L608`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L608)
- `new_body(*args)` — [`L476`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L476)
- `pass_scalars_as_refs(jaxpr: jax_core.Jaxpr, args: Sequence[Any], in_avals: Sequence[jax_core.AbstractValue], out_avals: Sequence[jax_core.AbstractValue], mesh, copy_to_smem: bool = False)` — [`L420`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L420) — Rewrites a jaxpr to pass scalars as refs instead of values.

## Module values
- `ARBITRARY` — [`L69`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L69)
- `CORE_PARALLEL` — [`L67`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L67)
- `DimensionSemantics` — [`L75`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L75)
- `LiteralDimensionSemantics` — [`L72`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L72)
- `PARALLEL` — [`L66`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L66)
- `SUBCORE_PARALLEL` — [`L68`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L68)
- `_out_shape_to_aval_mapping` — [`L44`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L44)
- `no_block_spec` — [`L43`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L43)
- `unsafe_map` — [`L40`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L40)
- `unsafe_zip` — [`L41`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/core.py#L41)


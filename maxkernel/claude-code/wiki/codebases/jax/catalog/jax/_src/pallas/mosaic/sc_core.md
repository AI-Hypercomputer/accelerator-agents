---
title: 'Module: jax/_src/pallas/mosaic/sc_core.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/mosaic/sc_core.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.mosaic.sc_core`/
symbols:
  VectorSubcoreMesh.check_is_compatible_with: VectorSubcoreMesh#check_is_compatible_with().
  ScalarSubcoreMesh.check_is_compatible_with: ScalarSubcoreMesh#check_is_compatible_with().
  get_sparse_core_info: get_sparse_core_info().
  _scalar_subcore_mesh_discharge_rule: _scalar_subcore_mesh_discharge_rule().
  _vector_subcore_mesh_discharge_rule: _vector_subcore_mesh_discharge_rule().
  VectorSubcoreMesh.__post_init__: VectorSubcoreMesh#__post_init__().
  VectorSubcoreMesh.supported_memory_spaces: VectorSubcoreMesh#supported_memory_spaces().
  ScalarSubcoreMesh.num_cores: ScalarSubcoreMesh#num_cores.
  VectorSubcoreMesh.num_cores: VectorSubcoreMesh#num_cores.
  VectorSubcoreMesh.shape: VectorSubcoreMesh#shape().
  supported_shapes: supported_shapes().
  VectorSubcoreMesh: VectorSubcoreMesh#
  ScalarSubcoreMesh.supported_memory_spaces: ScalarSubcoreMesh#supported_memory_spaces().
  VectorSubcoreMesh.dimension_semantics: VectorSubcoreMesh#dimension_semantics().
  ScalarSubcoreMesh: ScalarSubcoreMesh#
  Indices: Indices#
  VectorSubcoreMesh.num_subcores: VectorSubcoreMesh#num_subcores.
  ScalarSubcoreMesh.dimension_semantics: ScalarSubcoreMesh#dimension_semantics().
  ScalarSubcoreMesh.__post_init__: ScalarSubcoreMesh#__post_init__().
  Indices.pretty_print: Indices#pretty_print().
  ScalarSubcoreMesh.shape: ScalarSubcoreMesh#shape().
  ScalarSubcoreMesh.axis_name: ScalarSubcoreMesh#axis_name.
  VectorSubcoreMesh.core_axis_name: VectorSubcoreMesh#core_axis_name.
  Indices.ignored_value: Indices#ignored_value.
  ScalarSubcoreMesh.core_type: ScalarSubcoreMesh#core_type().
  ScalarSubcoreMesh.default_memory_space: ScalarSubcoreMesh#default_memory_space().
  VectorSubcoreMesh.core_type: VectorSubcoreMesh#core_type().
  VectorSubcoreMesh.default_memory_space: VectorSubcoreMesh#default_memory_space().
  VectorSubcoreMesh.size: VectorSubcoreMesh#size().
  VectorSubcoreMesh.subcore_axis_name: VectorSubcoreMesh#subcore_axis_name.
  Indices.values: Indices#values.
  ScalarSubcoreMesh.size: ScalarSubcoreMesh#size().
  ScalarSubcoreMesh.discharges_effect: ScalarSubcoreMesh#discharges_effect().
  ScalarSubcoreMesh.tracing_context: ScalarSubcoreMesh#tracing_context().
  VectorSubcoreMesh.discharges_effect: VectorSubcoreMesh#discharges_effect().
  VectorSubcoreMesh.tracing_context: VectorSubcoreMesh#tracing_context().
---
# Module: [`jax/_src/pallas/mosaic/sc_core.py`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_core.py)

## Classes
### `Indices`
- def: [`jax/_src/pallas/mosaic/sc_core.py:315`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_core.py#L315)
- doc: Indices for a gather or a scatter on SparseCore.
- signature: `class Indices:`
- members:
  - `pretty_print(self, context: jax_core.JaxprPpContext, *, print_dtype: bool = True)` — [`L328`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_core.py#L328)
  - `ignored_value` — [`L324`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_core.py#L324)
  - `values` — [`L323`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_core.py#L323)
- uses (calls/refs, reference-scoped): [`JaxprPpContext`](../../core.md#JaxprPpContext), [`register_dataclass`](../../tree_util.md#register_dataclass)
- used by: [`_extract_indirect_offsets_from_indices`](sc_lowering.md#_extract_indirect_offsets_from_indices), [`_dma_start_lowering_rule`](sc_lowering.md#_dma_start_lowering_rule), [`_extract_indirect_offsets`](sc_lowering.md#_extract_indirect_offsets)

### `ScalarSubcoreMesh`  ·  implements/extends Mesh
- def: [`jax/_src/pallas/mosaic/sc_core.py:46`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_core.py#L46)
- signature: `class ScalarSubcoreMesh(pallas_core.Mesh):`
- members:
  - `check_is_compatible_with(self, other_mesh)` — [`L84`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_core.py#L84)
  - `core_type(self)` — [`L61`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_core.py#L61)
  - `default_memory_space(self)` — [`L65`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_core.py#L65)
  - `dimension_semantics(self)` — [`L77`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_core.py#L77)
  - `discharges_effect(self, effect)` — [`L80`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_core.py#L80)
  - `shape(self)` — [`L69`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_core.py#L69)
  - `size(self)` — [`L73`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_core.py#L73)
  - `supported_memory_spaces(self)` — [`L105`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_core.py#L105)
  - `tracing_context(self)` — [`L113`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_core.py#L113)
  - `axis_name` — [`L47`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_core.py#L47)
  - `num_cores` — [`L48`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_core.py#L48)
- protocol/private: `__post_init__`[`L52`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_core.py#L52)
- uses (calls/refs, reference-scoped): [`CoreType`](core.md#CoreType), [`MemorySpace`](core.md#MemorySpace), [`Mesh`](../core.md#Mesh), [`HBM`](core.md#MemorySpace.HBM), [`SMEM`](core.md#MemorySpace.SMEM), [`get_sparse_core_info`](sc_core.md#get_sparse_core_info), [`check_is_compatible_with`](../core.md#Mesh.check_is_compatible_with), [`GridDimensionSemantics`](core.md#GridDimensionSemantics), [`DimensionSemantics`](core.md#DimensionSemantics), [`TensorCoreMesh`](core.md#TensorCoreMesh), [`SEMAPHORE`](core.md#MemorySpace.SEMAPHORE), [`SC_SCALAR_SUBCORE`](core.md#CoreType.SC_SCALAR_SUBCORE), [`num_cores`](sc_core.md#VectorSubcoreMesh.num_cores), [`VectorSubcoreMesh`](sc_core.md#VectorSubcoreMesh), [`num_cores`](tpu_info.md#SparseCoreInfo.num_cores), [`VMEM_SHARED`](core.md#MemorySpace.VMEM_SHARED), [`axis_names`](core.md#TensorCoreMesh.axis_names), [`core_axis_name`](sc_core.md#VectorSubcoreMesh.core_axis_name), [`CORE_PARALLEL`](core.md#GridDimensionSemantics.CORE_PARALLEL)
- used by: [`Mesh`](../core.md#Mesh), [`jax_fn`](lowering.md#_device_id_to_logical.jax_fn), [`core_type`](../core.md#Mesh.core_type), [`_get_mesh_shape_and_semantics`](lowering.md#_get_mesh_shape_and_semantics), [`check_is_compatible_with`](sc_core.md#VectorSubcoreMesh.check_is_compatible_with), [`shape`](../core.md#Mesh.shape), [`check_is_compatible_with`](../core.md#Mesh.check_is_compatible_with), [`_scalar_subcore_mesh_discharge_rule`](sc_core.md#_scalar_subcore_mesh_discharge_rule), [`default_memory_space`](../core.md#Mesh.default_memory_space), [`discharges_effect`](../core.md#Mesh.discharges_effect), [`supported_memory_spaces`](../core.md#Mesh.supported_memory_spaces), [`tracing_context`](../core.md#Mesh.tracing_context)

### `VectorSubcoreMesh`
- def: [`jax/_src/pallas/mosaic/sc_core.py:169`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_core.py#L169)
- members:
  - `check_is_compatible_with(self, other_mesh)` — [`L222`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_core.py#L222)
  - `core_type(self)` — [`L193`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_core.py#L193)
  - `default_memory_space(self)` — [`L197`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_core.py#L197)
  - `dimension_semantics(self)` — [`L212`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_core.py#L212)
  - `discharges_effect(self, effect)` — [`L218`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_core.py#L218)
  - `shape(self)` — [`L201`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_core.py#L201)
  - `size(self)` — [`L208`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_core.py#L208)
  - `supported_memory_spaces(self)` — [`L248`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_core.py#L248)
  - `tracing_context(self)` — [`L257`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_core.py#L257)
  - `core_axis_name` — [`L170`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_core.py#L170)
  - `num_cores` — [`L172`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_core.py#L172)
  - `num_subcores` — [`L175`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_core.py#L175)
  - `subcore_axis_name` — [`L171`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_core.py#L171)
- protocol/private: `__post_init__`[`L179`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_core.py#L179)
- uses (calls/refs, reference-scoped): [`CoreType`](core.md#CoreType), [`MemorySpace`](core.md#MemorySpace), [`Mesh`](../core.md#Mesh), [`HBM`](core.md#MemorySpace.HBM), [`SMEM`](core.md#MemorySpace.SMEM), [`get_sparse_core_info`](sc_core.md#get_sparse_core_info), [`VMEM`](core.md#MemorySpace.VMEM), [`SC_VECTOR_SUBCORE`](core.md#CoreType.SC_VECTOR_SUBCORE), [`check_is_compatible_with`](../core.md#Mesh.check_is_compatible_with), [`GridDimensionSemantics`](core.md#GridDimensionSemantics), [`DimensionSemantics`](core.md#DimensionSemantics), [`TensorCoreMesh`](core.md#TensorCoreMesh), [`SEMAPHORE`](core.md#MemorySpace.SEMAPHORE), [`num_cores`](sc_core.md#ScalarSubcoreMesh.num_cores), [`num_cores`](tpu_info.md#SparseCoreInfo.num_cores), [`ScalarSubcoreMesh`](sc_core.md#ScalarSubcoreMesh), [`VMEM_SHARED`](core.md#MemorySpace.VMEM_SHARED), [`num_subcores`](tpu_info.md#SparseCoreInfo.num_subcores), [`axis_name`](sc_core.md#ScalarSubcoreMesh.axis_name), [`axis_names`](core.md#TensorCoreMesh.axis_names), [`CORE_PARALLEL`](core.md#GridDimensionSemantics.CORE_PARALLEL), [`SUBCORE_PARALLEL`](core.md#GridDimensionSemantics.SUBCORE_PARALLEL)
- used by: [`jax_fn`](lowering.md#_device_id_to_logical.jax_fn), [`core_type`](../core.md#Mesh.core_type), [`_scatter_lowering_rule`](sc_primitives.md#_scatter_lowering_rule), [`_get_mesh_shape_and_semantics`](lowering.md#_get_mesh_shape_and_semantics), [`check_is_compatible_with`](sc_core.md#ScalarSubcoreMesh.check_is_compatible_with), [`shape`](../core.md#Mesh.shape), [`check_is_compatible_with`](../core.md#Mesh.check_is_compatible_with), [`default_memory_space`](../core.md#Mesh.default_memory_space), [`discharges_effect`](../core.md#Mesh.discharges_effect), [`_vector_subcore_mesh_discharge_rule`](sc_core.md#_vector_subcore_mesh_discharge_rule), [`supported_memory_spaces`](../core.md#Mesh.supported_memory_spaces), [`tracing_context`](../core.md#Mesh.tracing_context)

## Functions
- `_scalar_subcore_mesh_discharge_rule(in_avals, out_avals, *args, mesh, jaxpr, compiler_params, interpret, debug, cost_estimate, name, metadata)` — [`L117`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_core.py#L117)
- `_vector_subcore_mesh_discharge_rule(in_avals, out_avals, *args, mesh, jaxpr, compiler_params, interpret, debug, cost_estimate, name, metadata)` — [`L260`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_core.py#L260)
- `get_sparse_core_info()` — [`L33`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_core.py#L33) — Returns the SparseCore information for the current device.
- `supported_shapes(dtype: jax.typing.DTypeLike)` — [`L300`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_core.py#L300) — Returns all supported array shapes for the given dtype on SparseCore.


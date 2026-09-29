---
title: 'Module: jax/_src/named_sharding.py'
type: catalog
provenance: extracted
module: jax/_src/named_sharding.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.named_sharding`/
symbols:
  NamedSharding: NamedSharding#
  NamedSharding.mesh: NamedSharding#mesh.
  NamedSharding.spec: NamedSharding#spec.
  named_sharding_to_sdy_sharding: named_sharding_to_sdy_sharding().
  UnspecifiedValue: UnspecifiedValue#
  UNSPECIFIED: UNSPECIFIED.
  NamedSharding.update: NamedSharding#update().
  named_sharding_to_xla_hlo_sharding: named_sharding_to_xla_hlo_sharding().
  SdyArray.build: SdyArray#build().
  NamedSharding.__init__: NamedSharding#__init__().
  SdyArray: SdyArray#
  _check_unique_resources: _check_unique_resources().
  NamedSharding.memory_kind: NamedSharding#memory_kind().
  NamedSharding._hash: NamedSharding#_hash.
  get_array_mapping: get_array_mapping().
  NamedSharding.is_fully_replicated: NamedSharding#is_fully_replicated().
  flatten_spec: flatten_spec().
  SdyArray.logical_device_ids: SdyArray#logical_device_ids.
  NamedSharding.__reduce__: NamedSharding#__reduce__().
  NamedSharding.__eq__: NamedSharding#__eq__().
  SdyArray.dim_shardings: SdyArray#dim_shardings.
  SdyArray.__repr__: SdyArray#__repr__().
  SdyArray.mesh_shape: SdyArray#mesh_shape.
  remove_size_one_mesh_axis: remove_size_one_mesh_axis().
  NamedSharding.device_set: NamedSharding#device_set().
  check_pspec: check_pspec().
  _check_mesh_resource_axis: _check_mesh_resource_axis().
  NamedSharding._logical_device_ids: NamedSharding#_logical_device_ids.
  get_replicated_axes: get_replicated_axes().
  NamedSharding.__repr__: NamedSharding#__repr__().
  NamedSharding._device_assignment: NamedSharding#_device_assignment().
  NamedSharding.addressable_devices: NamedSharding#addressable_devices().
  NamedSharding.replicated_axes: NamedSharding#replicated_axes().
  NamedSharding._to_sdy_sharding: NamedSharding#_to_sdy_sharding().
  get_non_one_sized_mesh_spec: get_non_one_sized_mesh_spec().
  array_mapping_to_axis_resources: array_mapping_to_axis_resources().
  _check_mesh_unreduced: _check_mesh_unreduced().
  SdyDim: SdyDim#
  SdyDim.axes: SdyDim#axes.
  NamedSharding.is_fully_addressable: NamedSharding#is_fully_addressable().
  NamedSharding.is_equivalent_to: NamedSharding#is_equivalent_to().
  SdyDim.is_open: SdyDim#is_open.
  SdyDim._custom_repr: SdyDim#_custom_repr().
  ArrayMappingOrAutoOrUnspecified: ArrayMappingOrAutoOrUnspecified.
  NamedSharding._is_concrete: NamedSharding#_is_concrete().
  SdyDim.build: SdyDim#build().
  ArrayMapping: ArrayMapping.
  NamedSharding._to_xla_hlo_sharding: NamedSharding#_to_xla_hlo_sharding().
  NamedSharding.check_compatible_aval: NamedSharding#check_compatible_aval().
  NamedSharding.num_devices: NamedSharding#num_devices().
  NamedSharding.with_memory_kind: NamedSharding#with_memory_kind().
  MeshAxisName: MeshAxisName.
  SdyArray.replicated_axes: SdyArray#replicated_axes.
  SdyArray.unreduced_axes: SdyArray#unreduced_axes.
  XLADeviceAssignment: XLADeviceAssignment.
  _unpickle_named_sharding: _unpickle_named_sharding().
  Device: Device.
  DuplicateSpecError: DuplicateSpecError#
  NamedSharding.__hash__: NamedSharding#__hash__().
  SdyDim.__repr__: SdyDim#__repr__().
  DuplicateSpecError.__str__: DuplicateSpecError#__str__().
  DuplicateSpecError.mesh: DuplicateSpecError#mesh.
  NamedSharding._memory_kind: NamedSharding#_memory_kind.
  _get_axes: _get_axes().
  DuplicateSpecError.pspec: DuplicateSpecError#pspec.
  Shape: Shape.
  SdyDim.replace: SdyDim#replace.
  DuplicateSpecError.message: DuplicateSpecError#message.
  Index: Index.
  UnspecifiedValue.__repr__: UnspecifiedValue#__repr__().
  SdyArray.replace: SdyArray#replace.
  DuplicateSpecError.__init__: DuplicateSpecError#__init__().
---
# Module: [`jax/_src/named_sharding.py`](../../../../../../raw/code/jax/jax/_src/named_sharding.py)

## Classes
### `DuplicateSpecError`  ·  implements/extends Exception
- def: [`jax/_src/named_sharding.py:519`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L519)
- signature: `class DuplicateSpecError(Exception):`
- members:
  - `mesh` — [`L523`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L523)
  - `message` — [`L522`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L522)
  - `pspec` — [`L524`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L524)
- protocol/private: `__init__`[`L520`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L520), `__str__`[`L526`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L526)
- used by: [`call_shape_dtype_sharding_rule`](lax/utils.md#call_shape_dtype_sharding_rule), [`_check_unique_resources`](named_sharding.md#_check_unique_resources)

### `NamedSharding`
- def: [`jax/_src/named_sharding.py:71`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L71) — documented in [jax-_src-named_sharding](../../../concepts/jax-_src-named_sharding.md)
- members:
  - `_to_sdy_sharding(self, num_dimensions: int, modify_wrt_axis_types: bool = False)` — [`L249`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L249) — Lowers to shardy's representation of NamedSharding.
  - `addressable_devices(self)` — [`L200`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L200)
  - `check_compatible_aval(self, aval_shape: Shape)` — [`L158`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L158)
  - `device_set(self)` — [`L172`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L172)
  - `is_equivalent_to(self, other, ndim: int)` — [`L239`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L239)
  - `is_fully_addressable(self)` — [`L186`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L186)
  - `is_fully_replicated(self)` — [`L209`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L209)
  - `memory_kind(self)` — [`L136`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L136)
  - `num_devices(self)` — [`L168`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L168)
  - `replicated_axes(self)` — [`L222`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L222)
  - `update(self, **kwargs)` — [`L228`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L228)
  - `with_memory_kind(self, kind: str)` — [`L225`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L225)
  - `mesh` — [`L109`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L109) — documented in [jax-_src-named_sharding](../../../concepts/jax-_src-named_sharding.md)
  - `spec` — [`L110`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L110) — documented in [jax-_src-named_sharding](../../../concepts/jax-_src-named_sharding.md)
- protocol/private: `__eq__`[`L147`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L147), `__hash__`[`L140`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L140), `__init__`[`L115`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L115), `__reduce__`[`L131`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L131), `__repr__`[`L124`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L124), `_device_assignment`[`L179`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L179), `_hash`[`L142`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L142), `_is_concrete`[`L194`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L194), `_logical_device_ids`[`L112`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L112), `_memory_kind`[`L111`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L111), `_to_xla_hlo_sharding`[`L246`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L246)
- uses (calls/refs, reference-scoped): [`PartitionSpec`](partition_spec.md#PartitionSpec), [`Sharding`](sharding.md#Sharding), [`Mesh`](mesh.md#Mesh), [`AbstractMesh`](mesh.md#AbstractMesh), [`named_sharding_to_sdy_sharding`](named_sharding.md#named_sharding_to_sdy_sharding), [`unreduced`](partition_spec.md#P.unreduced), [`use_cpp_method`](util.md#use_cpp_method), [`shape`](mesh.md#Mesh.shape), [`named_sharding_to_xla_hlo_sharding`](named_sharding.md#named_sharding_to_xla_hlo_sharding), [`SdyArray`](named_sharding.md#SdyArray), [`use_cpp_class`](util.md#use_cpp_class), [`common_is_equivalent_to`](sharding.md#common_is_equivalent_to), [`get_array_mapping`](named_sharding.md#get_array_mapping), [`size`](mesh.md#Mesh.size), [`check_pspec`](named_sharding.md#check_pspec), [`get_replicated_axes`](named_sharding.md#get_replicated_axes), [`_internal_device_list`](sharding.md#Sharding._internal_device_list), [`_flat_devices_tuple`](mesh.md#Mesh._flat_devices_tuple), [`MeshAxisName`](named_sharding.md#MeshAxisName), [`XLADeviceAssignment`](named_sharding.md#XLADeviceAssignment), [`_flat_devices_set`](mesh.md#Mesh._flat_devices_set), [`_local_devices_set`](mesh.md#Mesh._local_devices_set), [`_unpickle_named_sharding`](named_sharding.md#_unpickle_named_sharding), [`Device`](named_sharding.md#Device), [`Shape`](named_sharding.md#Shape)
- used by: [`bind`](core.md#Primitive.bind), [`full_like`](lax/lax.md#full_like), [`Device`](../__init__.md#Device), [`_index_to_gather`](numpy/indexing.md#_index_to_gather), [`array`](numpy/array_constructors.md#array), [`asarray`](numpy/array_constructors.md#asarray), [`_export_lowered`](export/_export.md#_export_lowered), [`canonicalize_sharding`](sharding_impls.md#canonicalize_sharding), [`arange`](numpy/lax_numpy.md#arange), [`rewriting_take`](numpy/indexing.md#rewriting_take), [`reshape`](basearray.md#Array.reshape), [`uniform`](random/core.md#uniform), [`_cached_shard_map`](pmap.md#_cached_shard_map), [`arange`](../numpy/__init__.pyi.md#arange), [`empty_like`](lax/lax.md#empty_like), [`_handle_array_process_allgather`](../experimental/multihost_utils.md#_handle_array_process_allgather), [`with_sharding_constraint`](pjit.md#with_sharding_constraint), [`reshape`](lax/lax.md#reshape), [`full`](numpy/array_creation.md#full), [`bincount`](numpy/lax_numpy.md#bincount), [`to_named_sharding_with_abstract_mesh`](export/_export.md#to_named_sharding_with_abstract_mesh), [`conv_general_dilated`](lax/convolution.md#conv_general_dilated), [`array`](../numpy/__init__.pyi.md#array), [`reshard`](pjit.md#reshard), [`full_like`](numpy/array_creation.md#full_like), [`zeros`](../numpy/__init__.pyi.md#zeros), [`normal`](random/core.md#normal), [`_get_named_sharding`](export/_export.md#_get_named_sharding), [`einsum`](../numpy/__init__.pyi.md#einsum), [`zeros`](numpy/array_creation.md#zeros), [`_arange`](numpy/lax_numpy.md#_arange), [`randint`](random/core.md#randint), [`_callback_op_sharding`](callback.md#_callback_op_sharding), [`_ragged_dot_general_impl`](lax/lax.md#_ragged_dot_general_impl), [`broadcast_to`](../numpy/__init__.pyi.md#broadcast_to), [`matmul`](numpy/tensor_contractions.md#matmul), [`zeros_like`](../numpy/__init__.pyi.md#zeros_like), [`_get_new_mesh`](pjit.md#_get_new_mesh), [`broadcast_to`](lax/lax.md#broadcast_to), [`_unshard_shaped_array`](shard_map.md#_unshard_shaped_array)  (+161 more)

### `SdyArray`
- def: [`jax/_src/named_sharding.py:322`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L322)
- members:
  - `build(self, attr_cache: dict[SdyArray, sdy.TensorShardingAttr])` — [`L331`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L331)
  - `dim_shardings` — [`L324`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L324)
  - `logical_device_ids` — [`L325`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L325)
  - `mesh_shape` — [`L323`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L323)
  - `replace` — [`L329`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L329)
  - `replicated_axes` — [`L326`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L326)
  - `unreduced_axes` — [`L327`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L327)
- protocol/private: `__repr__`[`L356`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L356)
- uses (calls/refs, reference-scoped): [`SdyDim`](named_sharding.md#SdyDim), [`_custom_repr`](named_sharding.md#SdyDim._custom_repr), [`build`](named_sharding.md#SdyDim.build), [`_get_axes`](named_sharding.md#_get_axes)
- used by: [`_callback_op_sharding`](callback.md#_callback_op_sharding), [`emit_python_callback`](callback.md#emit_python_callback), [`debug_callback_lowering`](debugging.md#debug_callback_lowering), [`named_sharding_to_sdy_sharding`](named_sharding.md#named_sharding_to_sdy_sharding), [`_get_sdy_array_list_for_callbacks`](callback.md#_get_sdy_array_list_for_callbacks), [`receive_from_host`](callback.md#receive_from_host), [`send_to_host`](callback.md#send_to_host), [`_shardy_shard_map_sharding`](shard_map.md#_shardy_shard_map_sharding), [`_to_sdy_sharding`](sharding_impls.md#GSPMDSharding._to_sdy_sharding), [`_to_sdy_sharding`](sharding_impls.md#SingleDeviceSharding._to_sdy_sharding), [`_get_token_sharding`](shard_map.md#_get_token_sharding), [`shardings`](sharding_impls.md#SdyArrayList.shardings), [`_to_sdy_sharding`](named_sharding.md#NamedSharding._to_sdy_sharding), [`build`](sharding_impls.md#SdyArrayList.build)

### `SdyDim`
- def: [`jax/_src/named_sharding.py:291`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L291)
- signature: `class SdyDim:`
- members:
  - `build(self)` — [`L297`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L297)
  - `axes` — [`L292`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L292)
  - `is_open` — [`L293`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L293)
  - `replace` — [`L295`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L295)
- protocol/private: `__repr__`[`L302`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L302), `_custom_repr`[`L305`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L305)
- used by: [`named_sharding_to_sdy_sharding`](named_sharding.md#named_sharding_to_sdy_sharding), [`_get_sdy_array_list_for_callbacks`](callback.md#_get_sdy_array_list_for_callbacks), [`build`](named_sharding.md#SdyArray.build), [`_to_sdy_sharding`](sharding_impls.md#SingleDeviceSharding._to_sdy_sharding), [`__repr__`](named_sharding.md#SdyArray.__repr__), [`dim_shardings`](named_sharding.md#SdyArray.dim_shardings)

### `UnspecifiedValue`
- def: [`jax/_src/named_sharding.py:38`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L38)
- signature: `class UnspecifiedValue:`
- protocol/private: `__repr__`[`L39`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L39)
- used by: [`_parse_jit_arguments`](pjit.md#_parse_jit_arguments), [`with_sharding_constraint`](pjit.md#with_sharding_constraint), [`to_named_sharding_with_abstract_mesh`](export/_export.md#to_named_sharding_with_abstract_mesh), [`_pjit_partial_eval`](pjit.md#_pjit_partial_eval), [`_resolve_in_shardings`](pjit.md#_resolve_in_shardings), [`prepare_axis_resources`](sharding_impls.md#prepare_axis_resources), [`_pjit_batcher_for_sharding`](pjit.md#_pjit_batcher_for_sharding), [`_pjit_linearize`](pjit.md#_pjit_linearize), [`_pjit_lower_jaxpr_to_fun`](pjit.md#_pjit_lower_jaxpr_to_fun), [`UNSPECIFIED`](named_sharding.md#UNSPECIFIED), [`_pjit_pp_rule`](pjit.md#_pjit_pp_rule), [`_pjit_state_discharge_rule`](state/discharge.md#_pjit_state_discharge_rule), [`_check_and_canonicalize_out_shardings`](pjit.md#_check_and_canonicalize_out_shardings), [`_process_in_axis_resources`](pjit.md#_process_in_axis_resources), [`pjit_staging_rule`](pjit.md#pjit_staging_rule), [`_create_sharding_for_array`](pjit.md#_create_sharding_for_array), [`get_array_mapping`](named_sharding.md#get_array_mapping), [`pjit_check_aval_sharding`](pjit.md#pjit_check_aval_sharding), [`_resolve_in_layouts`](pjit.md#_resolve_in_layouts), [`finalize_arg_sharding`](pjit.md#finalize_arg_sharding), [`PjitSharding`](pjit.md#PjitSharding), [`LoweringSharding`](export/_export.md#LoweringSharding), [`ArrayMappingOrAutoOrUnspecified`](named_sharding.md#ArrayMappingOrAutoOrUnspecified)

## Functions
- `_check_mesh_resource_axis(mesh, pspec)` — [`L551`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L551)
- `_check_mesh_unreduced(mesh, pspec)` — [`L568`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L568)
- `_check_unique_resources(pspec: PartitionSpec, arg_name: str, mesh=None)` — [`L529`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L529)
- `_get_axes(axes, mesh_shape)` — [`L313`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L313)
- `_unpickle_named_sharding(mesh, spec, memory_kind, logical_device_ids)` — [`L65`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L65)
- `array_mapping_to_axis_resources(array_mapping: ArrayMapping)` — [`L494`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L494)
- `check_pspec(mesh, spec, _manual_axes=frozenset())` — [`L514`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L514)
- `flatten_spec(spec)` — [`L269`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L269)
- `get_array_mapping(axis_resources)` — [`L278`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L278)
- `get_non_one_sized_mesh_spec(mesh, spec)` — [`L382`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L382)
- `get_replicated_axes(spec, mesh)` — [`L263`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L263)
- `named_sharding_to_sdy_sharding(self, num_dimensions: int, modify_wrt_axis_types: bool)` — [`L467`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L467) — documented in [jax-_src-named_sharding](../../../concepts/jax-_src-named_sharding.md)
- `named_sharding_to_xla_hlo_sharding(self, num_dimensions: int)` — [`L391`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L391)
- `remove_size_one_mesh_axis(spec, mesh)` — [`L368`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L368)

## Module values
- `ArrayMapping` — [`L61`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L61)
- `ArrayMappingOrAutoOrUnspecified` — [`L62`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L62)
- `Device` — [`L33`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L33)
- `Index` — [`L34`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L34)
- `MeshAxisName` — [`L44`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L44)
- `Shape` — [`L32`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L32)
- `UNSPECIFIED` — [`L41`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L41)
- `XLADeviceAssignment` — [`L35`](../../../../../../raw/code/jax/jax/_src/named_sharding.py#L35)


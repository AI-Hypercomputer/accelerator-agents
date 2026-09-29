---
title: 'Module: jax/_src/sharding_impls.py'
type: catalog
provenance: extracted
module: jax/_src/sharding_impls.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.sharding_impls`/
symbols:
  canonicalize_sharding: canonicalize_sharding().
  prepare_axis_resources: prepare_axis_resources().
  SPMDAxisContext: SPMDAxisContext#
  parse_flatten_op_sharding: parse_flatten_op_sharding().
  GSPMDSharding._hlo_sharding: GSPMDSharding#_hlo_sharding.
  SPMDAxisContext.mesh: SPMDAxisContext#mesh.
  logical_sharding: logical_sharding().
  _gspmd_to_named_sharding_via_mesh: _gspmd_to_named_sharding_via_mesh().
  set_mesh.__init__: set_mesh#__init__().
  GSPMDSharding: GSPMDSharding#
  get_process_index_and_count: get_process_index_and_count().
  make_key_array_phys_sharding: make_key_array_phys_sharding().
  num_addressable_indices: num_addressable_indices().
  SingleDeviceSharding: SingleDeviceSharding#
  check_replicated_trailing_dims: check_replicated_trailing_dims().
  ShardingContext: ShardingContext#
  GSPMDSharding._to_sdy_sharding: GSPMDSharding#_to_sdy_sharding().
  SdyArrayList: SdyArrayList#
  cached_named_sharding: cached_named_sharding().
  SingleDeviceSharding._to_sdy_sharding: SingleDeviceSharding#_to_sdy_sharding().
  SingleDeviceSharding._device: SingleDeviceSharding#_device.
  make_mesh: make_mesh().
  SPMDAxisContext.manual_axes: SPMDAxisContext#manual_axes.
  _internal_use_concrete_mesh: _internal_use_concrete_mesh().
  GSPMDSharding.__eq__: GSPMDSharding#__eq__().
  device_replica_id_map: device_replica_id_map().
  local_to_global_shape: local_to_global_shape().
  physical_sharding: physical_sharding().
  set_mesh: set_mesh#
  make_single_device_sharding: make_single_device_sharding().
  GSPMDSharding._hash: GSPMDSharding#_hash.
  get_logical_gspmd_sharding: get_logical_gspmd_sharding().
  SdyArrayList.shardings: SdyArrayList#shardings.
  SingleDeviceSharding.devices_indices_map: SingleDeviceSharding#devices_indices_map().
  GSPMDSharding.__init__: GSPMDSharding#__init__().
  GSPMDSharding.__reduce__: GSPMDSharding#__reduce__().
  SdyArrayList.build: SdyArrayList#build().
  GSPMDSharding.memory_kind: GSPMDSharding#memory_kind().
  Device: Device.
  SingleDeviceSharding._hash: SingleDeviceSharding#_hash.
  GSPMDSharding._hlo_sharding_hash: GSPMDSharding#_hlo_sharding_hash().
  unflatten_array: unflatten_array().
  physical_hlo_sharding: physical_hlo_sharding().
  get_mesh: get_mesh().
  SingleDeviceSharding._memory_kind: SingleDeviceSharding#_memory_kind.
  SingleDeviceSharding.__init__: SingleDeviceSharding#__init__().
  SingleDeviceSharding.__reduce__: SingleDeviceSharding#__reduce__().
  SingleDeviceSharding.__eq__: SingleDeviceSharding#__eq__().
  GSPMDSharding._internal_device_list: GSPMDSharding#_internal_device_list.
  GSPMDSharding.check_compatible_aval: GSPMDSharding#check_compatible_aval().
  GSPMDSharding.with_memory_kind: GSPMDSharding#with_memory_kind().
  set_mesh.__exit__: set_mesh#__exit__().
  hashed_index: hashed_index().
  GSPMDSharding.is_fully_replicated: GSPMDSharding#is_fully_replicated().
  SingleDeviceSharding.memory_kind: SingleDeviceSharding#memory_kind().
  Shape: Shape.
  SingleDeviceSharding.device_set: SingleDeviceSharding#device_set().
  ShardingContext.num_devices: ShardingContext#num_devices.
  ShardingContext.device_assignment: ShardingContext#device_assignment.
  SingleDeviceSharding.__repr__: SingleDeviceSharding#__repr__().
  SingleDeviceSharding.with_memory_kind: SingleDeviceSharding#with_memory_kind().
  SingleDeviceSharding._device_assignment: SingleDeviceSharding#_device_assignment().
  SingleDeviceSharding.is_fully_addressable: SingleDeviceSharding#is_fully_addressable().
  GSPMDSharding._devices: GSPMDSharding#_devices.
  GSPMDSharding._memory_kind: GSPMDSharding#_memory_kind.
  GSPMDSharding.__repr__: GSPMDSharding#__repr__().
  GSPMDSharding.device_set: GSPMDSharding#device_set().
  GSPMDSharding._device_assignment: GSPMDSharding#_device_assignment().
  ShardingContext.__post_init__: ShardingContext#__post_init__().
  XLADeviceAssignment: XLADeviceAssignment.
  ShardingContext.abstract_mesh: ShardingContext#abstract_mesh.
  replicated_hlo_sharding: replicated_hlo_sharding.
  _unpickle_single_device_sharding: _unpickle_single_device_sharding().
  _unpickle_gspmd_sharding: _unpickle_gspmd_sharding().
  GSPMDSharding._to_xla_hlo_sharding: GSPMDSharding#_to_xla_hlo_sharding().
  GSPMDSharding.get_replicated: GSPMDSharding#get_replicated().
  unflatten_superdims: unflatten_superdims().
  explode_superdims: explode_superdims().
  NonUniformShardingError: NonUniformShardingError#
  XLACompatibleSharding: XLACompatibleSharding.
  SingleDeviceSharding.num_devices: SingleDeviceSharding#num_devices().
  SingleDeviceSharding._to_xla_hlo_sharding: SingleDeviceSharding#_to_xla_hlo_sharding().
  SingleDeviceSharding.check_compatible_aval: SingleDeviceSharding#check_compatible_aval().
  GSPMDSharding.num_devices: GSPMDSharding#num_devices().
  GSPMDSharding.is_fully_addressable: GSPMDSharding#is_fully_addressable().
  Index: Index.
  strides_for_sizes: strides_for_sizes().
  MeshAxisName: MeshAxisName.
  unflatten_superdims.check: unflatten_superdims().check().
  _slice_as_tuple: _slice_as_tuple().
  config_ext: config_ext.
  SingleDeviceSharding.__hash__: SingleDeviceSharding#__hash__().
  SingleDeviceSharding.is_fully_replicated: SingleDeviceSharding#is_fully_replicated().
  GSPMDSharding.__hash__: GSPMDSharding#__hash__().
  set_mesh.__slots__: set_mesh#__slots__.
  set_mesh.__enter__: set_mesh#__enter__().
---
# Module: [`jax/_src/sharding_impls.py`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py)

## Classes
### `GSPMDSharding`
- def: [`jax/_src/sharding_impls.py:198`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L198)
- members:
  - `check_compatible_aval(self, aval_shape: Shape)` — [`L247`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L247)
  - `device_set(self)` — [`L260`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L260)
  - `get_replicated(cls, device_assignment, *, memory_kind: str | None = None)` — [`L308`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L308)
  - `is_fully_addressable(self)` — [`L304`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L304)
  - `is_fully_replicated(self)` — [`L300`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L300)
  - `memory_kind(self)` — [`L264`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L264)
  - `num_devices(self)` — [`L256`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L256)
  - `with_memory_kind(self, kind: str)` — [`L267`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L267)
- protocol/private: `__eq__`[`L228`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L228), `__hash__`[`L237`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L237), `__init__`[`L205`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L205), `__reduce__`[`L218`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L218), `__repr__`[`L243`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L243), `_device_assignment`[`L271`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L271), `_devices`[`L199`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L199), `_hash`[`L239`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L239), `_hlo_sharding`[`L200`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L200), `_hlo_sharding_hash`[`L223`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L223), `_internal_device_list`[`L202`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L202), `_memory_kind`[`L201`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L201), `_to_sdy_sharding`[`L277`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L277), `_to_xla_hlo_sharding`[`L274`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L274)
- uses (calls/refs, reference-scoped): [`NamedSharding`](named_sharding.md#NamedSharding), [`PartitionSpec`](partition_spec.md#PartitionSpec), [`Sharding`](sharding.md#Sharding), [`AbstractMesh`](mesh.md#AbstractMesh), [`use_cpp_method`](util.md#use_cpp_method), [`_gspmd_to_named_sharding_via_mesh`](sharding_impls.md#_gspmd_to_named_sharding_via_mesh), [`SdyArray`](named_sharding.md#SdyArray), [`use_cpp_class`](util.md#use_cpp_class), [`_to_sdy_sharding`](named_sharding.md#NamedSharding._to_sdy_sharding), [`are_hlo_shardings_equal`](op_shardings.md#are_hlo_shardings_equal), [`Device`](sharding_impls.md#Device), [`get_num_ways_dim_sharded`](op_shardings.md#get_num_ways_dim_sharded), [`is_hlo_sharding_replicated`](op_shardings.md#is_hlo_sharding_replicated), [`Shape`](sharding_impls.md#Shape), [`XLADeviceAssignment`](sharding_impls.md#XLADeviceAssignment), [`replicated_hlo_sharding`](sharding_impls.md#replicated_hlo_sharding), [`_unpickle_gspmd_sharding`](sharding_impls.md#_unpickle_gspmd_sharding)
- used by: [`_handle_array_process_allgather`](../experimental/multihost_utils.md#_handle_array_process_allgather), [`to_named_sharding_with_abstract_mesh`](export/_export.md#to_named_sharding_with_abstract_mesh), [`_resolve_in_shardings`](pjit.md#_resolve_in_shardings), [`_pjit_batcher_for_sharding`](pjit.md#_pjit_batcher_for_sharding), [`_gspmd_to_named_sharding_via_mesh`](sharding_impls.md#_gspmd_to_named_sharding_via_mesh), [`device_set`](sharding.md#Sharding.device_set), [`make_key_array_phys_sharding`](sharding_impls.md#make_key_array_phys_sharding), [`wrap_with_sharding`](export/_export.md#wrap_with_sharding), [`_hlo_sharding_callback`](debugging.md#_inspect_sharding_lowering_rule._hlo_sharding_callback), [`_device_assignment`](sharding.md#Sharding._device_assignment), [`_to_xla_hlo_sharding`](sharding.md#Sharding._to_xla_hlo_sharding), [`is_fully_replicated`](sharding.md#Sharding.is_fully_replicated), [`to_mesh_pspec_sharding`](custom_partitioning.md#_custom_partitioning_lowering_rule.to_mesh_pspec_sharding), [`memory_kind`](sharding.md#Sharding.memory_kind), [`num_devices`](sharding.md#Sharding.num_devices), [`is_fully_addressable`](sharding.md#Sharding.is_fully_addressable), [`get_logical_gspmd_sharding`](sharding_impls.md#get_logical_gspmd_sharding), [`PjitSharding`](pjit.md#PjitSharding), [`with_memory_kind`](sharding.md#Sharding.with_memory_kind), [`_to_sdy_sharding`](sharding.md#Sharding._to_sdy_sharding), [`_hlo_sharding_to_gspmd_sharding`](export/_export.md#_hlo_sharding_to_gspmd_sharding), [`_unpickle_gspmd_sharding`](sharding_impls.md#_unpickle_gspmd_sharding)

### `NonUniformShardingError`  ·  implements/extends ValueError
- def: [`jax/_src/sharding_impls.py:532`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L532)
- doc: Raised when sharding is not uniform across processes.
- signature: `class NonUniformShardingError(ValueError):`
- used by: [`get_process_index_and_count`](sharding_impls.md#get_process_index_and_count), [`local_to_global_shape`](sharding_impls.md#local_to_global_shape)

### `SPMDAxisContext`
- def: [`jax/_src/sharding_impls.py:353`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L353)
- doc: A hardware axis context for parallel computations that use the GSPMD partitioner.
- signature: `class SPMDAxisContext:`
- members:
  - `manual_axes` — [`L361`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L361)
  - `mesh` — [`L360`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L360)
- uses (calls/refs, reference-scoped): [`Mesh`](mesh.md#Mesh), [`MeshAxisName`](sharding_impls.md#MeshAxisName)
- used by: [`_call_exported_lowering`](export/_export.md#_call_exported_lowering), [`mpmd_map_tpu_lowering_rule`](pallas/mosaic/pallas_call_registration.md#mpmd_map_tpu_lowering_rule), [`pallas_call_lowering`](pallas/mosaic_gpu/pallas_call_registration.md#pallas_call_lowering), [`_custom_partitioning_lowering_rule`](custom_partitioning.md#_custom_partitioning_lowering_rule), [`_pjit_lowering`](pjit.md#_pjit_lowering), [`pallas_call_tpu_lowering_rule`](pallas/mosaic/pallas_call_registration.md#pallas_call_tpu_lowering_rule), [`_shard_map_lowering_shardy`](shard_map.md#_shard_map_lowering_shardy), [`_callback_op_sharding`](callback.md#_callback_op_sharding), [`_tpu_custom_call_lowering`](tpu_custom_call.md#_tpu_custom_call_lowering), [`debug_callback_lowering`](debugging.md#debug_callback_lowering), [`_mosaic_gpu_lowering_rule`](../experimental/mosaic/gpu/core.md#_mosaic_gpu_lowering_rule), [`_inspect_sharding_lowering_rule`](debugging.md#_inspect_sharding_lowering_rule), [`_sharding_constraint_hlo_lowering`](pjit.md#_sharding_constraint_hlo_lowering), [`_shard_map_lowering`](shard_map.md#_shard_map_lowering), [`_custom_partitioning_partition`](custom_partitioning.md#_custom_partitioning_partition), [`_build_axis_index_lowering_hlo`](lax/parallel.md#_build_axis_index_lowering_hlo), [`_try_mesh_axes_replica_group`](lax/parallel.md#_try_mesh_axes_replica_group), [`_allreduce_lowering`](lax/parallel.md#_allreduce_lowering), [`_make_scoped_manual_sharding`](shard_map.md#_make_scoped_manual_sharding), [`_pbroadcast_lowering`](lax/parallel.md#_pbroadcast_lowering), [`_all_gather_lowering`](lax/parallel.md#_all_gather_lowering), [`_pcollectives_lowering_common`](lax/parallel.md#_pcollectives_lowering_common), [`_all_to_all_lowering`](lax/parallel.md#_all_to_all_lowering), [`_ragged_all_to_all_lowering`](lax/parallel.md#_ragged_all_to_all_lowering), [`_reduce_scatter_lowering`](lax/parallel.md#_reduce_scatter_lowering), [`_precv_lowering_gpu`](lax/parallel.md#_precv_lowering_gpu), [`_psend_lowering_gpu`](lax/parallel.md#_psend_lowering_gpu)

### `SdyArrayList`
- def: [`jax/_src/sharding_impls.py:89`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L89)
- signature: `class SdyArrayList:`
- members:
  - `build(self, cache: dict[SdyArray, sdy.TensorShardingAttr])` — [`L92`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L92)
  - `shardings` — [`L90`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L90)
- uses (calls/refs, reference-scoped): [`build`](named_sharding.md#SdyArray.build), [`SdyArray`](named_sharding.md#SdyArray)
- used by: [`_shard_map_lowering_shardy`](shard_map.md#_shard_map_lowering_shardy), [`_callback_op_sharding`](callback.md#_callback_op_sharding), [`emit_python_callback`](callback.md#emit_python_callback), [`debug_callback_lowering`](debugging.md#debug_callback_lowering), [`_get_sdy_array_list_for_callbacks`](callback.md#_get_sdy_array_list_for_callbacks), [`receive_from_host`](callback.md#receive_from_host), [`send_to_host`](callback.md#send_to_host), [`_emit_tpu_python_callback`](callback.md#_emit_tpu_python_callback)

### `ShardingContext`
- def: [`jax/_src/sharding_impls.py:365`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L365)
- doc: A hardware axis context for parallel computations that use the sharding
- signature: `class ShardingContext:`
- members:
  - `abstract_mesh` — [`L373`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L373)
  - `device_assignment` — [`L372`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L372)
  - `num_devices` — [`L371`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L371)
- protocol/private: `__post_init__`[`L375`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L375)
- uses (calls/refs, reference-scoped): [`AbstractMesh`](mesh.md#AbstractMesh)
- used by: [`_call_exported_lowering`](export/_export.md#_call_exported_lowering), [`_custom_partitioning_lowering_rule`](custom_partitioning.md#_custom_partitioning_lowering_rule), [`_pjit_lowering`](pjit.md#_pjit_lowering), [`_callback_op_sharding`](callback.md#_callback_op_sharding), [`_tpu_custom_call_lowering`](tpu_custom_call.md#_tpu_custom_call_lowering), [`debug_callback_lowering`](debugging.md#debug_callback_lowering), [`_mosaic_gpu_lowering_rule`](../experimental/mosaic/gpu/core.md#_mosaic_gpu_lowering_rule), [`_inspect_sharding_lowering_rule`](debugging.md#_inspect_sharding_lowering_rule), [`_build_axis_index_lowering_hlo`](lax/parallel.md#_build_axis_index_lowering_hlo), [`_wrap_main_func`](export/_export.md#_wrap_main_func), [`_allreduce_lowering`](lax/parallel.md#_allreduce_lowering), [`_pbroadcast_lowering`](lax/parallel.md#_pbroadcast_lowering), [`_all_gather_lowering`](lax/parallel.md#_all_gather_lowering), [`_all_to_all_lowering`](lax/parallel.md#_all_to_all_lowering), [`_ragged_all_to_all_lowering`](lax/parallel.md#_ragged_all_to_all_lowering), [`_reduce_scatter_lowering`](lax/parallel.md#_reduce_scatter_lowering)

### `SingleDeviceSharding`
- def: [`jax/_src/sharding_impls.py:106`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L106)
- members:
  - `check_compatible_aval(self, aval_shape: Shape)` — [`L184`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L184)
  - `device_set(self)` — [`L151`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L151)
  - `devices_indices_map(self, global_shape: Shape)` — [`L161`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L161)
  - `is_fully_addressable(self)` — [`L181`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L181)
  - `is_fully_replicated(self)` — [`L177`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L177)
  - `memory_kind(self)` — [`L155`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L155)
  - `num_devices(self)` — [`L147`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L147)
  - `with_memory_kind(self, kind: str)` — [`L158`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L158)
- protocol/private: `__eq__`[`L138`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L138), `__hash__`[`L133`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L133), `__init__`[`L122`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L122), `__reduce__`[`L126`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L126), `__repr__`[`L129`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L129), `_device`[`L118`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L118), `_device_assignment`[`L165`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L165), `_hash`[`L135`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L135), `_memory_kind`[`L119`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L119), `_to_sdy_sharding`[`L171`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L171), `_to_xla_hlo_sharding`[`L168`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L168)
- uses (calls/refs, reference-scoped): [`Sharding`](sharding.md#Sharding), [`use_cpp_method`](util.md#use_cpp_method), [`SdyArray`](named_sharding.md#SdyArray), [`use_cpp_class`](util.md#use_cpp_class), [`dim_shardings`](named_sharding.md#SdyArray.dim_shardings), [`mesh_shape`](named_sharding.md#SdyArray.mesh_shape), [`Device`](sharding_impls.md#Device), [`process_index`](xla_bridge.md#process_index), [`SdyDim`](named_sharding.md#SdyDim), [`axes`](named_sharding.md#SdyDim.axes), [`is_open`](named_sharding.md#SdyDim.is_open), [`Shape`](sharding_impls.md#Shape), [`XLADeviceAssignment`](sharding_impls.md#XLADeviceAssignment), [`replicated_hlo_sharding`](sharding_impls.md#replicated_hlo_sharding), [`_unpickle_single_device_sharding`](sharding_impls.md#_unpickle_single_device_sharding), [`Index`](sharding_impls.md#Index)
- used by: [`full`](numpy/array_creation.md#full), [`to_named_sharding_with_abstract_mesh`](export/_export.md#to_named_sharding_with_abstract_mesh), [`full_like`](numpy/array_creation.md#full_like), [`_callback_op_sharding`](callback.md#_callback_op_sharding), [`device_set`](sharding.md#Sharding.device_set), [`devices_indices_map`](sharding.md#Sharding.devices_indices_map), [`_device_assignment`](sharding.md#Sharding._device_assignment), [`_to_xla_hlo_sharding`](sharding.md#Sharding._to_xla_hlo_sharding), [`is_fully_replicated`](sharding.md#Sharding.is_fully_replicated), [`memory_kind`](sharding.md#Sharding.memory_kind), [`num_devices`](sharding.md#Sharding.num_devices), [`is_fully_addressable`](sharding.md#Sharding.is_fully_addressable), [`make_single_device_sharding`](sharding_impls.md#make_single_device_sharding), [`with_memory_kind`](sharding.md#Sharding.with_memory_kind), [`_to_sdy_sharding`](sharding.md#Sharding._to_sdy_sharding), [`_unpickle_single_device_sharding`](sharding_impls.md#_unpickle_single_device_sharding)

### `set_mesh`
- def: [`jax/_src/sharding_impls.py:977`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L977)
- doc: Sets a concrete mesh in a thread-local context.
- signature: `class set_mesh:`
- protocol/private: `__enter__`[`L1018`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L1018), `__exit__`[`L1021`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L1021), `__init__`[`L1000`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L1000), `__slots__`[`L998`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L998)
- uses (calls/refs, reference-scoped): [`Mesh`](mesh.md#Mesh), [`abstract_mesh`](mesh.md#Mesh.abstract_mesh), [`empty_abstract_mesh`](mesh.md#empty_abstract_mesh), [`_any_axis_manual`](mesh.md#BaseMesh._any_axis_manual), [`trace_state_clean`](core.md#trace_state_clean), [`abstract_mesh_context_manager`](config.md#abstract_mesh_context_manager), [`empty_concrete_mesh`](mesh.md#empty_concrete_mesh), [`device_context`](config.md#device_context)
- used by: [`Device`](../__init__.md#Device), [`_run_example`](../experimental/pallas/ops/gpu/collective_matmul_mgpu.md#_run_example), [`_handle_array_process_allgather`](../experimental/multihost_utils.md#_handle_array_process_allgather), [`_run_example`](../experimental/pallas/ops/gpu/all_gather_mgpu.md#_run_example), [`_run_example`](../experimental/pallas/ops/gpu/reduce_scatter_mgpu.md#_run_example), [`broadcast_one_to_all`](../experimental/multihost_utils.md#broadcast_one_to_all), [`_reshard_impl`](pjit.md#_reshard_impl)

## Functions
- `_gspmd_to_named_sharding_via_mesh(out_s: GSPMDSharding, mesh: Mesh | AbstractMesh)` — [`L835`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L835)
- `_internal_use_concrete_mesh(mesh: Mesh)` — [`L1035`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L1035)
- `_slice_as_tuple(s: slice)` — [`L527`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L527)
- `_unpickle_gspmd_sharding(devices, op_sharding, memory_kind)` — [`L194`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L194)
- `_unpickle_single_device_sharding(device, memory_kind)` — [`L101`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L101)
- `cached_named_sharding(mesh: Mesh | AbstractMesh, pspec: PartitionSpec, memory_kind: str | None = None)` — [`L829`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L829)
- `canonicalize_sharding(sharding: NamedSharding | PartitionSpec | None, api_name: str, check_mesh_consistency: bool = True)` — [`L843`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L843)
- `check(cond)` — [`L430`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L430)
- `check_replicated_trailing_dims(sharding: jsharding.Sharding, logical_shape, dtype)` — [`L794`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L794)
- `device_replica_id_map(sharding, global_shape: Shape)` — [`L69`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L69)
- `explode_superdims(sizes, dims)` — [`L451`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L451) — Explode superdims to fit a known shape.
- `get_logical_gspmd_sharding(logical_shape, dtype, phys_sharding)` — [`L781`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L781)
- `get_mesh()` — [`L1026`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L1026)
- `get_process_index_and_count(tensor_sharding: jsharding.Sharding, dim: int, ndims: int)` — [`L537`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L537) — Get current process index and number of unique processes for given dimension.
- `hashed_index(x)` — [`L63`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L63)
- `local_to_global_shape(sharding: jsharding.Sharding, local_shape: Shape)` — [`L652`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L652) — Computes the global shape given the per process if possible.
- `logical_sharding(logical_shape, dtype, phys_sharding)` — [`L808`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L808)
- `make_key_array_phys_sharding(aval, sharding)` — [`L764`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L764)
- `make_mesh(axis_shapes: Sequence[int], axis_names: Sequence[str], axis_types: tuple[AxisType, ...] | None = None, *, devices: Sequence[xc.Device] | None = None)` — [`L899`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L899) — Creates an efficient mesh with the shape and axis names specified.
- `make_single_device_sharding(device, *, memory_kind=None)` — [`L190`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L190)
- `num_addressable_indices(tensor_sharding: jsharding.Sharding, dim: int, global_shape: Shape)` — [`L711`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L711) — Returns the number of indices for given dimension this host has access to.
- `parse_flatten_op_sharding(hlo_sharding: xc.OpSharding | xc.HloSharding, mesh: Mesh | AbstractMesh)` — [`L479`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L479)
- `physical_hlo_sharding(aval, hlo_sharding: xc.HloSharding)` — [`L754`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L754)
- `physical_sharding(aval, sharding: jsharding.Sharding)` — [`L777`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L777)
- `prepare_axis_resources(axis_resources, arg_name, allow_unconstrained_dims=False)` — [`L316`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L316)
- `strides_for_sizes(sizes)` — [`L386`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L386) — Returns an array of strides for major-to-minor sizes.
- `unflatten_array(named_sizes, assignment)` — [`L390`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L390) — Recovers the ordering of axis names based on a device assignment.
- `unflatten_superdims(assignment)` — [`L416`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L416) — Unflatten a list of dimension sizes and their strides that generates assignment.

## Module values
- `Device` — [`L56`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L56)
- `Index` — [`L57`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L57)
- `MeshAxisName` — [`L313`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L313)
- `Shape` — [`L55`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L55)
- `XLACompatibleSharding` — [`L60`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L60)
- `XLADeviceAssignment` — [`L58`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L58)
- `config_ext` — [`L53`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L53)
- `replicated_hlo_sharding` — [`L98`](../../../../../../raw/code/jax/jax/_src/sharding_impls.py#L98)


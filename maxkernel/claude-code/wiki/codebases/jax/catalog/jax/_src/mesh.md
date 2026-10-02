---
title: 'Module: jax/_src/mesh.py'
type: catalog
provenance: extracted
module: jax/_src/mesh.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.mesh`/
symbols:
  Mesh: Mesh#
  AbstractMesh: AbstractMesh#
  AxisType: AxisType#
  get_abstract_mesh: get_abstract_mesh().
  use_abstract_mesh: use_abstract_mesh#
  Mesh.abstract_mesh: Mesh#abstract_mesh().
  Mesh.axis_names: Mesh#axis_names.
  BaseMesh.axis_types: BaseMesh#axis_types.
  AbstractMesh.empty: AbstractMesh#empty().
  Mesh.shape: Mesh#shape().
  Mesh.empty: Mesh#empty().
  BaseMesh.manual_axes: BaseMesh#manual_axes().
  get_concrete_mesh: get_concrete_mesh().
  BaseMesh.axis_names: BaseMesh#axis_names.
  AxisType.Explicit: AxisType#Explicit.
  empty_abstract_mesh: empty_abstract_mesh.
  Mesh.__enter__: Mesh#__enter__().
  BaseMesh.explicit_axes: BaseMesh#explicit_axes().
  Mesh.devices: Mesh#devices.
  ResourceEnv.physical_mesh: ResourceEnv#physical_mesh.
  thread_resources: thread_resources.
  _get_local_mesh: _get_local_mesh().
  BaseMesh.are_all_axes_explicit: BaseMesh#are_all_axes_explicit().
  BaseMesh._name_to_type: BaseMesh#_name_to_type().
  Mesh.__exit__: Mesh#__exit__().
  Mesh.size: Mesh#size.
  AxisType.Auto: AxisType#Auto.
  AxisType.Manual: AxisType#Manual.
  BaseMesh._any_axis_manual: BaseMesh#_any_axis_manual().
  BaseMesh._any_axis_explicit: BaseMesh#_any_axis_explicit().
  ResourceEnv.with_mesh: ResourceEnv#with_mesh().
  BaseMesh.auto_axes: BaseMesh#auto_axes().
  Mesh.shape_tuple: Mesh#shape_tuple().
  AbstractMesh.update: AbstractMesh#update().
  AbstractMesh.update_axis_types: AbstractMesh#update_axis_types().
  BaseMesh.are_all_axes_auto: BaseMesh#are_all_axes_auto().
  BaseMesh._are_all_axes_auto_or_manual: BaseMesh#_are_all_axes_auto_or_manual().
  Mesh.local_mesh: Mesh#local_mesh().
  AbstractMesh.shape_tuple: AbstractMesh#shape_tuple().
  use_abstract_mesh.__enter__: use_abstract_mesh#__enter__().
  AbstractMesh.abstract_device: AbstractMesh#abstract_device.
  BaseMesh.are_all_axes_manual: BaseMesh#are_all_axes_manual().
  Mesh._repr: Mesh#_repr().
  _ThreadResourcesLocalState.env: _ThreadResourcesLocalState#env.
  BaseMesh._are_all_axes_explicit_or_manual: BaseMesh#_are_all_axes_explicit_or_manual().
  BaseMesh._any_axis_auto: BaseMesh#_any_axis_auto().
  BaseMesh._any_axis_auto_or_manual: BaseMesh#_any_axis_auto_or_manual().
  Mesh.__new__: Mesh#__new__().
  Mesh.is_scalar: Mesh#is_scalar().
  Mesh.update: Mesh#update().
  AbstractMesh.__new__: AbstractMesh#__new__().
  AbstractMesh.__getnewargs_ex__: AbstractMesh#__getnewargs_ex__().
  _ThreadResourcesLocalState.stack: _ThreadResourcesLocalState#stack.
  Mesh.device_ids: Mesh#device_ids().
  AbstractMesh.axis_sizes: AbstractMesh#axis_sizes.
  ResourceEnv.physical_resource_axes: ResourceEnv#physical_resource_axes().
  AbstractDevice._repr: AbstractDevice#_repr().
  empty_concrete_mesh: empty_concrete_mesh.
  ResourceEnv.local_shape: ResourceEnv#local_shape().
  Mesh.__getnewargs_ex__: Mesh#__getnewargs_ex__().
  Mesh.__str__: Mesh#__str__().
  AbstractMesh.__repr__: AbstractMesh#__repr__().
  AbstractMesh.shape: AbstractMesh#shape().
  _normalize_axis_types: _normalize_axis_types().
  _raise_value_error: _raise_value_error().
  Mesh._flat_devices_tuple: Mesh#_flat_devices_tuple().
  ResourceEnv.resource_axes: ResourceEnv#resource_axes().
  EMPTY_ENV: EMPTY_ENV.
  all_axis_types_match: all_axis_types_match().
  any_axis_types_match: any_axis_types_match().
  Mesh.axis_sizes: Mesh#axis_sizes().
  AbstractMesh.size: AbstractMesh#size.
  ResourceEnv.shape: ResourceEnv#shape().
  ResourceEnv.__repr__: ResourceEnv#__repr__().
  Mesh.is_multi_process: Mesh#is_multi_process().
  use_abstract_mesh.__init__: use_abstract_mesh#__init__().
  use_abstract_mesh.__exit__: use_abstract_mesh#__exit__().
  AbstractDevice: AbstractDevice#
  Mesh.local_devices: Mesh#local_devices().
  MeshAxisName: MeshAxisName.
  BaseMesh: BaseMesh#
  AbstractDevice.num_cores: AbstractDevice#num_cores.
  Mesh._create: Mesh#_create().
  Mesh._local_mesh: Mesh#_local_mesh().
  Mesh._local_devices_set: Mesh#_local_devices_set().
  Mesh._flat_devices_set: Mesh#_flat_devices_set().
  AbstractMesh._create: AbstractMesh#_create().
  AbstractDevice.device_kind: AbstractDevice#device_kind.
  AbstractDevice.platform: AbstractDevice#platform.
  Mesh._internal_device_list: Mesh#_internal_device_list().
  Mesh.__repr__: Mesh#__repr__().
  AbstractDevice.__repr__: AbstractDevice#__repr__().
  AbstractMesh.devices: AbstractMesh#devices().
  AbstractMesh.device_ids: AbstractMesh#device_ids().
  AbstractMesh.is_multi_process: AbstractMesh#is_multi_process().
  AbstractMesh.local_devices: AbstractMesh#local_devices().
  AbstractMesh.local_mesh: AbstractMesh#local_mesh().
  AbstractMesh.__enter__: AbstractMesh#__enter__().
  AbstractMesh.__exit__: AbstractMesh#__exit__().
  unsafe_zip: unsafe_zip.
  ResourceAxisName: ResourceAxisName.
  show_axes: show_axes().
  config_ext: config_ext.
  ResourceEnv: ResourceEnv#
  _ThreadResourcesLocalState: _ThreadResourcesLocalState#
  AbstractMesh.abstract_mesh: AbstractMesh#abstract_mesh().
  AxisType.__repr__: AxisType#__repr__().
  BaseMesh.shape_tuple: BaseMesh#shape_tuple.
  _ThreadResourcesLocalState.__init__: _ThreadResourcesLocalState#__init__().
  AbstractMesh._internal_device_list: AbstractMesh#_internal_device_list().
  use_abstract_mesh.__slots__: use_abstract_mesh#__slots__.
---
# Module: [`jax/_src/mesh.py`](../../../../../../raw/code/jax/jax/_src/mesh.py)

## Classes
### `AbstractDevice`
- def: [`jax/_src/mesh.py:439`](../../../../../../raw/code/jax/jax/_src/mesh.py#L439)
- members:
  - `device_kind` — [`L440`](../../../../../../raw/code/jax/jax/_src/mesh.py#L440)
  - `num_cores` — [`L441`](../../../../../../raw/code/jax/jax/_src/mesh.py#L441)
  - `platform` — [`L442`](../../../../../../raw/code/jax/jax/_src/mesh.py#L442)
- protocol/private: `__repr__`[`L444`](../../../../../../raw/code/jax/jax/_src/mesh.py#L444), `_repr`[`L447`](../../../../../../raw/code/jax/jax/_src/mesh.py#L447)
- used by: [`abstract_mesh`](mesh.md#Mesh.abstract_mesh), [`_serialize_abstract_device`](export/serialization.md#_serialize_abstract_device), [`_deserialize_abstract_device`](export/serialization.md#_deserialize_abstract_device)

### `AbstractMesh`
- def: [`jax/_src/mesh.py:453`](../../../../../../raw/code/jax/jax/_src/mesh.py#L453) — documented in [jax-_src-mesh](../../../concepts/jax-_src-mesh.md)
- members:
  - `abstract_mesh(self)` — [`L538`](../../../../../../raw/code/jax/jax/_src/mesh.py#L538)
  - `device_ids(self)` — [`L551`](../../../../../../raw/code/jax/jax/_src/mesh.py#L551)
  - `devices(self)` — [`L547`](../../../../../../raw/code/jax/jax/_src/mesh.py#L547)
  - `empty(self)` — [`L534`](../../../../../../raw/code/jax/jax/_src/mesh.py#L534)
  - `is_multi_process(self)` — [`L555`](../../../../../../raw/code/jax/jax/_src/mesh.py#L555)
  - `local_devices(self)` — [`L559`](../../../../../../raw/code/jax/jax/_src/mesh.py#L559)
  - `local_mesh(self)` — [`L563`](../../../../../../raw/code/jax/jax/_src/mesh.py#L563)
  - `shape(self)` — [`L520`](../../../../../../raw/code/jax/jax/_src/mesh.py#L520)
  - `shape_tuple(self)` — [`L524`](../../../../../../raw/code/jax/jax/_src/mesh.py#L524)
  - `update(self, axis_sizes=None, axis_names=None, axis_types=None, **kwargs)` — [`L508`](../../../../../../raw/code/jax/jax/_src/mesh.py#L508)
  - `update_axis_types(self, name_to_type: dict[MeshAxisName, AxisType])` — [`L541`](../../../../../../raw/code/jax/jax/_src/mesh.py#L541)
  - `abstract_device` — [`L474`](../../../../../../raw/code/jax/jax/_src/mesh.py#L474)
  - `axis_sizes` — [`L473`](../../../../../../raw/code/jax/jax/_src/mesh.py#L473)
  - `size` — [`L475`](../../../../../../raw/code/jax/jax/_src/mesh.py#L475)
- protocol/private: `__enter__`[`L566`](../../../../../../raw/code/jax/jax/_src/mesh.py#L566), `__exit__`[`L569`](../../../../../../raw/code/jax/jax/_src/mesh.py#L569), `__getnewargs_ex__`[`L496`](../../../../../../raw/code/jax/jax/_src/mesh.py#L496), `__new__`[`L488`](../../../../../../raw/code/jax/jax/_src/mesh.py#L488), `__repr__`[`L500`](../../../../../../raw/code/jax/jax/_src/mesh.py#L500), `_create`[`L479`](../../../../../../raw/code/jax/jax/_src/mesh.py#L479), `_internal_device_list`[`L530`](../../../../../../raw/code/jax/jax/_src/mesh.py#L530)
- uses (calls/refs, reference-scoped): [`safe_zip`](util.md#safe_zip), [`AxisType`](mesh.md#AxisType), [`axis_types`](mesh.md#BaseMesh.axis_types), [`immutable`](util.md#immutable), [`axis_names`](mesh.md#BaseMesh.axis_names), [`weak_value_interner`](util.md#weak_value_interner), [`_normalize_axis_types`](mesh.md#_normalize_axis_types), [`_raise_value_error`](mesh.md#_raise_value_error), [`BaseMesh`](mesh.md#BaseMesh), [`MeshAxisName`](mesh.md#MeshAxisName)
- used by: [`_pallas_call_batching_rule`](pallas/pallas_call.md#_pallas_call_batching_rule), [`mesh`](named_sharding.md#NamedSharding.mesh), [`canonicalize_sharding`](sharding_impls.md#canonicalize_sharding), [`get_abstract_mesh`](mesh.md#get_abstract_mesh), [`with_sharding_constraint`](pjit.md#with_sharding_constraint), [`to_named_sharding_with_abstract_mesh`](export/_export.md#to_named_sharding_with_abstract_mesh), [`_custom_partitioning_lowering_rule`](custom_partitioning.md#_custom_partitioning_lowering_rule), [`vmap_f`](api.md#vmap.vmap_f), [`_get_named_sharding`](export/_export.md#_get_named_sharding), [`_get_new_mesh`](pjit.md#_get_new_mesh), [`pvary`](core.md#pvary), [`_unshard_shaped_array`](shard_map.md#_unshard_shaped_array), [`abstract_mesh`](mesh.md#Mesh.abstract_mesh), [`_shard_map_impl`](shard_map.md#_shard_map_impl), [`shard_map`](shard_map.md#shard_map), [`_inspect_sharding_lowering_rule`](debugging.md#_inspect_sharding_lowering_rule), [`_sharding_constraint_impl`](pjit.md#_sharding_constraint_impl), [`_spec_divisibility_error`](shard_map.md#_spec_divisibility_error), [`_check_specs_vs_args`](shard_map.md#_check_specs_vs_args), [`_inout_vma_error`](shard_map.md#_inout_vma_error), [`_pjit_batcher_for_sharding`](pjit.md#_pjit_batcher_for_sharding), [`_get_vjp_fun`](export/_export.md#_get_vjp_fun), [`_shmap_checks`](shard_map.md#_shmap_checks), [`parse_flatten_op_sharding`](sharding_impls.md#parse_flatten_op_sharding), [`create_from_uniques`](export/serialization.md#_SerializedUniques.create_from_uniques), [`_as_manual_mesh`](shard_map.md#_as_manual_mesh), [`_gspmd_to_named_sharding_via_mesh`](sharding_impls.md#_gspmd_to_named_sharding_via_mesh), [`reduced_vary_cast`](core.md#reduced_vary_cast), [`vary_unreduced_cast`](lax/parallel.md#vary_unreduced_cast), [`in_shardings_jax`](export/_export.md#Exported.in_shardings_jax), [`out_shardings_jax`](export/_export.md#Exported.out_shardings_jax), [`_serialize_abstract_mesh`](export/serialization.md#_serialize_abstract_mesh), [`check_avals_context_mesh`](core.md#check_avals_context_mesh), [`named_sharding_to_xla_hlo_sharding`](named_sharding.md#named_sharding_to_xla_hlo_sharding), [`preduced`](lax/parallel.md#preduced), [`_all_newly_manual_mesh_names`](shard_map.md#_all_newly_manual_mesh_names), [`_shard_map`](shard_map.md#_shard_map), [`call_sharding_rule`](lax/utils.md#call_sharding_rule), [`get_ctx_mesh`](pjit.md#get_ctx_mesh), [`get_mesh_from_args`](shard_map.md#get_mesh_from_args)  (+33 more)

### `AxisType`  ·  implements/extends Enum
- def: [`jax/_src/mesh.py:111`](../../../../../../raw/code/jax/jax/_src/mesh.py#L111) — documented in [jax-_src-mesh](../../../concepts/jax-_src-mesh.md)
- signature: `class AxisType(enum.Enum):`
- members:
  - `Auto` — [`L112`](../../../../../../raw/code/jax/jax/_src/mesh.py#L112)
  - `Explicit` — [`L113`](../../../../../../raw/code/jax/jax/_src/mesh.py#L113)
  - `Manual` — [`L114`](../../../../../../raw/code/jax/jax/_src/mesh.py#L114)
- protocol/private: `__repr__`[`L116`](../../../../../../raw/code/jax/jax/_src/mesh.py#L116)
- used by: [`canonicalize_sharding`](sharding_impls.md#canonicalize_sharding), [`wrapped`](shard_map.md#_shard_map.wrapped), [`_run_example`](../experimental/pallas/ops/gpu/collective_matmul_mgpu.md#_run_example), [`_get_new_mesh`](pjit.md#_get_new_mesh), [`named_sharding_to_sdy_sharding`](named_sharding.md#named_sharding_to_sdy_sharding), [`_run_example`](../experimental/pallas/ops/gpu/all_gather_mgpu.md#_run_example), [`_run_example`](../experimental/pallas/ops/gpu/reduce_scatter_mgpu.md#_run_example), [`_shmap_checks`](shard_map.md#_shmap_checks), [`axis_types`](mesh.md#BaseMesh.axis_types), [`decorator`](pjit.md#_auto_axes.decorator), [`decorator`](pjit.md#_explicit_axes.decorator), [`_as_manual_mesh`](shard_map.md#_as_manual_mesh), [`check_shardings_are_auto`](pjit.md#check_shardings_are_auto), [`manual_axes`](mesh.md#BaseMesh.manual_axes), [`explicit_axes`](mesh.md#BaseMesh.explicit_axes), [`_make_scoped_manual_sharding`](shard_map.md#_make_scoped_manual_sharding), [`are_all_axes_explicit`](mesh.md#BaseMesh.are_all_axes_explicit), [`_check_unreduced`](shard_map.md#_check_unreduced), [`get_mat`](core.md#get_mat), [`_any_axis_explicit`](mesh.md#BaseMesh._any_axis_explicit), [`_any_axis_manual`](mesh.md#BaseMesh._any_axis_manual), [`make_mesh`](sharding_impls.md#make_mesh), [`auto_axes`](mesh.md#BaseMesh.auto_axes), [`update_axis_types`](mesh.md#AbstractMesh.update_axis_types), [`_are_all_axes_auto_or_manual`](mesh.md#BaseMesh._are_all_axes_auto_or_manual), [`are_all_axes_auto`](mesh.md#BaseMesh.are_all_axes_auto), [`modify_spec_for_auto_manual`](core.md#modify_spec_for_auto_manual), [`_axis_type_to_enum`](export/serialization.md#_axis_type_to_enum), [`_check_mesh_resource_axis`](named_sharding.md#_check_mesh_resource_axis), [`are_all_axes_manual`](mesh.md#BaseMesh.are_all_axes_manual), [`__new__`](mesh.md#AbstractMesh.__new__), [`__new__`](mesh.md#Mesh.__new__), [`_any_axis_auto`](mesh.md#BaseMesh._any_axis_auto), [`_any_axis_auto_or_manual`](mesh.md#BaseMesh._any_axis_auto_or_manual), [`_are_all_axes_explicit_or_manual`](mesh.md#BaseMesh._are_all_axes_explicit_or_manual), [`_check_mesh_unreduced`](named_sharding.md#_check_mesh_unreduced), [`_normalize_axis_types`](mesh.md#_normalize_axis_types), [`all_axis_types_match`](mesh.md#all_axis_types_match), [`any_axis_types_match`](mesh.md#any_axis_types_match)

### `BaseMesh`
- def: [`jax/_src/mesh.py:146`](../../../../../../raw/code/jax/jax/_src/mesh.py#L146)
- signature: `class BaseMesh:`
- members:
  - `are_all_axes_auto(self)` — [`L156`](../../../../../../raw/code/jax/jax/_src/mesh.py#L156)
  - `are_all_axes_explicit(self)` — [`L160`](../../../../../../raw/code/jax/jax/_src/mesh.py#L160)
  - `are_all_axes_manual(self)` — [`L152`](../../../../../../raw/code/jax/jax/_src/mesh.py#L152)
  - `auto_axes(self)` — [`L197`](../../../../../../raw/code/jax/jax/_src/mesh.py#L197)
  - `explicit_axes(self)` — [`L202`](../../../../../../raw/code/jax/jax/_src/mesh.py#L202)
  - `manual_axes(self)` — [`L207`](../../../../../../raw/code/jax/jax/_src/mesh.py#L207)
  - `axis_names` — [`L147`](../../../../../../raw/code/jax/jax/_src/mesh.py#L147)
  - `axis_types` — [`L149`](../../../../../../raw/code/jax/jax/_src/mesh.py#L149)
  - `shape_tuple` — [`L148`](../../../../../../raw/code/jax/jax/_src/mesh.py#L148)
- protocol/private: `_any_axis_auto`[`L182`](../../../../../../raw/code/jax/jax/_src/mesh.py#L182), `_any_axis_auto_or_manual`[`L190`](../../../../../../raw/code/jax/jax/_src/mesh.py#L190), `_any_axis_explicit`[`L186`](../../../../../../raw/code/jax/jax/_src/mesh.py#L186), `_any_axis_manual`[`L178`](../../../../../../raw/code/jax/jax/_src/mesh.py#L178), `_are_all_axes_auto_or_manual`[`L164`](../../../../../../raw/code/jax/jax/_src/mesh.py#L164), `_are_all_axes_explicit_or_manual`[`L171`](../../../../../../raw/code/jax/jax/_src/mesh.py#L171), `_name_to_type`[`L212`](../../../../../../raw/code/jax/jax/_src/mesh.py#L212)
- uses (calls/refs, reference-scoped): [`safe_zip`](util.md#safe_zip), [`AxisType`](mesh.md#AxisType), [`Explicit`](mesh.md#AxisType.Explicit), [`Auto`](mesh.md#AxisType.Auto), [`Manual`](mesh.md#AxisType.Manual), [`all_axis_types_match`](mesh.md#all_axis_types_match), [`any_axis_types_match`](mesh.md#any_axis_types_match), [`MeshAxisName`](mesh.md#MeshAxisName)
- used by: [`bind`](core.md#Primitive.bind), [`full_like`](lax/lax.md#full_like), [`Mesh`](mesh.md#Mesh), [`canonicalize_sharding`](sharding_impls.md#canonicalize_sharding), [`wrapped`](shard_map.md#_shard_map.wrapped), [`rewriting_take`](numpy/indexing.md#rewriting_take), [`AbstractMesh`](mesh.md#AbstractMesh), [`empty_like`](lax/lax.md#empty_like), [`with_sharding_constraint`](pjit.md#with_sharding_constraint), [`_custom_partitioning_lowering_rule`](custom_partitioning.md#_custom_partitioning_lowering_rule), [`_get_named_sharding`](export/_export.md#_get_named_sharding), [`_tpu_custom_call_lowering`](tpu_custom_call.md#_tpu_custom_call_lowering), [`_get_new_mesh`](pjit.md#_get_new_mesh), [`pvary`](core.md#pvary), [`_unshard_shaped_array`](shard_map.md#_unshard_shaped_array), [`abstract_mesh`](mesh.md#Mesh.abstract_mesh), [`repeat`](numpy/lax_numpy.md#repeat), [`_inspect_sharding_lowering_rule`](debugging.md#_inspect_sharding_lowering_rule), [`canonicalize_sharding_for_samplers`](random/core.md#canonicalize_sharding_for_samplers), [`_pallas_call_abstract_eval`](pallas/pallas_call.md#_pallas_call_abstract_eval), [`_shmap_checks`](shard_map.md#_shmap_checks), [`decorator`](pjit.md#_auto_axes.decorator), [`percentile`](numpy/reductions.md#percentile), [`check_shardings_are_auto`](pjit.md#check_shardings_are_auto), [`_axis_index_effectful_abstract_eval`](lax/parallel.md#_axis_index_effectful_abstract_eval), [`insert_reduced_reshard`](core.md#insert_reduced_reshard), [`reduced_vary_cast`](core.md#reduced_vary_cast), [`vary_unreduced_cast`](lax/parallel.md#vary_unreduced_cast), [`__init__`](sharding_impls.md#set_mesh.__init__), [`_scan_leaf`](lax/control_flow/loops.md#_scan_leaf), [`_serialize_abstract_mesh`](export/serialization.md#_serialize_abstract_mesh), [`check_avals_context_mesh`](core.md#check_avals_context_mesh), [`named_sharding_to_xla_hlo_sharding`](named_sharding.md#named_sharding_to_xla_hlo_sharding), [`preduced`](lax/parallel.md#preduced), [`_all_newly_manual_mesh_names`](shard_map.md#_all_newly_manual_mesh_names), [`call_sharding_rule`](lax/utils.md#call_sharding_rule), [`get_ctx_mesh`](pjit.md#get_ctx_mesh), [`_remainder_leaf`](lax/control_flow/loops.md#_remainder_leaf), [`check_replicated_trailing_dims`](sharding_impls.md#check_replicated_trailing_dims), [`top_level_all_gather`](shard_map.md#top_level_all_gather)  (+12 more)

### `Mesh`
- def: [`jax/_src/mesh.py:217`](../../../../../../raw/code/jax/jax/_src/mesh.py#L217) — documented in [jax-_src-mesh](../../../concepts/jax-_src-mesh.md)
- members:
  - `abstract_mesh(self)` — [`L407`](../../../../../../raw/code/jax/jax/_src/mesh.py#L407)
  - `axis_sizes(self)` — [`L344`](../../../../../../raw/code/jax/jax/_src/mesh.py#L344)
  - `device_ids(self)` — [`L363`](../../../../../../raw/code/jax/jax/_src/mesh.py#L363)
  - `empty(self)` — [`L348`](../../../../../../raw/code/jax/jax/_src/mesh.py#L348)
  - `is_multi_process(self)` — [`L352`](../../../../../../raw/code/jax/jax/_src/mesh.py#L352)
  - `is_scalar(self)` — [`L293`](../../../../../../raw/code/jax/jax/_src/mesh.py#L293)
  - `local_devices(self)` — [`L402`](../../../../../../raw/code/jax/jax/_src/mesh.py#L402)
  - `local_mesh(self)` — [`L356`](../../../../../../raw/code/jax/jax/_src/mesh.py#L356)
  - `shape(self)` — [`L332`](../../../../../../raw/code/jax/jax/_src/mesh.py#L332)
  - `shape_tuple(self)` — [`L338`](../../../../../../raw/code/jax/jax/_src/mesh.py#L338)
  - `update(self, devices=None, axis_names=None, axis_types=None)` — [`L322`](../../../../../../raw/code/jax/jax/_src/mesh.py#L322)
  - `axis_names` — [`L251`](../../../../../../raw/code/jax/jax/_src/mesh.py#L251)
  - `devices` — [`L250`](../../../../../../raw/code/jax/jax/_src/mesh.py#L250)
  - `size` — [`L252`](../../../../../../raw/code/jax/jax/_src/mesh.py#L252)
- protocol/private: `__enter__`[`L299`](../../../../../../raw/code/jax/jax/_src/mesh.py#L299), `__exit__`[`L314`](../../../../../../raw/code/jax/jax/_src/mesh.py#L314), `__getnewargs_ex__`[`L296`](../../../../../../raw/code/jax/jax/_src/mesh.py#L296), `__new__`[`L266`](../../../../../../raw/code/jax/jax/_src/mesh.py#L266), `__repr__`[`L398`](../../../../../../raw/code/jax/jax/_src/mesh.py#L398), `__str__`[`L383`](../../../../../../raw/code/jax/jax/_src/mesh.py#L383), `_create`[`L256`](../../../../../../raw/code/jax/jax/_src/mesh.py#L256), `_flat_devices_set`[`L380`](../../../../../../raw/code/jax/jax/_src/mesh.py#L380), `_flat_devices_tuple`[`L372`](../../../../../../raw/code/jax/jax/_src/mesh.py#L372), `_internal_device_list`[`L376`](../../../../../../raw/code/jax/jax/_src/mesh.py#L376), `_local_devices_set`[`L368`](../../../../../../raw/code/jax/jax/_src/mesh.py#L368), `_local_mesh`[`L359`](../../../../../../raw/code/jax/jax/_src/mesh.py#L359), `_repr`[`L391`](../../../../../../raw/code/jax/jax/_src/mesh.py#L391)
- uses (calls/refs, reference-scoped): [`safe_zip`](util.md#safe_zip), [`AbstractMesh`](mesh.md#AbstractMesh), [`AxisType`](mesh.md#AxisType), [`axis_types`](mesh.md#BaseMesh.axis_types), [`immutable`](util.md#immutable), [`empty_abstract_mesh`](mesh.md#empty_abstract_mesh), [`physical_mesh`](mesh.md#ResourceEnv.physical_mesh), [`thread_resources`](mesh.md#thread_resources), [`_get_local_mesh`](mesh.md#_get_local_mesh), [`with_mesh`](mesh.md#ResourceEnv.with_mesh), [`env`](mesh.md#_ThreadResourcesLocalState.env), [`weak_value_interner`](util.md#weak_value_interner), [`stack`](mesh.md#_ThreadResourcesLocalState.stack), [`process_index`](xla_bridge.md#process_index), [`_normalize_axis_types`](mesh.md#_normalize_axis_types), [`AbstractDevice`](mesh.md#AbstractDevice), [`mesh_context_manager`](config.md#mesh_context_manager), [`BaseMesh`](mesh.md#BaseMesh), [`MeshAxisName`](mesh.md#MeshAxisName), [`num_cores`](mesh.md#AbstractDevice.num_cores), [`disallow_mesh_context_manager`](config.md#disallow_mesh_context_manager), [`device_kind`](mesh.md#AbstractDevice.device_kind), [`platform`](mesh.md#AbstractDevice.platform)
- used by: [`full_like`](lax/lax.md#full_like), [`lower_jaxpr_into_pipelined_module`](pallas/mosaic/lowering.md#lower_jaxpr_into_pipelined_module), [`_trace_for_jit`](pjit.md#_trace_for_jit), [`lower_jaxpr_to_module`](pallas/mosaic_gpu/lowering.md#lower_jaxpr_to_module), [`_export_lowered`](export/_export.md#_export_lowered), [`mesh`](named_sharding.md#NamedSharding.mesh), [`_call_exported_lowering`](export/_export.md#_call_exported_lowering), [`canonicalize_sharding`](sharding_impls.md#canonicalize_sharding), [`lower_pipelined_jaxpr_to_module`](pallas/mosaic_gpu/lowering.md#lower_pipelined_jaxpr_to_module), [`lower_jaxpr_into_unpipelined_module`](pallas/mosaic/lowering.md#lower_jaxpr_into_unpipelined_module), [`_cached_shard_map`](pmap.md#_cached_shard_map), [`empty_like`](lax/lax.md#empty_like), [`_handle_array_process_allgather`](../experimental/multihost_utils.md#_handle_array_process_allgather), [`with_sharding_constraint`](pjit.md#with_sharding_constraint), [`__call__`](custom_partitioning.md#custom_partitioning.__call__), [`to_named_sharding_with_abstract_mesh`](export/_export.md#to_named_sharding_with_abstract_mesh), [`_custom_partitioning_lowering_rule`](custom_partitioning.md#_custom_partitioning_lowering_rule), [`_pjit_lowering`](pjit.md#_pjit_lowering), [`reshard`](pjit.md#reshard), [`_get_named_sharding`](export/_export.md#_get_named_sharding), [`_callback_op_sharding`](callback.md#_callback_op_sharding), [`_tpu_custom_call_lowering`](tpu_custom_call.md#_tpu_custom_call_lowering), [`_shard_map_staging`](shard_map.md#_shard_map_staging), [`__init__`](pallas/mosaic/lowering.md#MosaicGridMapping.__init__), [`process_shard_map`](shard_map.md#ShardMapTrace.process_shard_map), [`_get_new_mesh`](pjit.md#_get_new_mesh), [`_unshard_shaped_array`](shard_map.md#_unshard_shaped_array), [`global_array_to_host_local_array_impl`](../experimental/multihost_utils.md#global_array_to_host_local_array_impl), [`host_local_array_to_global_array_impl`](../experimental/multihost_utils.md#host_local_array_to_global_array_impl), [`_kernel_to_module`](../experimental/mosaic/gpu/core.md#_kernel_to_module), [`debug_callback_lowering`](debugging.md#debug_callback_lowering), [`_shard_map_impl`](shard_map.md#_shard_map_impl), [`_mosaic_gpu_lowering_rule`](../experimental/mosaic/gpu/core.md#_mosaic_gpu_lowering_rule), [`_resolve_in_shardings`](pjit.md#_resolve_in_shardings), [`prepare_axis_resources`](sharding_impls.md#prepare_axis_resources), [`shard_map`](shard_map.md#shard_map), [`_inspect_sharding_lowering_rule`](debugging.md#_inspect_sharding_lowering_rule), [`_sharding_constraint_impl`](pjit.md#_sharding_constraint_impl), [`_spec_divisibility_error`](shard_map.md#_spec_divisibility_error), [`_check_specs_vs_args`](shard_map.md#_check_specs_vs_args)  (+72 more)

### `ResourceEnv`  ·  implements/extends NamedTuple
- def: [`jax/_src/mesh.py:49`](../../../../../../raw/code/jax/jax/_src/mesh.py#L49)
- signature: `class ResourceEnv(NamedTuple):`
- members:
  - `local_shape(self)` — [`L73`](../../../../../../raw/code/jax/jax/_src/mesh.py#L73)
  - `physical_resource_axes(self)` — [`L61`](../../../../../../raw/code/jax/jax/_src/mesh.py#L61)
  - `resource_axes(self)` — [`L65`](../../../../../../raw/code/jax/jax/_src/mesh.py#L65)
  - `shape(self)` — [`L69`](../../../../../../raw/code/jax/jax/_src/mesh.py#L69)
  - `with_mesh(self, mesh: Mesh)` — [`L52`](../../../../../../raw/code/jax/jax/_src/mesh.py#L52)
  - `physical_mesh` — [`L50`](../../../../../../raw/code/jax/jax/_src/mesh.py#L50)
- protocol/private: `__repr__`[`L76`](../../../../../../raw/code/jax/jax/_src/mesh.py#L76)
- uses (calls/refs, reference-scoped): [`Mesh`](mesh.md#Mesh), [`axis_names`](mesh.md#Mesh.axis_names), [`shape`](mesh.md#Mesh.shape), [`local_mesh`](mesh.md#Mesh.local_mesh), [`ResourceAxisName`](mesh.md#ResourceAxisName), [`show_axes`](mesh.md#show_axes)
- used by: [`with_sharding_constraint`](pjit.md#with_sharding_constraint), [`__call__`](custom_partitioning.md#custom_partitioning.__call__), [`_custom_partitioning_lowering_rule`](custom_partitioning.md#_custom_partitioning_lowering_rule), [`_kernel_to_module`](../experimental/mosaic/gpu/core.md#_kernel_to_module), [`_inspect_sharding_lowering_rule`](debugging.md#_inspect_sharding_lowering_rule), [`get_ctx_mesh`](pjit.md#get_ctx_mesh), [`__enter__`](mesh.md#Mesh.__enter__), [`__exit__`](mesh.md#Mesh.__exit__), [`EMPTY_ENV`](mesh.md#EMPTY_ENV)

### `_ThreadResourcesLocalState`  ·  implements/extends local
- def: [`jax/_src/mesh.py:429`](../../../../../../raw/code/jax/jax/_src/mesh.py#L429)
- signature: `class _ThreadResourcesLocalState(threading.local):`
- members:
  - `env` — [`L433`](../../../../../../raw/code/jax/jax/_src/mesh.py#L433)
  - `stack` — [`L432`](../../../../../../raw/code/jax/jax/_src/mesh.py#L432)
- protocol/private: `__init__`[`L431`](../../../../../../raw/code/jax/jax/_src/mesh.py#L431)
- uses (calls/refs, reference-scoped): [`EMPTY_ENV`](mesh.md#EMPTY_ENV)
- used by: [`with_sharding_constraint`](pjit.md#with_sharding_constraint), [`__call__`](custom_partitioning.md#custom_partitioning.__call__), [`_custom_partitioning_lowering_rule`](custom_partitioning.md#_custom_partitioning_lowering_rule), [`_kernel_to_module`](../experimental/mosaic/gpu/core.md#_kernel_to_module), [`_inspect_sharding_lowering_rule`](debugging.md#_inspect_sharding_lowering_rule), [`get_ctx_mesh`](pjit.md#get_ctx_mesh), [`__enter__`](mesh.md#Mesh.__enter__), [`thread_resources`](mesh.md#thread_resources), [`__exit__`](mesh.md#Mesh.__exit__)

### `use_abstract_mesh`
- def: [`jax/_src/mesh.py:581`](../../../../../../raw/code/jax/jax/_src/mesh.py#L581)
- doc: Sets a abstract mesh in a thread-local context.
- signature: `class use_abstract_mesh:`
- protocol/private: `__enter__`[`L616`](../../../../../../raw/code/jax/jax/_src/mesh.py#L616), `__exit__`[`L627`](../../../../../../raw/code/jax/jax/_src/mesh.py#L627), `__init__`[`L609`](../../../../../../raw/code/jax/jax/_src/mesh.py#L609), `__slots__`[`L607`](../../../../../../raw/code/jax/jax/_src/mesh.py#L607)
- uses (calls/refs, reference-scoped): [`AbstractMesh`](mesh.md#AbstractMesh), [`empty`](mesh.md#AbstractMesh.empty), [`abstract_mesh_context_manager`](config.md#abstract_mesh_context_manager), [`size`](mesh.md#AbstractMesh.size), [`config_ext`](mesh.md#config_ext)
- used by: [`_trace_for_jit`](pjit.md#_trace_for_jit), [`_partial_eval_jaxpr_custom_rule`](shard_map.md#_partial_eval_jaxpr_custom_rule), [`process_primitive`](shard_map.md#ShardMapTrace.process_primitive), [`_shard_map_staging`](shard_map.md#_shard_map_staging), [`process_shard_map`](shard_map.md#ShardMapTrace.process_shard_map), [`_shard_map_dce`](shard_map.md#_shard_map_dce), [`_shard_map_to_lojax`](shard_map.md#_shard_map_to_lojax), [`_shard_map_linearize`](shard_map.md#_shard_map_linearize), [`_shard_map_discharge`](shard_map.md#_shard_map_discharge), [`_shard_map_remat`](shard_map.md#_shard_map_remat), [`decorator`](pjit.md#_auto_axes.decorator), [`decorator`](pjit.md#_explicit_axes.decorator), [`to_concrete_value`](shard_map.md#ShardMapTracer.to_concrete_value), [`_run_shmap_lu`](shard_map.md#_run_shmap_lu), [`__str__`](shard_map.md#ShardMapTracer.__str__), [`debug_callback_impl`](debugging.md#debug_callback_impl), [`_unmatch_spec`](shard_map.md#_unmatch_spec), [`_run_shmap`](shard_map.md#_run_shmap), [`_match_spec2`](shard_map.md#_match_spec2), [`_unmatch_spec2`](shard_map.md#_unmatch_spec2), [`shard_device_array`](array.md#shard_device_array), [`_empty_array`](lax/control_flow/loops.md#_empty_array), [`body_fun`](lax/control_flow/loops.md#_scan_impl.body_fun), [`_reshard_transpose_fancy`](pjit.md#_reshard_transpose_fancy), [`empty_like_shaped_array`](lax/lax.md#empty_like_shaped_array)

## Functions
- `_get_local_mesh(global_mesh: Mesh, process_index: int)` — [`L83`](../../../../../../raw/code/jax/jax/_src/mesh.py#L83)
- `_normalize_axis_types(axis_names, axis_types, name)` — [`L119`](../../../../../../raw/code/jax/jax/_src/mesh.py#L119)
- `_raise_value_error(name)` — [`L575`](../../../../../../raw/code/jax/jax/_src/mesh.py#L575)
- `all_axis_types_match(axis_types, ty: AxisType)` — [`L135`](../../../../../../raw/code/jax/jax/_src/mesh.py#L135)
- `any_axis_types_match(axis_types, ty: AxisType)` — [`L140`](../../../../../../raw/code/jax/jax/_src/mesh.py#L140)
- `get_abstract_mesh()` — [`L631`](../../../../../../raw/code/jax/jax/_src/mesh.py#L631) — documented in [jax-_src-mesh](../../../concepts/jax-_src-mesh.md)
- `get_concrete_mesh()` — [`L635`](../../../../../../raw/code/jax/jax/_src/mesh.py#L635)
- `show_axes(axes)` — [`L45`](../../../../../../raw/code/jax/jax/_src/mesh.py#L45)

## Module values
- `EMPTY_ENV` — [`L427`](../../../../../../raw/code/jax/jax/_src/mesh.py#L427)
- `MeshAxisName` — [`L41`](../../../../../../raw/code/jax/jax/_src/mesh.py#L41)
- `ResourceAxisName` — [`L42`](../../../../../../raw/code/jax/jax/_src/mesh.py#L42)
- `config_ext` — [`L39`](../../../../../../raw/code/jax/jax/_src/mesh.py#L39)
- `empty_abstract_mesh` — [`L578`](../../../../../../raw/code/jax/jax/_src/mesh.py#L578)
- `empty_concrete_mesh` — [`L579`](../../../../../../raw/code/jax/jax/_src/mesh.py#L579)
- `thread_resources` — [`L435`](../../../../../../raw/code/jax/jax/_src/mesh.py#L435)
- `unsafe_zip` — [`L38`](../../../../../../raw/code/jax/jax/_src/mesh.py#L38)


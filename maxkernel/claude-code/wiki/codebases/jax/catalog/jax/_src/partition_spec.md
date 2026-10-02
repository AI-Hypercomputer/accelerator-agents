---
title: 'Module: jax/_src/partition_spec.py'
type: catalog
provenance: extracted
module: jax/_src/partition_spec.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.partition_spec`/
symbols:
  PartitionSpec: PartitionSpec.
  P.unreduced: P#unreduced.
  P.UNCONSTRAINED: P#UNCONSTRAINED.
  P.reduced: P#reduced.
  P._normalized_spec_for_aval: P#_normalized_spec_for_aval().
  P._partitions: P#_partitions.
  P.partitions: P#partitions().
  P: P#
  P.update: P#update().
  _UNCONSTRAINED_PARTITION: _UNCONSTRAINED_PARTITION.
  P.__repr__: P#__repr__().
  P.__add__: P#__add__().
  P.__new__: P#__new__().
  P.__getnewargs_ex__: P#__getnewargs_ex__().
  P.__getitem__: P#__getitem__().
  P.__iter__: P#__iter__().
  P.__radd__: P#__radd__().
  P.to_ct_spec: P#to_ct_spec().
  P._create: P#_create().
  P.index: P#index().
  P.count: P#count().
  _get_default_unconstrained: _get_default_unconstrained().
  P._check_compatible_wrt_shape: P#_check_compatible_wrt_shape().
  AxisName: AxisName.
  UnconstrainedSingleton.__reduce__: UnconstrainedSingleton#__reduce__().
  P.__len__: P#__len__().
  _canonicalize_partition: _canonicalize_partition.
  _check: _check().
  _get_ur_str: _get_ur_str().
  _canonicalize_partitions: _canonicalize_partitions.
  UnconstrainedSingleton: UnconstrainedSingleton#
  UnconstrainedSingleton.__repr__: UnconstrainedSingleton#__repr__().
  P.__slots__: P#__slots__.
  P.__init_subclass__: P#__init_subclass__().
  P.to_lo: P#to_lo().
  P.to_tangent_spec: P#to_tangent_spec().
---
# Module: [`jax/_src/partition_spec.py`](../../../../../../raw/code/jax/jax/_src/partition_spec.py)

## Classes
### `P`
- def: [`jax/_src/partition_spec.py:75`](../../../../../../raw/code/jax/jax/_src/partition_spec.py#L75)
- doc: Tuple describing how to partition an array across a mesh of devices.
- signature: `class P:`
- members:
  - `count(self, value)` — [`L193`](../../../../../../raw/code/jax/jax/_src/partition_spec.py#L193)
  - `index(self, value)` — [`L190`](../../../../../../raw/code/jax/jax/_src/partition_spec.py#L190)
  - `partitions(self)` — [`L126`](../../../../../../raw/code/jax/jax/_src/partition_spec.py#L126)
  - `to_ct_spec(self)` — [`L208`](../../../../../../raw/code/jax/jax/_src/partition_spec.py#L208)
  - `to_lo(self)` — [`L202`](../../../../../../raw/code/jax/jax/_src/partition_spec.py#L202)
  - `to_tangent_spec(self)` — [`L205`](../../../../../../raw/code/jax/jax/_src/partition_spec.py#L205)
  - `update(self, partitions=None, unreduced=None, reduced=None)` — [`L196`](../../../../../../raw/code/jax/jax/_src/partition_spec.py#L196)
  - `UNCONSTRAINED` — [`L90`](../../../../../../raw/code/jax/jax/_src/partition_spec.py#L90)
  - `reduced` — [`L87`](../../../../../../raw/code/jax/jax/_src/partition_spec.py#L87)
  - `unreduced` — [`L86`](../../../../../../raw/code/jax/jax/_src/partition_spec.py#L86)
- protocol/private: `__add__`[`L158`](../../../../../../raw/code/jax/jax/_src/partition_spec.py#L158), `__getitem__`[`L141`](../../../../../../raw/code/jax/jax/_src/partition_spec.py#L141), `__getnewargs_ex__`[`L137`](../../../../../../raw/code/jax/jax/_src/partition_spec.py#L137), `__init_subclass__`[`L122`](../../../../../../raw/code/jax/jax/_src/partition_spec.py#L122), `__iter__`[`L148`](../../../../../../raw/code/jax/jax/_src/partition_spec.py#L148), `__len__`[`L155`](../../../../../../raw/code/jax/jax/_src/partition_spec.py#L155), `__new__`[`L104`](../../../../../../raw/code/jax/jax/_src/partition_spec.py#L104), `__radd__`[`L176`](../../../../../../raw/code/jax/jax/_src/partition_spec.py#L176), `__repr__`[`L129`](../../../../../../raw/code/jax/jax/_src/partition_spec.py#L129), `__slots__`[`L84`](../../../../../../raw/code/jax/jax/_src/partition_spec.py#L84), `_check_compatible_wrt_shape`[`L218`](../../../../../../raw/code/jax/jax/_src/partition_spec.py#L218), `_create`[`L94`](../../../../../../raw/code/jax/jax/_src/partition_spec.py#L94), `_normalized_spec_for_aval`[`L211`](../../../../../../raw/code/jax/jax/_src/partition_spec.py#L211), `_partitions`[`L85`](../../../../../../raw/code/jax/jax/_src/partition_spec.py#L85)
- uses (calls/refs, reference-scoped): [`_UNCONSTRAINED_PARTITION`](partition_spec.md#_UNCONSTRAINED_PARTITION), [`weak_value_interner`](util.md#weak_value_interner), [`AxisName`](partition_spec.md#AxisName), [`_canonicalize_partition`](partition_spec.md#_canonicalize_partition), [`_canonicalize_partitions`](partition_spec.md#_canonicalize_partitions), [`_check`](partition_spec.md#_check), [`_get_ur_str`](partition_spec.md#_get_ur_str)
- used by: [`PartitionSpec`](partition_spec.md#PartitionSpec), [`Device`](../__init__.md#Device), [`canonicalize_sharding`](sharding_impls.md#canonicalize_sharding), [`wrapped`](shard_map.md#_shard_map.wrapped), [`reshard`](pjit.md#reshard), [`broadcast_to`](lax/lax.md#broadcast_to), [`prepare_axis_resources`](sharding_impls.md#prepare_axis_resources), [`named_sharding_to_sdy_sharding`](named_sharding.md#named_sharding_to_sdy_sharding), [`_sharding_constraint_hlo_lowering`](pjit.md#_sharding_constraint_hlo_lowering), [`_pjit_batcher_for_sharding`](pjit.md#_pjit_batcher_for_sharding), [`__repr__`](array.md#ArrayImpl.__repr__), [`update_aval_with_sharding`](core.md#update_aval_with_sharding), [`check_shardings_are_auto`](pjit.md#check_shardings_are_auto), [`_sharding_constraint_batcher`](pjit.md#_sharding_constraint_batcher), [`_top_level_ag`](shard_map.md#_top_level_ag), [`__format__`](array.md#ArrayImpl.__format__), [`named_sharding_to_xla_hlo_sharding`](named_sharding.md#named_sharding_to_xla_hlo_sharding), [`assert_shardings_equal`](pjit.md#assert_shardings_equal), [`__str__`](array.md#ArrayImpl.__str__), [`add_implicit_pvary_and_unreduced`](shard_map.md#_shard_map.wrapped.f_wrapped.add_implicit_pvary_and_unreduced), [`is_replicated_or_unreduced`](numpy/lax_numpy.md#is_replicated_or_unreduced), [`_check_unique_resources`](named_sharding.md#_check_unique_resources), [`_convert_element_type_ur_rule`](lax/lax.md#_convert_element_type_ur_rule), [`_manual_spec`](shard_map.md#_manual_spec), [`_serialize_partition_spec`](export/serialization.md#_serialize_partition_spec), [`get_array_mapping`](named_sharding.md#get_array_mapping), [`is_fully_replicated`](named_sharding.md#NamedSharding.is_fully_replicated), [`flatten_spec`](named_sharding.md#flatten_spec), [`remove_size_one_mesh_axis`](named_sharding.md#remove_size_one_mesh_axis), [`_check_mesh_resource_axis`](named_sharding.md#_check_mesh_resource_axis), [`get_unconstrained_dims`](pjit.md#get_unconstrained_dims), [`get_replicated_axes`](named_sharding.md#get_replicated_axes), [`_spec_to_names`](shard_map.md#_spec_to_names), [`check_spec`](shard_map.md#_check_specs.check_spec), [`_check_sharding`](core.md#_check_sharding), [`f_shmap`](shard_map.md#_top_level_ag.f_shmap)

### `UnconstrainedSingleton`
- def: [`jax/_src/partition_spec.py:67`](../../../../../../raw/code/jax/jax/_src/partition_spec.py#L67)
- signature: `class UnconstrainedSingleton:`
- protocol/private: `__reduce__`[`L69`](../../../../../../raw/code/jax/jax/_src/partition_spec.py#L69), `__repr__`[`L68`](../../../../../../raw/code/jax/jax/_src/partition_spec.py#L68)
- uses (calls/refs, reference-scoped): [`_get_default_unconstrained`](partition_spec.md#_get_default_unconstrained)
- used by: [`_UNCONSTRAINED_PARTITION`](partition_spec.md#_UNCONSTRAINED_PARTITION)

## Functions
- `_check(partitions, unreduced, reduced)` — [`L23`](../../../../../../raw/code/jax/jax/_src/partition_spec.py#L23)
- `_get_default_unconstrained()` — [`L65`](../../../../../../raw/code/jax/jax/_src/partition_spec.py#L65)
- `_get_ur_str(unreduced, reduced)` — [`L53`](../../../../../../raw/code/jax/jax/_src/partition_spec.py#L53)

## Module values
- `AxisName` — [`L21`](../../../../../../raw/code/jax/jax/_src/partition_spec.py#L21)
- `PartitionSpec` — [`L228`](../../../../../../raw/code/jax/jax/_src/partition_spec.py#L228)
- `_UNCONSTRAINED_PARTITION` — [`L71`](../../../../../../raw/code/jax/jax/_src/partition_spec.py#L71)
- `_canonicalize_partition` — [`L62`](../../../../../../raw/code/jax/jax/_src/partition_spec.py#L62)
- `_canonicalize_partitions` — [`L63`](../../../../../../raw/code/jax/jax/_src/partition_spec.py#L63)


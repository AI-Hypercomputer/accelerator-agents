---
title: 'Module: jax/_src/export/serialization.py'
type: catalog
provenance: extracted
module: jax/_src/export/serialization.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.export.serialization`/
symbols:
  _serialize_exported: _serialize_exported().
  _deserialize_exported: _deserialize_exported().
  _dtype_to_dtype_kind: _dtype_to_dtype_kind.
  _SerializedUniques.create_from_uniques: _SerializedUniques#create_from_uniques().
  _serialize_named_sharding: _serialize_named_sharding().
  _deserialize_aval: _deserialize_aval().
  _serialize_abstract_mesh: _serialize_abstract_mesh().
  _SerializedUniques.create_from_exported: _SerializedUniques#create_from_exported().
  _serialize_array: _serialize_array().
  _SerializedUniques.add_named_sharding: _SerializedUniques#add_named_sharding().
  _serialize_aval: _serialize_aval().
  _serialize_partition_spec: _serialize_partition_spec().
  _serialize_pytreedef: _serialize_pytreedef().
  _deserialize_named_sharding: _deserialize_named_sharding().
  _memory_space_to_enum: _memory_space_to_enum.
  _axis_type_to_enum: _axis_type_to_enum.
  _deserialize_sharding: _deserialize_sharding().
  _serialize_abstract_device: _serialize_abstract_device().
  _deserialize_exported.get_aval_by_idx: _deserialize_exported().get_aval_by_idx().
  _deserialize_pytreedef: _deserialize_pytreedef().
  _deserialize_pytreedef_to_pytree: _deserialize_pytreedef_to_pytree().
  _deserialize_abstract_mesh: _deserialize_abstract_mesh().
  _serialize_sharding: _serialize_sharding().
  _SerializedUniques: _SerializedUniques#
  _SerializedUniques.unique_abstract_meshes: _SerializedUniques#unique_abstract_meshes.
  _SerializedUniques.add_aval: _SerializedUniques#add_aval().
  _serialize_disabled_safety_check: _serialize_disabled_safety_check().
  _deserialize_disabled_safety_check: _deserialize_disabled_safety_check().
  _SerializedUniques.unique_avals: _SerializedUniques#unique_avals.
  _SerializedUniques.avals_map: _SerializedUniques#avals_map.
  _SerializedUniques.unique_named_shardings: _SerializedUniques#unique_named_shardings.
  _SerializedUniques.named_shardings_map: _SerializedUniques#named_shardings_map.
  serialize: serialize().
  deserialize: deserialize().
  _deserialize_effect: _deserialize_effect().
  _SerializedUniques.abstract_meshes_map: _SerializedUniques#abstract_meshes_map.
  _deserialize_abstract_device: _deserialize_abstract_device().
  _deserialize_partition_spec: _deserialize_partition_spec().
  register_dtype_kind: register_dtype_kind().
  _deserialize_exported.sharding_by_idx: _deserialize_exported().sharding_by_idx().
  _dtype_kind_to_dtype: _dtype_kind_to_dtype.
  _serialize_effect: _serialize_effect().
  _memory_space_from_enum: _memory_space_from_enum.
  _axis_type_from_enum: _axis_type_from_enum.
  _serialize_partition_spec_one_axis: _serialize_partition_spec_one_axis().
  e: e.
  T: T.
  _serialize_pytreedef.serialize_key: _serialize_pytreedef().serialize_key().
  get_platform_from_device_kind: get_platform_from_device_kind().
  _deserialize_partition_spec_one_axis: _deserialize_partition_spec_one_axis().
  SerT: SerT.
  _SERIALIZATION_VERSION: _SERIALIZATION_VERSION.
---
# Module: [`jax/_src/export/serialization.py`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py)

## Classes
### `_SerializedUniques`
- def: [`jax/_src/export/serialization.py:79`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L79)
- signature: `class _SerializedUniques:`
- members:
  - `add_aval(self, aval: core.AbstractValue)` — [`L111`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L111)
  - `add_named_sharding(self, sharding: named_sharding.NamedSharding | None)` — [`L116`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L116)
  - `create_from_exported(exp: _export.Exported)` — [`L89`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L89)
  - `create_from_uniques(unique_avals: list[core.AbstractValue], unique_abstract_meshes: list[mesh.AbstractMesh], unique_named_shardings: list[named_sharding.NamedSharding])` — [`L99`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L99)
  - `abstract_meshes_map` — [`L84`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L84)
  - `avals_map` — [`L82`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L82)
  - `named_shardings_map` — [`L86`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L86)
  - `unique_abstract_meshes` — [`L83`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L83)
  - `unique_avals` — [`L81`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L81)
  - `unique_named_shardings` — [`L85`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L85)
- uses (calls/refs, reference-scoped): [`NamedSharding`](../named_sharding.md#NamedSharding), [`AbstractValue`](../core.md#AbstractValue), [`mesh`](../named_sharding.md#NamedSharding.mesh), [`AbstractMesh`](../mesh.md#AbstractMesh), [`abstract_mesh`](../mesh.md#Mesh.abstract_mesh), [`Exported`](_export.md#Exported), [`in_avals`](_export.md#Exported.in_avals), [`_out_named_shardings`](_export.md#Exported._out_named_shardings), [`out_avals`](_export.md#Exported.out_avals), [`_in_named_shardings`](_export.md#Exported._in_named_shardings)
- used by: [`_serialize_exported`](serialization.md#_serialize_exported), [`_deserialize_exported`](serialization.md#_deserialize_exported), [`_serialize_named_sharding`](serialization.md#_serialize_named_sharding), [`_deserialize_named_sharding`](serialization.md#_deserialize_named_sharding), [`_deserialize_sharding`](serialization.md#_deserialize_sharding), [`_serialize_sharding`](serialization.md#_serialize_sharding), [`get_aval_by_idx`](serialization.md#_deserialize_exported.get_aval_by_idx), [`sharding_by_idx`](serialization.md#_deserialize_exported.sharding_by_idx)

## Functions
- `_deserialize_abstract_device(ser_abs_device: ser_flatbuf.AbstractDevice | None)` — [`L659`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L659)
- `_deserialize_abstract_mesh(ser_mesh: ser_flatbuf.AbstractMesh)` — [`L701`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L701)
- `_deserialize_aval(aval: ser_flatbuf.AbstractValue, *, scope: shape_poly.SymbolicScope, sharding: named_sharding.NamedSharding | None)` — [`L821`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L821)
- `_deserialize_disabled_safety_check(sc: ser_flatbuf.DisabledSafetyCheck)` — [`L938`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L938)
- `_deserialize_effect(eff: ser_flatbuf.Effect)` — [`L898`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L898)
- `_deserialize_exported(exp: ser_flatbuf.Exported)` — [`L281`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L281)
- `_deserialize_named_sharding(s: ser_flatbuf.NamedSharding, *, uniques: _SerializedUniques)` — [`L789`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L789)
- `_deserialize_partition_spec(spec: ser_flatbuf.PartitionSpec)` — [`L756`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L756)
- `_deserialize_partition_spec_one_axis(spec: ser_flatbuf.PartitionSpecOneAxis)` — [`L729`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L729)
- `_deserialize_pytreedef(p: ser_flatbuf.PyTreeDef, py_tree_leaves: Sequence[Any] | None = None)` — [`L517`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L517)
- `_deserialize_pytreedef_to_pytree(p: ser_flatbuf.PyTreeDef, leaf_iterator: Iterator[Any])` — [`L538`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L538) — Deserializes a PyTreeDef into a PyTree using an iterator over leaves.
- `_deserialize_sharding(s: ser_flatbuf.Sharding, *, uniques: _SerializedUniques)` — [`L856`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L856)
- `_serialize_abstract_device(builder: flatbuffers.Builder, device: mesh.AbstractDevice | None)` — [`L633`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L633)
- `_serialize_abstract_mesh(builder: flatbuffers.Builder, mesh: mesh.AbstractMesh)` — [`L673`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L673)
- `_serialize_array(builder: flatbuffers.Builder, serialize_one: Callable[[flatbuffers.Builder, T], int], elements: Iterable[T])` — [`L268`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L268)
- `_serialize_aval(builder: flatbuffers.Builder, aval: core.ShapedArray)` — [`L803`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L803)
- `_serialize_disabled_safety_check(builder: flatbuffers.Builder, check: _export.DisabledSafetyCheck)` — [`L916`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L916)
- `_serialize_effect(builder: flatbuffers.Builder, eff: core.Effect)` — [`L872`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L872)
- `_serialize_exported(builder: flatbuffers.Builder, exp: _export.Exported, vjp_order: int)` — [`L154`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L154)
- `_serialize_named_sharding(builder: flatbuffers.Builder, sharding: named_sharding.NamedSharding, *, uniques: _SerializedUniques)` — [`L769`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L769)
- `_serialize_partition_spec(builder: flatbuffers.Builder, spec: partition_spec.PartitionSpec)` — [`L738`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L738)
- `_serialize_partition_spec_one_axis(builder: flatbuffers.Builder, spec: str | tuple[str, ...] | None)` — [`L714`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L714)
- `_serialize_pytreedef(builder: flatbuffers.Builder, p: tree_util.PyTreeDef)` — [`L450`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L450)
- `_serialize_sharding(builder: flatbuffers.Builder, s: _export.NamedSharding | None, *, uniques: _SerializedUniques)` — [`L842`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L842)
- `deserialize(ser: bytearray)` — [`L144`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L144) — Deserializes an Exported.
- `get_aval_by_idx(idx, sharding: _export.NamedSharding | None)` — [`L346`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L346)
- `get_platform_from_device_kind(device_kind)` — [`L648`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L648)
- `register_dtype_kind(dtype: Any, kind: int)` — [`L612`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L612)
- `serialize(exp: _export.Exported, vjp_order: int = 0)` — [`L128`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L128) — Serializes an Exported.
- `serialize_key(builder, k)` — [`L477`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L477)
- `sharding_by_idx(idx)` — [`L308`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L308)

## Module values
- `SerT` — [`L49`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L49)
- `T` — [`L48`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L48)
- `_SERIALIZATION_VERSION` — [`L75`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L75)
- `_axis_type_from_enum` — [`L630`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L630)
- `_axis_type_to_enum` — [`L625`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L625)
- `_dtype_kind_to_dtype` — [`L607`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L607)
- `_dtype_to_dtype_kind` — [`L577`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L577)
- `_memory_space_from_enum` — [`L622`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L622)
- `_memory_space_to_enum` — [`L617`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L617)
- `e` — [`L29`](../../../../../../../raw/code/jax/jax/_src/export/serialization.py#L29)


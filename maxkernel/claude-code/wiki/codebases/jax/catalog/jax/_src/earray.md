---
title: 'Module: jax/_src/earray.py'
type: catalog
provenance: extracted
module: jax/_src/earray.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.earray`/
symbols:
  EArray: EArray#
  EArray.sharding: EArray#sharding().
  EArray.device: EArray#device().
  EArray.copy: EArray#copy().
  EArray.__len__: EArray#__len__().
  EArray.aval: EArray#aval().
  _earray_shard_arg_handler: _earray_shard_arg_handler().
  EArray.__init__: EArray#__init__().
  EArray.block_until_ready: EArray#block_until_ready().
  EArray.copy_to_host_async: EArray#copy_to_host_async().
  EArray.__repr__: EArray#__repr__().
  EArray.__iter__: EArray#__iter__().
  EArray.committed: EArray#committed().
  EArray.addressable_data: EArray#addressable_data().
  unsafe_map: unsafe_map.
  unsafe_zip: unsafe_zip.
  EArray.__slots__: EArray#__slots__.
  EArray.__hash__: EArray#__hash__.
  EArray.__array_priority__: EArray#__array_priority__.
  EArray.shape: EArray#shape.
  EArray.dtype: EArray#dtype.
  EArray.ndim: EArray#ndim.
  EArray.size: EArray#size.
  EArray.itemsize: EArray#itemsize.
  EArray.devices: EArray#devices.
  EArray._committed: EArray#_committed.
  EArray.is_fully_addressable: EArray#is_fully_addressable.
  EArray.is_fully_replicated: EArray#is_fully_replicated.
  EArray.delete: EArray#delete.
  EArray.is_deleted: EArray#is_deleted.
  EArray.on_device_size_in_bytes: EArray#on_device_size_in_bytes.
  EArray.unsafe_buffer_pointer: EArray#unsafe_buffer_pointer.
  EArray.addressable_shards: EArray#addressable_shards().
  EArray.global_shards: EArray#global_shards().
---
# Module: [`jax/_src/earray.py`](../../../../../../raw/code/jax/jax/_src/earray.py)

## Classes
### `EArray`  ·  implements/extends Array
- def: [`jax/_src/earray.py:31`](../../../../../../raw/code/jax/jax/_src/earray.py#L31)
- signature: `class EArray(basearray.Array):`
- members:
  - `addressable_data(self, index: int)` — [`L101`](../../../../../../raw/code/jax/jax/_src/earray.py#L101)
  - `addressable_shards(self)` — [`L105`](../../../../../../raw/code/jax/jax/_src/earray.py#L105)
  - `aval(self)` — [`L41`](../../../../../../raw/code/jax/jax/_src/earray.py#L41)
  - `block_until_ready(self)` — [`L44`](../../../../../../raw/code/jax/jax/_src/earray.py#L44)
  - `committed(self)` — [`L90`](../../../../../../raw/code/jax/jax/_src/earray.py#L90)
  - `copy(self)` — [`L51`](../../../../../../raw/code/jax/jax/_src/earray.py#L51)
  - `copy_to_host_async(self)` — [`L48`](../../../../../../raw/code/jax/jax/_src/earray.py#L48)
  - `device(self)` — [`L94`](../../../../../../raw/code/jax/jax/_src/earray.py#L94)
  - `global_shards(self)` — [`L109`](../../../../../../raw/code/jax/jax/_src/earray.py#L109)
  - `sharding(self)` — [`L85`](../../../../../../raw/code/jax/jax/_src/earray.py#L85)
  - `delete` — [`L78`](../../../../../../raw/code/jax/jax/_src/earray.py#L78)
  - `devices` — [`L74`](../../../../../../raw/code/jax/jax/_src/earray.py#L74)
  - `dtype` — [`L63`](../../../../../../raw/code/jax/jax/_src/earray.py#L63)
  - `is_deleted` — [`L79`](../../../../../../raw/code/jax/jax/_src/earray.py#L79)
  - `is_fully_addressable` — [`L76`](../../../../../../raw/code/jax/jax/_src/earray.py#L76)
  - `is_fully_replicated` — [`L77`](../../../../../../raw/code/jax/jax/_src/earray.py#L77)
  - `itemsize` — [`L68`](../../../../../../raw/code/jax/jax/_src/earray.py#L68)
  - `ndim` — [`L66`](../../../../../../raw/code/jax/jax/_src/earray.py#L66)
  - `on_device_size_in_bytes` — [`L80`](../../../../../../raw/code/jax/jax/_src/earray.py#L80)
  - `shape` — [`L62`](../../../../../../raw/code/jax/jax/_src/earray.py#L62)
  - `size` — [`L67`](../../../../../../raw/code/jax/jax/_src/earray.py#L67)
  - `unsafe_buffer_pointer` — [`L81`](../../../../../../raw/code/jax/jax/_src/earray.py#L81)
- protocol/private: `__array_priority__`[`L34`](../../../../../../raw/code/jax/jax/_src/earray.py#L34), `__hash__`[`L33`](../../../../../../raw/code/jax/jax/_src/earray.py#L33), `__init__`[`L36`](../../../../../../raw/code/jax/jax/_src/earray.py#L36), `__iter__`[`L57`](../../../../../../raw/code/jax/jax/_src/earray.py#L57), `__len__`[`L69`](../../../../../../raw/code/jax/jax/_src/earray.py#L69), `__repr__`[`L54`](../../../../../../raw/code/jax/jax/_src/earray.py#L54), `__slots__`[`L32`](../../../../../../raw/code/jax/jax/_src/earray.py#L32), `_committed`[`L75`](../../../../../../raw/code/jax/jax/_src/earray.py#L75)
- uses (calls/refs, reference-scoped): [`Array`](basearray.md#Array), [`shape`](basearray.md#Array.shape), [`dtype`](basearray.md#Array.dtype), [`ndim`](basearray.md#Array.ndim), [`logical_sharding`](sharding_impls.md#logical_sharding), [`device_set`](sharding.md#Sharding.device_set)
- used by: [`Array`](basearray.md#Array), [`sharding`](basearray.md#Array.sharding), [`device`](basearray.md#Array.device), [`addressable_data`](basearray.md#Array.addressable_data), [`addressable_shards`](basearray.md#Array.addressable_shards), [`global_shards`](basearray.md#Array.global_shards), [`aval`](basearray.md#Array.aval), [`block_until_ready`](basearray.md#Array.block_until_ready), [`committed`](basearray.md#Array.committed), [`copy`](basearray.md#Array.copy), [`copy_to_host_async`](basearray.md#Array.copy_to_host_async)

## Functions
- `_earray_shard_arg_handler(xs, shardings, layouts, copy_semantics)` — [`L114`](../../../../../../raw/code/jax/jax/_src/earray.py#L114)

## Module values
- `unsafe_map` — [`L27`](../../../../../../raw/code/jax/jax/_src/earray.py#L27)
- `unsafe_zip` — [`L28`](../../../../../../raw/code/jax/jax/_src/earray.py#L28)


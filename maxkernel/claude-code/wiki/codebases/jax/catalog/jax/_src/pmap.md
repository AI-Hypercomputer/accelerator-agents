---
title: 'Module: jax/_src/pmap.py'
type: catalog
provenance: extracted
module: jax/_src/pmap.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pmap`/
symbols:
  _cached_shard_map: _cached_shard_map().
  global_array_to_host_local_array: global_array_to_host_local_array().
  pmap.infer_params: pmap().infer_params().
  host_local_array_to_global_array: host_local_array_to_global_array().
  pmap: pmap().
  _cached_shard_map._fun: _cached_shard_map()._fun().
  _get_mesh_devices: _get_mesh_devices().
  _prepare_pmap: _prepare_pmap().
  _local_to_global_aval: _local_to_global_aval().
  _global_to_local_aval: _global_to_local_aval().
  pmap.wrapped: pmap().wrapped().
  _cached_shard_map.out_local_shardings_thunk: _cached_shard_map().out_local_shardings_thunk().
  _get_in_axes_flat: _get_in_axes_flat().
  pmap.lower: pmap().lower().
  CachedShardMap.out_local_shardings_thunk: CachedShardMap#out_local_shardings_thunk.
  _pmap_wrap_init: _pmap_wrap_init().
  _get_donated_invars: _get_donated_invars().
  _get_out_shardings: _get_out_shardings().
  unsafe_zip: unsafe_zip.
  _local_device_indices: _local_device_indices().
  _is_sharding_equivalent: _is_sharding_equivalent().
  AxisName: AxisName.
  CachedShardMap.in_specs_flat: CachedShardMap#in_specs_flat.
  CachedShardMap.in_local_shardings: CachedShardMap#in_local_shardings.
  CachedShardMap.in_global_shardings: CachedShardMap#in_global_shardings.
  CachedShardMap.mesh: CachedShardMap#mesh.
  _get_dyn_args: _get_dyn_args().
  unsafe_map: unsafe_map.
  CachedShardMap: CachedShardMap#
  CachedShardMap.pmapped: CachedShardMap#pmapped.
  CachedShardMap.local_devices: CachedShardMap#local_devices.
  CachedShardMap.out_specs: CachedShardMap#out_specs.
  CachedShardMap.donate_argnums: CachedShardMap#donate_argnums.
  CachedShardMap.out_global_shardings: CachedShardMap#out_global_shardings.
  CachedShardMap.jitted_f: CachedShardMap#jitted_f.
  CachedShardMap.jitted_f_with_shardings: CachedShardMap#jitted_f_with_shardings.
  _mapped_axis_size: _mapped_axis_size().
---
# Module: [`jax/_src/pmap.py`](../../../../../../raw/code/jax/jax/_src/pmap.py)

## Classes
### `CachedShardMap`  ·  implements/extends NamedTuple
- def: [`jax/_src/pmap.py:365`](../../../../../../raw/code/jax/jax/_src/pmap.py#L365)
- doc: Core cached pmap result.
- signature: `class CachedShardMap(NamedTuple):`
- members:
  - `donate_argnums` — [`L395`](../../../../../../raw/code/jax/jax/_src/pmap.py#L395)
  - `in_global_shardings` — [`L388`](../../../../../../raw/code/jax/jax/_src/pmap.py#L388)
  - `in_local_shardings` — [`L387`](../../../../../../raw/code/jax/jax/_src/pmap.py#L387)
  - `in_specs_flat` — [`L385`](../../../../../../raw/code/jax/jax/_src/pmap.py#L385)
  - `jitted_f` — [`L397`](../../../../../../raw/code/jax/jax/_src/pmap.py#L397)
  - `jitted_f_with_shardings` — [`L398`](../../../../../../raw/code/jax/jax/_src/pmap.py#L398)
  - `local_devices` — [`L386`](../../../../../../raw/code/jax/jax/_src/pmap.py#L386)
  - `mesh` — [`L389`](../../../../../../raw/code/jax/jax/_src/pmap.py#L389)
  - `out_global_shardings` — [`L396`](../../../../../../raw/code/jax/jax/_src/pmap.py#L396)
  - `out_local_shardings_thunk` — [`L391`](../../../../../../raw/code/jax/jax/_src/pmap.py#L391)
  - `out_specs` — [`L390`](../../../../../../raw/code/jax/jax/_src/pmap.py#L390)
  - `pmapped` — [`L384`](../../../../../../raw/code/jax/jax/_src/pmap.py#L384)
- uses (calls/refs, reference-scoped): [`NamedSharding`](named_sharding.md#NamedSharding), [`PartitionSpec`](partition_spec.md#PartitionSpec), [`Mesh`](mesh.md#Mesh)
- used by: [`_cached_shard_map`](pmap.md#_cached_shard_map)

## Functions
- `_cached_shard_map(fun, in_tree, in_axes_flat, out_axes_flat, out_axes_tree, donated_invars, mesh_devices, axis_name)` — [`L402`](../../../../../../raw/code/jax/jax/_src/pmap.py#L402)
- `_fun(*flat_args)` — [`L410`](../../../../../../raw/code/jax/jax/_src/pmap.py#L410)
- `_get_donated_invars(donate_tuple, in_tree, num_flat_args)` — [`L650`](../../../../../../raw/code/jax/jax/_src/pmap.py#L650) — Compute donation vector for arguments.
- `_get_dyn_args(wrapped_f, static_broadcasted_tuple, args)` — [`L525`](../../../../../../raw/code/jax/jax/_src/pmap.py#L525) — Extract dynamic args and argnums after handling static args.
- `_get_in_axes_flat(in_axes, dyn_argnums, dyn_args, kwargs, num_flat_args, in_tree)` — [`L561`](../../../../../../raw/code/jax/jax/_src/pmap.py#L561) — Compute flat in_axes tuple from in_axes prefix and args structure.
- `_get_mesh_devices(devices, backend, local_axis_size, axis_size, trace_state_clean)` — [`L669`](../../../../../../raw/code/jax/jax/_src/pmap.py#L669) — Compute effective mesh devices based on context.
- `_get_out_shardings(out_tree, pspecs, out_shardings_thunk)` — [`L769`](../../../../../../raw/code/jax/jax/_src/pmap.py#L769) — Get flattened output shardings, combining pspec flattening and sharding lookup.
- `_global_to_local_aval(shape, dtype, sharding)` — [`L745`](../../../../../../raw/code/jax/jax/_src/pmap.py#L745) — Compute local aval from global shape.
- `_is_sharding_equivalent(sharding_a, sharding_b, ndim)` — [`L763`](../../../../../../raw/code/jax/jax/_src/pmap.py#L763) — Check if sharding is equivalent to NamedSharding(mesh.local_mesh, pspec).
- `_local_device_indices(local_sharding, shape)` — [`L757`](../../../../../../raw/code/jax/jax/_src/pmap.py#L757) — Cached device indices for slicing arrays.
- `_local_to_global_aval(shape, dtype, sharding)` — [`L733`](../../../../../../raw/code/jax/jax/_src/pmap.py#L733) — Compute global aval from local shape.
- `_mapped_axis_size(args, in_axes)` — [`L474`](../../../../../../raw/code/jax/jax/_src/pmap.py#L474) — Infer axis size from the first mapped argument.
- `_pmap_wrap_init(f, static_broadcasted_tuple)` — [`L501`](../../../../../../raw/code/jax/jax/_src/pmap.py#L501) — Create a wrapped function with DebugInfo for pmap.
- `_prepare_pmap(fun, axis_name, static_broadcasted_argnums, donate_argnums, in_axes, out_axes)` — [`L344`](../../../../../../raw/code/jax/jax/_src/pmap.py#L344)
- `global_array_to_host_local_array(out, cached, trace_state_clean)` — [`L867`](../../../../../../raw/code/jax/jax/_src/pmap.py#L867) — Convert global arrays to host-local arrays for multihost pmap output.
- `host_local_array_to_global_array(dyn_args_flat, cached, trace_state_clean, donated_invars)` — [`L777`](../../../../../../raw/code/jax/jax/_src/pmap.py#L777) — Convert host-local arrays to global arrays for multihost pmap.
- `infer_params(*args, **kwargs)` — [`L289`](../../../../../../raw/code/jax/jax/_src/pmap.py#L289)
- `lower(*args, **kwargs)` — [`L324`](../../../../../../raw/code/jax/jax/_src/pmap.py#L324)
- `out_local_shardings_thunk(pspec)` — [`L440`](../../../../../../raw/code/jax/jax/_src/pmap.py#L440)
- `pmap(fun: Callable, axis_name: AxisName | None = None, *, in_axes: int | None | Sequence[Any] = 0, out_axes: Any = 0, static_broadcasted_argnums: int | Iterable[int] = (), devices: Sequence[xc.Device] | None = None, backend: str | None = None, axis_size: int | None = None, donate_argnums: int | Iterable[int] = ())` — [`L58`](../../../../../../raw/code/jax/jax/_src/pmap.py#L58) — Old way of doing parallel map. Use :py:func:`jax.shard_map` instead.
- `wrapped(*args, **kwargs)` — [`L316`](../../../../../../raw/code/jax/jax/_src/pmap.py#L316)

## Module values
- `AxisName` — [`L54`](../../../../../../raw/code/jax/jax/_src/pmap.py#L54)
- `unsafe_map` — [`L49`](../../../../../../raw/code/jax/jax/_src/pmap.py#L49)
- `unsafe_zip` — [`L50`](../../../../../../raw/code/jax/jax/_src/pmap.py#L50)


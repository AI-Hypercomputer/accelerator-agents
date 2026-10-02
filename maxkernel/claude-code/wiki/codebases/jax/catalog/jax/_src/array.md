---
title: 'Module: jax/_src/array.py'
type: catalog
provenance: extracted
module: jax/_src/array.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.array`/
symbols:
  make_array_from_callback: make_array_from_callback().
  ArrayImpl: ArrayImpl#
  ArrayImpl._value: ArrayImpl#_value().
  ArrayImpl.sharding: ArrayImpl#sharding().
  ArrayImpl.__repr__: ArrayImpl#__repr__().
  make_array_from_single_device_arrays: make_array_from_single_device_arrays().
  ArrayImpl.shape: ArrayImpl#shape().
  ArrayImpl.__dlpack_device__: ArrayImpl#__dlpack_device__().
  ArrayImpl.__init__: ArrayImpl#__init__().
  ArrayImpl._arrays: ArrayImpl#_arrays.
  ArrayImpl.copy_to_host_async: ArrayImpl#copy_to_host_async().
  _array_from_process_local_data: _array_from_process_local_data().
  ArrayImpl.__format__: ArrayImpl#__format__().
  ArrayImpl.global_shards: ArrayImpl#global_shards().
  _validate_shape_and_dtype_for_per_device_arrays: _validate_shape_and_dtype_for_per_device_arrays().
  _array_shard_arg: _array_shard_arg().
  ArrayImpl.__iter__: ArrayImpl#__iter__().
  ArrayImpl.__str__: ArrayImpl#__str__().
  make_array_from_process_local_data: make_array_from_process_local_data().
  shard_device_array: shard_device_array().
  _get_device: _get_device().
  ArrayImpl.addressable_shards: ArrayImpl#addressable_shards().
  ArrayImpl._check_if_deleted: ArrayImpl#_check_if_deleted().
  ArrayImpl.dtype: ArrayImpl#dtype().
  ArrayImpl.is_fully_addressable: ArrayImpl#is_fully_addressable().
  ArrayImpl.format: ArrayImpl#format().
  _get_and_check_dtype: _get_and_check_dtype().
  make_array_from_callback.get_data: make_array_from_callback().get_data().
  _array_global_result_handler: _array_global_result_handler().
  ArrayImpl.is_fully_replicated: ArrayImpl#is_fully_replicated().
  Shard.index: Shard#index().
  ArrayImpl.on_device_size_in_bytes: ArrayImpl#on_device_size_in_bytes().
  Shard.__init__: Shard#__init__().
  ArrayImpl.aval: ArrayImpl#aval.
  Shard.replica_id: Shard#replica_id().
  ArrayImpl.devices: ArrayImpl#devices().
  ArrayImpl.addressable_data: ArrayImpl#addressable_data().
  ArrayImpl.delete: ArrayImpl#delete().
  ArrayImpl.block_until_ready: ArrayImpl#block_until_ready().
  ArrayImpl._npy_value: ArrayImpl#_npy_value.
  ArrayImpl.is_deleted: ArrayImpl#is_deleted().
  Shard.__repr__: Shard#__repr__().
  ArrayImpl.__dlpack__: ArrayImpl#__dlpack__().
  ArrayImpl.__reduce__: ArrayImpl#__reduce__().
  _process_has_full_value_in_mcjax: _process_has_full_value_in_mcjax().
  ArrayImpl.unsafe_buffer_pointer: ArrayImpl#unsafe_buffer_pointer().
  ArrayImpl.__cuda_array_interface__: ArrayImpl#__cuda_array_interface__().
  ArrayImpl._copy_single_device_array_to_host_async: ArrayImpl#_copy_single_device_array_to_host_async().
  _get_shape_from_index: _get_shape_from_index().
  _array_from_process_local_data.cb: _array_from_process_local_data().cb().
  _token_global_result_handler: _token_global_result_handler().
  Shape: Shape.
  _cached_index_calc: _cached_index_calc().
  ArrayImpl.device: ArrayImpl#device().
  Shard.device: Shard#device().
  ArrayImpl.ndim: ArrayImpl#ndim().
  Index: Index.
  ArrayImpl._check_and_rearrange: ArrayImpl#_check_and_rearrange().
  shard_sharded_device_array_slow_path: shard_sharded_device_array_slow_path().
  _sharding_indices_and_eq: _sharding_indices_and_eq().
  _token_shard_arg: _token_shard_arg().
  Shard.data: Shard#data().
  ArrayImpl._sharding: ArrayImpl#_sharding.
  Shard: Shard#
  ArrayImpl.weak_type: ArrayImpl#weak_type().
  ArrayImpl.__bool__: ArrayImpl#__bool__().
  ArrayImpl.__float__: ArrayImpl#__float__().
  ArrayImpl.__int__: ArrayImpl#__int__().
  ArrayImpl.__complex__: ArrayImpl#__complex__().
  ArrayImpl.__hex__: ArrayImpl#__hex__().
  ArrayImpl.__oct__: ArrayImpl#__oct__().
  ArrayImpl.__index__: ArrayImpl#__index__().
  ArrayImpl.__getitem__: ArrayImpl#__getitem__().
  _reconstruct_array: _reconstruct_array().
  ArrayImpl.size: ArrayImpl#size().
  ArrayImpl._single_device_array_to_np_array_did_copy: ArrayImpl#_single_device_array_to_np_array_did_copy().
  _get_aval_array: _get_aval_array().
  as_slice_indices: as_slice_indices().
  _fallback_check_via_indices: _fallback_check_via_indices().
  _array_global_result_handler.handler: _array_global_result_handler().handler().
  _token_global_result_handler.wrapper: _token_global_result_handler().wrapper().
  Device: Device.
  Shard._sharding: Shard#_sharding.
  ArrayImpl._committed: ArrayImpl#_committed.
  ArrayImpl.committed: ArrayImpl#committed().
  ArrayImpl.__len__: ArrayImpl#__len__().
  ArrayImpl.tobytes: ArrayImpl#tobytes().
  ArrayImpl.tolist: ArrayImpl#tolist().
  ArrayImpl.__array__: ArrayImpl#__array__().
  unsafe_zip: unsafe_zip.
  Shard._global_shape: Shard#_global_shape.
  PRNGKeyArray: PRNGKeyArray.
  Shard._device: Shard#_device.
  Shard._data: Shard#_data.
  _array_from_process_local_data.local_slice: _array_from_process_local_data().local_slice().
  _array_mlir_constant_handler: _array_mlir_constant_handler().
  ArrayImpl._skip_checks: ArrayImpl#_skip_checks.
  ArrayImpl.device_buffer: ArrayImpl#device_buffer().
  ArrayImpl.device_buffers: ArrayImpl#device_buffers().
---
# Module: [`jax/_src/array.py`](../../../../../../raw/code/jax/jax/_src/array.py)

## Classes
### `ArrayImpl`
- def: [`jax/_src/array.py:179`](../../../../../../raw/code/jax/jax/_src/array.py#L179)
- members:
  - `addressable_data(self, index: int)` — [`L525`](../../../../../../raw/code/jax/jax/_src/array.py#L525)
  - `addressable_shards(self)` — [`L532`](../../../../../../raw/code/jax/jax/_src/array.py#L532)
  - `block_until_ready(self)` — [`L605`](../../../../../../raw/code/jax/jax/_src/array.py#L605)
  - `committed(self)` — [`L285`](../../../../../../raw/code/jax/jax/_src/array.py#L285)
  - `copy_to_host_async(self)` — [`L620`](../../../../../../raw/code/jax/jax/_src/array.py#L620)
  - `delete(self)` — [`L582`](../../../../../../raw/code/jax/jax/_src/array.py#L582)
  - `device(self)` — [`L274`](../../../../../../raw/code/jax/jax/_src/array.py#L274)
  - `device_buffer(self)` — [`L516`](../../../../../../raw/code/jax/jax/_src/array.py#L516)
  - `device_buffers(self)` — [`L521`](../../../../../../raw/code/jax/jax/_src/array.py#L521)
  - `devices(self)` — [`L511`](../../../../../../raw/code/jax/jax/_src/array.py#L511)
  - `dtype(self)` — [`L258`](../../../../../../raw/code/jax/jax/_src/array.py#L258)
  - `format(self)` — [`L546`](../../../../../../raw/code/jax/jax/_src/array.py#L546)
  - `global_shards(self)` — [`L561`](../../../../../../raw/code/jax/jax/_src/array.py#L561) — Returns list of all `Shard`s of the Array across all devices.
  - `is_deleted(self)` — [`L591`](../../../../../../raw/code/jax/jax/_src/array.py#L591)
  - `is_fully_addressable(self)` — [`L396`](../../../../../../raw/code/jax/jax/_src/array.py#L396) — Is this Array fully addressable?
  - `is_fully_replicated(self)` — [`L357`](../../../../../../raw/code/jax/jax/_src/array.py#L357)
  - `ndim(self)` — [`L262`](../../../../../../raw/code/jax/jax/_src/array.py#L262)
  - `on_device_size_in_bytes(self)` — [`L505`](../../../../../../raw/code/jax/jax/_src/array.py#L505) — Returns the total global on-device size of the array in bytes.
  - `shape(self)` — [`L254`](../../../../../../raw/code/jax/jax/_src/array.py#L254)
  - `sharding(self)` — [`L270`](../../../../../../raw/code/jax/jax/_src/array.py#L270)
  - `size(self)` — [`L266`](../../../../../../raw/code/jax/jax/_src/array.py#L266)
  - `tobytes(self, order="C")` — [`L322`](../../../../../../raw/code/jax/jax/_src/array.py#L322)
  - `tolist(self)` — [`L325`](../../../../../../raw/code/jax/jax/_src/array.py#L325)
  - `unsafe_buffer_pointer(self)` — [`L490`](../../../../../../raw/code/jax/jax/_src/array.py#L490)
  - `weak_type(self)` — [`L281`](../../../../../../raw/code/jax/jax/_src/array.py#L281)
  - `aval` — [`L180`](../../../../../../raw/code/jax/jax/_src/array.py#L180)
- protocol/private: `__array__`[`L409`](../../../../../../raw/code/jax/jax/_src/array.py#L409), `__bool__`[`L294`](../../../../../../raw/code/jax/jax/_src/array.py#L294), `__complex__`[`L306`](../../../../../../raw/code/jax/jax/_src/array.py#L306), `__cuda_array_interface__`[`L498`](../../../../../../raw/code/jax/jax/_src/array.py#L498), `__dlpack__`[`L416`](../../../../../../raw/code/jax/jax/_src/array.py#L416), `__dlpack_device__`[`L436`](../../../../../../raw/code/jax/jax/_src/array.py#L436), `__float__`[`L298`](../../../../../../raw/code/jax/jax/_src/array.py#L298), `__format__`[`L328`](../../../../../../raw/code/jax/jax/_src/array.py#L328), `__getitem__`[`L338`](../../../../../../raw/code/jax/jax/_src/array.py#L338), `__hex__`[`L310`](../../../../../../raw/code/jax/jax/_src/array.py#L310), `__index__`[`L318`](../../../../../../raw/code/jax/jax/_src/array.py#L318), `__init__`[`L188`](../../../../../../raw/code/jax/jax/_src/array.py#L188), `__int__`[`L302`](../../../../../../raw/code/jax/jax/_src/array.py#L302), `__iter__`[`L344`](../../../../../../raw/code/jax/jax/_src/array.py#L344), `__len__`[`L288`](../../../../../../raw/code/jax/jax/_src/array.py#L288), `__oct__`[`L314`](../../../../../../raw/code/jax/jax/_src/array.py#L314), `__reduce__`[`L484`](../../../../../../raw/code/jax/jax/_src/array.py#L484), `__repr__`[`L360`](../../../../../../raw/code/jax/jax/_src/array.py#L360), `__str__`[`L386`](../../../../../../raw/code/jax/jax/_src/array.py#L386), `_arrays`[`L182`](../../../../../../raw/code/jax/jax/_src/array.py#L182), `_check_and_rearrange`[`L207`](../../../../../../raw/code/jax/jax/_src/array.py#L207), `_check_if_deleted`[`L599`](../../../../../../raw/code/jax/jax/_src/array.py#L599), `_committed`[`L183`](../../../../../../raw/code/jax/jax/_src/array.py#L183), `_copy_single_device_array_to_host_async`[`L616`](../../../../../../raw/code/jax/jax/_src/array.py#L616), `_npy_value`[`L185`](../../../../../../raw/code/jax/jax/_src/array.py#L185), `_sharding`[`L181`](../../../../../../raw/code/jax/jax/_src/array.py#L181), `_single_device_array_to_np_array_did_copy`[`L612`](../../../../../../raw/code/jax/jax/_src/array.py#L612), `_skip_checks`[`L184`](../../../../../../raw/code/jax/jax/_src/array.py#L184), `_value`[`L631`](../../../../../../raw/code/jax/jax/_src/array.py#L631)
- uses (calls/refs, reference-scoped): [`Array`](basearray.md#Array), [`ShapedArray`](core.md#ShapedArray), [`NamedSharding`](named_sharding.md#NamedSharding), [`shape`](core.md#ShapedArray.shape), [`dtype`](core.md#ShapedArray.dtype), [`Sharding`](sharding.md#Sharding), [`rewriting_take`](numpy/indexing.md#rewriting_take), [`spec`](named_sharding.md#NamedSharding.spec), [`device_put`](api.md#device_put), [`Format`](layout.md#Format), [`str_short`](core.md#ShapedArray.str_short), [`unreduced`](partition_spec.md#P.unreduced), [`use_cpp_method`](util.md#use_cpp_method), [`weak_type`](core.md#ShapedArray.weak_type), [`_validate_shape_and_dtype_for_per_device_arrays`](array.md#_validate_shape_and_dtype_for_per_device_arrays), [`to_dlpack`](dlpack.md#to_dlpack), [`enable_checks`](config.md#enable_checks), [`device_set`](sharding.md#Sharding.device_set), [`Layout`](layout.md#Layout), [`DLDeviceType`](typing.md#DLDeviceType), [`_get_device`](array.md#_get_device), [`check_integer_conversion`](core.md#check_integer_conversion), [`use_cpp_class`](util.md#use_cpp_class), [`is_fully_replicated`](sharding.md#Sharding.is_fully_replicated), [`memory_kind`](sharding.md#Sharding.memory_kind), [`num_devices`](sharding.md#Sharding.num_devices), [`is_fully_addressable`](sharding.md#Sharding.is_fully_addressable), [`check_scalar_conversion`](core.md#check_scalar_conversion), [`_process_has_full_value_in_mcjax`](array.md#_process_has_full_value_in_mcjax), [`Shape`](array.md#Shape), [`_cached_index_calc`](array.md#_cached_index_calc), [`has_addressable_devices`](sharding.md#Sharding.has_addressable_devices), [`check_bool_conversion`](core.md#check_bool_conversion), [`Shard`](array.md#Shard), [`platform`](basearray.md#Array.platform), [`_reconstruct_array`](array.md#_reconstruct_array), [`from_pjrt_layout`](layout.md#Layout.from_pjrt_layout), [`Device`](array.md#Device), [`kDLCUDAHost`](typing.md#DLDeviceType.kDLCUDAHost), [`kDLTPUHost`](typing.md#DLDeviceType.kDLTPUHost)  (+3 more)
- used by: [`shape`](basearray.md#Array.shape), [`dtype`](basearray.md#Array.dtype), [`ndim`](basearray.md#Array.ndim), [`size`](basearray.md#Array.size), [`full`](lax/lax.md#full), [`make_array_from_callback`](array.md#make_array_from_callback), [`_handle_array_process_allgather`](../experimental/multihost_utils.md#_handle_array_process_allgather), [`global_array_to_host_local_array_impl`](../experimental/multihost_utils.md#global_array_to_host_local_array_impl), [`host_local_array_to_global_array_impl`](../experimental/multihost_utils.md#host_local_array_to_global_array_impl), [`like`](core.md#ShapeDtypeStruct.like), [`global_array_to_host_local_array`](pmap.md#global_array_to_host_local_array), [`make_array_from_single_device_arrays`](array.md#make_array_from_single_device_arrays), [`host_local_array_to_global_array`](pmap.md#host_local_array_to_global_array), [`_array_from_process_local_data`](array.md#_array_from_process_local_data), [`_validate_shape_and_dtype_for_per_device_arrays`](array.md#_validate_shape_and_dtype_for_per_device_arrays), [`sharding`](basearray.md#Array.sharding), [`to_dlpack`](dlpack.md#to_dlpack), [`_infer_src_sharding`](api.md#_infer_src_sharding), [`_get_device`](array.md#_get_device), [`t`](lax/lax.md#t), [`get_data`](array.md#make_array_from_callback.get_data), [`__init__`](array.md#Shard.__init__), [`device`](basearray.md#Array.device), [`addressable_data`](basearray.md#Array.addressable_data), [`is_sharded`](numpy/indexing.md#NDIndexer.is_sharded), [`concat_in_tree`](../experimental/multihost_utils.md#assert_equal.concat_in_tree), [`addressable_shards`](basearray.md#Array.addressable_shards), [`global_shards`](basearray.md#Array.global_shards), [`is_literalable`](core.md#is_literalable), [`block_until_ready`](api.md#block_until_ready), [`block_until_ready`](basearray.md#Array.block_until_ready), [`devices`](basearray.md#Array.devices), [`committed`](basearray.md#Array.committed), [`copy_to_host_async`](basearray.md#Array.copy_to_host_async), [`delete`](basearray.md#Array.delete), [`is_deleted`](basearray.md#Array.is_deleted), [`is_fully_addressable`](basearray.md#Array.is_fully_addressable), [`is_fully_replicated`](basearray.md#Array.is_fully_replicated), [`on_device_size_in_bytes`](basearray.md#Array.on_device_size_in_bytes), [`tobytes`](basearray.md#Array.tobytes)  (+3 more)

### `Shard`
- def: [`jax/_src/array.py:70`](../../../../../../raw/code/jax/jax/_src/array.py#L70)
- doc: A single data shard of an Array.
- signature: `class Shard:`
- members:
  - `data(self)` — [`L117`](../../../../../../raw/code/jax/jax/_src/array.py#L117)
  - `device(self)` — [`L113`](../../../../../../raw/code/jax/jax/_src/array.py#L113)
  - `index(self)` — [`L97`](../../../../../../raw/code/jax/jax/_src/array.py#L97)
  - `replica_id(self)` — [`L109`](../../../../../../raw/code/jax/jax/_src/array.py#L109)
- protocol/private: `__init__`[`L82`](../../../../../../raw/code/jax/jax/_src/array.py#L82), `__repr__`[`L89`](../../../../../../raw/code/jax/jax/_src/array.py#L89), `_data`[`L87`](../../../../../../raw/code/jax/jax/_src/array.py#L87), `_device`[`L84`](../../../../../../raw/code/jax/jax/_src/array.py#L84), `_global_shape`[`L86`](../../../../../../raw/code/jax/jax/_src/array.py#L86), `_sharding`[`L85`](../../../../../../raw/code/jax/jax/_src/array.py#L85)
- uses (calls/refs, reference-scoped): [`Sharding`](sharding.md#Sharding), [`ArrayImpl`](array.md#ArrayImpl), [`devices_indices_map`](sharding.md#Sharding.devices_indices_map), [`device_replica_id_map`](sharding_impls.md#device_replica_id_map), [`Shape`](array.md#Shape), [`Index`](array.md#Index), [`Device`](array.md#Device), [`PRNGKeyArray`](array.md#PRNGKeyArray)
- used by: [`Device`](../__init__.md#Device), [`host_local_array_to_global_array_impl`](../experimental/multihost_utils.md#host_local_array_to_global_array_impl), [`global_shards`](array.md#ArrayImpl.global_shards), [`addressable_shards`](array.md#ArrayImpl.addressable_shards)

## Functions
- `_array_from_process_local_data(local_data: np.ndarray, sharding: Sharding, global_shape: Shape | None = None)` — [`L978`](../../../../../../raw/code/jax/jax/_src/array.py#L978)
- `_array_global_result_handler(global_aval, out_sharding, committed)` — [`L1276`](../../../../../../raw/code/jax/jax/_src/array.py#L1276)
- `_array_mlir_constant_handler(val, aval)` — [`L1120`](../../../../../../raw/code/jax/jax/_src/array.py#L1120)
- `_array_shard_arg(xs, shardings, layouts, copy_semantics)` — [`L1223`](../../../../../../raw/code/jax/jax/_src/array.py#L1223)
- `_cached_index_calc(s, shape)` — [`L131`](../../../../../../raw/code/jax/jax/_src/array.py#L131)
- `_fallback_check_via_indices(src_sharding, dst_sharding, shape)` — [`L1209`](../../../../../../raw/code/jax/jax/_src/array.py#L1209)
- `_get_and_check_dtype(arrays: Sequence[basearray.Array | np.ndarray | literals.TypedNdArray], dtype: DTypeLike | ExtendedDType | None, fname: str)` — [`L675`](../../../../../../raw/code/jax/jax/_src/array.py#L675)
- `_get_aval_array(self)` — [`L1115`](../../../../../../raw/code/jax/jax/_src/array.py#L1115)
- `_get_device(a: ArrayImpl)` — [`L61`](../../../../../../raw/code/jax/jax/_src/array.py#L61)
- `_get_shape_from_index(slc: Index, shape: Shape)` — [`L667`](../../../../../../raw/code/jax/jax/_src/array.py#L667)
- `_process_has_full_value_in_mcjax(s, shape)` — [`L144`](../../../../../../raw/code/jax/jax/_src/array.py#L144)
- `_reconstruct_array(fun, args, arr_state, aval_state)` — [`L121`](../../../../../../raw/code/jax/jax/_src/array.py#L121) — Method to reconstruct a device array from a serialized state.
- `_sharding_indices_and_eq(src_sharding, dst_sharding, ndim)` — [`L1215`](../../../../../../raw/code/jax/jax/_src/array.py#L1215)
- `_token_global_result_handler(global_aval, out_sharding, committed)` — [`L1308`](../../../../../../raw/code/jax/jax/_src/array.py#L1308)
- `_token_shard_arg(xs, shardings, layouts, copy_semantics)` — [`L1294`](../../../../../../raw/code/jax/jax/_src/array.py#L1294)
- `_validate_shape_and_dtype_for_per_device_arrays(arrays: Sequence[ArrayImpl | np.ndarray | literals.TypedNdArray], sharding: Sharding, aval: core.ShapedArray, expected_shape: Shape)` — [`L156`](../../../../../../raw/code/jax/jax/_src/array.py#L156) — Validates that per-device arrays are valid and consistent.
- `as_slice_indices(arr: Any, idx: Index)` — [`L1142`](../../../../../../raw/code/jax/jax/_src/array.py#L1142) — Returns start_indices, limit_indices, removed_dims
- `cb(index: Index | None)` — [`L1028`](../../../../../../raw/code/jax/jax/_src/array.py#L1028)
- `get_data(index: Index | None)` — [`L761`](../../../../../../raw/code/jax/jax/_src/array.py#L761)
- `handler(xs)` — [`L1278`](../../../../../../raw/code/jax/jax/_src/array.py#L1278)
- `local_slice(i, start)` — [`L1021`](../../../../../../raw/code/jax/jax/_src/array.py#L1021)
- `make_array_from_callback(shape: Shape, sharding: Sharding | Format, data_callback: Callable[[Index | None], ArrayLike], dtype: DTypeLike | None = None)` — [`L703`](../../../../../../raw/code/jax/jax/_src/array.py#L703) — Returns a ``jax.Array`` via data fetched from ``data_callback``.
- `make_array_from_process_local_data(sharding, local_data, global_shape=None)` — [`L838`](../../../../../../raw/code/jax/jax/_src/array.py#L838) — Creates distributed tensor using the data available in process.
- `make_array_from_single_device_arrays(shape: Shape, sharding: Sharding, arrays: Sequence[basearray.Array], *, dtype: DTypeLike | None = None)` — [`L1039`](../../../../../../raw/code/jax/jax/_src/array.py#L1039) — Returns a ``jax.Array`` from a sequence of ``jax.Array``\s each on a single device.
- `shard_device_array(x, devices, indices, sharding)` — [`L1167`](../../../../../../raw/code/jax/jax/_src/array.py#L1167)
- `shard_sharded_device_array_slow_path(x, devices, indices, sharding)` — [`L1182`](../../../../../../raw/code/jax/jax/_src/array.py#L1182)
- `wrapper(array)` — [`L1311`](../../../../../../raw/code/jax/jax/_src/array.py#L1311)

## Module values
- `Device` — [`L57`](../../../../../../raw/code/jax/jax/_src/array.py#L57)
- `Index` — [`L58`](../../../../../../raw/code/jax/jax/_src/array.py#L58)
- `PRNGKeyArray` — [`L59`](../../../../../../raw/code/jax/jax/_src/array.py#L59)
- `Shape` — [`L56`](../../../../../../raw/code/jax/jax/_src/array.py#L56)
- `unsafe_zip` — [`L54`](../../../../../../raw/code/jax/jax/_src/array.py#L54)


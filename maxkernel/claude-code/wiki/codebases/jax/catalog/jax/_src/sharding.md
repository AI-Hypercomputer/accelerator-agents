---
title: 'Module: jax/_src/sharding.py'
type: catalog
provenance: extracted
module: jax/_src/sharding.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.sharding`/
symbols:
  Sharding: Sharding#
  common_devices_indices_map: common_devices_indices_map().
  _addressable_devices_indices_map: _addressable_devices_indices_map().
  Sharding.device_set: Sharding#device_set().
  Sharding.devices_indices_map: Sharding#devices_indices_map().
  _common_shard_shape: _common_shard_shape().
  common_is_equivalent_to: common_is_equivalent_to().
  Sharding._device_assignment: Sharding#_device_assignment().
  Sharding._to_xla_hlo_sharding: Sharding#_to_xla_hlo_sharding().
  Sharding.is_fully_replicated: Sharding#is_fully_replicated().
  Sharding.addressable_devices_indices_map: Sharding#addressable_devices_indices_map().
  Sharding.num_devices: Sharding#num_devices().
  Sharding.memory_kind: Sharding#memory_kind().
  Sharding.shard_shape: Sharding#shard_shape().
  Sharding.is_fully_addressable: Sharding#is_fully_addressable().
  Sharding.addressable_devices: Sharding#addressable_devices().
  Sharding.with_memory_kind: Sharding#with_memory_kind().
  Sharding._internal_device_list: Sharding#_internal_device_list().
  Sharding._addressable_device_assignment: Sharding#_addressable_device_assignment().
  Sharding._to_sdy_sharding: Sharding#_to_sdy_sharding().
  Sharding._is_concrete: Sharding#_is_concrete().
  Sharding.is_equivalent_to: Sharding#is_equivalent_to().
  Shape: Shape.
  Device: Device.
  Sharding.has_addressable_devices: Sharding#has_addressable_devices().
  IndivisibleError: IndivisibleError#
  XLADeviceAssignment: XLADeviceAssignment.
  Index: Index.
---
# Module: [`jax/_src/sharding.py`](../../../../../../raw/code/jax/jax/_src/sharding.py)

## Classes
### `IndivisibleError`  ·  implements/extends ValueError
- def: [`jax/_src/sharding.py:32`](../../../../../../raw/code/jax/jax/_src/sharding.py#L32)
- signature: `class IndivisibleError(ValueError):`
- used by: [`parse_flatten_op_sharding`](sharding_impls.md#parse_flatten_op_sharding), [`_common_shard_shape`](sharding.md#_common_shard_shape), [`pjit_check_aval_sharding`](pjit.md#pjit_check_aval_sharding)

### `Sharding`
- def: [`jax/_src/sharding.py:94`](../../../../../../raw/code/jax/jax/_src/sharding.py#L94)
- members:
  - `addressable_devices(self)` — [`L164`](../../../../../../raw/code/jax/jax/_src/sharding.py#L164) — The set of devices in the :class:`Sharding` that are addressable by the
  - `addressable_devices_indices_map(self, global_shape: Shape)` — [`L174`](../../../../../../raw/code/jax/jax/_src/sharding.py#L174) — A mapping from addressable devices to the slice of array data each contains.
  - `device_set(self)` — [`L100`](../../../../../../raw/code/jax/jax/_src/sharding.py#L100) — The set of devices that this :class:`Sharding` spans.
  - `devices_indices_map(self, global_shape: Shape)` — [`L183`](../../../../../../raw/code/jax/jax/_src/sharding.py#L183) — Returns a mapping from devices to the array slices each contains.
  - `has_addressable_devices(self)` — [`L192`](../../../../../../raw/code/jax/jax/_src/sharding.py#L192)
  - `is_equivalent_to(self: Sharding, other: Sharding, ndim: int)` — [`L209`](../../../../../../raw/code/jax/jax/_src/sharding.py#L209) — Returns ``True`` if two shardings are equivalent.
  - `is_fully_addressable(self)` — [`L118`](../../../../../../raw/code/jax/jax/_src/sharding.py#L118) — Is this sharding fully addressable?
  - `is_fully_replicated(self)` — [`L109`](../../../../../../raw/code/jax/jax/_src/sharding.py#L109) — Is this sharding fully replicated?
  - `memory_kind(self)` — [`L133`](../../../../../../raw/code/jax/jax/_src/sharding.py#L133) — Returns the memory kind of the sharding.
  - `num_devices(self)` — [`L128`](../../../../../../raw/code/jax/jax/_src/sharding.py#L128) — Number of devices that the sharding contains.
  - `shard_shape(self, global_shape: Shape)` — [`L201`](../../../../../../raw/code/jax/jax/_src/sharding.py#L201) — Returns the shape of the data on each device.
  - `with_memory_kind(self, kind: str)` — [`L137`](../../../../../../raw/code/jax/jax/_src/sharding.py#L137) — Returns a new Sharding instance with the specified memory kind.
- protocol/private: `_addressable_device_assignment`[`L196`](../../../../../../raw/code/jax/jax/_src/sharding.py#L196), `_device_assignment`[`L142`](../../../../../../raw/code/jax/jax/_src/sharding.py#L142), `_internal_device_list`[`L146`](../../../../../../raw/code/jax/jax/_src/sharding.py#L146), `_is_concrete`[`L160`](../../../../../../raw/code/jax/jax/_src/sharding.py#L160), `_to_sdy_sharding`[`L152`](../../../../../../raw/code/jax/jax/_src/sharding.py#L152), `_to_xla_hlo_sharding`[`L149`](../../../../../../raw/code/jax/jax/_src/sharding.py#L149)
- uses (calls/refs, reference-scoped): [`common_devices_indices_map`](sharding.md#common_devices_indices_map), [`process_count`](xla_bridge.md#process_count), [`_addressable_devices_indices_map`](sharding.md#_addressable_devices_indices_map), [`_to_sdy_sharding`](sharding_impls.md#GSPMDSharding._to_sdy_sharding), [`memory_kind`](named_sharding.md#NamedSharding.memory_kind), [`_common_shard_shape`](sharding.md#_common_shard_shape), [`use_cpp_class`](util.md#use_cpp_class), [`common_is_equivalent_to`](sharding.md#common_is_equivalent_to), [`_to_sdy_sharding`](sharding_impls.md#SingleDeviceSharding._to_sdy_sharding), [`is_fully_replicated`](named_sharding.md#NamedSharding.is_fully_replicated), [`device_set`](named_sharding.md#NamedSharding.device_set), [`_device_assignment`](named_sharding.md#NamedSharding._device_assignment), [`_to_sdy_sharding`](named_sharding.md#NamedSharding._to_sdy_sharding), [`addressable_devices`](named_sharding.md#NamedSharding.addressable_devices), [`devices_indices_map`](sharding_impls.md#SingleDeviceSharding.devices_indices_map), [`memory_kind`](sharding_impls.md#GSPMDSharding.memory_kind), [`Device`](sharding.md#Device), [`Shape`](sharding.md#Shape), [`is_equivalent_to`](named_sharding.md#NamedSharding.is_equivalent_to), [`is_fully_addressable`](named_sharding.md#NamedSharding.is_fully_addressable), [`with_memory_kind`](sharding_impls.md#GSPMDSharding.with_memory_kind), [`is_fully_replicated`](sharding_impls.md#GSPMDSharding.is_fully_replicated), [`memory_kind`](sharding_impls.md#SingleDeviceSharding.memory_kind), [`_is_concrete`](named_sharding.md#NamedSharding._is_concrete), [`device_set`](sharding_impls.md#SingleDeviceSharding.device_set), [`_to_xla_hlo_sharding`](named_sharding.md#NamedSharding._to_xla_hlo_sharding), [`_device_assignment`](sharding_impls.md#GSPMDSharding._device_assignment), [`_device_assignment`](sharding_impls.md#SingleDeviceSharding._device_assignment), [`device_set`](sharding_impls.md#GSPMDSharding.device_set), [`is_fully_addressable`](sharding_impls.md#SingleDeviceSharding.is_fully_addressable), [`num_devices`](named_sharding.md#NamedSharding.num_devices), [`with_memory_kind`](named_sharding.md#NamedSharding.with_memory_kind), [`with_memory_kind`](sharding_impls.md#SingleDeviceSharding.with_memory_kind), [`XLADeviceAssignment`](sharding.md#XLADeviceAssignment), [`Index`](sharding.md#Index), [`_to_xla_hlo_sharding`](sharding_impls.md#GSPMDSharding._to_xla_hlo_sharding), [`_to_xla_hlo_sharding`](sharding_impls.md#SingleDeviceSharding._to_xla_hlo_sharding), [`is_fully_addressable`](sharding_impls.md#GSPMDSharding.is_fully_addressable), [`num_devices`](sharding_impls.md#GSPMDSharding.num_devices), [`num_devices`](sharding_impls.md#SingleDeviceSharding.num_devices)  (+1 more)
- used by: [`NamedSharding`](named_sharding.md#NamedSharding), [`astype`](basearray.md#Array.astype), [`full_like`](lax/lax.md#full_like), [`array`](numpy/array_constructors.md#array), [`asarray`](numpy/array_constructors.md#asarray), [`io_callback`](callback.md#io_callback), [`_convert_element_type`](lax/lax.md#_convert_element_type), [`arange`](numpy/lax_numpy.md#arange), [`full`](lax/lax.md#full), [`make_array_from_callback`](array.md#make_array_from_callback), [`_linspace`](numpy/array_creation.md#_linspace), [`arange`](../numpy/__init__.pyi.md#arange), [`empty_like`](lax/lax.md#empty_like), [`_handle_array_process_allgather`](../experimental/multihost_utils.md#_handle_array_process_allgather), [`device_put`](api.md#device_put), [`full`](numpy/array_creation.md#full), [`array`](../numpy/__init__.pyi.md#array), [`full_like`](numpy/array_creation.md#full_like), [`zeros`](../numpy/__init__.pyi.md#zeros), [`zeros`](numpy/array_creation.md#zeros), [`_callback_op_sharding`](callback.md#_callback_op_sharding), [`zeros_like`](../numpy/__init__.pyi.md#zeros_like), [`_value`](array.md#ArrayImpl._value), [`global_array_to_host_local_array_impl`](../experimental/multihost_utils.md#global_array_to_host_local_array_impl), [`host_local_array_to_global_array_impl`](../experimental/multihost_utils.md#host_local_array_to_global_array_impl), [`_reshape_pull_rule`](pallas/fuser/block_spec.md#_reshape_pull_rule), [`ones`](numpy/array_creation.md#ones), [`zeros_like`](numpy/array_creation.md#zeros_like), [`astype`](numpy/lax_numpy.md#astype), [`manual_sharding_spec`](../experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.md#SplashAttentionKernel.manual_sharding_spec), [`prepare_axis_resources`](sharding_impls.md#prepare_axis_resources), [`ones_like`](numpy/array_creation.md#ones_like), [`empty`](numpy/array_creation.md#empty), [`_pjit_batcher_for_sharding`](pjit.md#_pjit_batcher_for_sharding), [`_reshape_push_rule`](pallas/fuser/block_spec.md#_reshape_push_rule), [`_check_sharding`](api.md#_check_sharding), [`from_dlpack`](dlpack.md#from_dlpack), [`__repr__`](array.md#ArrayImpl.__repr__), [`make_array_from_single_device_arrays`](array.md#make_array_from_single_device_arrays), [`empty_like`](numpy/array_creation.md#empty_like)  (+87 more)

## Functions
- `_addressable_devices_indices_map(sharding: Sharding, global_shape: Shape)` — [`L36`](../../../../../../raw/code/jax/jax/_src/sharding.py#L36)
- `_common_shard_shape(self, global_shape: Shape)` — [`L60`](../../../../../../raw/code/jax/jax/_src/sharding.py#L60)
- `common_devices_indices_map(s: Sharding, global_shape: Shape)` — [`L45`](../../../../../../raw/code/jax/jax/_src/sharding.py#L45)
- `common_is_equivalent_to(s1: Sharding, s2: Sharding, ndim: int, check_devices: bool = True)` — [`L81`](../../../../../../raw/code/jax/jax/_src/sharding.py#L81)

## Module values
- `Device` — [`L28`](../../../../../../raw/code/jax/jax/_src/sharding.py#L28)
- `Index` — [`L29`](../../../../../../raw/code/jax/jax/_src/sharding.py#L29)
- `Shape` — [`L27`](../../../../../../raw/code/jax/jax/_src/sharding.py#L27)
- `XLADeviceAssignment` — [`L30`](../../../../../../raw/code/jax/jax/_src/sharding.py#L30)


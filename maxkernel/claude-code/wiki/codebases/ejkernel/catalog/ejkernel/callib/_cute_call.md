---
title: 'Module: ejkernel/callib/_cute_call.py'
type: catalog
provenance: extracted
module: ejkernel/callib/_cute_call.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.callib._cute_call`/
symbols:
  cute_call: cute_call().
  _arg_contract_key: _arg_contract_key().
  CAN_USE_CUTE: CAN_USE_CUTE.
  _validate_out_leaves: _validate_out_leaves().
  _leaf_shape_dtype: _leaf_shape_dtype().
  _array_device_set: _array_device_set().
  _assert_single_device_args: _assert_single_device_args().
  _shape_specs_from_out_leaves: _shape_specs_from_out_leaves().
  _out_contract_key: _out_contract_key().
  cute_call._coerce_function_output: cute_call()._coerce_function_output().
  _device_set_from_sharding: _device_set_from_sharding().
  _shape_key: _shape_key().
  _NAMED_CALL_CACHE._NAMED_CALL_CACHE: _NAMED_CALL_CACHE._NAMED_CALL_CACHE.
  _NAMED_CALL_CACHE_LOCK: _NAMED_CALL_CACHE_LOCK.
  _has_multi_accelerators: _has_multi_accelerators().
  _in_shard_map_context: _in_shard_map_context().
  __all__: __all__.
---
# Module: [`ejkernel/callib/_cute_call.py`](../../../../../../raw/code/ejkernel/ejkernel/callib/_cute_call.py)

## Functions
- `_arg_contract_key(arg: Any)` — [`L358`](../../../../../../raw/code/ejkernel/ejkernel/callib/_cute_call.py#L358) — Build a cache-key fragment for an argument.
- `_array_device_set(arg: Any)` — [`L78`](../../../../../../raw/code/ejkernel/ejkernel/callib/_cute_call.py#L78) — Return the device set for a JAX array or tracer.
- `_assert_single_device_args(array_args: Sequence[Any], device_index: int | None, *, allow_sharded_tracers: bool)` — [`L128`](../../../../../../raw/code/ejkernel/ejkernel/callib/_cute_call.py#L128) — Validate that array arguments are placed on one logical device.
- `_coerce_function_output(function_out: Any)` — [`L472`](../../../../../../raw/code/ejkernel/ejkernel/callib/_cute_call.py#L472) — Validate and restructure the callable's raw output.
- `_device_set_from_sharding(sharding: Any)` — [`L48`](../../../../../../raw/code/ejkernel/ejkernel/callib/_cute_call.py#L48) — Extract the participating devices from a sharding object.
- `_has_multi_accelerators()` — [`L198`](../../../../../../raw/code/ejkernel/ejkernel/callib/_cute_call.py#L198) — Check whether more than one non-CPU accelerator is available.
- `_in_shard_map_context()` — [`L213`](../../../../../../raw/code/ejkernel/ejkernel/callib/_cute_call.py#L213) — Check whether execution is currently inside a ``jax.shard_map`` context.
- `_leaf_shape_dtype(leaf: Any)` — [`L245`](../../../../../../raw/code/ejkernel/ejkernel/callib/_cute_call.py#L245) — Read ``(shape, dtype)`` from an output leaf or tracer.
- `_out_contract_key(output_contract_shapes: Sequence[jax.ShapeDtypeStruct] | None)` — [`L380`](../../../../../../raw/code/ejkernel/ejkernel/callib/_cute_call.py#L380) — Build a stable key for expected output contracts.
- `_shape_key(shape: Any)` — [`L346`](../../../../../../raw/code/ejkernel/ejkernel/callib/_cute_call.py#L346) — Normalize shape values to a stable, hashable key.
- `_shape_specs_from_out_leaves(flat_out: Sequence[Any])` — [`L325`](../../../../../../raw/code/ejkernel/ejkernel/callib/_cute_call.py#L325) — Build shape/dtype structs from explicit ``out`` leaves.
- `_validate_out_leaves(flat_out: Sequence[Any], flat_out_shapes: Sequence[jax.ShapeDtypeStruct] | None)` — [`L272`](../../../../../../raw/code/ejkernel/ejkernel/callib/_cute_call.py#L272) — Validate explicit output leaves and optional output shape contracts.
- `cute_call(*args: Any, call: Callable[..., Any] | None = None, out_shape: ShapeDtype | Sequence[ShapeDtype] | None = None, out: Any | None = None, name: str | None = None, device: int | None = None)` — [`L396`](../../../../../../raw/code/ejkernel/ejkernel/callib/_cute_call.py#L396) — Execute a CuTe DSL kernel and return its output(s).

## Module values
- `CAN_USE_CUTE` — [`L35`](../../../../../../raw/code/ejkernel/ejkernel/callib/_cute_call.py#L35)
- `_NAMED_CALL_CACHE` — [`L44`](../../../../../../raw/code/ejkernel/ejkernel/callib/_cute_call.py#L44)
- `_NAMED_CALL_CACHE_LOCK` — [`L45`](../../../../../../raw/code/ejkernel/ejkernel/callib/_cute_call.py#L45)
- `__all__` — [`L537`](../../../../../../raw/code/ejkernel/ejkernel/callib/_cute_call.py#L537)


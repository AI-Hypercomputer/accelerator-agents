---
title: 'Module: ejkernel/callib/_tilelang_call.py'
type: catalog
provenance: extracted
module: ejkernel/callib/_tilelang_call.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.callib._tilelang_call`/
symbols:
  build_tilelang_call: build_tilelang_call().
  tilelang_call: tilelang_call().
  _freeze_key: _freeze_key().
  _arg_contract_key: _arg_contract_key().
  CAN_USE_TILELANG: CAN_USE_TILELANG.
  _NAMED_CALL_CACHE._NAMED_CALL_CACHE: _NAMED_CALL_CACHE._NAMED_CALL_CACHE.
  _validate_out_leaves: _validate_out_leaves().
  _out_contract_key: _out_contract_key().
  _leaf_shape_dtype: _leaf_shape_dtype().
  _array_device_set: _array_device_set().
  _assert_single_device_args: _assert_single_device_args().
  _shape_specs_from_out_leaves: _shape_specs_from_out_leaves().
  tilelang_call._coerce_function_output: tilelang_call()._coerce_function_output().
  _NAMED_CALL_CACHE_LOCK: _NAMED_CALL_CACHE_LOCK.
  _env_int: _env_int().
  _device_set_from_sharding: _device_set_from_sharding().
  _shape_key: _shape_key().
  build_tilelang_call._builder: build_tilelang_call()._builder().
  _env_optional_float: _env_optional_float().
  _env_target: _env_target().
  _env_compile_flags: _env_compile_flags().
  _has_multi_accelerators: _has_multi_accelerators().
  _in_shard_map_context: _in_shard_map_context().
  _shape_dtype_tree: _shape_dtype_tree().
  __all__: __all__.
---
# Module: [`ejkernel/callib/_tilelang_call.py`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_call.py)

## Functions
- `_arg_contract_key(arg: Any)` — [`L450`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_call.py#L450) — Build a cache-key fragment for an argument.
- `_array_device_set(arg: Any)` — [`L170`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_call.py#L170) — Return the device set for a JAX array or tracer.
- `_assert_single_device_args(array_args: Sequence[Any], device_index: int | None, *, allow_sharded_tracers: bool)` — [`L220`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_call.py#L220) — Validate that all array arguments live on a single logical device.
- `_builder(**config: Any)` — [`L671`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_call.py#L671) — Build a prim-func from static metadata plus one candidate config.
- `_coerce_function_output(function_out: Any)` — [`L842`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_call.py#L842) — Validate and restructure the callable's raw output.
- `_device_set_from_sharding(sharding: Any)` — [`L139`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_call.py#L139) — Extract the participating devices from a sharding object.
- `_env_compile_flags(value: list[str] | tuple[str, ...] | None)` — [`L122`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_call.py#L122) — Resolve TileLang compile flags from an explicit value or the environment.
- `_env_int(name: str, value: int | None, default: int, minimum: int)` — [`L52`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_call.py#L52) — Resolve an integer option from an explicit value or environment variable.
- `_env_optional_float(name: str, value: float | None, default: float | None)` — [`L78`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_call.py#L78) — Resolve an optional float option from an explicit value or environment variable.
- `_env_target(value: str | None)` — [`L105`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_call.py#L105) — Resolve the TileLang compile target from an explicit value or the environment.
- `_freeze_key(value: Any)` — [`L500`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_call.py#L500) — Convert arbitrary static metadata into a stable hashable cache fragment.
- `_has_multi_accelerators()` — [`L290`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_call.py#L290) — Check whether more than one non-CPU accelerator is available.
- `_in_shard_map_context()` — [`L305`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_call.py#L305) — Detect whether execution is currently inside a ``jax.shard_map`` context.
- `_leaf_shape_dtype(leaf: Any)` — [`L337`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_call.py#L337) — Read ``(shape, dtype)`` from an output leaf or tracer.
- `_out_contract_key(output_contract_shapes: Sequence[jax.ShapeDtypeStruct] | None)` — [`L472`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_call.py#L472) — Build a stable key for expected output contracts.
- `_shape_dtype_tree(value: Any)` — [`L488`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_call.py#L488) — Convert an output contract tree to ``jax.ShapeDtypeStruct`` leaves.
- `_shape_key(shape: Any)` — [`L438`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_call.py#L438) — Normalize shape values to a stable, hashable key.
- `_shape_specs_from_out_leaves(flat_out: Sequence[Any])` — [`L417`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_call.py#L417) — Build shape/dtype structs from explicit ``out`` leaves.
- `_validate_out_leaves(flat_out: Sequence[Any], flat_out_shapes: Sequence[jax.ShapeDtypeStruct] | None)` — [`L364`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_call.py#L364) — Validate explicit output leaves and optional output shape contracts.
- `build_tilelang_call(prim_func: Any | None = None, *, kernel: Callable[..., Any] | None = None, out_shape: ShapeDtype | Sequence[ShapeDtype] | None = None, output_shape_dtype: ShapeDtype | Sequence[ShapeDtype] | None = None, args: Sequence[Any] | None = None, meta: dict[str, Any] | None = None, configs: Sequence[dict[str, Any]] | None = None, name: str | None = None, input_output_aliases: dict[int, int] | None = None, target: str | None = None, target_host: str | None = None, pass_configs: dict[str, Any] | None = None, compile_flags: list[str] | tuple[str, ...] | None = None, warmup: int | None = None, rep: int | None = None, timeout: float | None = None, autotune: bool | None = None, cache_key: Any | None = None)` — [`L536`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_call.py#L536) — Build, cache, and optionally autotune a primitive-backed TileLang call.
- `tilelang_call(*args: Any, call: Callable[..., Any] | None = None, kernel: Callable[..., Any] | None = None, out_shape: ShapeDtype | Sequence[ShapeDtype] | None = None, out: Any | None = None, name: str | None = None, device: int | None = None, meta: dict[str, Any] | None = None, configs: Sequence[dict[str, Any]] | None = None, input_output_aliases: dict[int, int] | None = None, target: str | None = None, target_host: str | None = None, pass_configs: dict[str, Any] | None = None, compile_flags: list[str] | tuple[str, ...] | None = None, warmup: int | None = None, rep: int | None = None, timeout: float | None = None, autotune: bool | None = None, cache_key: Any | None = None)` — [`L714`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_call.py#L714) — Execute a tile-lang kernel and return its output(s).

## Module values
- `CAN_USE_TILELANG` — [`L40`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_call.py#L40)
- `_NAMED_CALL_CACHE` — [`L48`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_call.py#L48)
- `_NAMED_CALL_CACHE_LOCK` — [`L49`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_call.py#L49)
- `__all__` — [`L907`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_call.py#L907)


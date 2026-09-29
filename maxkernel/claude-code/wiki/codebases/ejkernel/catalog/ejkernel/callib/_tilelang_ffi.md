---
title: 'Module: ejkernel/callib/_tilelang_ffi.py'
type: catalog
provenance: extracted
module: ejkernel/callib/_tilelang_ffi.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.callib._tilelang_ffi`/
symbols:
  _compile_or_get_kernel: _compile_or_get_kernel().
  _register_target_once: _register_target_once().
  tilelang_kernel_call_p: tilelang_kernel_call_p.
  _tilelang_kernel_call_impl: _tilelang_kernel_call_impl().
  autotune_tilelang_ffi: autotune_tilelang_ffi().
  _CompiledTilelangKernel.target_name: _CompiledTilelangKernel#target_name.
  build_tilelang_ffi_call: build_tilelang_ffi_call().
  has_tilelang_ffi_support: has_tilelang_ffi_support().
  CAN_USE_TILELANG_PRIMITIVE: CAN_USE_TILELANG_PRIMITIVE.
  _HAS_JAX_TVM_FFI: _HAS_JAX_TVM_FFI.
  _CompiledTilelangKernel.adapter_func: _CompiledTilelangKernel#adapter_func.
  _COMPILE_CACHE._COMPILE_CACHE: _COMPILE_CACHE._COMPILE_CACHE.
  _CompiledTilelangKernel: _CompiledTilelangKernel#
  build_tilelang_ffi_call._call: build_tilelang_ffi_call()._call().
  _shape_dtype_key: _shape_dtype_key().
  _raise_on_jvp: _raise_on_jvp().
  _AUTOTUNE_LOCK: _AUTOTUNE_LOCK.
  _REGISTERED_TARGETS._REGISTERED_TARGETS: _REGISTERED_TARGETS._REGISTERED_TARGETS.
  _AUTOTUNE_CACHE._AUTOTUNE_CACHE: _AUTOTUNE_CACHE._AUTOTUNE_CACHE.
  _CompiledTilelangKernel.kernel: _CompiledTilelangKernel#kernel.
  _COMPILE_LOCK: _COMPILE_LOCK.
  _REGISTERED_TARGETS_LOCK: _REGISTERED_TARGETS_LOCK.
  _to_shape_dtype_struct: _to_shape_dtype_struct().
  _cache_key_hash: _cache_key_hash().
  _raise_on_vmap: _raise_on_vmap().
  _tilelang_kernel_call_abstract_eval: _tilelang_kernel_call_abstract_eval().
  __all__: __all__.
---
# Module: [`ejkernel/callib/_tilelang_ffi.py`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_ffi.py)

## Classes
### `_CompiledTilelangKernel`
- def: [`ejkernel/callib/_tilelang_ffi.py:65`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_ffi.py#L65)
- doc: Container for a compiled tile-lang adapter and its FFI target name.
- signature: `class _CompiledTilelangKernel:`
- members:
  - `adapter_func` — [`L77`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_ffi.py#L77)
  - `kernel` — [`L76`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_ffi.py#L76)
  - `target_name` — [`L75`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_ffi.py#L75)
- used by: [`_compile_or_get_kernel`](_tilelang_ffi.md#_compile_or_get_kernel), [`_register_target_once`](_tilelang_ffi.md#_register_target_once), [`_tilelang_kernel_call_impl`](_tilelang_ffi.md#_tilelang_kernel_call_impl), [`_COMPILE_CACHE`](_tilelang_ffi.md#_COMPILE_CACHE._COMPILE_CACHE)

## Functions
- `_cache_key_hash(cache_key: tuple[Any, ...])` — [`L113`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_ffi.py#L113) — Build a deterministic SHA-256 hex digest from a compile cache key.
- `_call(*args)` — [`L413`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_ffi.py#L413) — Dispatch the tile-lang kernel through the JAX primitive.
- `_compile_or_get_kernel(*, prim_func: Any, in_shaped: tuple[Any, ...], out_shaped: tuple[jax.ShapeDtypeStruct, ...], target: str | None, target_host: str | None, pass_configs: tuple[tuple[str, Any], ...] | None, compile_flags: tuple[str, ...] | None)` — [`L125`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_ffi.py#L125) — Compile (or fetch cached) tile-lang prim_func and FFI target metadata.
- `_raise_on_jvp(*args, **kwargs)` — [`L317`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_ffi.py#L317) — Raise for unsupported automatic differentiation through the primitive.
- `_raise_on_vmap(*args, **kwargs)` — [`L338`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_ffi.py#L338) — Raise for unsupported batching through the primitive.
- `_register_target_once(kernel: _CompiledTilelangKernel)` — [`L204`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_ffi.py#L204) — Register a compiled tile-lang callable as a JAX FFI target exactly once.
- `_shape_dtype_key(shaped: Any)` — [`L99`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_ffi.py#L99) — Build a stable compile-cache key fragment from a shaped value.
- `_tilelang_kernel_call_abstract_eval(*_, out_shape_dtype_flat, **__)` — [`L302`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_ffi.py#L302) — Primitive abstract evaluation returning output avals from the contract.
- `_tilelang_kernel_call_impl(*args_flat, prim_func: Any, out_shape_dtype_flat: tuple[jax.ShapeDtypeStruct, ...], input_output_aliases: tuple[tuple[int, int], ...], target: str | None, target_host: str | None, pass_configs: tuple[tuple[str, Any], ...] | None, compile_flags: tuple[str, ...] | None)` — [`L242`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_ffi.py#L242) — Primitive implementation shared by eager and lowering paths.
- `_to_shape_dtype_struct(out_shape: Any)` — [`L86`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_ffi.py#L86) — Normalize output descriptors into ``jax.ShapeDtypeStruct`` leaves.
- `autotune_tilelang_ffi(prim_func_builder: Any, configs: list[dict[str, Any]], *, example_inputs: list, output_shape_dtype: Any, cache_key: Any, input_output_aliases: dict[int, int] | None = None, target: str | None = None, target_host: str | None = None, pass_configs: dict[str, Any] | None = None, compile_flags: list[str] | tuple[str, ...] | None = None, warmup: int = 3, iters: int = 10, rep: int | None = None, timeout: float | None = None)` — [`L452`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_ffi.py#L452) — Compile every candidate config, time it on-device, return the fastest.
- `build_tilelang_ffi_call(prim_func: Any, *, output_shape_dtype: Any, input_output_aliases: dict[int, int] | None = None, target: str | None = None, target_host: str | None = None, pass_configs: dict[str, Any] | None = None, compile_flags: list[str] | tuple[str, ...] | None = None)` — [`L367`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_ffi.py#L367) — Create a callable that dispatches a tile-lang kernel through a JAX primitive.
- `has_tilelang_ffi_support()` — [`L438`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_ffi.py#L438) — Return whether the tile-lang TVM-FFI primitive path can be used.

## Module values
- `CAN_USE_TILELANG_PRIMITIVE` — [`L55`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_ffi.py#L55)
- `_AUTOTUNE_CACHE` — [`L448`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_ffi.py#L448)
- `_AUTOTUNE_LOCK` — [`L449`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_ffi.py#L449)
- `_COMPILE_CACHE` — [`L80`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_ffi.py#L80)
- `_COMPILE_LOCK` — [`L81`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_ffi.py#L81)
- `_HAS_JAX_TVM_FFI` — [`L47`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_ffi.py#L47)
- `_REGISTERED_TARGETS` — [`L82`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_ffi.py#L82)
- `_REGISTERED_TARGETS_LOCK` — [`L83`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_ffi.py#L83)
- `__all__` — [`L556`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_ffi.py#L556)
- `tilelang_kernel_call_p` — [`L296`](../../../../../../raw/code/ejkernel/ejkernel/callib/_tilelang_ffi.py#L296)


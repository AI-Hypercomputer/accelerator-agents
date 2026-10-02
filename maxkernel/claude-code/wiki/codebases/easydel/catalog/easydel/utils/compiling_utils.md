---
title: 'Module: easydel/utils/compiling_utils.py'
type: catalog
provenance: extracted
module: easydel/utils/compiling_utils.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.utils.compiling_utils`/
symbols:
  hash_fn: hash_fn().
  smart_compile: smart_compile().
  COMPILE_FUNC_DIR: COMPILE_FUNC_DIR.
  load_cached_functions: load_cached_functions().
  COMPILED_FILE_NAME: COMPILED_FILE_NAME.
  NoCompileContext.__enter__: NoCompileContext#__enter__().
  ECACHE_COMPILES: ECACHE_COMPILES.
  NoCompileContext._original_func: NoCompileContext#_original_func.
  RECOMPILE_FORCE: RECOMPILE_FORCE.
  CACHE_DIR: CACHE_DIR.
  save_compiled_fn: save_compiled_fn().
  load_compiled_fn: load_compiled_fn().
  NoCompileContext.wrapper: NoCompileContext#wrapper().
  get_safe_hash_int: get_safe_hash_int().
  _get_args_signature: _get_args_signature().
  NoCompileContext.__exit__: NoCompileContext#__exit__().
  SIGNATURE_FILE_NAME: SIGNATURE_FILE_NAME.
  _get_leaf_signature: _get_leaf_signature().
  get_hash_of_lowering: get_hash_of_lowering().
  NoCompileContext: NoCompileContext#
  NoCompileContext.message: NoCompileContext#message.
  COMPILED_CACHE.COMPILED_CACHE: COMPILED_CACHE.COMPILED_CACHE.
  __all__: __all__.
  P: P.
  R: R.
  _get_hardware_signature: _get_hardware_signature().
  NoCompileContext.__init__: NoCompileContext#__init__().
---
# Module: [`easydel/utils/compiling_utils.py`](../../../../../../raw/code/EasyDeL/easydel/utils/compiling_utils.py)

## Classes
### `NoCompileContext`
- def: [`easydel/utils/compiling_utils.py:391`](../../../../../../raw/code/EasyDeL/easydel/utils/compiling_utils.py#L391)
- doc: Context manager that fails if JAX triggers a new compilation.
- signature: `class NoCompileContext:`
- members:
  - `__enter__(self)` — [`L402`](../../../../../../raw/code/EasyDeL/easydel/utils/compiling_utils.py#L402) — Patch JAX's cached lowering to detect compilation cache misses.
  - `__exit__(self, exc_type, exc_value, traceback)` — [`L424`](../../../../../../raw/code/EasyDeL/easydel/utils/compiling_utils.py#L424) — Restore the cached lowering function.
  - `__init__(self, message: str = "JAX attempted to compile a new executable inside ForbidCompile.")` — [`L397`](../../../../../../raw/code/EasyDeL/easydel/utils/compiling_utils.py#L397) — Initialize the guard with a custom failure message.
  - `wrapper(*args, **kwargs)` — [`L409`](../../../../../../raw/code/EasyDeL/easydel/utils/compiling_utils.py#L409)
  - `message` — [`L399`](../../../../../../raw/code/EasyDeL/easydel/utils/compiling_utils.py#L399)
- protocol/private: `_original_func`[`L400`](../../../../../../raw/code/EasyDeL/easydel/utils/compiling_utils.py#L400)

## Functions
- `_get_args_signature(args: tuple, kwargs: dict)` — [`L131`](../../../../../../raw/code/EasyDeL/easydel/utils/compiling_utils.py#L131) — Create signature for function arguments.
- `_get_hardware_signature()` — [`L106`](../../../../../../raw/code/EasyDeL/easydel/utils/compiling_utils.py#L106) — Create signature for current JAX hardware environment.
- `_get_leaf_signature(leaf: tp.Any)` — [`L115`](../../../../../../raw/code/EasyDeL/easydel/utils/compiling_utils.py#L115) — Generate hashable signature for PyTree leaf.
- `get_hash_of_lowering(lowered_func: Lowered)` — [`L295`](../../../../../../raw/code/EasyDeL/easydel/utils/compiling_utils.py#L295) — Compute a SHA-256 hash of a lowered JAX function's text representation.
- `get_safe_hash_int(text, algorithm="md5")` — [`L271`](../../../../../../raw/code/EasyDeL/easydel/utils/compiling_utils.py#L271) — Generate an integer hash of text using the specified algorithm.
- `hash_fn(self)` — [`L255`](../../../../../../raw/code/EasyDeL/easydel/utils/compiling_utils.py#L255) — Generate a deterministic hash for an object based on its ``__dict__`` values. — documented in [easydel-infra-base_config](../../../concepts/easydel-infra-base_config.md)
- `load_cached_functions(verbose: bool = True)` — [`L149`](../../../../../../raw/code/EasyDeL/easydel/utils/compiling_utils.py#L149) — Pre-loads all valid cached functions from disk into the persistent L2 cache.
- `load_compiled_fn(path: str | os.PathLike, prefix: str | None = None)` — [`L228`](../../../../../../raw/code/EasyDeL/easydel/utils/compiling_utils.py#L228) — Load a previously saved compiled JAX function from disk.
- `save_compiled_fn(path: str | os.PathLike, fn: Compiled, prefix: str | None = None)` — [`L177`](../../../../../../raw/code/EasyDeL/easydel/utils/compiling_utils.py#L177) — Save a compiled JAX function to disk for later reuse.
- `smart_compile(lowered_func: Lowered, tag: str | None = None, verbose: bool = True, cache_key: tuple[str, tuple] | None = None)` — [`L313`](../../../../../../raw/code/EasyDeL/easydel/utils/compiling_utils.py#L313) — Compile a lowered JAX function with two-level (disk + memory) caching.

## Module values
- `CACHE_DIR` — [`L95`](../../../../../../raw/code/EasyDeL/easydel/utils/compiling_utils.py#L95)
- `COMPILED_CACHE` — [`L103`](../../../../../../raw/code/EasyDeL/easydel/utils/compiling_utils.py#L103)
- `COMPILED_FILE_NAME` — [`L101`](../../../../../../raw/code/EasyDeL/easydel/utils/compiling_utils.py#L101)
- `COMPILE_FUNC_DIR` — [`L97`](../../../../../../raw/code/EasyDeL/easydel/utils/compiling_utils.py#L97)
- `ECACHE_COMPILES` — [`L93`](../../../../../../raw/code/EasyDeL/easydel/utils/compiling_utils.py#L93)
- `P` — [`L89`](../../../../../../raw/code/EasyDeL/easydel/utils/compiling_utils.py#L89)
- `R` — [`L90`](../../../../../../raw/code/EasyDeL/easydel/utils/compiling_utils.py#L90)
- `RECOMPILE_FORCE` — [`L92`](../../../../../../raw/code/EasyDeL/easydel/utils/compiling_utils.py#L92)
- `SIGNATURE_FILE_NAME` — [`L102`](../../../../../../raw/code/EasyDeL/easydel/utils/compiling_utils.py#L102)
- `__all__` — [`L77`](../../../../../../raw/code/EasyDeL/easydel/utils/compiling_utils.py#L77)


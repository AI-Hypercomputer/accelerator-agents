---
title: 'Module: jax/_src/compilation_cache.py'
type: catalog
provenance: extracted
module: jax/_src/compilation_cache.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.compilation_cache`/
symbols:
  _initialize_cache: _initialize_cache().
  put_executable_and_time: put_executable_and_time().
  reset_cache: reset_cache().
  _get_cache: _get_cache().
  VerificationCache.put: VerificationCache#put().
  get_file_cache: get_file_cache().
  get_executable_and_time: get_executable_and_time().
  _cache._cache: _cache._cache.
  VerificationCache: VerificationCache#
  is_cache_used: is_cache_used().
  VerificationCache.get: VerificationCache#get().
  logger: logger.
  VerificationCache._path: VerificationCache#_path().
  is_persistent_cache_enabled: is_persistent_cache_enabled().
  set_cache_dir: set_cache_dir().
  _cache_checked._cache_checked: _cache_checked._cache_checked.
  _cache_used._cache_used: _cache_used._cache_used.
  _is_cache_enabled: _is_cache_enabled().
  extract_executable_and_time: extract_executable_and_time().
  VerificationCache._base_cache: VerificationCache#_base_cache.
  initialize_cache: initialize_cache().
  is_executable_in_cache: is_executable_in_cache().
  _cache_initialized._cache_initialized: _cache_initialized._cache_initialized.
  _cache_initialized_mutex: _cache_initialized_mutex.
  VerificationCache._verified_keys: VerificationCache#_verified_keys.
  combine_executable_and_time: combine_executable_and_time().
  _TIME_BYTES: _TIME_BYTES.
  decompress_executable: decompress_executable().
  VerificationCache.__init__: VerificationCache#__init__().
  VerificationCache.clear: VerificationCache#clear().
  is_initialized: is_initialized().
  VerificationCache.base_cache_hits: VerificationCache#base_cache_hits.
  CacheVerificationError: CacheVerificationError#
  default_min_cache_entry_size: default_min_cache_entry_size().
  compress_executable: compress_executable().
  _UNSUPPORTED_RUNTIMES._UNSUPPORTED_RUNTIMES: _UNSUPPORTED_RUNTIMES._UNSUPPORTED_RUNTIMES.
  CacheVerificationError.__init__: CacheVerificationError#__init__().
  CacheVerificationError.cache_key: CacheVerificationError#cache_key.
  CacheVerificationError.executable_on_disk: CacheVerificationError#executable_on_disk.
  CacheVerificationError.executable_new: CacheVerificationError#executable_new.
  get_cache_key: get_cache_key().
---
# Module: [`jax/_src/compilation_cache.py`](../../../../../../raw/code/jax/jax/_src/compilation_cache.py)

## Classes
### `CacheVerificationError`  ·  implements/extends RuntimeError
- def: [`jax/_src/compilation_cache.py:113`](../../../../../../raw/code/jax/jax/_src/compilation_cache.py#L113)
- doc: Error raised when a freshly compiled executable does not exactly
- signature: `class CacheVerificationError(RuntimeError):`
- members:
  - `cache_key` — [`L126`](../../../../../../raw/code/jax/jax/_src/compilation_cache.py#L126)
  - `executable_new` — [`L128`](../../../../../../raw/code/jax/jax/_src/compilation_cache.py#L128)
  - `executable_on_disk` — [`L127`](../../../../../../raw/code/jax/jax/_src/compilation_cache.py#L127)
- protocol/private: `__init__`[`L118`](../../../../../../raw/code/jax/jax/_src/compilation_cache.py#L118)
- used by: [`put`](compilation_cache.md#VerificationCache.put)

### `VerificationCache`
- def: [`jax/_src/compilation_cache.py:131`](../../../../../../raw/code/jax/jax/_src/compilation_cache.py#L131)
- members:
  - `clear(self)` — [`L189`](../../../../../../raw/code/jax/jax/_src/compilation_cache.py#L189)
  - `get(self, key: str)` — [`L154`](../../../../../../raw/code/jax/jax/_src/compilation_cache.py#L154)
  - `put(self, key: str, value: bytes)` — [`L165`](../../../../../../raw/code/jax/jax/_src/compilation_cache.py#L165)
  - `base_cache_hits` — [`L144`](../../../../../../raw/code/jax/jax/_src/compilation_cache.py#L144)
- protocol/private: `__init__`[`L141`](../../../../../../raw/code/jax/jax/_src/compilation_cache.py#L141), `_base_cache`[`L142`](../../../../../../raw/code/jax/jax/_src/compilation_cache.py#L142), `_path`[`L147`](../../../../../../raw/code/jax/jax/_src/compilation_cache.py#L147), `_verified_keys`[`L143`](../../../../../../raw/code/jax/jax/_src/compilation_cache.py#L143)
- uses (calls/refs, reference-scoped): [`CacheInterface`](compilation_cache_interface.md#CacheInterface), [`get`](compilation_cache_interface.md#CacheInterface.get), [`put`](compilation_cache_interface.md#CacheInterface.put), [`extract_executable_and_time`](compilation_cache.md#extract_executable_and_time), [`_path`](compilation_cache_interface.md#CacheInterface._path), [`decompress_executable`](compilation_cache.md#decompress_executable), [`CacheVerificationError`](compilation_cache.md#CacheVerificationError)
- used by: [`get_file_cache`](compilation_cache.md#get_file_cache), [`get`](compilation_cache_interface.md#CacheInterface.get), [`put`](compilation_cache_interface.md#CacheInterface.put)

## Functions
- `_get_cache(backend)` — [`L281`](../../../../../../raw/code/jax/jax/_src/compilation_cache.py#L281)
- `_initialize_cache()` — [`L241`](../../../../../../raw/code/jax/jax/_src/compilation_cache.py#L241)
- `_is_cache_enabled()` — [`L237`](../../../../../../raw/code/jax/jax/_src/compilation_cache.py#L237)
- `combine_executable_and_time(serialized_executable: bytes, compile_time: int)` — [`L460`](../../../../../../raw/code/jax/jax/_src/compilation_cache.py#L460) — Given the serialized executable and the compilation time, produce a cache
- `compress_executable(executable: bytes)` — [`L296`](../../../../../../raw/code/jax/jax/_src/compilation_cache.py#L296)
- `decompress_executable(executable: bytes)` — [`L305`](../../../../../../raw/code/jax/jax/_src/compilation_cache.py#L305)
- `default_min_cache_entry_size()` — [`L232`](../../../../../../raw/code/jax/jax/_src/compilation_cache.py#L232) — Returns the minimum size below which the entry should not be cached.
- `extract_executable_and_time(executable_and_time: bytes)` — [`L477`](../../../../../../raw/code/jax/jax/_src/compilation_cache.py#L477) — Given the cache entry in the format shown below, extract the serialized
- `get_cache_key(module: ir.Module, devices: np.ndarray, compile_options, backend, ignore_custom_partitioning: bool = False)` — [`L414`](../../../../../../raw/code/jax/jax/_src/compilation_cache.py#L414)
- `get_executable_and_time(cache_key: str, compile_options, backend, executable_devices, host_callbacks: Sequence[Any] = ())` — [`L326`](../../../../../../raw/code/jax/jax/_src/compilation_cache.py#L326) — Returns the cached executable and its compilation time if present, or None
- `get_file_cache(path: str)` — [`L104`](../../../../../../raw/code/jax/jax/_src/compilation_cache.py#L104) — Returns the file cache and the path to the cache.
- `initialize_cache(path)` — [`L213`](../../../../../../raw/code/jax/jax/_src/compilation_cache.py#L213) — This API is deprecated; use set_cache_dir instead.
- `is_cache_used(backend: xla_client.Client)` — [`L68`](../../../../../../raw/code/jax/jax/_src/compilation_cache.py#L68) — Check if cache is used and report adoption metrics one-time per task.
- `is_executable_in_cache(backend, cache_key: str)` — [`L315`](../../../../../../raw/code/jax/jax/_src/compilation_cache.py#L315) — Checks if the executable is in the cache.
- `is_initialized()` — [`L431`](../../../../../../raw/code/jax/jax/_src/compilation_cache.py#L431) — Deprecated.
- `is_persistent_cache_enabled()` — [`L276`](../../../../../../raw/code/jax/jax/_src/compilation_cache.py#L276)
- `put_executable_and_time(cache_key: str, module_name: str, executable: xla_client.LoadedExecutable, backend, compile_time: int)` — [`L358`](../../../../../../raw/code/jax/jax/_src/compilation_cache.py#L358) — Adds the 'executable' and its compilation time to the cache, possibly
- `reset_cache()` — [`L442`](../../../../../../raw/code/jax/jax/_src/compilation_cache.py#L442) — Get back to pristine, uninitialized state.
- `set_cache_dir(path)` — [`L193`](../../../../../../raw/code/jax/jax/_src/compilation_cache.py#L193) — Sets the persistent compilation cache directory.

## Module values
- `_TIME_BYTES` — [`L66`](../../../../../../raw/code/jax/jax/_src/compilation_cache.py#L66)
- `_UNSUPPORTED_RUNTIMES` — [`L64`](../../../../../../raw/code/jax/jax/_src/compilation_cache.py#L64)
- `_cache` — [`L53`](../../../../../../raw/code/jax/jax/_src/compilation_cache.py#L53)
- `_cache_checked` — [`L57`](../../../../../../raw/code/jax/jax/_src/compilation_cache.py#L57)
- `_cache_initialized` — [`L55`](../../../../../../raw/code/jax/jax/_src/compilation_cache.py#L55)
- `_cache_initialized_mutex` — [`L62`](../../../../../../raw/code/jax/jax/_src/compilation_cache.py#L62)
- `_cache_used` — [`L59`](../../../../../../raw/code/jax/jax/_src/compilation_cache.py#L59)
- `logger` — [`L51`](../../../../../../raw/code/jax/jax/_src/compilation_cache.py#L51)


---
title: 'Module: jax/_src/compilation_cache_interface.py'
type: catalog
provenance: extracted
module: jax/_src/compilation_cache_interface.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.compilation_cache_interface`/CacheInterface#
symbols:
  CacheInterface: ''
  CacheInterface.get: get().
  CacheInterface.put: put().
  CacheInterface._path: _path.
---
# Module: [`jax/_src/compilation_cache_interface.py`](../../../../../../raw/code/jax/jax/_src/compilation_cache_interface.py)

## Classes
### `CacheInterface`  ·  implements/extends StrictABC
- def: [`jax/_src/compilation_cache_interface.py:23`](../../../../../../raw/code/jax/jax/_src/compilation_cache_interface.py#L23)
- signature: `class CacheInterface(util.StrictABC):`
- members:
  - `get(self, key: str)` — [`L27`](../../../../../../raw/code/jax/jax/_src/compilation_cache_interface.py#L27)
  - `put(self, key: str, value: bytes)` — [`L31`](../../../../../../raw/code/jax/jax/_src/compilation_cache_interface.py#L31)
- protocol/private: `_path`[`L24`](../../../../../../raw/code/jax/jax/_src/compilation_cache_interface.py#L24)
- uses (calls/refs, reference-scoped): [`put`](lru_cache.md#LRUCache.put), [`get`](lru_cache.md#LRUCache.get), [`put`](compilation_cache.md#VerificationCache.put), [`StrictABC`](util.md#StrictABC), [`get`](compilation_cache.md#VerificationCache.get), [`LRUCache`](lru_cache.md#LRUCache)
- used by: [`put_executable_and_time`](compilation_cache.md#put_executable_and_time), [`reset_cache`](compilation_cache.md#reset_cache), [`_get_cache`](compilation_cache.md#_get_cache), [`put`](compilation_cache.md#VerificationCache.put), [`path`](lru_cache.md#LRUCache.path), [`StrictABC`](util.md#StrictABC), [`get_file_cache`](compilation_cache.md#get_file_cache), [`get_executable_and_time`](compilation_cache.md#get_executable_and_time), [`_cache`](compilation_cache.md#_cache._cache), [`VerificationCache`](compilation_cache.md#VerificationCache), [`get`](compilation_cache.md#VerificationCache.get), [`_path`](compilation_cache.md#VerificationCache._path), [`filelock`](lru_cache.md#filelock.filelock), [`is_executable_in_cache`](compilation_cache.md#is_executable_in_cache), [`LRUCache`](lru_cache.md#LRUCache), [`__init__`](compilation_cache.md#VerificationCache.__init__)


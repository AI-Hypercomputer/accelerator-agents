---
title: 'Module: jax/_src/lru_cache.py'
type: catalog
provenance: extracted
module: jax/_src/lru_cache.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.lru_cache`/
symbols:
  LRUCache.put: LRUCache#put().
  LRUCache.get: LRUCache#get().
  LRUCache._evict_if_needed: LRUCache#_evict_if_needed().
  LRUCache.path: LRUCache#path.
  LRUCache.eviction_enabled: LRUCache#eviction_enabled.
  LRUCache.lock: LRUCache#lock.
  LRUCache.lock_path: LRUCache#lock_path.
  filelock.filelock: filelock.filelock.
  _CACHE_SUFFIX: _CACHE_SUFFIX.
  LRUCache: LRUCache#
  _ATIME_SUFFIX: _ATIME_SUFFIX.
  logger: logger.
  LRUCache.max_size: LRUCache#max_size.
  LRUCache.__init__: LRUCache#__init__().
  _is_local_filesystem: _is_local_filesystem().
  LRUCache.lock_timeout_secs: LRUCache#lock_timeout_secs.
---
# Module: [`jax/_src/lru_cache.py`](../../../../../../raw/code/jax/jax/_src/lru_cache.py)

## Classes
### `LRUCache`  ·  implements/extends CacheInterface
- def: [`jax/_src/lru_cache.py:43`](../../../../../../raw/code/jax/jax/_src/lru_cache.py#L43)
- doc: Bounded cache with least-recently-used (LRU) eviction policy.
- signature: `class LRUCache(CacheInterface):`
- members:
  - `__init__(self, path: str, *, max_size: int, lock_timeout_secs: float | None = 10)` — [`L54`](../../../../../../raw/code/jax/jax/_src/lru_cache.py#L54) — Args:
  - `_evict_if_needed(self, *, additional_size: int = 0)` — [`L163`](../../../../../../raw/code/jax/jax/_src/lru_cache.py#L163) — Evicts the least recently used items from the cache if necessary
  - `get(self, key: str)` — [`L84`](../../../../../../raw/code/jax/jax/_src/lru_cache.py#L84) — Retrieves the cached value for the given key.
  - `put(self, key: str, value: bytes)` — [`L121`](../../../../../../raw/code/jax/jax/_src/lru_cache.py#L121) — Adds a new entry to the cache.
  - `eviction_enabled` — [`L69`](../../../../../../raw/code/jax/jax/_src/lru_cache.py#L69)
  - `lock` — [`L80`](../../../../../../raw/code/jax/jax/_src/lru_cache.py#L80)
  - `lock_path` — [`L78`](../../../../../../raw/code/jax/jax/_src/lru_cache.py#L78)
  - `lock_timeout_secs` — [`L76`](../../../../../../raw/code/jax/jax/_src/lru_cache.py#L76)
  - `max_size` — [`L75`](../../../../../../raw/code/jax/jax/_src/lru_cache.py#L75)
  - `path` — [`L66`](../../../../../../raw/code/jax/jax/_src/lru_cache.py#L66)
- uses (calls/refs, reference-scoped): [`CacheInterface`](compilation_cache_interface.md#CacheInterface), [`filelock`](lru_cache.md#filelock.filelock), [`_CACHE_SUFFIX`](lru_cache.md#_CACHE_SUFFIX), [`_ATIME_SUFFIX`](lru_cache.md#_ATIME_SUFFIX), [`_path`](compilation_cache_interface.md#CacheInterface._path), [`logger`](lru_cache.md#logger), [`_is_local_filesystem`](lru_cache.md#_is_local_filesystem)
- used by: [`get_file_cache`](compilation_cache.md#get_file_cache), [`CacheInterface`](compilation_cache_interface.md#CacheInterface), [`get`](compilation_cache_interface.md#CacheInterface.get), [`put`](compilation_cache_interface.md#CacheInterface.put)

## Functions
- `_is_local_filesystem(path: str)` — [`L39`](../../../../../../raw/code/jax/jax/_src/lru_cache.py#L39)

## Module values
- `_ATIME_SUFFIX` — [`L36`](../../../../../../raw/code/jax/jax/_src/lru_cache.py#L36)
- `_CACHE_SUFFIX` — [`L35`](../../../../../../raw/code/jax/jax/_src/lru_cache.py#L35)
- `filelock` — [`L23`](../../../../../../raw/code/jax/jax/_src/lru_cache.py#L23)
- `logger` — [`L32`](../../../../../../raw/code/jax/jax/_src/lru_cache.py#L32)


---
title: 'Module: easydel/data/execution/cache.py'
type: catalog
provenance: extracted
module: easydel/data/execution/cache.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.data.execution.cache`/
symbols:
  CacheMetadata.to_dict: CacheMetadata#to_dict().
  DiskCache.get: DiskCache#get().
  CacheMetadata: CacheMetadata#
  MemoryCache._cache: MemoryCache#_cache.
  TreeCacheManager.stats: TreeCacheManager#stats().
  MemoryCache.stats: MemoryCache#stats().
  MemoryCache.get: MemoryCache#get().
  DiskCache.put: DiskCache#put().
  TreeCacheManager._layers: TreeCacheManager#_layers.
  TreeCacheManager.get_or_compute: TreeCacheManager#get_or_compute().
  DatasetCache.get: DatasetCache#get().
  CacheLayer: CacheLayer#
  CacheLayer.get: CacheLayer#get().
  CacheLayer.put: CacheLayer#put().
  MemoryCache.put: MemoryCache#put().
  DiskCache.cache_dir: DiskCache#cache_dir.
  DiskCache.compression: DiskCache#compression.
  DiskCache.invalidate: DiskCache#invalidate().
  TreeCacheManager._disk: TreeCacheManager#_disk.
  DiskCache._get_paths: DiskCache#_get_paths().
  DatasetCache.cache_dir: DatasetCache#cache_dir.
  DatasetCache._get_dataset_path: DatasetCache#_get_dataset_path().
  TreeCacheManager.get: TreeCacheManager#get().
  TreeCacheManager.put: TreeCacheManager#put().
  DatasetCache.invalidate: DatasetCache#invalidate().
  MemoryCache: MemoryCache#
  DiskCache: DiskCache#
  TreeCacheManager.cache_dir: TreeCacheManager#cache_dir.
  CacheMetadata.is_valid_for: CacheMetadata#is_valid_for().
  CacheLayer.contains: CacheLayer#contains().
  CacheLayer.invalidate: CacheLayer#invalidate().
  CacheLayer.get_metadata: CacheLayer#get_metadata().
  DiskCache.contains: DiskCache#contains().
  DatasetCache.put: DatasetCache#put().
  TreeCacheManager._memory: TreeCacheManager#_memory.
  logger: logger.
  MemoryCache._hits: MemoryCache#_hits.
  DiskCache.expiry_seconds: DiskCache#expiry_seconds.
  TreeCacheManager: TreeCacheManager#
  CacheMetadata.from_dict: CacheMetadata#from_dict().
  DiskCache._compress: DiskCache#_compress().
  DiskCache._decompress: DiskCache#_decompress().
  TreeCacheManager.invalidate: TreeCacheManager#invalidate().
  MemoryCache._misses: MemoryCache#_misses.
  MemoryCache.contains: MemoryCache#contains().
  MemoryCache.invalidate: MemoryCache#invalidate().
  TreeCacheManager.contains: TreeCacheManager#contains().
  DatasetCache.contains: DatasetCache#contains().
  DatasetLike: DatasetLike.
  CacheMetadata.source_hash: CacheMetadata#source_hash.
  CacheMetadata.config_hash: CacheMetadata#config_hash.
  MemoryCache.max_size: MemoryCache#max_size.
  DatasetCache: DatasetCache#
  CacheMetadata.version: CacheMetadata#version.
  CacheMetadata.created_at: CacheMetadata#created_at.
  CacheMetadata.tokenizer_hash: CacheMetadata#tokenizer_hash.
  CacheMetadata.transform_hash: CacheMetadata#transform_hash.
  CacheMetadata.num_examples: CacheMetadata#num_examples.
  CacheMetadata.extra: CacheMetadata#extra.
  MemoryCache.__init__: MemoryCache#__init__().
  DiskCache.__init__: DiskCache#__init__().
  TreeCacheManager.__init__: TreeCacheManager#__init__().
  TreeCacheManager.compute_key: TreeCacheManager#compute_key().
  DatasetCache.__init__: DatasetCache#__init__().
---
# Module: [`easydel/data/execution/cache.py`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py)

## Classes
### `CacheLayer`  ·  implements/extends ABC
- def: [`easydel/data/execution/cache.py:125`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L125)
- doc: Abstract interface for a single cache layer.
- signature: `class CacheLayer(ABC):`
- members:
  - `contains(self, key: str)` — [`L158`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L158) — Check if a key exists in the cache.
  - `get(self, key: str)` — [`L133`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L133) — Get item from cache.
  - `get_metadata(self, key: str)` — [`L178`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L178) — Get metadata for a key without loading the full data.
  - `invalidate(self, key: str | None = None)` — [`L170`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L170) — Invalidate a specific key or the entire cache.
  - `put(self, key: str, value: tp.Any, metadata: CacheMetadata | None = None)` — [`L142`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L142) — Store an item in cache with optional metadata.
- uses (calls/refs, reference-scoped): [`get`](cache.md#DiskCache.get), [`CacheMetadata`](cache.md#CacheMetadata), [`get`](cache.md#MemoryCache.get), [`put`](cache.md#DiskCache.put), [`put`](cache.md#MemoryCache.put), [`invalidate`](cache.md#DiskCache.invalidate), [`DiskCache`](cache.md#DiskCache), [`MemoryCache`](cache.md#MemoryCache), [`contains`](cache.md#DiskCache.contains), [`contains`](cache.md#MemoryCache.contains), [`invalidate`](cache.md#MemoryCache.invalidate)
- used by: [`DiskCache`](cache.md#DiskCache), [`MemoryCache`](cache.md#MemoryCache)

### `CacheMetadata`
- def: [`easydel/data/execution/cache.py:47`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L47)
- doc: Metadata stored alongside cached data for validation and invalidation.
- signature: `class CacheMetadata:`
- members:
  - `from_dict(cls, data: dict)` — [`L88`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L88) — Deserialize metadata from a dictionary.
  - `is_valid_for(self, config_hash: str, source_hash: str | None = None)` — [`L108`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L108) — Check if this cache entry is still valid for the given configuration.
  - `to_dict(self)` — [`L70`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L70) — Serialize metadata to a plain dictionary for JSON storage.
  - `config_hash` — [`L67`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L67)
  - `created_at` — [`L62`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L62)
  - `extra` — [`L68`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L68)
  - `num_examples` — [`L66`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L66)
  - `source_hash` — [`L63`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L63)
  - `tokenizer_hash` — [`L64`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L64)
  - `transform_hash` — [`L65`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L65)
  - `version` — [`L61`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L61)
- used by: [`get`](cache.md#DiskCache.get), [`_cache`](cache.md#MemoryCache._cache), [`get`](cache.md#MemoryCache.get), [`get_or_compute`](cache.md#TreeCacheManager.get_or_compute), [`put`](cache.md#DiskCache.put), [`get`](cache.md#CacheLayer.get), [`put`](cache.md#CacheLayer.put), [`put`](cache.md#MemoryCache.put), [`get`](cache.md#TreeCacheManager.get), [`put`](cache.md#TreeCacheManager.put), [`get_metadata`](cache.md#CacheLayer.get_metadata)

### `DatasetCache`
- def: [`easydel/data/execution/cache.py:598`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L598)
- doc: Specialized cache for HuggingFace datasets.
- signature: `class DatasetCache:`
- members:
  - `__init__(self, cache_dir: str | Path)` — [`L607`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L607) — Initialize DatasetCache.
  - `_get_dataset_path(self, key: str)` — [`L616`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L616) — Get the filesystem path for a cached dataset.
  - `contains(self, key: str)` — [`L667`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L667) — Check if a dataset is cached on disk.
  - `get(self, key: str)` — [`L628`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L628) — Load a cached dataset from disk.
  - `invalidate(self, key: str | None = None)` — [`L679`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L679) — Invalidate cached dataset(s) by removing them from disk.
  - `put(self, key: str, dataset: DatasetLike)` — [`L655`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L655) — Save a dataset to the disk cache.
  - `cache_dir` — [`L613`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L613)
- uses (calls/refs, reference-scoped): [`logger`](cache.md#logger), [`DatasetLike`](cache.md#DatasetLike)

### `DiskCache`  ·  implements/extends CacheLayer
- def: [`easydel/data/execution/cache.py:256`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L256)
- doc: Disk-based cache with optional compression and expiry.
- signature: `class DiskCache(CacheLayer):`
- members:
  - `__init__(self, cache_dir: str | Path, compression: str = "none", expiry_seconds: int | None = None)` — [`L265`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L265) — Initialize DiskCache.
  - `_compress(self, data: bytes)` — [`L290`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L290) — Compress data using configured algorithm.
  - `_decompress(self, data: bytes)` — [`L312`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L312) — Decompress data using configured algorithm.
  - `_get_paths(self, key: str)` — [`L283`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L283) — Get data and metadata file paths for a key.
  - `contains(self, key: str)` — [`L398`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L398) — Check if key exists on disk and has not expired.
  - `get(self, key: str)` — [`L334`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L334) — Load data and metadata from disk, checking expiry and decompressing.
  - `invalidate(self, key: str | None = None)` — [`L412`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L412) — Remove cached files from disk for a key, or clear the entire cache directory.
  - `put(self, key: str, value: tp.Any, metadata: CacheMetadata | None = None)` — [`L380`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L380) — Serialize, compress, and write data to disk with optional metadata.
  - `cache_dir` — [`L278`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L278)
  - `compression` — [`L280`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L280)
  - `expiry_seconds` — [`L281`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L281)
- uses (calls/refs, reference-scoped): [`to_dict`](cache.md#CacheMetadata.to_dict), [`CacheMetadata`](cache.md#CacheMetadata), [`CacheLayer`](cache.md#CacheLayer), [`logger`](cache.md#logger), [`from_dict`](cache.md#CacheMetadata.from_dict)
- used by: [`stats`](cache.md#TreeCacheManager.stats), [`CacheLayer`](cache.md#CacheLayer), [`get`](cache.md#CacheLayer.get), [`put`](cache.md#CacheLayer.put), [`_disk`](cache.md#TreeCacheManager._disk), [`contains`](cache.md#CacheLayer.contains), [`invalidate`](cache.md#CacheLayer.invalidate)

### `MemoryCache`  ·  implements/extends CacheLayer
- def: [`easydel/data/execution/cache.py:191`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L191)
- doc: In-memory LRU cache layer.
- signature: `class MemoryCache(CacheLayer):`
- members:
  - `__init__(self, max_size: int = 1000)` — [`L197`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L197) — Initialize MemoryCache.
  - `contains(self, key: str)` — [`L232`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L232) — Check if key exists in memory cache.
  - `get(self, key: str)` — [`L208`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L208) — Retrieve an item from memory, promoting it to most-recently-used.
  - `invalidate(self, key: str | None = None)` — [`L236`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L236) — Remove a specific key or clear the entire memory cache.
  - `put(self, key: str, value: tp.Any, metadata: CacheMetadata | None = None)` — [`L217`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L217) — Store an item, evicting the oldest entry if at capacity.
  - `stats(self)` — [`L244`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L244) — Return cache statistics.
  - `max_size` — [`L203`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L203)
- protocol/private: `_cache`[`L204`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L204), `_hits`[`L205`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L205), `_misses`[`L206`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L206)
- uses (calls/refs, reference-scoped): [`CacheMetadata`](cache.md#CacheMetadata), [`CacheLayer`](cache.md#CacheLayer)
- used by: [`stats`](cache.md#TreeCacheManager.stats), [`CacheLayer`](cache.md#CacheLayer), [`get`](cache.md#CacheLayer.get), [`put`](cache.md#CacheLayer.put), [`contains`](cache.md#CacheLayer.contains), [`invalidate`](cache.md#CacheLayer.invalidate), [`_memory`](cache.md#TreeCacheManager._memory)

### `TreeCacheManager`
- def: [`easydel/data/execution/cache.py:426`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L426)
- doc: Multi-layer hierarchical cache for processed data (Levanter-style).
- signature: `class TreeCacheManager:`
- members:
  - `__init__(self, cache_dir: str | Path, memory_size: int = 100, disk_expiry: int | None = 86400, compression: str = "none")` — [`L437`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L437) — Initialize TreeCacheManager.
  - `compute_key(config: dict, prefix: str = "", include_content_hash: bool = False, content: str | None = None)` — [`L558`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L558) — Compute a cache key from configuration.
  - `contains(self, key: str)` — [`L500`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L500) — Check if key exists in any cache layer.
  - `get(self, key: str)` — [`L463`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L463) — Get from cache, checking layers from fastest to slowest.
  - `get_or_compute(self, key: str, compute_fn: tp.Callable[[], tp.Any], metadata: CacheMetadata | None = None, validate_fn: tp.Callable[[CacheMetadata], bool] | None = None)` — [`L520`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L520) — Get from cache or compute and store.
  - `invalidate(self, key: str | None = None)` — [`L511`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L511) — Invalidate a key from all cache layers.
  - `put(self, key: str, value: tp.Any, metadata: CacheMetadata | None = None)` — [`L484`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L484) — Store data in all cache layers simultaneously.
  - `stats(self)` — [`L587`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L587) — Return cache statistics.
  - `cache_dir` — [`L452`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L452)
- protocol/private: `_disk`[`L456`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L456), `_layers`[`L461`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L461), `_memory`[`L455`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L455)
- uses (calls/refs, reference-scoped): [`CacheMetadata`](cache.md#CacheMetadata), [`stats`](cache.md#MemoryCache.stats), [`compression`](cache.md#DiskCache.compression), [`DiskCache`](cache.md#DiskCache), [`MemoryCache`](cache.md#MemoryCache)
- used by: [`cache_manager`](../core/protocols.md#PipelineContext.cache_manager)

## Module values
- `DatasetLike` — [`L40`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L40)
- `logger` — [`L43`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/cache.py#L43)


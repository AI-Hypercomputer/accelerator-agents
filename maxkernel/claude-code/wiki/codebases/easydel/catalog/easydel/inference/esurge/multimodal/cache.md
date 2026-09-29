---
title: 'Module: easydel/inference/esurge/multimodal/cache.py'
type: catalog
provenance: extracted
module: easydel/inference/esurge/multimodal/cache.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.esurge.multimodal.cache`/VisionEncoderCache#
symbols:
  VisionEncoderCache.get_stats: get_stats().
  VisionEncoderCache.clear: clear().
  VisionEncoderCache._cache: _cache.
  VisionEncoderCache.get: get().
  VisionEncoderCache.put: put().
  VisionEncoderCache._lock: _lock.
  VisionEncoderCache.hit_rate: hit_rate().
  VisionEncoderCache.current_size: current_size.
  VisionEncoderCache.hits: hits.
  VisionEncoderCache.contains: contains().
  VisionEncoderCache.__len__: __len__().
  VisionEncoderCache: ''
  VisionEncoderCache.misses: misses.
  VisionEncoderCache.size_mb: size_mb().
  VisionEncoderCache.capacity_bytes: capacity_bytes.
  VisionEncoderCache.__init__: __init__().
  VisionEncoderCache.compute_hash: compute_hash().
---
# Module: [`easydel/inference/esurge/multimodal/cache.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/cache.py)

## Classes
### `VisionEncoderCache`
- def: [`easydel/inference/esurge/multimodal/cache.py:78`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/cache.py#L78)
- doc: Thread-safe LRU cache for vision encoder outputs.
- signature: `class VisionEncoderCache:`
- members:
  - `__init__(self, capacity_mb: int = 1024)` — [`L118`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/cache.py#L118) — Initialize VisionEncoderCache.
  - `__len__(self)` — [`L295`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/cache.py#L295) — Return number of cached entries.
  - `clear(self)` — [`L276`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/cache.py#L276) — Clear all cached entries and reset statistics.
  - `compute_hash(self, pixel_values: np.ndarray)` — [`L140`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/cache.py#L140) — Compute content hash for image pixel values.
  - `contains(self, hash_key: str)` — [`L253`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/cache.py#L253) — Check if hash key is in cache without updating LRU order.
  - `get(self, hash_key: str)` — [`L176`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/cache.py#L176) — Retrieve cached embeddings by hash key.
  - `get_stats(self)` — [`L340`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/cache.py#L340) — Return comprehensive cache statistics.
  - `hit_rate(self)` — [`L324`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/cache.py#L324) — Return cache hit rate (0.0 to 1.0).
  - `put(self, hash_key: str, embeddings: jax.Array)` — [`L209`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/cache.py#L209) — Cache embeddings with LRU eviction.
  - `size_mb(self)` — [`L310`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/cache.py#L310) — Return current cache size in megabytes.
  - `capacity_bytes` — [`L132`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/cache.py#L132)
  - `current_size` — [`L133`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/cache.py#L133)
  - `hits` — [`L137`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/cache.py#L137)
  - `misses` — [`L138`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/cache.py#L138)
- protocol/private: `_cache`[`L134`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/cache.py#L134), `_lock`[`L135`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/cache.py#L135)
- used by: [`cache`](manager.md#MultiModalManager.cache), [`clear_cache`](manager.md#MultiModalManager.clear_cache), [`get_cache_stats`](manager.md#MultiModalManager.get_cache_stats)


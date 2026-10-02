---
title: 'Module: tpu_inference/runner/kv_cache.py'
type: catalog
provenance: extracted
module: tpu_inference/runner/kv_cache.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.runner.kv_cache`/
symbols:
  create_kv_caches: create_kv_caches().
  get_kv_cache_shape_with_mesh: get_kv_cache_shape_with_mesh().
  get_attention_page_size_bytes: get_attention_page_size_bytes().
  KVCacheMetadata.count: KVCacheMetadata#count.
  KVCacheMetadata.shape: KVCacheMetadata#shape.
  DEFAULT_KV_CACHE_DTYPE: DEFAULT_KV_CACHE_DTYPE.
  KVCacheMetadata: KVCacheMetadata#
  logger: logger.
  KVCacheMetadata.dtype: KVCacheMetadata#dtype.
  KVCacheMetadata.sharding: KVCacheMetadata#sharding.
  create_kv_caches._allocate: create_kv_caches()._allocate().
---
# Module: [`tpu_inference/runner/kv_cache.py`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/kv_cache.py)

## Classes
### `KVCacheMetadata`
- def: [`tpu_inference/runner/kv_cache.py:38`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/kv_cache.py#L38)
- doc: Used to store metadata about the KV cache for logging in the KV cache manager.
- signature: `class KVCacheMetadata:`
- members:
  - `count` — [`L44`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/kv_cache.py#L44)
  - `dtype` — [`L46`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/kv_cache.py#L46)
  - `shape` — [`L45`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/kv_cache.py#L45)
  - `sharding` — [`L47`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/kv_cache.py#L47)
- used by: [`initialize_kv_cache`](kv_cache_manager.md#KVCacheManager.initialize_kv_cache)

## Functions
- `_allocate()` — [`L142`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/kv_cache.py#L142)
- `create_kv_caches(num_blocks: int, block_size: int, num_kv_heads: int, head_size: int, mesh: Mesh, layer_names: List[str], cache_dtype: jnp.dtype = DEFAULT_KV_CACHE_DTYPE, use_mla: bool = False)` — [`L92`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/kv_cache.py#L92) — Creates a list of KV cache where each array mapps to single attention layer.
- `get_attention_page_size_bytes(mesh, block_size, num_kv_heads, head_size, dtype, use_mla)` — [`L155`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/kv_cache.py#L155)
- `get_kv_cache_shape_with_mesh(mesh: Mesh, total_num_pages: int, block_size: int, actual_num_kv_heads: int, actual_head_dim: int, kv_dtype: any, use_mla: bool = False)` — [`L50`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/kv_cache.py#L50) — Gets the KV cache shape based on the mesh configuration.

## Module values
- `DEFAULT_KV_CACHE_DTYPE` — [`L34`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/kv_cache.py#L34)
- `logger` — [`L32`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/kv_cache.py#L32)


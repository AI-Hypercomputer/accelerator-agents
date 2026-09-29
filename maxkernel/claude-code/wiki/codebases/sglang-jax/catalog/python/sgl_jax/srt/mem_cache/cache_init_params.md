---
title: 'Module: python/sgl_jax/srt/mem_cache/cache_init_params.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/mem_cache/cache_init_params.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.mem_cache.cache_init_params`/CacheInitParams#
symbols:
  CacheInitParams: ''
  CacheInitParams.req_to_token_pool: req_to_token_pool.
  CacheInitParams.token_to_kv_pool_allocator: token_to_kv_pool_allocator.
  CacheInitParams.page_size: page_size.
  CacheInitParams.is_eagle: is_eagle.
  CacheInitParams.sliding_window_size: sliding_window_size.
---
# Module: [`python/sgl_jax/srt/mem_cache/cache_init_params.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/cache_init_params.py)

## Classes
### `CacheInitParams`
- def: [`python/sgl_jax/srt/mem_cache/cache_init_params.py:12`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/cache_init_params.py#L12)
- signature: `class CacheInitParams:`
- members:
  - `is_eagle` — [`L17`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/cache_init_params.py#L17)
  - `page_size` — [`L15`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/cache_init_params.py#L15)
  - `req_to_token_pool` — [`L13`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/cache_init_params.py#L13)
  - `sliding_window_size` — [`L19`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/cache_init_params.py#L19)
  - `token_to_kv_pool_allocator` — [`L14`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/cache_init_params.py#L14)
- uses (calls/refs, reference-scoped): [`ReqToTokenPool`](memory_pool.md#ReqToTokenPool), [`BaseTokenToKVPoolAllocator`](allocator.md#BaseTokenToKVPoolAllocator)
- used by: [`default_radix_cache_factory`](registry.md#default_radix_cache_factory), [`build_kv_cache`](kv_cache_builder.md#build_kv_cache), [`__init__`](unified_cache_components/recurrent_component.md#RecurrentComponent.__init__), [`__init__`](unified_cache_components/full_component.md#FullComponent.__init__), [`__init__`](unified_cache_components/tree_component.md#TreeComponent.__init__), [`params`](registry.md#TreeCacheBuildContext.params)


---
title: 'Module: python/sgl_jax/srt/mem_cache/chunk_cache.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/mem_cache/chunk_cache.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.mem_cache.chunk_cache`/
symbols:
  ChunkCache: ChunkCache#
  ChunkCache.cache_finished_req: ChunkCache#cache_finished_req().
  ChunkCache.match_prefix: ChunkCache#match_prefix().
  ChunkCache.cache_unfinished_req: ChunkCache#cache_unfinished_req().
  SWAChunkCache.__init__: SWAChunkCache#__init__().
  ChunkCache.__init__: ChunkCache#__init__().
  ChunkCache.evict: ChunkCache#evict().
  ChunkCache.inc_lock_ref: ChunkCache#inc_lock_ref().
  SWAChunkCache: SWAChunkCache#
  ChunkCache.dec_lock_ref: ChunkCache#dec_lock_ref().
  ChunkCache.req_to_token_pool: ChunkCache#req_to_token_pool.
  ChunkCache.token_to_kv_pool_allocator: ChunkCache#token_to_kv_pool_allocator.
  ChunkCache.page_size: ChunkCache#page_size.
  ChunkCache.reset: ChunkCache#reset().
  ChunkCache.pretty_print: ChunkCache#pretty_print().
  SWAChunkCache.sliding_window_size: SWAChunkCache#sliding_window_size.
  SWAChunkCache.supports_swa: SWAChunkCache#supports_swa().
  SWAChunkCache.full_evictable_size: SWAChunkCache#full_evictable_size().
  SWAChunkCache.swa_evictable_size: SWAChunkCache#swa_evictable_size().
  SWAChunkCache.full_protected_size: SWAChunkCache#full_protected_size().
  SWAChunkCache.swa_protected_size: SWAChunkCache#swa_protected_size().
---
# Module: [`python/sgl_jax/srt/mem_cache/chunk_cache.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/chunk_cache.py)

## Classes
### `ChunkCache`  ·  implements/extends BasePrefixCache
- def: [`python/sgl_jax/srt/mem_cache/chunk_cache.py:26`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/chunk_cache.py#L26)
- signature: `class ChunkCache(BasePrefixCache):`
- members:
  - `cache_finished_req(self, req: Req, is_insert: bool = True)` — [`L48`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/chunk_cache.py#L48)
  - `cache_unfinished_req(self, req: Req)` — [`L59`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/chunk_cache.py#L59)
  - `dec_lock_ref(self, node: Any, params: DecLockRefParams | None = None)` — [`L70`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/chunk_cache.py#L70)
  - `evict(self, params: EvictParams)` — [`L64`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/chunk_cache.py#L64)
  - `inc_lock_ref(self, node: Any)` — [`L67`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/chunk_cache.py#L67)
  - `match_prefix(self, params: MatchPrefixParams)` — [`L40`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/chunk_cache.py#L40)
  - `pretty_print(self)` — [`L73`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/chunk_cache.py#L73)
  - `reset(self)` — [`L37`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/chunk_cache.py#L37)
  - `page_size` — [`L35`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/chunk_cache.py#L35)
  - `req_to_token_pool` — [`L33`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/chunk_cache.py#L33)
  - `token_to_kv_pool_allocator` — [`L34`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/chunk_cache.py#L34)
- protocol/private: `__init__`[`L27`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/chunk_cache.py#L27)
- uses (calls/refs, reference-scoped): `Req`, `prefix_indices`, [`BasePrefixCache`](base_prefix_cache.md#BasePrefixCache), `dp_rank`, `fill_ids`, `req_pool_idx`, [`ReqToTokenPool`](memory_pool.md#ReqToTokenPool), [`BaseTokenToKVPoolAllocator`](allocator.md#BaseTokenToKVPoolAllocator), [`EvictParams`](base_prefix_cache.md#EvictParams), [`MatchPrefixParams`](base_prefix_cache.md#MatchPrefixParams), [`IncLockRefResult`](base_prefix_cache.md#IncLockRefResult), [`req_to_token`](memory_pool.md#ReqToTokenPool.req_to_token), [`free`](allocator.md#BaseTokenToKVPoolAllocator.free), [`MatchResult`](base_prefix_cache.md#MatchResult), [`DecLockRefParams`](base_prefix_cache.md#DecLockRefParams), [`EvictResult`](base_prefix_cache.md#EvictResult), [`last_device_node`](base_prefix_cache.md#MatchResult.last_device_node), [`device_indices`](base_prefix_cache.md#MatchResult.device_indices), [`last_host_node`](base_prefix_cache.md#MatchResult.last_host_node), [`best_match_node`](base_prefix_cache.md#MatchResult.best_match_node), `pop_committed_kv_cache`, [`delta`](base_prefix_cache.md#IncLockRefResult.delta), [`SWAChunkCache`](chunk_cache.md#SWAChunkCache)
- used by: [`get_internal_state`](../managers/scheduler.md#Scheduler.get_internal_state), [`default_radix_cache_factory`](registry.md#default_radix_cache_factory), `init_new`, `maybe_evict_swa`, [`BasePrefixCache`](base_prefix_cache.md#BasePrefixCache), `_evict_tree_cache_if_needed`, [`evict_from_tree_cache`](common.md#evict_from_tree_cache), [`evict`](base_prefix_cache.md#BasePrefixCache.evict), [`match_prefix`](base_prefix_cache.md#BasePrefixCache.match_prefix), [`inc_lock_ref`](base_prefix_cache.md#BasePrefixCache.inc_lock_ref), [`dec_lock_ref`](base_prefix_cache.md#BasePrefixCache.dec_lock_ref), [`cache_unfinished_req`](base_prefix_cache.md#BasePrefixCache.cache_unfinished_req), [`cache_finished_req`](base_prefix_cache.md#BasePrefixCache.cache_finished_req), [`pretty_print`](base_prefix_cache.md#BasePrefixCache.pretty_print), [`reset`](base_prefix_cache.md#BasePrefixCache.reset), [`__init__`](chunk_cache.md#SWAChunkCache.__init__), [`SWAChunkCache`](chunk_cache.md#SWAChunkCache)

### `SWAChunkCache`  ·  implements/extends ChunkCache
- def: [`python/sgl_jax/srt/mem_cache/chunk_cache.py:77`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/chunk_cache.py#L77)
- doc: ChunkCache with support for sliding window attention.
- signature: `class SWAChunkCache(ChunkCache):`
- members:
  - `full_evictable_size(self, dp_rank: int = 0)` — [`L96`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/chunk_cache.py#L96)
  - `full_protected_size(self, dp_rank: int = 0)` — [`L102`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/chunk_cache.py#L102)
  - `supports_swa(self)` — [`L93`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/chunk_cache.py#L93)
  - `swa_evictable_size(self, dp_rank: int = 0)` — [`L99`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/chunk_cache.py#L99)
  - `swa_protected_size(self, dp_rank: int = 0)` — [`L105`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/chunk_cache.py#L105)
  - `sliding_window_size` — [`L91`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/chunk_cache.py#L91)
- protocol/private: `__init__`[`L83`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/chunk_cache.py#L83)
- uses (calls/refs, reference-scoped): [`ReqToTokenPool`](memory_pool.md#ReqToTokenPool), [`ChunkCache`](chunk_cache.md#ChunkCache), [`SWATokenToKVPoolAllocator`](allocator.md#SWATokenToKVPoolAllocator), [`__init__`](chunk_cache.md#ChunkCache.__init__)
- used by: [`default_radix_cache_factory`](registry.md#default_radix_cache_factory), [`ChunkCache`](chunk_cache.md#ChunkCache), [`full_evictable_size`](base_prefix_cache.md#BasePrefixCache.full_evictable_size), [`swa_evictable_size`](base_prefix_cache.md#BasePrefixCache.swa_evictable_size), [`full_protected_size`](base_prefix_cache.md#BasePrefixCache.full_protected_size), [`swa_protected_size`](base_prefix_cache.md#BasePrefixCache.swa_protected_size)


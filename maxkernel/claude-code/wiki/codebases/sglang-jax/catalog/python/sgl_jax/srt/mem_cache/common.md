---
title: 'Module: python/sgl_jax/srt/mem_cache/common.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/mem_cache/common.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.mem_cache.common`/
symbols:
  evict_from_tree_cache: evict_from_tree_cache().
  alloc_token_slots: alloc_token_slots().
  release_kv_cache: release_kv_cache().
  alloc_paged_token_slots_extend: alloc_paged_token_slots_extend().
  available_and_evictable_str: available_and_evictable_str().
  logger: logger.
---
# Module: [`python/sgl_jax/srt/mem_cache/common.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/common.py)

## Functions
- `alloc_paged_token_slots_extend(tree_cache: BasePrefixCache, prefix_lens: list[int], seq_lens: list[int], last_loc: list[int], extend_num_tokens: int, backup_state: bool = False, dp_rank: int = 0)` — [`L41`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/common.py#L41)
- `alloc_token_slots(tree_cache: BasePrefixCache, num_tokens: int, backup_state: bool = False, dp_rank: int = 0)` — [`L13`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/common.py#L13)
- `available_and_evictable_str(tree_cache, dp_rank: int = 0)` — [`L103`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/common.py#L103)
- `evict_from_tree_cache(tree_cache: BasePrefixCache | None, num_tokens: int, dp_rank: int = 0)` — [`L72`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/common.py#L72)
- `release_kv_cache(req, tree_cache: BasePrefixCache, is_insert: bool = True, allow_overallocated: bool = False)` — [`L120`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/common.py#L120) — Single entry point for releasing a request's KV cache (sglang #12224).

## Module values
- `logger` — [`L8`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/common.py#L8)


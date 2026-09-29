---
title: 'Module: python/sgl_jax/srt/mem_cache/registry.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/mem_cache/registry.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.mem_cache.registry`/
symbols:
  default_radix_cache_factory: default_radix_cache_factory().
  create_tree_cache: create_tree_cache().
  TreeCacheBuildContext.model_config: TreeCacheBuildContext#model_config.
  TreeCacheBuildContext.server_args: TreeCacheBuildContext#server_args.
  TreeCacheBuildContext.disable_radix_cache: TreeCacheBuildContext#disable_radix_cache.
  TreeCacheBuildContext.params: TreeCacheBuildContext#params.
  TreeCacheBuildContext: TreeCacheBuildContext#
  TreeCacheBuildContext.is_hybrid_swa: TreeCacheBuildContext#is_hybrid_swa.
  TreeCacheBuildContext.tp_size: TreeCacheBuildContext#tp_size.
  TreeCacheBuildContext.is_hybrid_recurrent: TreeCacheBuildContext#is_hybrid_recurrent.
  TreeCacheBuildContext.effective_chunked_prefill_size: TreeCacheBuildContext#effective_chunked_prefill_size.
  logger: logger.
---
# Module: [`python/sgl_jax/srt/mem_cache/registry.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/registry.py)

## Classes
### `TreeCacheBuildContext`
- def: [`python/sgl_jax/srt/mem_cache/registry.py:18`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/registry.py#L18)
- signature: `class TreeCacheBuildContext:`
- members:
  - `disable_radix_cache` — [`L22`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/registry.py#L22)
  - `effective_chunked_prefill_size` — [`L23`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/registry.py#L23)
  - `is_hybrid_recurrent` — [`L26`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/registry.py#L26)
  - `is_hybrid_swa` — [`L21`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/registry.py#L21)
  - `model_config` — [`L24`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/registry.py#L24)
  - `params` — [`L20`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/registry.py#L20)
  - `server_args` — [`L19`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/registry.py#L19)
  - `tp_size` — [`L25`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/registry.py#L25)
- uses (calls/refs, reference-scoped): [`ServerArgs`](../server_args.md#ServerArgs), [`ModelConfig`](../configs/model_config.md#ModelConfig), [`CacheInitParams`](cache_init_params.md#CacheInitParams)
- used by: [`default_radix_cache_factory`](registry.md#default_radix_cache_factory), [`build_kv_cache`](kv_cache_builder.md#build_kv_cache), [`create_tree_cache`](registry.md#create_tree_cache)

## Functions
- `create_tree_cache(ctx: TreeCacheBuildContext)` — [`L120`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/registry.py#L120)
- `default_radix_cache_factory(ctx: TreeCacheBuildContext)` — [`L29`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/registry.py#L29) — documented in [python-sgl_jax-srt-mem_cache-base_prefix_cache](../../../../../concepts/python-sgl_jax-srt-mem_cache-base_prefix_cache.md)

## Module values
- `logger` — [`L14`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/registry.py#L14)


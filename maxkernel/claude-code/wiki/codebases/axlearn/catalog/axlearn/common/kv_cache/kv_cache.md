---
title: 'Module: axlearn/common/kv_cache/kv_cache.py'
type: catalog
provenance: extracted
module: axlearn/common/kv_cache/kv_cache.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.kv_cache.kv_cache`/
symbols:
  KVCache: KVCache#
  KVCache.init_states: KVCache#init_states().
  KVCache.extend_step: KVCache#extend_step().
  _INVALID_KV_POSITION: _INVALID_KV_POSITION.
  KVCache.update_single: KVCache#update_single().
---
# Module: [`axlearn/common/kv_cache/kv_cache.py`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/kv_cache.py)

## Classes
### `KVCache`  ·  implements/extends BaseKVCache
- def: [`axlearn/common/kv_cache/kv_cache.py:20`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/kv_cache.py#L20)
- doc: Default KV cache.
- signature: `class KVCache(BaseKVCache):`
- members:
  - `extend_step(self, cached_states: Nested[Tensor], *, k_proj: Tensor, v_proj: Tensor, key_positions: Tensor, segment_ids: Optional[Tensor] = None, page_pool: Optional[Nested[Tensor]] = None)` — [`L46`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/kv_cache.py#L46)
  - `init_states(self, shape: BaseKVCache.Shape, *, dtype: jnp.dtype)` — [`L29`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/kv_cache.py#L29)
  - `update_single(cached_kv_slice, kv_proj_slice, time_idx)` — [`L102`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/kv_cache.py#L102)
- uses (calls/refs, reference-scoped): [`Tensor`](../utils.md#Tensor), [`Nested`](../utils.md#Nested), [`config`](../config.md#Configurable.config), [`BaseKVCache`](base_kv_cache.md#BaseKVCache), [`nowrap`](../module.md#nowrap), [`maybe_shard`](../utils.md#maybe_shard), [`PagedKVCache`](paged_kv_cache.md#PagedKVCache), [`extend_step`](paged_kv_cache.md#PagedKVCache.extend_step), [`init_states`](paged_kv_cache.md#PagedKVCache.init_states), [`Output`](base_kv_cache.md#BaseKVCache.Output), [`_cache_dtype`](base_kv_cache.md#BaseKVCache._cache_dtype), [`Shape`](base_kv_cache.md#BaseKVCache.Shape), [`batch_size`](base_kv_cache.md#BaseKVCache.Shape.batch_size), [`num_kv_heads`](base_kv_cache.md#BaseKVCache.Shape.num_kv_heads), [`per_head_dim`](base_kv_cache.md#BaseKVCache.Shape.per_head_dim), [`kv_len`](base_kv_cache.md#BaseKVCache.Shape.kv_len), [`_INVALID_KV_POSITION`](kv_cache.md#_INVALID_KV_POSITION)
- used by: [`_mask_tpl`](../rattention/rattention.md#RAttention._mask_tpl), `_test_extend_step`, `test_extend_step`, [`__call__`](../flash_attention/gpu_attention.md#CuDNNGPUFlashAttention.__call__), [`BaseKVCache`](base_kv_cache.md#BaseKVCache), [`_prepare_layers`](../flash_attention/layer_test.md#_prepare_layers), `test_gqa_extend_step`, [`_test_extend_step`](../flash_attention/layer_test.md#TestFlashAttention._test_extend_step), `_test_prefill_states`, [`is_supported`](../flash_attention/common.md#BaseSingleStepDecoding.is_supported), [`_kv_cache_type`](../flash_attention/gpu_attention.md#CuDNNGPUFlashAttention._kv_cache_type), [`init_states`](../attention.md#MultiheadAttention.init_states), [`is_supported`](../flash_attention/tpu_attention.md#TPUIncrementalSplashAttention.is_supported), [`bench_flash_attention`](../flash_attention/gpu_attention_benchmark.md#bench_flash_attention), [`init_states`](../rattention/rattention.md#RAttention.init_states), [`is_supported`](../flash_attention/tpu_decoding.md#TPUDecoding.is_supported), [`kv_cache`](../attention.md#MultiheadAttention.Config.kv_cache), [`is_supported`](../flash_attention/gpu_decoding.md#GPUDecoding.is_supported), [`PagedKVCache`](paged_kv_cache.md#PagedKVCache), [`init_states`](base_kv_cache.md#BaseKVCache.init_states), [`extend_step`](base_kv_cache.md#BaseKVCache.extend_step), [`extend_step`](paged_kv_cache.md#PagedKVCache.extend_step), [`init_states`](paged_kv_cache.md#PagedKVCache.init_states)

## Module values
- `_INVALID_KV_POSITION` — [`L17`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/kv_cache.py#L17)


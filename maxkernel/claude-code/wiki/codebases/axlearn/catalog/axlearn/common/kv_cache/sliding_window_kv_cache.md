---
title: 'Module: axlearn/common/kv_cache/sliding_window_kv_cache.py'
type: catalog
provenance: extracted
module: axlearn/common/kv_cache/sliding_window_kv_cache.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.kv_cache.sliding_window_kv_cache`/
symbols:
  SlidingWindowKVCache.init_states: SlidingWindowKVCache#init_states().
  enable_sliding_window_attention: enable_sliding_window_attention().
  SlidingWindowKVCache.extend_step: SlidingWindowKVCache#extend_step().
  SlidingWindowKVCache.Config: SlidingWindowKVCache#Config#
  SlidingWindowKVCache: SlidingWindowKVCache#
  SlidingWindowKVCache.Config.cached_kv_length: SlidingWindowKVCache#Config#cached_kv_length.
  SlidingWindowKVCache._invaild_position: SlidingWindowKVCache#_invaild_position().
  SlidingWindowKVCache.prepare_proj: SlidingWindowKVCache#prepare_proj().
  SlidingWindowKVCache.update_cache: SlidingWindowKVCache#update_cache().
---
# Module: [`axlearn/common/kv_cache/sliding_window_kv_cache.py`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/sliding_window_kv_cache.py)

## Classes
### `Config`  ·  implements/extends Config
- def: [`axlearn/common/kv_cache/sliding_window_kv_cache.py:36`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/sliding_window_kv_cache.py#L36)
- doc: Configures SlidingWindowKVCache.
- signature: `class Config(BaseKVCache.Config):`
- members:
  - `cached_kv_length` — [`L40`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/sliding_window_kv_cache.py#L40)
- uses (calls/refs, reference-scoped): [`config_class`](../config.md#config_class), [`REQUIRED`](../config.md#REQUIRED), [`Required`](../config.md#Required), [`BaseKVCache`](base_kv_cache.md#BaseKVCache), [`Config`](base_kv_cache.md#BaseKVCache.Config)

### `SlidingWindowKVCache`  ·  implements/extends BaseKVCache
- def: [`axlearn/common/kv_cache/sliding_window_kv_cache.py:19`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/sliding_window_kv_cache.py#L19)
- doc: KV cache for sliding window attention.
- signature: `class SlidingWindowKVCache(BaseKVCache):`
- members:
  - `extend_step(self, cached_states: Nested[Tensor], *, k_proj: Tensor, v_proj: Tensor, key_positions: Tensor, segment_ids: Optional[Tensor] = None, page_pool: Optional[Nested[Tensor]] = None)` — [`L71`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/sliding_window_kv_cache.py#L71) — Updates the sliding window KV cache per extend step.
  - `init_states(self, shape: BaseKVCache.Shape, *, dtype: jnp.dtype)` — [`L47`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/sliding_window_kv_cache.py#L47)
  - `prepare_proj(k_proj, v_proj, key_positions)` — [`L146`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/sliding_window_kv_cache.py#L146)
  - `update_cache(k_proj, v_proj, key_positions)` — [`L117`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/sliding_window_kv_cache.py#L117)
- protocol/private: `_invaild_position`[`L42`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/sliding_window_kv_cache.py#L42)
- uses (calls/refs, reference-scoped): [`Tensor`](../utils.md#Tensor), [`Nested`](../utils.md#Nested), [`config`](../config.md#Configurable.config), [`BaseKVCache`](base_kv_cache.md#BaseKVCache), [`nowrap`](../module.md#nowrap), [`maybe_shard`](../utils.md#maybe_shard), [`Output`](base_kv_cache.md#BaseKVCache.Output), [`_cache_dtype`](base_kv_cache.md#BaseKVCache._cache_dtype), [`Shape`](base_kv_cache.md#BaseKVCache.Shape), [`batch_size`](base_kv_cache.md#BaseKVCache.Shape.batch_size), [`num_kv_heads`](base_kv_cache.md#BaseKVCache.Shape.num_kv_heads), [`per_head_dim`](base_kv_cache.md#BaseKVCache.Shape.per_head_dim)
- used by: [`BaseKVCache`](base_kv_cache.md#BaseKVCache), [`_test_extend_step`](../flash_attention/layer_test.md#TestFlashAttention._test_extend_step), [`enable_sliding_window_attention`](sliding_window_kv_cache.md#enable_sliding_window_attention), [`init_states`](base_kv_cache.md#BaseKVCache.init_states), [`extend_step`](base_kv_cache.md#BaseKVCache.extend_step)

## Functions
- `enable_sliding_window_attention(cfg: MultiheadAttention.Config, sliding_window_size: int)` — [`L159`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/sliding_window_kv_cache.py#L159) — Enable sliding window attention.


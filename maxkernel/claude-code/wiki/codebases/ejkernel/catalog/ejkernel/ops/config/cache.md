---
title: 'Module: ejkernel/ops/config/cache.py'
type: catalog
provenance: extracted
module: ejkernel/ops/config/cache.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.ops.config.cache`/
symbols:
  ConfigCache: ConfigCache#
  ConfigCache.put: ConfigCache#put().
  overlay_cache._ejk_user_context: overlay_cache#_ejk_user_context.
  overlay_cache.__enter__: overlay_cache#__enter__().
  overlay_cache.__exit__: overlay_cache#__exit__().
  ConfigCache._data: ConfigCache#_data.
  _cache_overlay._cache_overlay: _cache_overlay._cache_overlay.
  ConfigCache.get: ConfigCache#get().
  Cfg: Cfg.
  overlay_cache: overlay_cache#
  ConfigCache.clear: ConfigCache#clear().
  ConfigCache.size: ConfigCache#size().
  ConfigCache.keys: ConfigCache#keys().
  overlay_cache.token: overlay_cache#token.
  overlay_cache.mapping: overlay_cache#mapping.
  ConfigCache.__init__: ConfigCache#__init__().
  overlay_cache.__init__: overlay_cache#__init__().
---
# Module: [`ejkernel/ops/config/cache.py`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/cache.py)

## Classes
### `ConfigCache`  ·  implements/extends Generic
- def: [`ejkernel/ops/config/cache.py:44`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/cache.py#L44)
- doc: Thread-safe cache for storing kernel configurations.
- signature: `class ConfigCache(Generic[Cfg]):`
- members:
  - `__init__(self)` — [`L63`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/cache.py#L63) — Initialize an empty configuration cache.
  - `clear(self)` — [`L91`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/cache.py#L91) — Clear all cached configurations.
  - `get(self, dev: str, op_id: str, call_key: str)` — [`L67`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/cache.py#L67) — Retrieve a cached configuration.
  - `keys(self)` — [`L103`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/cache.py#L103) — Get all cache keys.
  - `put(self, dev: str, op_id: str, call_key: str, cfg: Cfg)` — [`L80`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/cache.py#L80) — Store a configuration in the cache.
  - `size(self)` — [`L95`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/cache.py#L95) — Get the number of cached configurations.
- protocol/private: `_data`[`L65`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/cache.py#L65)
- uses (calls/refs, reference-scoped): [`Cfg`](cache.md#Cfg)
- used by: [`choose`](selection.md#ConfigSelectorChain.choose), [`autotune_lowered`](../execution/offline.md#autotune_lowered), [`_quantized_matmul_executor`](../../modules/operations/quantized_matmul.md#_quantized_matmul_executor._quantized_matmul_executor), [`_executor`](../../modules/operations/ragged_gated_delta_rule_v2.md#_executor._executor), [`_executor`](../../modules/operations/ragged_causal_conv1d.md#_executor._executor), [`_all_gather_matmul_executor`](../../modules/operations/all_gather_matmul.md#_all_gather_matmul_executor._all_gather_matmul_executor), [`_chunked_prefill_paged_decode_executor`](../../modules/operations/chunked_prefill_paged_decode.md#_chunked_prefill_paged_decode_executor._chunked_prefill_paged_decode_executor), [`_decode_attention_executor`](../../modules/operations/decode_attention.md#_decode_attention_executor._decode_attention_executor), [`_dsa_executor`](../../modules/operations/deepseek_attn.md#_dsa_executor._dsa_executor), [`_executor`](../../modules/operations/attention.md#_executor._executor), [`_executor`](../../modules/operations/blocksparse_attention.md#_executor._executor), [`_executor`](../../modules/operations/fused_conv_decode.md#_executor._executor), [`_executor`](../../modules/operations/fused_cross_entropy.md#_executor._executor), [`_executor`](../../modules/operations/fused_kl_divergence.md#_executor._executor), [`_executor`](../../modules/operations/gated_delta_rule_grouped_decode.md#_executor._executor), [`_executor`](../../modules/operations/gated_delta_rule.md#_executor._executor), [`_executor`](../../modules/operations/gdn_compute_schedule_v2.md#_executor._executor), [`_executor`](../../modules/operations/kernel_delta_attention.md#_executor._executor), [`_executor`](../../modules/operations/ragged_gated_delta_rule.md#_executor._executor), [`_executor`](../../modules/operations/rwkv4.md#_executor._executor), [`_executor`](../../modules/operations/rwkv6.md#_executor._executor), [`_executor`](../../modules/operations/scaled_dot_product_attention.md#_executor._executor), [`_executor_rwkv7`](../../modules/operations/rwkv7.md#_executor_rwkv7._executor_rwkv7), [`_executor_rwkv7_mul`](../../modules/operations/rwkv7.md#_executor_rwkv7_mul._executor_rwkv7_mul), [`_flash_executor`](../../modules/operations/flash_attention.md#_flash_executor._flash_executor), [`_gla_executor`](../../modules/operations/gated_linear_attention.md#_gla_executor._gla_executor), [`_grouped_matmul_executor`](../../modules/operations/grouped_matmul.md#_grouped_matmul_executor._grouped_matmul_executor), [`_lightning_executor`](../../modules/operations/lightning_attention.md#_lightning_executor._lightning_executor), [`_mean_pooling_executor`](../../modules/operations/pooling.md#_mean_pooling_executor._mean_pooling_executor), [`_mla_executor`](../../modules/operations/multi_head_latent_attention.md#_mla_executor._mla_executor), [`_mlrpa_executor`](../../modules/operations/multi_latent_ragged_page_attention.md#_mlrpa_executor._mlrpa_executor), [`_mlrpa_v2_executor`](../../modules/operations/multi_latent_ragged_page_attention_v2.md#_mlrpa_v2_executor._mlrpa_v2_executor), [`_page_attention_executor`](../../modules/operations/page_attention.md#_page_attention_executor._page_attention_executor), [`_prefill_page_attention_executor`](../../modules/operations/prefill_page_attention.md#_prefill_page_attention_executor._prefill_page_attention_executor), [`_ragged_decode_attention_executor`](../../modules/operations/ragged_decode_attention.md#_ragged_decode_attention_executor._ragged_decode_attention_executor), [`_ragged_page_attention_executor`](../../modules/operations/ragged_page_attention_v2.md#_ragged_page_attention_executor._ragged_page_attention_executor), [`_ragged_page_attention_executor`](../../modules/operations/ragged_page_attention_v3.md#_ragged_page_attention_executor._ragged_page_attention_executor), [`_ragged_page_attention_v2_tq_executor`](../../modules/operations/ragged_page_attention_v2_turboquant.md#_ragged_page_attention_v2_tq_executor._ragged_page_attention_v2_tq_executor), [`_ragged_page_attention_v3_tq_executor`](../../modules/operations/ragged_page_attention_v3_turboquant.md#_ragged_page_attention_v3_tq_executor._ragged_page_attention_v3_tq_executor), [`_recurrent_executor`](../../modules/operations/recurrent.md#_recurrent_executor._recurrent_executor)  (+8 more)

### `overlay_cache`
- def: [`ejkernel/ops/config/cache.py:118`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/cache.py#L118)
- doc: Context manager for temporarily overriding configuration cache entries.
- signature: `class overlay_cache:`
- members:
  - `__enter__(self)` — [`L159`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/cache.py#L159) — Activate the overlay by pushing the mapping onto the context-variable stack.
  - `__exit__(self, *exc)` — [`L184`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/cache.py#L184) — Deactivate the overlay by restoring the context-variable stack token.
  - `__init__(self, mapping: dict[tuple[str, str, str], Any])` — [`L150`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/cache.py#L150) — Initialize cache overlay with mapping.
  - `mapping` — [`L156`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/cache.py#L156)
  - `token` — [`L157`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/cache.py#L157)
- protocol/private: `_ejk_user_context`[`L176`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/cache.py#L176)
- uses (calls/refs, reference-scoped): [`_cache_overlay`](cache.md#_cache_overlay._cache_overlay)
- used by: [`_ctx`](../execution/tuning.md#AutotuningResult._ctx), [`__exit__`](../execution/tuning.md#AutotuningResult.__exit__)

## Module values
- `Cfg` — [`L41`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/cache.py#L41)
- `_cache_overlay` — [`L112`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/cache.py#L112)


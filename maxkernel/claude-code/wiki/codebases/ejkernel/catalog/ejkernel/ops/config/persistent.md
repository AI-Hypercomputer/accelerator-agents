---
title: 'Module: ejkernel/ops/config/persistent.py'
type: catalog
provenance: extracted
module: ejkernel/ops/config/persistent.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.ops.config.persistent`/
symbols:
  PersistentCache: PersistentCache#
  PersistentCache.put: PersistentCache#put().
  PersistentCache.get: PersistentCache#get().
  PersistentCache._data: PersistentCache#_data.
  PersistentCache._disabled: PersistentCache#_disabled.
  Cfg: Cfg.
  PersistentCache.cfg_type: PersistentCache#cfg_type.
  PersistentCache.path: PersistentCache#path.
  PersistentCache.__init__: PersistentCache#__init__().
  PersistentCache.loader: PersistentCache#loader.
  PersistentCache.dumper: PersistentCache#dumper.
  PersistentCache._key: PersistentCache#_key().
---
# Module: [`ejkernel/ops/config/persistent.py`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/persistent.py)

## Classes
### `PersistentCache`  ·  implements/extends Generic
- def: [`ejkernel/ops/config/persistent.py:61`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/persistent.py#L61)
- doc: Disk-backed JSON cache for kernel configurations.
- signature: `class PersistentCache(Generic[Cfg]):`
- members:
  - `__init__(self, opname: str, path: str | None = None, loader: Callable[[Any], Cfg] | None = None, dumper: Callable[[Cfg], Any] | None = None, cfg_type: type[Cfg] | None = None)` — [`L98`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/persistent.py#L98) — Initialize persistent cache with file path and optional serializers.
  - `_key(self, device: str, op_id: str, call_key: str)` — [`L167`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/persistent.py#L167) — Generate internal storage key from cache coordinates.
  - `get(self, device: str, op_id: str, call_key: str)` — [`L180`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/persistent.py#L180) — Retrieve cached configuration for the given coordinates.
  - `put(self, device: str, op_id: str, call_key: str, cfg: Cfg)` — [`L206`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/persistent.py#L206) — Store configuration in the cache with atomic file update.
  - `cfg_type` — [`L157`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/persistent.py#L157)
  - `dumper` — [`L156`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/persistent.py#L156)
  - `loader` — [`L155`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/persistent.py#L155)
  - `path` — [`L154`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/persistent.py#L154)
- protocol/private: `_data`[`L159`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/persistent.py#L159), `_disabled`[`L126`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/persistent.py#L126)
- uses (calls/refs, reference-scoped): [`Cfg`](persistent.md#Cfg)
- used by: [`choose`](selection.md#ConfigSelectorChain.choose), [`autotune_lowered`](../execution/offline.md#autotune_lowered), [`_quantized_matmul_executor`](../../modules/operations/quantized_matmul.md#_quantized_matmul_executor._quantized_matmul_executor), [`_executor`](../../modules/operations/ragged_gated_delta_rule_v2.md#_executor._executor), [`_executor`](../../modules/operations/ragged_causal_conv1d.md#_executor._executor), [`_all_gather_matmul_executor`](../../modules/operations/all_gather_matmul.md#_all_gather_matmul_executor._all_gather_matmul_executor), [`_chunked_prefill_paged_decode_executor`](../../modules/operations/chunked_prefill_paged_decode.md#_chunked_prefill_paged_decode_executor._chunked_prefill_paged_decode_executor), [`_decode_attention_executor`](../../modules/operations/decode_attention.md#_decode_attention_executor._decode_attention_executor), [`_dsa_executor`](../../modules/operations/deepseek_attn.md#_dsa_executor._dsa_executor), [`_executor`](../../modules/operations/attention.md#_executor._executor), [`_executor`](../../modules/operations/blocksparse_attention.md#_executor._executor), [`_executor`](../../modules/operations/fused_conv_decode.md#_executor._executor), [`_executor`](../../modules/operations/fused_cross_entropy.md#_executor._executor), [`_executor`](../../modules/operations/fused_kl_divergence.md#_executor._executor), [`_executor`](../../modules/operations/gated_delta_rule_grouped_decode.md#_executor._executor), [`_executor`](../../modules/operations/gated_delta_rule.md#_executor._executor), [`_executor`](../../modules/operations/gdn_compute_schedule_v2.md#_executor._executor), [`_executor`](../../modules/operations/kernel_delta_attention.md#_executor._executor), [`_executor`](../../modules/operations/ragged_gated_delta_rule.md#_executor._executor), [`_executor`](../../modules/operations/rwkv4.md#_executor._executor), [`_executor`](../../modules/operations/rwkv6.md#_executor._executor), [`_executor`](../../modules/operations/scaled_dot_product_attention.md#_executor._executor), [`_executor_rwkv7`](../../modules/operations/rwkv7.md#_executor_rwkv7._executor_rwkv7), [`_executor_rwkv7_mul`](../../modules/operations/rwkv7.md#_executor_rwkv7_mul._executor_rwkv7_mul), [`_flash_executor`](../../modules/operations/flash_attention.md#_flash_executor._flash_executor), [`_gla_executor`](../../modules/operations/gated_linear_attention.md#_gla_executor._gla_executor), [`_grouped_matmul_executor`](../../modules/operations/grouped_matmul.md#_grouped_matmul_executor._grouped_matmul_executor), [`_lightning_executor`](../../modules/operations/lightning_attention.md#_lightning_executor._lightning_executor), [`_mean_pooling_executor`](../../modules/operations/pooling.md#_mean_pooling_executor._mean_pooling_executor), [`_mla_executor`](../../modules/operations/multi_head_latent_attention.md#_mla_executor._mla_executor), [`_mlrpa_executor`](../../modules/operations/multi_latent_ragged_page_attention.md#_mlrpa_executor._mlrpa_executor), [`_mlrpa_v2_executor`](../../modules/operations/multi_latent_ragged_page_attention_v2.md#_mlrpa_v2_executor._mlrpa_v2_executor), [`_page_attention_executor`](../../modules/operations/page_attention.md#_page_attention_executor._page_attention_executor), [`_prefill_page_attention_executor`](../../modules/operations/prefill_page_attention.md#_prefill_page_attention_executor._prefill_page_attention_executor), [`_ragged_decode_attention_executor`](../../modules/operations/ragged_decode_attention.md#_ragged_decode_attention_executor._ragged_decode_attention_executor), [`_ragged_page_attention_executor`](../../modules/operations/ragged_page_attention_v2.md#_ragged_page_attention_executor._ragged_page_attention_executor), [`_ragged_page_attention_executor`](../../modules/operations/ragged_page_attention_v3.md#_ragged_page_attention_executor._ragged_page_attention_executor), [`_ragged_page_attention_v2_tq_executor`](../../modules/operations/ragged_page_attention_v2_turboquant.md#_ragged_page_attention_v2_tq_executor._ragged_page_attention_v2_tq_executor), [`_ragged_page_attention_v3_tq_executor`](../../modules/operations/ragged_page_attention_v3_turboquant.md#_ragged_page_attention_v3_tq_executor._ragged_page_attention_v3_tq_executor), [`_recurrent_executor`](../../modules/operations/recurrent.md#_recurrent_executor._recurrent_executor)  (+8 more)

## Module values
- `Cfg` — [`L58`](../../../../../../../raw/code/ejkernel/ejkernel/ops/config/persistent.py#L58)


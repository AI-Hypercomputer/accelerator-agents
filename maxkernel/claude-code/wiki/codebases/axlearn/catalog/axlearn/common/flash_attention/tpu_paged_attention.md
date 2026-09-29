---
title: 'Module: axlearn/common/flash_attention/tpu_paged_attention.py'
type: catalog
provenance: extracted
module: axlearn/common/flash_attention/tpu_paged_attention.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.flash_attention.tpu_paged_attention`/
symbols:
  TPUPagedAttention.__call__: TPUPagedAttention#__call__().
  TPUPagedAttention.is_supported: TPUPagedAttention#is_supported().
  TPUPagedAttention.megacore_mode_heuristic: TPUPagedAttention#megacore_mode_heuristic().
  TPUPagedAttention.sparse_mode_heuristic: TPUPagedAttention#sparse_mode_heuristic().
  TPUPagedAttention.Config: TPUPagedAttention#Config#
  TPUPagedAttention: TPUPagedAttention#
  _get_tpu_cores_per_chip: _get_tpu_cores_per_chip().
  TPUPagedAttention.Config.megacore_mode: TPUPagedAttention#Config#megacore_mode.
---
# Module: [`axlearn/common/flash_attention/tpu_paged_attention.py`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention.py)

## Classes
### `Config`  ·  implements/extends Config
- def: [`axlearn/common/flash_attention/tpu_paged_attention.py:75`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention.py#L75)
- doc: Configures TPUPagedAttention.
- signature: `class Config(BasePagedAttention.Config):`
- members:
  - `megacore_mode` — [`L78`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention.py#L78)
- uses (calls/refs, reference-scoped): [`config_class`](../config.md#config_class), [`BasePagedAttention`](common.md#BasePagedAttention), [`Config`](common.md#BasePagedAttention.Config)

### `TPUPagedAttention`  ·  implements/extends BasePagedAttention
- def: [`axlearn/common/flash_attention/tpu_paged_attention.py:71`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention.py#L71)
- doc: Wraps TPU paged flash attention kernel.
- signature: `class TPUPagedAttention(BasePagedAttention):`
- members:
  - `__call__(self, input_batch: Nested[Tensor | BaseAttentionBias])` — [`L142`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention.py#L142) — See `BasePagedAttention.__call__`.
  - `is_supported(self, input_batch: Nested[Tensor | BaseAttentionBias], kv_cache_type: Optional[type[BaseKVCache]])` — [`L121`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention.py#L121) — See `BasePagedAttention.is_supported`.
  - `megacore_mode_heuristic(self, input_batch: Nested[Tensor | BaseAttentionBias])` — [`L80`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention.py#L80) — Simple heuristic to enable megacore parallelism on TPUs with 2 cores.
  - `sparse_mode_heuristic(self, mask: BaseAttentionBias, max_length: int)` — [`L105`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention.py#L105) — Simple heuristic of whether to use block-sparse kernel.
- uses (calls/refs, reference-scoped): [`Tensor`](../utils.md#Tensor), [`Nested`](../utils.md#Nested), [`BaseAttentionBias`](../attention_bias.md#BaseAttentionBias), [`MaskFnAttentionBias`](../attention_bias.md#MaskFnAttentionBias), [`_log_unsupported`](common.md#BaseFlashAttention._log_unsupported), [`BaseKVCache`](../kv_cache/base_kv_cache.md#BaseKVCache), [`cfg`](common.md#BaseFlashAttention.cfg), [`split`](../attention_bias.md#split), [`is_supported`](common.md#BasePagedAttention.is_supported), [`SlidingWindowAttentionBias`](../attention_bias.md#SlidingWindowAttentionBias), [`value`](../attention_bias.md#BaseAttentionBias.value), [`prepare_block_sparse_map`](tpu_paged_attention_kernel.md#prepare_block_sparse_map), [`_check_block_size`](common.md#BasePagedAttention._check_block_size), [`BasePagedAttention`](common.md#BasePagedAttention), [`_make_index_map`](tpu_paged_attention_kernel.md#_make_index_map), [`softmax_scale`](common.md#BaseFlashAttention.Config.softmax_scale), [`interpret`](common.md#BaseFlashAttention.Config.interpret), [`_make_index_map_for_logit_sink`](tpu_paged_attention_kernel.md#_make_index_map_for_logit_sink), [`tpu_block_size`](common.md#BaseFlashAttention.Config.tpu_block_size), [`_paged_flash_attention_kernel`](tpu_paged_attention_kernel.md#_paged_flash_attention_kernel), [`_paged_flash_attention_sparse_kernel`](tpu_paged_attention_kernel.md#_paged_flash_attention_sparse_kernel), [`sliding_window_size`](../attention_bias.md#SlidingWindowAttentionBias.sliding_window_size), [`_get_tpu_cores_per_chip`](tpu_paged_attention.md#_get_tpu_cores_per_chip)
- used by: [`flash_attention_implementation`](utils.md#flash_attention_implementation), [`is_supported`](common.md#BasePagedAttention.is_supported), [`BasePagedAttention`](common.md#BasePagedAttention), [`PAGED_ATTN_BACKENDS`](utils.md#PAGED_ATTN_BACKENDS)

## Functions
- `_get_tpu_cores_per_chip(interpret: bool = False)` — [`L45`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention.py#L45) — Return number of physical cores per TPU if available.


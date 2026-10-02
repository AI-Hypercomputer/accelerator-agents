---
title: 'Module: axlearn/common/flash_attention/tpu_decoding.py'
type: catalog
provenance: extracted
module: axlearn/common/flash_attention/tpu_decoding.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.flash_attention.tpu_decoding`/
symbols:
  TPUDecoding.__call__: TPUDecoding#__call__().
  TPUDecoding.is_supported: TPUDecoding#is_supported().
  _tpu_decoding_kernel: _tpu_decoding_kernel().
  TPUDecoding: TPUDecoding#
  _tpu_decoding_kernel.init: _tpu_decoding_kernel().init().
  _tpu_decoding_kernel.compute: _tpu_decoding_kernel().compute().
  TPUDecoding.kv_index_map: TPUDecoding#kv_index_map().
  TPUDecoding.bias_index_map: TPUDecoding#bias_index_map().
  _tpu_decoding_kernel.final: _tpu_decoding_kernel().final().
---
# Module: [`axlearn/common/flash_attention/tpu_decoding.py`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_decoding.py)

## Classes
### `TPUDecoding`  ·  implements/extends BaseSingleStepDecoding
- def: [`axlearn/common/flash_attention/tpu_decoding.py:147`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_decoding.py#L147)
- doc: Wraps the TPU decoding kernel.
- signature: `class TPUDecoding(BaseSingleStepDecoding):`
- members:
  - `__call__(self, input_batch: Nested[Tensor | BaseAttentionBias])` — [`L170`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_decoding.py#L170) — See `BaseFlashAttention.__call__`.
  - `bias_index_map(batch_idx, head_idx, kv_block_idx, kv_seq_len, kv_block_offset, kv_block_offset_size)` — [`L240`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_decoding.py#L240)
  - `is_supported(self, input_batch: Nested[Tensor | BaseAttentionBias], kv_cache_type: Optional[type[BaseKVCache]])` — [`L150`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_decoding.py#L150) — See `BaseFlashAttention.is_supported`.
  - `kv_index_map(batch_idx, head_idx, kv_block_idx, kv_seq_len, kv_block_offset, kv_block_offset_size)` — [`L226`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_decoding.py#L226)
- uses (calls/refs, reference-scoped): [`Tensor`](../utils.md#Tensor), [`Nested`](../utils.md#Nested), [`BaseAttentionBias`](../attention_bias.md#BaseAttentionBias), [`MaskFnAttentionBias`](../attention_bias.md#MaskFnAttentionBias), [`_log_unsupported`](common.md#BaseFlashAttention._log_unsupported), [`BaseKVCache`](../kv_cache/base_kv_cache.md#BaseKVCache), [`cfg`](common.md#BaseFlashAttention.cfg), [`KVCache`](../kv_cache/kv_cache.md#KVCache), [`is_supported`](common.md#BaseSingleStepDecoding.is_supported), [`split`](../attention_bias.md#split), [`value`](../attention_bias.md#BaseAttentionBias.value), [`prepare_block_sparse_map`](tpu_paged_attention_kernel.md#prepare_block_sparse_map), [`BaseSingleStepDecoding`](common.md#BaseSingleStepDecoding), [`softmax_scale`](common.md#BaseFlashAttention.Config.softmax_scale), [`interpret`](common.md#BaseFlashAttention.Config.interpret), [`_tpu_decoding_kernel`](tpu_decoding.md#_tpu_decoding_kernel), [`tpu_block_size`](common.md#BaseFlashAttention.Config.tpu_block_size)
- used by: [`is_supported`](common.md#BaseSingleStepDecoding.is_supported), [`BACKENDS`](utils.md#BACKENDS), [`BaseSingleStepDecoding`](common.md#BaseSingleStepDecoding)

## Functions
- `_tpu_decoding_kernel(kv_seq_len_ref, kv_block_offset, kv_block_offset_size, q_ref, k_ref, v_ref, b_ref, logit_sink_ref, o_ref, m_i, l_i, o_scratch, softmax_scale: float, mask_fn: Optional[MaskFn])` — [`L51`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_decoding.py#L51)
- `compute()` — [`L99`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_decoding.py#L99)
- `final()` — [`L140`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_decoding.py#L140)
- `init()` — [`L80`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_decoding.py#L80)


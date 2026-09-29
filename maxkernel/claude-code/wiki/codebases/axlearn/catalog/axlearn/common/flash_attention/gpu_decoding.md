---
title: 'Module: axlearn/common/flash_attention/gpu_decoding.py'
type: catalog
provenance: extracted
module: axlearn/common/flash_attention/gpu_decoding.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.flash_attention.gpu_decoding`/
symbols:
  GPUDecoding.__call__: GPUDecoding#__call__().
  GPUDecoding.is_supported: GPUDecoding#is_supported().
  _attn_forward_kernel: _attn_forward_kernel().
  _decode_attn_unbatched: _decode_attn_unbatched().
  GPUDecoding: GPUDecoding#
  _attn_forward_kernel._compute.body: _attn_forward_kernel()._compute().body().
  _attn_forward_kernel._compute: _attn_forward_kernel()._compute().
  _attn_forward_kernel._compute.body.compute: _attn_forward_kernel()._compute().body().compute().
  _get_sm_count: _get_sm_count().
  _attn_forward_kernel._compute.body.no_compute: _attn_forward_kernel()._compute().body().no_compute().
---
# Module: [`axlearn/common/flash_attention/gpu_decoding.py`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_decoding.py)

## Classes
### `GPUDecoding`  ·  implements/extends BaseSingleStepDecoding
- def: [`axlearn/common/flash_attention/gpu_decoding.py:286`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_decoding.py#L286)
- doc: Implements GPU FlashDecoding with GQA support.
- signature: `class GPUDecoding(BaseSingleStepDecoding):`
- members:
  - `__call__(self, input_batch: Nested[Tensor | BaseAttentionBias])` — [`L304`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_decoding.py#L304) — See `BaseFlashAttention.__call__`.
  - `is_supported(self, input_batch: Nested[Tensor | BaseAttentionBias], kv_cache_type: Optional[type[BaseKVCache]])` — [`L289`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_decoding.py#L289) — See `BaseSingleStepDecoding.is_supported`.
- uses (calls/refs, reference-scoped): [`Tensor`](../utils.md#Tensor), [`Nested`](../utils.md#Nested), [`BaseAttentionBias`](../attention_bias.md#BaseAttentionBias), [`MaskFnAttentionBias`](../attention_bias.md#MaskFnAttentionBias), [`_log_unsupported`](common.md#BaseFlashAttention._log_unsupported), [`BaseKVCache`](../kv_cache/base_kv_cache.md#BaseKVCache), [`cfg`](common.md#BaseFlashAttention.cfg), [`KVCache`](../kv_cache/kv_cache.md#KVCache), [`is_supported`](common.md#BaseSingleStepDecoding.is_supported), [`split`](../attention_bias.md#split), [`value`](../attention_bias.md#BaseAttentionBias.value), [`BaseSingleStepDecoding`](common.md#BaseSingleStepDecoding), [`_decode_attn_unbatched`](gpu_decoding.md#_decode_attn_unbatched), [`gpu_block_size`](common.md#BaseFlashAttention.Config.gpu_block_size), [`softmax_scale`](common.md#BaseFlashAttention.Config.softmax_scale), [`interpret`](common.md#BaseFlashAttention.Config.interpret)
- used by: [`is_supported`](common.md#BaseSingleStepDecoding.is_supported), [`bench_flash_attention`](gpu_attention_benchmark.md#bench_flash_attention), [`BACKENDS`](utils.md#BACKENDS), [`BaseSingleStepDecoding`](common.md#BaseSingleStepDecoding)

## Functions
- `_attn_forward_kernel(q_ref, k_ref, v_ref, bias_ref, kv_seq_len_ref, o_ref, l_ref, m_ref, mask_fn: Optional[MaskFn], softmax_scale: float, block_k: int, block_h: int, qhead_per_kvhead: int)` — [`L71`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_decoding.py#L71)
- `_compute(block_kv_start_idx, block_kv_seqlen, o, m_i, l_i)` — [`L95`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_decoding.py#L95)
- `_decode_attn_unbatched(q, k, v, bias, kv_seq_len, softmax_scale: float, mask_fn: Optional[MaskFn], block_h: int, block_k: int, num_warps: int, num_stages: int, interpret: bool, debug: bool, batch_size: int)` — [`L192`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_decoding.py#L192)
- `_get_sm_count()` — [`L176`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_decoding.py#L176) — Returns number of SMs for the current GPU or 0 if unknown.
- `body(start_k, carry)` — [`L104`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_decoding.py#L104)
- `compute()` — [`L112`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_decoding.py#L112)
- `no_compute()` — [`L140`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_decoding.py#L140)


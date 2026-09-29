---
title: 'Module: axlearn/common/flash_attention/gpu_paged_attention.py'
type: catalog
provenance: extracted
module: axlearn/common/flash_attention/gpu_paged_attention.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.flash_attention.gpu_paged_attention`/
symbols:
  GPUPagedAttention.__call__: GPUPagedAttention#__call__().
  GPUPagedAttention.is_supported: GPUPagedAttention#is_supported().
  _paged_attention_unbatched: _paged_attention_unbatched().
  _paged_attention_kernel: _paged_attention_kernel().
  GPUPagedAttention: GPUPagedAttention#
  _paged_attention_kernel._compute: _paged_attention_kernel()._compute().
  _paged_attention_kernel._compute.body: _paged_attention_kernel()._compute().body().
  _largest_divisor_leq: _largest_divisor_leq().
---
# Module: [`axlearn/common/flash_attention/gpu_paged_attention.py`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_paged_attention.py)

## Classes
### `GPUPagedAttention`  ·  implements/extends BasePagedAttention
- def: [`axlearn/common/flash_attention/gpu_paged_attention.py:299`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_paged_attention.py#L299)
- doc: Implements GPU PagedAttention .
- signature: `class GPUPagedAttention(BasePagedAttention):`
- members:
  - `__call__(self, input_batch: Nested[Tensor | BaseAttentionBias])` — [`L325`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_paged_attention.py#L325) — See `BasePagedAttention.__call__`.
  - `is_supported(self, input_batch: Nested[Tensor | BaseAttentionBias], kv_cache_type: Optional[type[BaseKVCache]])` — [`L302`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_paged_attention.py#L302) — See `BasePagedAttention.is_supported`.
- uses (calls/refs, reference-scoped): [`Tensor`](../utils.md#Tensor), [`Nested`](../utils.md#Nested), [`NEG_INF`](../attention_bias.md#NEG_INF), [`BaseAttentionBias`](../attention_bias.md#BaseAttentionBias), [`MaskFnAttentionBias`](../attention_bias.md#MaskFnAttentionBias), [`_log_unsupported`](common.md#BaseFlashAttention._log_unsupported), [`BaseKVCache`](../kv_cache/base_kv_cache.md#BaseKVCache), [`cfg`](common.md#BaseFlashAttention.cfg), [`split`](../attention_bias.md#split), [`is_supported`](common.md#BasePagedAttention.is_supported), [`value`](../attention_bias.md#BaseAttentionBias.value), [`_check_block_size`](common.md#BasePagedAttention._check_block_size), [`BasePagedAttention`](common.md#BasePagedAttention), [`_paged_attention_unbatched`](gpu_paged_attention.md#_paged_attention_unbatched), [`gpu_block_size`](common.md#BaseFlashAttention.Config.gpu_block_size), [`softmax_scale`](common.md#BaseFlashAttention.Config.softmax_scale), [`interpret`](common.md#BaseFlashAttention.Config.interpret)
- used by: [`is_supported`](common.md#BasePagedAttention.is_supported), [`BasePagedAttention`](common.md#BasePagedAttention), [`PAGED_ATTN_BACKENDS`](utils.md#PAGED_ATTN_BACKENDS)

## Functions
- `_compute(start_page_idx, end_page_idx, o, m_i, l_i)` — [`L66`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_paged_attention.py#L66) — Computes attention for a range of pages.
- `_largest_divisor_leq(x: int, y: int)` — [`L155`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_paged_attention.py#L155) — Find the largest divisor of x <= y.
- `_paged_attention_kernel(q_ref, key_ref, value_ref, page_tables_ref, bias_ref, lengths_ref, o_ref, *residual_refs, pages_per_compute_block: int, mask_value: float, softmax_scale: float, mask_fn: Optional[MaskFn])` — [`L38`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_paged_attention.py#L38) — Computes attention outputs for the given block.
- `_paged_attention_unbatched(q, key, value, page_tables, bias, lengths, *, block_h: int, pages_per_compute_block: int, num_warps: int, num_stages: int, interpret: bool, debug: bool, mask_value: float, softmax_scale: float, batch_size: int, mask_fn: Optional[MaskFn])` — [`L164`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_paged_attention.py#L164) — Partition unbatched input and feed into the compute kernel.
- `body(start_k, carry)` — [`L88`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_paged_attention.py#L88)


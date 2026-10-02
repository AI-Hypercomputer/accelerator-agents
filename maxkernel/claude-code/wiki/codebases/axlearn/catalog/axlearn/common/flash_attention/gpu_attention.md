---
title: 'Module: axlearn/common/flash_attention/gpu_attention.py'
type: catalog
provenance: extracted
module: axlearn/common/flash_attention/gpu_attention.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.flash_attention.gpu_attention`/
symbols:
  CuDNNGPUFlashAttention.__call__: CuDNNGPUFlashAttention#__call__().
  PallasGPUFlashAttention.__call__: PallasGPUFlashAttention#__call__().
  CuDNNGPUFlashAttention._kv_cache_type: CuDNNGPUFlashAttention#_kv_cache_type.
  PallasGPUFlashAttention.is_supported: PallasGPUFlashAttention#is_supported().
  _mha_backward: _mha_backward().
  _flash_attention_impl: _flash_attention_impl().
  CuDNNGPUFlashAttention.is_supported: CuDNNGPUFlashAttention#is_supported().
  _mha_forward_kernel: _mha_forward_kernel().
  flash_attention: flash_attention().
  _mha_backward_kernel_dkdv: _mha_backward_kernel_dkdv().
  _mha_backward_kernel_dq: _mha_backward_kernel_dq().
  CuDNNGPUFlashAttention: CuDNNGPUFlashAttention#
  _mha_forward_kernel.body: _mha_forward_kernel().body().
  _mha_backward_kernel_dkdv.inner_loop_dkdv: _mha_backward_kernel_dkdv().inner_loop_dkdv().
  _mha_backward_kernel_dq.inner_loop_dq: _mha_backward_kernel_dq().inner_loop_dq().
  PallasGPUFlashAttention: PallasGPUFlashAttention#
  CuDNNGPUFlashAttentionWithExplicitBias: CuDNNGPUFlashAttentionWithExplicitBias#
  _segment_mask: _segment_mask().
  _key_value_iterator_indices: _key_value_iterator_indices().
  _mha_forward: _mha_forward().
  _mha_backward.call_kernel: _mha_backward().call_kernel().
  CuDNNGPUFlashAttention._allow_explicit_bias: CuDNNGPUFlashAttention#_allow_explicit_bias.
  CuDNNGPUFlashAttentionWithExplicitBias._allow_explicit_bias: CuDNNGPUFlashAttentionWithExplicitBias#_allow_explicit_bias.
---
# Module: [`axlearn/common/flash_attention/gpu_attention.py`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention.py)

## Classes
### `CuDNNGPUFlashAttention`  ·  implements/extends BaseFlashAttention
- def: [`axlearn/common/flash_attention/gpu_attention.py:792`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention.py#L792)
- doc: Wraps cuDNN FlashAttention and disallows explicit bias.
- signature: `class CuDNNGPUFlashAttention(BaseFlashAttention):`
- members:
  - `__call__(self, input_batch: Nested[Tensor | BaseAttentionBias])` — [`L871`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention.py#L871) — See `BaseFlashAttention.__call__`. — documented in [axlearn-common-flash_attention-common](../../../../concepts/axlearn-common-flash_attention-common.md)
  - `is_supported(self, input_batch: Nested[Tensor | BaseAttentionBias], kv_cache_type: Optional[type[BaseKVCache]])` — [`L805`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention.py#L805) — See `BaseFlashAttention.is_supported`.
- protocol/private: `_allow_explicit_bias`[`L803`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention.py#L803), `_kv_cache_type`[`L814`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention.py#L814)
- uses (calls/refs, reference-scoped): [`Tensor`](../utils.md#Tensor), [`Nested`](../utils.md#Nested), [`BaseAttentionBias`](../attention_bias.md#BaseAttentionBias), [`MaskFnAttentionBias`](../attention_bias.md#MaskFnAttentionBias), [`_log_unsupported`](common.md#BaseFlashAttention._log_unsupported), [`BaseKVCache`](../kv_cache/base_kv_cache.md#BaseKVCache), [`cfg`](common.md#BaseFlashAttention.cfg), [`CausalAttentionBias`](../attention_bias.md#CausalAttentionBias), [`KVCache`](../kv_cache/kv_cache.md#KVCache), [`is_supported`](common.md#BaseFlashAttention.is_supported), [`split`](../attention_bias.md#split), [`BaseFlashAttention`](common.md#BaseFlashAttention), [`SlidingWindowAttentionBias`](../attention_bias.md#SlidingWindowAttentionBias), [`value`](../attention_bias.md#BaseAttentionBias.value), [`has_value`](../attention_bias.md#BaseAttentionBias.has_value), [`causal_mask`](../attention_bias.md#causal_mask), [`_check_block_size`](common.md#BaseFlashAttention._check_block_size), [`repeat_kv_heads`](common.md#repeat_kv_heads), [`dropout_rate`](common.md#BaseFlashAttention.Config.dropout_rate), [`softmax_scale`](common.md#BaseFlashAttention.Config.softmax_scale), [`CuDNNGPUFlashAttentionWithExplicitBias`](gpu_attention.md#CuDNNGPUFlashAttentionWithExplicitBias), [`astype`](../attention_bias.md#BaseAttentionBias.astype)
- used by: [`is_supported`](common.md#BaseFlashAttention.is_supported), [`BaseFlashAttention`](common.md#BaseFlashAttention), [`BACKENDS`](utils.md#BACKENDS), [`CuDNNGPUFlashAttentionWithExplicitBias`](gpu_attention.md#CuDNNGPUFlashAttentionWithExplicitBias)

### `CuDNNGPUFlashAttentionWithExplicitBias`  ·  implements/extends CuDNNGPUFlashAttention
- def: [`axlearn/common/flash_attention/gpu_attention.py:995`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention.py#L995)
- doc: Wraps cuDNN FlashAttention and allows explicit bias.
- signature: `class CuDNNGPUFlashAttentionWithExplicitBias(CuDNNGPUFlashAttention):`
- protocol/private: `_allow_explicit_bias`[`L1002`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention.py#L1002)
- uses (calls/refs, reference-scoped): [`CuDNNGPUFlashAttention`](gpu_attention.md#CuDNNGPUFlashAttention)
- used by: [`bench_flash_attention`](gpu_attention_benchmark.md#bench_flash_attention), [`BACKENDS`](utils.md#BACKENDS), [`CuDNNGPUFlashAttention`](gpu_attention.md#CuDNNGPUFlashAttention)

### `PallasGPUFlashAttention`  ·  implements/extends BaseFlashAttention
- def: [`axlearn/common/flash_attention/gpu_attention.py:929`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention.py#L929)
- doc: Wraps Pallas implementation of GPU FlashAttention.
- signature: `class PallasGPUFlashAttention(BaseFlashAttention):`
- members:
  - `__call__(self, input_batch: Nested[Tensor | BaseAttentionBias])` — [`L964`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention.py#L964) — See `BaseFlashAttention.__call__`.
  - `is_supported(self, input_batch: Nested[Tensor | BaseAttentionBias], kv_cache_type: Optional[type[BaseKVCache]])` — [`L939`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention.py#L939) — See `BaseFlashAttention.is_supported`.
- uses (calls/refs, reference-scoped): [`Tensor`](../utils.md#Tensor), [`Nested`](../utils.md#Nested), [`BaseAttentionBias`](../attention_bias.md#BaseAttentionBias), [`MaskFnAttentionBias`](../attention_bias.md#MaskFnAttentionBias), [`_log_unsupported`](common.md#BaseFlashAttention._log_unsupported), [`BaseKVCache`](../kv_cache/base_kv_cache.md#BaseKVCache), [`cfg`](common.md#BaseFlashAttention.cfg), [`is_supported`](common.md#BaseFlashAttention.is_supported), [`split`](../attention_bias.md#split), [`BaseFlashAttention`](common.md#BaseFlashAttention), [`value`](../attention_bias.md#BaseAttentionBias.value), [`has_value`](../attention_bias.md#BaseAttentionBias.has_value), [`SegmentIdAttentionBias`](../attention_bias.md#SegmentIdAttentionBias), [`_check_block_size`](common.md#BaseFlashAttention._check_block_size), [`get_segment_ids`](common.md#get_segment_ids), [`repeat_kv_heads`](common.md#repeat_kv_heads), [`dropout_rate`](common.md#BaseFlashAttention.Config.dropout_rate), [`flash_attention`](gpu_attention.md#flash_attention), [`gpu_block_size`](common.md#BaseFlashAttention.Config.gpu_block_size), [`softmax_scale`](common.md#BaseFlashAttention.Config.softmax_scale), [`interpret`](common.md#BaseFlashAttention.Config.interpret)
- used by: [`is_supported`](common.md#BaseFlashAttention.is_supported), [`bench_flash_attention`](gpu_attention_benchmark.md#bench_flash_attention), [`BaseFlashAttention`](common.md#BaseFlashAttention), [`BACKENDS`](utils.md#BACKENDS)

## Functions
- `_flash_attention_impl(query: Tensor, key: Tensor, value: Tensor, bias: Optional[Tensor] = None, segment_ids: Optional[Tensor] = None, prng_key: Optional[Tensor] = None, softmax_scale: float = 1, mask_fn: Optional[MaskFn] = None, dropout_rate: float = 0, block_q: int = 128, block_k: int = 128, num_warps: Optional[int] = None, num_stages: Optional[int] = None, grid: Optional[Sequence[int]] = None, interpret: bool = False, debug: bool = False, output_activations: bool = False)` — [`L293`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention.py#L293) — Computes flash forward and residuals if output_activations is True.
- `_key_value_iterator_indices(block_mask_map: np.ndarray)` — [`L91`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention.py#L91) — build the iteration begin/end indices for the key/value dimension.
- `_mha_backward(softmax_scale: float, mask_fn: Optional[MaskFn], dropout_rate: float, block_q: int, block_k: int, num_warps: Optional[int], num_stages: int, grid: Any, interpret: bool, debug: bool, res, do)` — [`L621`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention.py#L621) — Calls Pallas kernels to compute dQ, dK and dV.
- `_mha_backward_kernel_dkdv(q_ref, k_ref, v_ref, b_ref, s_ref, dropout_mask_ref, do_scaled_ref, lse_ref, delta_ref, index_offset_ref, index_offset_size_ref, dk_ref, dv_ref, *, softmax_scale: float, mask_fn: Optional[MaskFn], dropout_rate: float, block_q: int, block_k: int)` — [`L436`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention.py#L436) — Computes dK and dV.
- `_mha_backward_kernel_dq(q_ref, k_ref, v_ref, b_ref, s_ref, dropout_mask_ref, do_scaled_ref, lse_ref, delta_ref, index_offset_ref, index_offset_size_ref, dq_ref, *, softmax_scale: float, mask_fn: Optional[MaskFn], dropout_rate: float, block_q: int, block_k: int)` — [`L533`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention.py#L533) — Computes dQ.
- `_mha_forward(*args: Any)` — [`L428`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention.py#L428) — Wraps flash_attention for custom vjp.
- `_mha_forward_kernel(q_ref, k_ref, v_ref, b_ref, s_ref, dropout_mask_ref, index_offset_ref, index_offset_size_ref, o_ref, *residual_refs, softmax_scale: float, mask_fn: Optional[MaskFn], dropout_rate: float, block_q: int, block_k: int, head_dim: int)` — [`L114`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention.py#L114) — Computes attention outputs for the given block.
- `_segment_mask(q_segment_ids: Tensor, kv_segment_ids: Tensor)` — [`L75`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention.py#L75) — Build the segment mask for the given query and key bias ids.
- `body(start_k, carry)` — [`L186`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention.py#L186)
- `call_kernel(*, kernel, grid, out_shape, in_specs, out_specs, index_offset, index_offset_size)` — [`L730`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention.py#L730)
- `flash_attention(query: Tensor, key: Tensor, value: Tensor, bias: Optional[Tensor] = None, segment_ids: Optional[Tensor] = None, prng_key: Optional[Tensor] = None, softmax_scale: float = 1, mask_fn: Optional[MaskFn] = None, dropout_rate: float = 0, block_q: int = 128, block_k: int = 128, num_warps: Optional[int] = None, num_stages: Optional[int] = None, grid: Optional[Sequence[int]] = None, interpret: bool = False, debug: bool = False)` — [`L250`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention.py#L250) — Computes attention outputs following FlashAttention.
- `inner_loop_dkdv(start_q, carry)` — [`L479`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention.py#L479)
- `inner_loop_dq(start_k, carry)` — [`L575`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention.py#L575)


---
title: 'Module: axlearn/common/flash_attention/neuron_attention.py'
type: catalog
provenance: extracted
module: axlearn/common/flash_attention/neuron_attention.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.flash_attention.neuron_attention`/
symbols:
  NeuronFlashAttention.__call__: NeuronFlashAttention#__call__().
  NeuronFlashAttention.is_supported: NeuronFlashAttention#is_supported().
  Tensor: Tensor.
  flash_attention: flash_attention().
  _mha_forward: _mha_forward().
  lnc: lnc.
  _mha_backward: _mha_backward().
  NeuronFlashAttention: NeuronFlashAttention#
---
# Module: [`axlearn/common/flash_attention/neuron_attention.py`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/neuron_attention.py)

## Classes
### `NeuronFlashAttention`  ·  implements/extends BaseFlashAttention
- def: [`axlearn/common/flash_attention/neuron_attention.py:225`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/neuron_attention.py#L225)
- doc: Wraps the Neuron attention kernel.
- signature: `class NeuronFlashAttention(BaseFlashAttention):`
- members:
  - `__call__(self, input_batch: Nested[Tensor | BaseAttentionBias])` — [`L243`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/neuron_attention.py#L243) — See `BaseFlashAttention.__call__`.
  - `is_supported(self, input_batch: Nested[Tensor | BaseAttentionBias], kv_cache_type: Optional[type[BaseKVCache]])` — [`L228`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/neuron_attention.py#L228) — See `BaseFlashAttention.is_supported`.
- uses (calls/refs, reference-scoped): [`Nested`](../utils.md#Nested), [`BaseAttentionBias`](../attention_bias.md#BaseAttentionBias), [`_log_unsupported`](common.md#BaseFlashAttention._log_unsupported), [`BaseKVCache`](../kv_cache/base_kv_cache.md#BaseKVCache), [`cfg`](common.md#BaseFlashAttention.cfg), [`CausalAttentionBias`](../attention_bias.md#CausalAttentionBias), [`is_supported`](common.md#BaseFlashAttention.is_supported), [`split`](../attention_bias.md#split), [`BaseFlashAttention`](common.md#BaseFlashAttention), [`value`](../attention_bias.md#BaseAttentionBias.value), [`has_value`](../attention_bias.md#BaseAttentionBias.has_value), [`Tensor`](neuron_attention.md#Tensor), [`repeat_kv_heads`](common.md#repeat_kv_heads), [`dropout_rate`](common.md#BaseFlashAttention.Config.dropout_rate), [`softmax_scale`](common.md#BaseFlashAttention.Config.softmax_scale), [`flash_attention`](neuron_attention.md#flash_attention)
- used by: [`flash_attention_implementation`](utils.md#flash_attention_implementation), [`is_supported`](common.md#BaseFlashAttention.is_supported), [`BaseFlashAttention`](common.md#BaseFlashAttention)

## Functions
- `_mha_backward(causal: bool, softmax_scale: float, dropout_rate: float, res, d_attn_output: Tensor)` — [`L154`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/neuron_attention.py#L154)
- `_mha_forward(query: Tensor, key: Tensor, value: Tensor, bias: Tensor, prng_key: Tensor, causal: bool, softmax_scale: float, dropout_rate: float)` — [`L60`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/neuron_attention.py#L60) — Computes attention outputs following FlashAttention.
- `flash_attention(query: Tensor, key: Tensor, value: Tensor, bias: Optional[Tensor] = None, prng_key: Optional[Tensor] = None, causal: bool = False, softmax_scale: float = 1, dropout_rate: float = 0)` — [`L31`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/neuron_attention.py#L31) — Wraps _mha_forward for custom vjp.

## Module values
- `Tensor` — [`L25`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/neuron_attention.py#L25)
- `lnc` — [`L26`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/neuron_attention.py#L26)


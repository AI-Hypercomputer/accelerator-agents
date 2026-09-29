---
title: 'Module: python/sgl_jax/srt/layers/radix_linear_attention.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/layers/radix_linear_attention.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.layers.radix_linear_attention`/RadixLinearAttention#
symbols:
  RadixLinearAttention.scale: scale.
  RadixLinearAttention: ''
  RadixLinearAttention.activation: activation.
  RadixLinearAttention.A_log: A_log.
  RadixLinearAttention.dt_bias: dt_bias.
  RadixLinearAttention.__call__: __call__().
  RadixLinearAttention.layer_id: layer_id.
  RadixLinearAttention.num_q_heads: num_q_heads.
  RadixLinearAttention.__init__: __init__().
  RadixLinearAttention.conv1d: conv1d.
  RadixLinearAttention.q_conv1d: q_conv1d.
  RadixLinearAttention.k_conv1d: k_conv1d.
  RadixLinearAttention.v_conv1d: v_conv1d.
  RadixLinearAttention.head_q_dim: head_q_dim.
  RadixLinearAttention.num_k_heads: num_k_heads.
  RadixLinearAttention.head_k_dim: head_k_dim.
  RadixLinearAttention.num_v_heads: num_v_heads.
  RadixLinearAttention.head_v_dim: head_v_dim.
  RadixLinearAttention.bias: bias.
---
# Module: [`python/sgl_jax/srt/layers/radix_linear_attention.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/radix_linear_attention.py)

## Classes
### `RadixLinearAttention`  ·  implements/extends Module
- def: [`python/sgl_jax/srt/layers/radix_linear_attention.py:15`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/radix_linear_attention.py#L15)
- signature: `class RadixLinearAttention(nnx.Module):`
- members:
  - `A_log` — [`L56`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/radix_linear_attention.py#L56)
  - `activation` — [`L55`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/radix_linear_attention.py#L55)
  - `bias` — [`L54`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/radix_linear_attention.py#L54)
  - `conv1d` — [`L53`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/radix_linear_attention.py#L53)
  - `dt_bias` — [`L57`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/radix_linear_attention.py#L57)
  - `head_k_dim` — [`L41`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/radix_linear_attention.py#L41)
  - `head_q_dim` — [`L40`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/radix_linear_attention.py#L40)
  - `head_v_dim` — [`L42`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/radix_linear_attention.py#L42)
  - `k_conv1d` — [`L49`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/radix_linear_attention.py#L49)
  - `layer_id` — [`L36`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/radix_linear_attention.py#L36)
  - `num_k_heads` — [`L38`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/radix_linear_attention.py#L38)
  - `num_q_heads` — [`L37`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/radix_linear_attention.py#L37)
  - `num_v_heads` — [`L39`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/radix_linear_attention.py#L39)
  - `q_conv1d` — [`L48`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/radix_linear_attention.py#L48)
  - `scale` — [`L58`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/radix_linear_attention.py#L58)
  - `v_conv1d` — [`L50`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/radix_linear_attention.py#L50)
- protocol/private: `__call__`[`L66`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/radix_linear_attention.py#L66), `__init__`[`L16`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/radix_linear_attention.py#L16)
- uses (calls/refs, reference-scoped): [`ForwardBatch`](../model_executor/forward_batch_info.md#ForwardBatch), [`LinearBase`](linear.md#LinearBase), [`attn_backend`](../model_executor/forward_batch_info.md#ForwardBatch.attn_backend)
- used by: [`__call__`](attention/linear/kda_backend.md#KDAAttnBackend.__call__), [`__call__`](attention/linear/gdn_backend.md#GDNAttnBackend.__call__), [`_forward_extend`](attention/linear/kda_backend.md#KDAAttnBackend._forward_extend), [`_forward_decode`](attention/linear/kda_backend.md#KDAAttnBackend._forward_decode), [`_fused_kda_gate`](attention/linear/kda_backend.md#KDAAttnBackend._fused_kda_gate)


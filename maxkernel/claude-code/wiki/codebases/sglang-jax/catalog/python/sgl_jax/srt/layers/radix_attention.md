---
title: 'Module: python/sgl_jax/srt/layers/radix_attention.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/layers/radix_attention.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.layers.radix_attention`/
symbols:
  RadixAttention: RadixAttention#
  RadixAttention.__call__: RadixAttention#__call__().
  RadixAttention.scaling: RadixAttention#scaling.
  RadixAttention.sliding_window_size: RadixAttention#sliding_window_size.
  RadixAttention.__init__: RadixAttention#__init__().
  RadixAttention.layer_id: RadixAttention#layer_id.
  AttentionType: AttentionType#
  RadixAttention.head_dim: RadixAttention#head_dim.
  RadixAttention.logit_cap: RadixAttention#logit_cap.
  RadixAttention.xai_temperature_len: RadixAttention#xai_temperature_len.
  RadixAttention.softmax_dtype: RadixAttention#softmax_dtype.
  RadixAttention.attn_type: RadixAttention#attn_type.
  RadixAttention.q_head_num: RadixAttention#q_head_num.
  RadixAttention.kv_head_num: RadixAttention#kv_head_num.
  AttentionType.DECODER: AttentionType#DECODER.
  AttentionType.ENCODER_ONLY: AttentionType#ENCODER_ONLY.
  RadixAttention.qk_head_dim: RadixAttention#qk_head_dim.
  RadixAttention.v_head_dim: RadixAttention#v_head_dim.
---
# Module: [`python/sgl_jax/srt/layers/radix_attention.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/radix_attention.py)

## Classes
### `AttentionType`  ·  implements/extends Enum
- def: [`python/sgl_jax/srt/layers/radix_attention.py:13`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/radix_attention.py#L13)
- doc: Attention type.
- signature: `class AttentionType(Enum):`
- members:
  - `DECODER` — [`L19`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/radix_attention.py#L19)
  - `ENCODER_ONLY` — [`L21`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/radix_attention.py#L21)
- used by: [`__call__`](attention/native_backend.md#NativeAttention.__call__), [`__init__`](radix_attention.md#RadixAttention.__init__)

### `RadixAttention`  ·  implements/extends Module
- def: [`python/sgl_jax/srt/layers/radix_attention.py:24`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/radix_attention.py#L24)
- doc: The attention layer implementation.
- signature: `class RadixAttention(nnx.Module):`
- members:
  - `attn_type` — [`L52`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/radix_attention.py#L52)
  - `head_dim` — [`L45`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/radix_attention.py#L45)
  - `kv_head_num` — [`L44`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/radix_attention.py#L44)
  - `layer_id` — [`L49`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/radix_attention.py#L49)
  - `logit_cap` — [`L51`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/radix_attention.py#L51)
  - `q_head_num` — [`L43`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/radix_attention.py#L43)
  - `qk_head_dim` — [`L46`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/radix_attention.py#L46)
  - `scaling` — [`L48`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/radix_attention.py#L48)
  - `sliding_window_size` — [`L50`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/radix_attention.py#L50)
  - `softmax_dtype` — [`L60`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/radix_attention.py#L60)
  - `v_head_dim` — [`L47`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/radix_attention.py#L47)
  - `xai_temperature_len` — [`L53`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/radix_attention.py#L53)
- protocol/private: `__call__`[`L62`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/radix_attention.py#L62), `__init__`[`L29`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/radix_attention.py#L29)
- uses (calls/refs, reference-scoped): [`ForwardBatch`](../model_executor/forward_batch_info.md#ForwardBatch), [`KVCache`](../mem_cache/memory_pool.md#KVCache), [`attn_backend`](../model_executor/forward_batch_info.md#ForwardBatch.attn_backend), [`AttentionType`](radix_attention.md#AttentionType), [`DECODER`](radix_attention.md#AttentionType.DECODER)
- used by: [`__call__`](attention/flashattention_backend.md#FlashAttention.__call__), [`__call__`](attention/native_backend.md#NativeAttention.__call__), [`__call__`](attention/mla_backend.md#MLAAttentionBackend.__call__), [`_ragged_paged_attention_with_fused_kv`](attention/flashattention_backend.md#FlashAttention._ragged_paged_attention_with_fused_kv), [`__call__`](attention/base_attn_backend.md#AttentionBackend.__call__)


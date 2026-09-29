---
title: 'Module: python/sgl_jax/srt/layers/attention/linear/kda_backend.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/layers/attention/linear/kda_backend.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.layers.attention.linear.kda_backend`/
symbols:
  KDAAttnBackend.__call__: KDAAttnBackend#__call__().
  KDAAttnBackend._forward_extend: KDAAttnBackend#_forward_extend().
  KDAAttnBackend._forward_decode: KDAAttnBackend#_forward_decode().
  KDAAttnBackend.get_state: KDAAttnBackend#get_state().
  KDAAttnBackend.set_ssm_state: KDAAttnBackend#set_ssm_state().
  KDAAttnBackend.set_conv_state: KDAAttnBackend#set_conv_state().
  KDAAttnBackend._call: KDAAttnBackend#_call().
  KDAAttnBackend._fused_kda_gate: KDAAttnBackend#_fused_kda_gate().
  KDAAttnBackend._short_conv_extend: KDAAttnBackend#_short_conv_extend().
  KDAAttnBackend: KDAAttnBackend#
  KDAAttnBackend._chunk_kda_call: KDAAttnBackend#_chunk_kda_call().
  KDAAttnBackend._decode_kernel: KDAAttnBackend#_decode_kernel().
  KDAAttnBackend.__init__: KDAAttnBackend#__init__().
  l2_normalize: l2_normalize().
  KDAAttnBackend._scatter: KDAAttnBackend#_scatter().
  KDAAttnBackend._unpack_conv_states: KDAAttnBackend#_unpack_conv_states().
  KDAAttnBackend._pack_conv_states: KDAAttnBackend#_pack_conv_states().
  __all__: __all__.
---
# Module: [`python/sgl_jax/srt/layers/attention/linear/kda_backend.py`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/kda_backend.py)

## Classes
### `KDAAttnBackend`  ·  implements/extends LinearRecurrentAttnBackend
- def: [`python/sgl_jax/srt/layers/attention/linear/kda_backend.py:29`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/kda_backend.py#L29)
- doc: Attention backend for KDA (Kimi Delta Attention) linear attention.
- signature: `class KDAAttnBackend(LinearRecurrentAttnBackend):`
- members:
  - `_forward_decode(self, q: jax.Array, k: jax.Array, v: jax.Array, g: jax.Array, beta: jax.Array, initial_state: jax.Array, layer: RadixLinearAttention, scale: float | None = None)` — [`L340`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/kda_backend.py#L340) — Single-step decode via naive JAX recurrence (Pallas decode TBD).
  - `_forward_extend(self, q: jax.Array, k: jax.Array, v: jax.Array, g: jax.Array, beta: jax.Array, initial_state: jax.Array, cu_seqlens: jax.Array, layer: RadixLinearAttention, scale: float | None = None)` — [`L267`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/kda_backend.py#L267) — Chunked prefill via Pallas kernel.
  - `_fused_kda_gate(self, layer: RadixLinearAttention, g: jax.Array)` — [`L386`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/kda_backend.py#L386) — JAX-side gate activation used by the DECODE path.
  - `_pack_conv_states(self, q_state: jax.Array, k_state: jax.Array, v_state: jax.Array)` — [`L410`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/kda_backend.py#L410) — Concat per-stream ``[B, D, K-1]`` caches → packed ``[B, proj_size, K-1]``.
  - `_short_conv_extend(self, x: jax.Array, weight: jax.Array, cache: jax.Array, cu_seqlens: jax.Array, activation)` — [`L234`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/kda_backend.py#L234) — EXTEND-path conv wrapped in shard_map.
  - `_unpack_conv_states(self, conv_states: jax.Array)` — [`L400`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/kda_backend.py#L400) — Slice ``[B, proj_size, K-1]`` into per-stream Q/K/V caches.
  - `get_state(self, recurrent_state_pool, layer_id, recurrent_indices)` — [`L146`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/kda_backend.py#L146) — Return per-request views of (ssm, conv) state for this layer.
  - `set_conv_state(self, recurrent_state_pool, layer_id, recurrent_indices, new_conv_packed)` — [`L206`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/kda_backend.py#L206) — Scatter per-request packed conv state. Same idx==0 guard as set_ssm_state.
  - `set_ssm_state(self, recurrent_state_pool, layer_id, recurrent_indices, new_recurrent)` — [`L181`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/kda_backend.py#L181) — Scatter per-request ``new_recurrent`` into the FULL pool buffer.
- protocol/private: `__call__`[`L37`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/kda_backend.py#L37), `__init__`[`L32`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/kda_backend.py#L32), `_call`[`L244`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/kda_backend.py#L244), `_chunk_kda_call`[`L288`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/kda_backend.py#L288), `_decode_kernel`[`L354`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/kda_backend.py#L354), `_scatter`[`L189`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/kda_backend.py#L189)
- uses (calls/refs, reference-scoped): [`ForwardMode`](../../../model_executor/forward_batch_info.md#ForwardMode), [`ForwardBatch`](../../../model_executor/forward_batch_info.md#ForwardBatch), [`named_scope`](../../../utils/profiling_utils.md#named_scope), [`chunk_kda_fwd`](../../../kernels/kda/kda.md#chunk_kda_fwd), [`EXTEND`](../../../model_executor/forward_batch_info.md#ForwardMode.EXTEND), [`forward_mode`](../../../model_executor/forward_batch_info.md#ForwardBatch.forward_mode), [`DECODE`](../../../model_executor/forward_batch_info.md#ForwardMode.DECODE), [`short_convolution`](short_convolution.md#short_convolution), [`LinearRecurrentAttnBackend`](../hybrid_linear_attn_backend.md#LinearRecurrentAttnBackend), [`mesh`](../hybrid_linear_attn_backend.md#LinearRecurrentAttnBackend.mesh), [`forward_metadata`](../hybrid_linear_attn_backend.md#LinearRecurrentAttnBackend.forward_metadata), [`scale`](../../radix_linear_attention.md#RadixLinearAttention.scale), [`weight`](../../linear.md#LinearBase.weight), [`RadixLinearAttention`](../../radix_linear_attention.md#RadixLinearAttention), [`has_initial_state`](../hybrid_linear_attn_backend.md#LinearRecurrentAttnBackendMetadata.has_initial_state), [`recurrent_indices`](../hybrid_linear_attn_backend.md#LinearRecurrentAttnBackendMetadata.recurrent_indices), [`RecurrentStatePool`](../../../mem_cache/recurrent_state_pool.md#RecurrentStatePool), [`activation`](../../radix_linear_attention.md#RadixLinearAttention.activation), [`naive_recurrent_kda`](../../../kernels/kda/naive.md#naive_recurrent_kda), [`A_log`](../../radix_linear_attention.md#RadixLinearAttention.A_log), [`cu_q_lens`](../hybrid_linear_attn_backend.md#LinearRecurrentAttnBackendMetadata.cu_q_lens), [`dt_bias`](../../radix_linear_attention.md#RadixLinearAttention.dt_bias), [`get_layer_cache`](../hybrid_linear_attn_backend.md#LinearRecurrentAttnBackend.get_layer_cache), [`layer_id`](../../radix_linear_attention.md#RadixLinearAttention.layer_id), [`__init__`](../hybrid_linear_attn_backend.md#LinearRecurrentAttnBackend.__init__), [`num_q_heads`](../../radix_linear_attention.md#RadixLinearAttention.num_q_heads), [`head_q_dim`](../../radix_linear_attention.md#RadixLinearAttention.head_q_dim), [`k_conv1d`](../../radix_linear_attention.md#RadixLinearAttention.k_conv1d), [`l2_normalize`](kda_backend.md#l2_normalize), [`q_conv1d`](../../radix_linear_attention.md#RadixLinearAttention.q_conv1d), [`v_conv1d`](../../radix_linear_attention.md#RadixLinearAttention.v_conv1d), [`head_k_dim`](../../radix_linear_attention.md#RadixLinearAttention.head_k_dim), [`head_v_dim`](../../radix_linear_attention.md#RadixLinearAttention.head_v_dim), [`num_k_heads`](../../radix_linear_attention.md#RadixLinearAttention.num_k_heads), [`num_v_heads`](../../radix_linear_attention.md#RadixLinearAttention.num_v_heads)
- used by: [`attn_backend_wrapper`](../hybrid_linear_attn_backend.md#attn_backend_wrapper), [`LinearRecurrentAttnBackend`](../hybrid_linear_attn_backend.md#LinearRecurrentAttnBackend)

## Functions
- `l2_normalize(x: jax.Array, epsilon: float = 0.000001)` — [`L23`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/kda_backend.py#L23) — L2-normalize along the last axis. Computed in fp32, cast back to input dtype.

## Module values
- `__all__` — [`L420`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/kda_backend.py#L420)


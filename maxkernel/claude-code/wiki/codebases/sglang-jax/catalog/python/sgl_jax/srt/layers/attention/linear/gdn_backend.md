---
title: 'Module: python/sgl_jax/srt/layers/attention/linear/gdn_backend.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/layers/attention/linear/gdn_backend.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.layers.attention.linear.gdn_backend`/
symbols:
  GDNAttnBackend.__call__: GDNAttnBackend#__call__().
  GDNAttnBackend.forward_extend: GDNAttnBackend#forward_extend().
  GDNAttnBackend.forward_decode: GDNAttnBackend#forward_decode().
  GDNAttnBackend.conv_dim: GDNAttnBackend#conv_dim.
  GDNAttnBackend._decode_local: GDNAttnBackend#_decode_local().
  GDNAttnBackend._extend_local: GDNAttnBackend#_extend_local().
  GDNAttnBackend: GDNAttnBackend#
  _mesh_tp_size: _mesh_tp_size().
  GDNAttnBackend.__init__: GDNAttnBackend#__init__().
  GDNAttnBackend.num_k_heads: GDNAttnBackend#num_k_heads.
  GDNAttnBackend.num_v_heads: GDNAttnBackend#num_v_heads.
  GDNAttnBackend.head_k_dim: GDNAttnBackend#head_k_dim.
  GDNAttnBackend.head_v_dim: GDNAttnBackend#head_v_dim.
  GDNAttnBackend.key_dim: GDNAttnBackend#key_dim.
  GDNAttnBackend.value_dim: GDNAttnBackend#value_dim.
  GDNAttnBackend.conv_kernel_size: GDNAttnBackend#conv_kernel_size.
---
# Module: [`python/sgl_jax/srt/layers/attention/linear/gdn_backend.py`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/gdn_backend.py)

## Classes
### `GDNAttnBackend`  ·  implements/extends LinearRecurrentAttnBackend
- def: [`python/sgl_jax/srt/layers/attention/linear/gdn_backend.py:60`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/gdn_backend.py#L60)
- doc: Gated-DeltaNet attention backend.
- signature: `class GDNAttnBackend(LinearRecurrentAttnBackend):`
- members:
  - `__call__(self, q: jax.Array, k: jax.Array, v: jax.Array, a: jax.Array, b: jax.Array, layer: RadixLinearAttention, forward_batch: ForwardBatch, recurrent_state_pool: RecurrentStatePool, **kwargs)` — [`L121`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/gdn_backend.py#L121) — Dispatch by ``forward_batch.forward_mode``.
  - `forward_decode(self, mixed_qkv: jax.Array, conv_state_in: jax.Array, recurrent_state_in: jax.Array, b: jax.Array, a: jax.Array, conv1d_weight: jax.Array, A_log: jax.Array, dt_bias: jax.Array)` — [`L196`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/gdn_backend.py#L196) — One token per request — single conv1d update + parallel single
  - `forward_extend(self, mixed_qkv: jax.Array, conv_state_in: jax.Array, recurrent_state_in: jax.Array, b: jax.Array, a: jax.Array, conv1d_weight: jax.Array, A_log: jax.Array, dt_bias: jax.Array)` — [`L292`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/gdn_backend.py#L292) — Packed ragged batch through ``ragged_gated_delta_rule_ref``.
  - `conv_dim` — [`L87`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/gdn_backend.py#L87)
  - `conv_kernel_size` — [`L83`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/gdn_backend.py#L83)
  - `head_k_dim` — [`L81`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/gdn_backend.py#L81)
  - `head_v_dim` — [`L82`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/gdn_backend.py#L82)
  - `key_dim` — [`L85`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/gdn_backend.py#L85)
  - `num_k_heads` — [`L79`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/gdn_backend.py#L79)
  - `num_v_heads` — [`L80`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/gdn_backend.py#L80)
  - `value_dim` — [`L86`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/gdn_backend.py#L86)
- protocol/private: `__init__`[`L69`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/gdn_backend.py#L69), `_decode_local`[`L217`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/gdn_backend.py#L217), `_extend_local`[`L314`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/gdn_backend.py#L314)
- uses (calls/refs, reference-scoped): [`ForwardBatch`](../../../model_executor/forward_batch_info.md#ForwardBatch), [`forward_mode`](../../../model_executor/forward_batch_info.md#ForwardBatch.forward_mode), [`is_decode`](../../../model_executor/forward_batch_info.md#ForwardMode.is_decode), [`LinearRecurrentAttnBackend`](../hybrid_linear_attn_backend.md#LinearRecurrentAttnBackend), [`mesh`](../hybrid_linear_attn_backend.md#LinearRecurrentAttnBackend.mesh), [`forward_metadata`](../hybrid_linear_attn_backend.md#LinearRecurrentAttnBackend.forward_metadata), [`decode_gated_delta_rule_ref`](../../../kernels/gdn/gated_delta.md#decode_gated_delta_rule_ref), [`weight`](../../linear.md#LinearBase.weight), [`ragged_gated_delta_rule_ref`](../../../kernels/gdn/gated_delta.md#ragged_gated_delta_rule_ref), [`RadixLinearAttention`](../../radix_linear_attention.md#RadixLinearAttention), [`has_initial_state`](../hybrid_linear_attn_backend.md#LinearRecurrentAttnBackendMetadata.has_initial_state), [`recurrent_indices`](../hybrid_linear_attn_backend.md#LinearRecurrentAttnBackendMetadata.recurrent_indices), [`RecurrentStatePool`](../../../mem_cache/recurrent_state_pool.md#RecurrentStatePool), [`jax_causal_conv1d_prefill`](../../../kernels/gdn/gated_delta.md#jax_causal_conv1d_prefill), [`jax_causal_conv1d_update`](../../../kernels/gdn/gated_delta.md#jax_causal_conv1d_update), [`A_log`](../../radix_linear_attention.md#RadixLinearAttention.A_log), [`cu_q_lens`](../hybrid_linear_attn_backend.md#LinearRecurrentAttnBackendMetadata.cu_q_lens), [`dt_bias`](../../radix_linear_attention.md#RadixLinearAttention.dt_bias), [`get_layer_cache`](../hybrid_linear_attn_backend.md#LinearRecurrentAttnBackend.get_layer_cache), [`_mesh_tp_size`](gdn_backend.md#_mesh_tp_size), [`layer_id`](../../radix_linear_attention.md#RadixLinearAttention.layer_id), [`__init__`](../hybrid_linear_attn_backend.md#LinearRecurrentAttnBackend.__init__), [`conv1d`](../../radix_linear_attention.md#RadixLinearAttention.conv1d)
- used by: [`attn_backend_wrapper`](../hybrid_linear_attn_backend.md#attn_backend_wrapper), [`LinearRecurrentAttnBackend`](../hybrid_linear_attn_backend.md#LinearRecurrentAttnBackend)

## Functions
- `_mesh_tp_size(mesh: jax.sharding.Mesh)` — [`L50`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/gdn_backend.py#L50) — TP size = mesh size on the ``"tensor"`` axis (1 if absent).


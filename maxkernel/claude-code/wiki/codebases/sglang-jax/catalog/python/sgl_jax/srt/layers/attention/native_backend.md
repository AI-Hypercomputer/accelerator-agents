---
title: 'Module: python/sgl_jax/srt/layers/attention/native_backend.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/layers/attention/native_backend.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.layers.attention.native_backend`/
symbols:
  NativeAttention.__call__: NativeAttention#__call__().
  NativeAttention._get_and_update_kv_cache: NativeAttention#_get_and_update_kv_cache().
  forward_attention: forward_attention().
  NativeAttention.tree_flatten: NativeAttention#tree_flatten().
  NativeAttention: NativeAttention#
  NativeAttention.kv_sharding: NativeAttention#kv_sharding.
  NativeAttention.num_kv_heads: NativeAttention#num_kv_heads.
  _apply_decode_mask: _apply_decode_mask().
  NativeAttention.mesh: NativeAttention#mesh.
  NativeAttention.get_forward_metadata: NativeAttention#get_forward_metadata().
  NativeAttention.num_heads: NativeAttention#num_heads.
  _apply_extend_mask: _apply_extend_mask().
  _apply_decode_mask.create_decode_sequence_mask: _apply_decode_mask().create_decode_sequence_mask().
  NativeAttention.__init__: NativeAttention#__init__().
  NativeAttention.tree_unflatten: NativeAttention#tree_unflatten().
  NativeAttention.get_max_running_reqests: NativeAttention#get_max_running_reqests().
---
# Module: [`python/sgl_jax/srt/layers/attention/native_backend.py`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/native_backend.py)

## Classes
### `NativeAttention`  ·  implements/extends AttentionBackend
- def: [`python/sgl_jax/srt/layers/attention/native_backend.py:15`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/native_backend.py#L15)
- doc: Native Attention layer for variable-length sequences using ForwardBatch.
- signature: `class NativeAttention(AttentionBackend):`
- members:
  - `__call__(self, q: jax.Array, k: jax.Array, v: jax.Array, layer: RadixAttention, forward_batch: ForwardBatch, token_to_kv_pool: KVCache, **kwargs)` — [`L54`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/native_backend.py#L54) — Args:
  - `_get_and_update_kv_cache(self, k: jax.Array, v: jax.Array, forward_batch: ForwardBatch, token_to_kv_pool: KVCache, layer_id: int)` — [`L118`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/native_backend.py#L118) — Update KV cache and return (k_3d, v_3d, fused_5d).
  - `get_forward_metadata(self, batch: ModelWorkerBatch)` — [`L49`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/native_backend.py#L49) — Init the metadata for a forward pass and return it.
  - `get_max_running_reqests(max_context_len: int, page_size: int)` — [`L163`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/native_backend.py#L163)
  - `tree_flatten(self)` — [`L32`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/native_backend.py#L32)
  - `tree_unflatten(cls, aux_data, children)` — [`L42`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/native_backend.py#L42)
  - `kv_sharding` — [`L30`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/native_backend.py#L30)
  - `mesh` — [`L29`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/native_backend.py#L29)
  - `num_heads` — [`L24`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/native_backend.py#L24)
  - `num_kv_heads` — [`L26`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/native_backend.py#L26)
- protocol/private: `__init__`[`L18`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/native_backend.py#L18)
- uses (calls/refs, reference-scoped): [`ForwardMode`](../../model_executor/forward_batch_info.md#ForwardMode), `ModelWorkerBatch`, [`ForwardBatch`](../../model_executor/forward_batch_info.md#ForwardBatch), [`is_extend`](../../model_executor/forward_batch_info.md#ForwardMode.is_extend), [`named_scope`](../../utils/profiling_utils.md#named_scope), [`KVCache`](../../mem_cache/memory_pool.md#KVCache), [`forward_mode`](../../model_executor/forward_batch_info.md#ForwardBatch.forward_mode), [`AttentionBackend`](base_attn_backend.md#AttentionBackend), [`DECODE`](../../model_executor/forward_batch_info.md#ForwardMode.DECODE), [`get_fused_kv_buffer`](../../mem_cache/memory_pool.md#KVCache.get_fused_kv_buffer), [`forward_attention`](native_backend.md#forward_attention), [`set_kv_buffer`](../../mem_cache/memory_pool.md#KVCache.set_kv_buffer), [`is_tpu_runtime`](../../utils/jax_utils.md#is_tpu_runtime), [`out_cache_loc`](../../model_executor/forward_batch_info.md#ForwardBatch.out_cache_loc), [`seq_lens`](../../model_executor/forward_batch_info.md#ForwardBatch.seq_lens), [`RadixAttention`](../radix_attention.md#RadixAttention), [`scaling`](../radix_attention.md#RadixAttention.scaling), [`cache_loc`](../../model_executor/forward_batch_info.md#ForwardBatch.cache_loc), [`sliding_window_size`](../radix_attention.md#RadixAttention.sliding_window_size), [`AttentionType`](../radix_attention.md#AttentionType), [`extend_prefix_lens`](../../model_executor/forward_batch_info.md#ForwardBatch.extend_prefix_lens), [`extend_seq_lens`](../../model_executor/forward_batch_info.md#ForwardBatch.extend_seq_lens), [`layer_id`](../radix_attention.md#RadixAttention.layer_id), [`head_dim`](../radix_attention.md#RadixAttention.head_dim), [`softmax_dtype`](../radix_attention.md#RadixAttention.softmax_dtype), [`ENCODER_ONLY`](../radix_attention.md#AttentionType.ENCODER_ONLY), [`attn_type`](../radix_attention.md#RadixAttention.attn_type), [`kv_head_num`](../radix_attention.md#RadixAttention.kv_head_num), [`q_head_num`](../radix_attention.md#RadixAttention.q_head_num)
- used by: [`_get_attention_backend`](../../model_executor/model_runner.md#ModelRunner._get_attention_backend), [`AttentionBackend`](base_attn_backend.md#AttentionBackend), [`get_forward_metadata`](base_attn_backend.md#AttentionBackend.get_forward_metadata), [`get_max_running_reqests`](base_attn_backend.md#AttentionBackend.get_max_running_reqests), [`get_attention_impl`](utils.md#get_attention_impl)

## Functions
- `_apply_decode_mask(attn_weights: jax.Array, seq_lengths: jax.Array, sliding_window_size: int | None = None, mesh: Mesh | None = None)` — [`L399`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/native_backend.py#L399) — Create a sequence mask that ensures tokens only attend within their sequence and window.
- `_apply_extend_mask(attn_weights: jax.Array, seq_lengths: jax.Array, extend_prefix_lens: jax.Array, extend_seq_lens: jax.Array, is_causal: bool = True, sliding_window_size: int | None = None, mesh: Mesh | None = None)` — [`L330`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/native_backend.py#L330) — Applies a block-diagonal and optionally a causal/SWA mask in a unified,
- `create_decode_sequence_mask()` — [`L413`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/native_backend.py#L413)
- `forward_attention(q: jax.Array, k_cache: jax.Array, v_cache: jax.Array, seq_lengths: jax.Array, loc: jax.Array, extend_prefix_lens: jax.Array, extend_seq_lens: jax.Array, num_heads, num_kv_heads, scale=None, is_causal=True, mode=ForwardMode.DECODE, kv_sharding=None, mesh: Mesh | None = None, xai_temperature_len: float | None = None, attention_sink: jax.Array | None = None, sliding_window_size: int | None = None, softmax_dtype: jnp.dtype | None = None)` — [`L169`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/native_backend.py#L169) — Forward pass using native JAX implementation with block-diagonal attention.


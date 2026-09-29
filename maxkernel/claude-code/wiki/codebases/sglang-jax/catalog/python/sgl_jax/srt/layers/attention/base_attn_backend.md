---
title: 'Module: python/sgl_jax/srt/layers/attention/base_attn_backend.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/layers/attention/base_attn_backend.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.layers.attention.base_attn_backend`/AttentionBackend
symbols:
  AttentionBackend: '#'
  AttentionBackend.get_forward_metadata: '#get_forward_metadata().'
  AttentionBackend.get_max_running_reqests: '#get_max_running_reqests().'
  AttentionBackend.__call__: '#__call__().'
  AttentionBackendMetadata: Metadata#
  AttentionBackendMetadata.tree_flatten: Metadata#tree_flatten().
  AttentionBackendMetadata.tree_unflatten: Metadata#tree_unflatten().
---
# Module: [`python/sgl_jax/srt/layers/attention/base_attn_backend.py`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/base_attn_backend.py)

## Classes
### `AttentionBackend`  ·  implements/extends Module
- def: [`python/sgl_jax/srt/layers/attention/base_attn_backend.py:35`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/base_attn_backend.py#L35)
- doc: The base class of attention backends
- signature: `class AttentionBackend(nnx.Module):`
- members:
  - `__call__(self, q: jax.Array, k: jax.Array, v: jax.Array, layer: RadixAttention, forward_batch: ForwardBatch, **kwargs)` — [`L43`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/base_attn_backend.py#L43) — Run forward on an attention layer.
  - `get_forward_metadata(self, batch: ModelWorkerBatch)` — [`L39`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/base_attn_backend.py#L39) — Init the metadata for a forward pass and return it
  - `get_max_running_reqests(max_context_len: int, page_size: int)` — [`L56`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/base_attn_backend.py#L56)
- uses (calls/refs, reference-scoped): `ModelWorkerBatch`, [`ForwardBatch`](../../model_executor/forward_batch_info.md#ForwardBatch), [`get_forward_metadata`](flashattention_backend.md#FlashAttention.get_forward_metadata), [`get_forward_metadata`](mla_backend.md#MLAAttentionBackend.get_forward_metadata), [`get_forward_metadata`](hybrid_linear_attn_backend.md#LinearRecurrentAttnBackend.get_forward_metadata), [`HybridLinearAttnBackend`](hybrid_linear_attn_backend.md#HybridLinearAttnBackend), [`get_forward_metadata`](hybrid_linear_attn_backend.md#HybridLinearAttnBackend.get_forward_metadata), [`LinearRecurrentAttnBackend`](hybrid_linear_attn_backend.md#LinearRecurrentAttnBackend), [`RadixAttention`](../radix_attention.md#RadixAttention), [`FlashAttention`](flashattention_backend.md#FlashAttention), [`NativeAttention`](native_backend.md#NativeAttention), [`get_max_running_reqests`](hybrid_linear_attn_backend.md#HybridLinearAttnBackend.get_max_running_reqests), [`MLAAttentionBackend`](mla_backend.md#MLAAttentionBackend), [`get_forward_metadata`](native_backend.md#NativeAttention.get_forward_metadata), [`get_max_running_reqests`](flashattention_backend.md#FlashAttention.get_max_running_reqests), [`get_max_running_reqests`](mla_backend.md#MLAAttentionBackend.get_max_running_reqests), [`get_max_running_reqests`](hybrid_linear_attn_backend.md#LinearRecurrentAttnBackend.get_max_running_reqests), [`get_max_running_reqests`](native_backend.md#NativeAttention.get_max_running_reqests)
- used by: [`forward_batch_generation`](../../managers/tp_worker.md#ModelWorker.forward_batch_generation), [`cur_sampling_info`](../../managers/tp_worker_overlap_thread.md#ModelWorkerClient.cur_sampling_info), [`attn_backend_wrapper`](hybrid_linear_attn_backend.md#attn_backend_wrapper), [`max_total_num_tokens`](../../managers/tp_worker.md#ModelWorker.max_total_num_tokens), [`HybridLinearAttnBackend`](hybrid_linear_attn_backend.md#HybridLinearAttnBackend), [`get_forward_metadata`](hybrid_linear_attn_backend.md#HybridLinearAttnBackend.get_forward_metadata), [`LinearRecurrentAttnBackend`](hybrid_linear_attn_backend.md#LinearRecurrentAttnBackend), [`logger`](../../model_executor/forward_batch_info.md#logger), [`set_forward_metadata`](../../managers/tp_worker.md#ModelWorker.set_forward_metadata), [`attn_backend`](../../model_executor/forward_batch_info.md#ForwardBatch.attn_backend), [`FlashAttention`](flashattention_backend.md#FlashAttention), [`NativeAttention`](native_backend.md#NativeAttention), [`__init__`](hybrid_linear_attn_backend.md#HybridLinearAttnBackend.__init__), [`get_max_running_reqests`](hybrid_linear_attn_backend.md#HybridLinearAttnBackend.get_max_running_reqests), [`MLAAttentionBackend`](mla_backend.md#MLAAttentionBackend)

### `AttentionBackendMetadata`
- def: [`python/sgl_jax/srt/layers/attention/base_attn_backend.py:19`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/base_attn_backend.py#L19)
- doc: Empty pytree base type for per-backend forward metadata.
- signature: `class AttentionBackendMetadata:`
- members:
  - `tree_flatten(self)` — [`L27`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/base_attn_backend.py#L27)
  - `tree_unflatten(cls, aux_data, children)` — [`L31`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/base_attn_backend.py#L31)
- used by: [`full_attn_metadata`](hybrid_linear_attn_backend.md#HybridLinearAttnBackendMetadata.full_attn_metadata)


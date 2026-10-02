---
title: 'Module: python/sgl_jax/srt/layers/attention/flashattention_backend.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/layers/attention/flashattention_backend.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.layers.attention.flashattention_backend`/
symbols:
  FlashAttention.get_eagle_forward_metadata: FlashAttention#get_eagle_forward_metadata().
  FlashAttention.get_eagle_multi_step_metadata: FlashAttention#get_eagle_multi_step_metadata().
  FlashAttention.__call__: FlashAttention#__call__().
  FlashAttention.get_forward_metadata: FlashAttention#get_forward_metadata().
  FlashAttention.page_size: FlashAttention#page_size.
  FlashAttention.tree_flatten: FlashAttention#tree_flatten().
  FlashAttentionMetadata.tree_flatten: FlashAttentionMetadata#tree_flatten().
  FlashAttentionMetadata.tree_unflatten: FlashAttentionMetadata#tree_unflatten().
  FlashAttention.forward_metadata: FlashAttention#forward_metadata.
  FlashAttentionMetadata.custom_mask: FlashAttentionMetadata#custom_mask.
  FlashAttention._ragged_paged_attention_with_fused_kv: FlashAttention#_ragged_paged_attention_with_fused_kv().
  FlashAttention.attention_data_partition_axis: FlashAttention#attention_data_partition_axis.
  FlashAttention._get_fused_kv_cache: FlashAttention#_get_fused_kv_cache().
  FlashAttentionMetadata: FlashAttentionMetadata#
  FlashAttentionMetadata.cu_q_lens: FlashAttentionMetadata#cu_q_lens.
  FlashAttentionMetadata.cu_kv_lens: FlashAttentionMetadata#cu_kv_lens.
  FlashAttentionMetadata.page_indices: FlashAttentionMetadata#page_indices.
  FlashAttentionMetadata.swa_page_indices: FlashAttentionMetadata#swa_page_indices.
  FlashAttentionMetadata.seq_lens: FlashAttentionMetadata#seq_lens.
  FlashAttentionMetadata.distribution: FlashAttentionMetadata#distribution.
  FlashAttention.kv_partition_axis: FlashAttention#kv_partition_axis.
  FlashAttention: FlashAttention#
  FlashAttention.mesh: FlashAttention#mesh.
  FlashAttention.num_kv_heads: FlashAttention#num_kv_heads.
  FlashAttention.head_dim: FlashAttention#head_dim.
  _per_dp_cumsum: _per_dp_cumsum().
  FlashAttention.tree_unflatten: FlashAttention#tree_unflatten().
  FlashAttention.get_max_running_reqests: FlashAttention#get_max_running_reqests().
  FlashAttention._dp_starts: FlashAttention#_dp_starts().
  FlashAttention.num_heads: FlashAttention#num_heads.
  logger: logger.
  FlashAttention.__init__: FlashAttention#__init__().
---
# Module: [`python/sgl_jax/srt/layers/attention/flashattention_backend.py`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/flashattention_backend.py)

## Classes
### `FlashAttention`  ·  implements/extends AttentionBackend
- def: [`python/sgl_jax/srt/layers/attention/flashattention_backend.py:86`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/flashattention_backend.py#L86)
- doc: Native Attention layer for variable-length sequences using ForwardBatch.
- signature: `class FlashAttention(AttentionBackend):`
- members:
  - `__call__(self, q: jax.Array, k: jax.Array, v: jax.Array, layer: RadixAttention, forward_batch: ForwardBatch, token_to_kv_pool: KVCache, causal: int = 1, attention_sink: jax.Array = None)` — [`L495`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/flashattention_backend.py#L495) — Args:
  - `get_eagle_forward_metadata(self, batch: ModelWorkerBatch)` — [`L206`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/flashattention_backend.py#L206) — Return the metadata for a forward pass. — documented in [python-sgl_jax-srt-speculative-eagle_util](../../../../../../concepts/python-sgl_jax-srt-speculative-eagle_util.md)
  - `get_eagle_multi_step_metadata(self, batch: ModelWorkerBatch)` — [`L356`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/flashattention_backend.py#L356)
  - `get_forward_metadata(self, batch: ModelWorkerBatch)` — [`L113`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/flashattention_backend.py#L113) — Return the metadata for a forward pass.
  - `get_max_running_reqests(max_context_len: int, page_size: int)` — [`L631`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/flashattention_backend.py#L631)
  - `tree_flatten(self)` — [`L465`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/flashattention_backend.py#L465)
  - `tree_unflatten(cls, aux_data, children)` — [`L479`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/flashattention_backend.py#L479)
  - `attention_data_partition_axis` — [`L107`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/flashattention_backend.py#L107)
  - `forward_metadata` — [`L108`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/flashattention_backend.py#L108)
  - `head_dim` — [`L104`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/flashattention_backend.py#L104)
  - `kv_partition_axis` — [`L106`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/flashattention_backend.py#L106)
  - `mesh` — [`L109`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/flashattention_backend.py#L109)
  - `num_heads` — [`L99`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/flashattention_backend.py#L99)
  - `num_kv_heads` — [`L101`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/flashattention_backend.py#L101)
  - `page_size` — [`L105`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/flashattention_backend.py#L105)
- protocol/private: `__init__`[`L89`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/flashattention_backend.py#L89), `_dp_starts`[`L394`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/flashattention_backend.py#L394), `_get_fused_kv_cache`[`L622`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/flashattention_backend.py#L622), `_ragged_paged_attention_with_fused_kv`[`L571`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/flashattention_backend.py#L571)
- uses (calls/refs, reference-scoped): [`ForwardMode`](../../model_executor/forward_batch_info.md#ForwardMode), `ModelWorkerBatch`, `seq_lens`, `spec_info_padded`, [`allocate_lens`](../../speculative/eagle_util.md#EagleDraftInput.allocate_lens), [`ForwardBatch`](../../model_executor/forward_batch_info.md#ForwardBatch), `forward_mode`, `dp_size`, [`device_array`](../../utils/jax_utils.md#device_array), [`EagleDraftInput`](../../speculative/eagle_util.md#EagleDraftInput), `per_dp_bs_size`, [`ragged_paged_attention`](../../kernels/ragged_paged_attention/ragged_paged_attention_v3.md#ragged_paged_attention), [`named_scope`](../../utils/profiling_utils.md#named_scope), [`KVCache`](../../mem_cache/memory_pool.md#KVCache), [`EXTEND`](../../model_executor/forward_batch_info.md#ForwardMode.EXTEND), [`forward_mode`](../../model_executor/forward_batch_info.md#ForwardBatch.forward_mode), [`AttentionBackend`](base_attn_backend.md#AttentionBackend), `extend_seq_lens`, [`is_none`](../../speculative/spec_info.md#SpeculativeAlgorithm.is_none), [`DECODE`](../../model_executor/forward_batch_info.md#ForwardMode.DECODE), `cache_loc`, `logits_indices_selector`, [`cdiv`](../../utils/common_utils.md#cdiv), [`custom_mask`](flashattention_backend.md#FlashAttentionMetadata.custom_mask), [`draft_token_num`](../../speculative/eagle_util.md#EagleVerifyInput.draft_token_num), [`get_fused_kv_buffer`](../../mem_cache/memory_pool.md#KVCache.get_fused_kv_buffer), `spec_algorithm`, [`is_target_verify`](../../model_executor/forward_batch_info.md#ForwardMode.is_target_verify), [`FlashAttentionMetadata`](flashattention_backend.md#FlashAttentionMetadata), [`RadixAttention`](../radix_attention.md#RadixAttention), [`cu_kv_lens`](flashattention_backend.md#FlashAttentionMetadata.cu_kv_lens), [`cu_q_lens`](flashattention_backend.md#FlashAttentionMetadata.cu_q_lens), [`distribution`](flashattention_backend.md#FlashAttentionMetadata.distribution), [`page_indices`](flashattention_backend.md#FlashAttentionMetadata.page_indices), [`seq_lens`](flashattention_backend.md#FlashAttentionMetadata.seq_lens), [`swa_page_indices`](flashattention_backend.md#FlashAttentionMetadata.swa_page_indices), [`scaling`](../radix_attention.md#RadixAttention.scaling), [`DRAFT_EXTEND`](../../model_executor/forward_batch_info.md#ForwardMode.DRAFT_EXTEND), [`TARGET_VERIFY`](../../model_executor/forward_batch_info.md#ForwardMode.TARGET_VERIFY), [`custom_mask`](../../speculative/eagle_util.md#EagleVerifyInput.custom_mask)  (+9 more)
- used by: [`_get_attention_backend`](../../model_executor/model_runner.md#ModelRunner._get_attention_backend), [`AttentionBackend`](base_attn_backend.md#AttentionBackend), [`get_forward_metadata`](base_attn_backend.md#AttentionBackend.get_forward_metadata), [`get_max_running_reqests`](base_attn_backend.md#AttentionBackend.get_max_running_reqests), [`get_attention_impl`](utils.md#get_attention_impl)

### `FlashAttentionMetadata`
- def: [`python/sgl_jax/srt/layers/attention/flashattention_backend.py:41`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/flashattention_backend.py#L41)
- doc: Metadata to be init once in the model forward pass,
- signature: `class FlashAttentionMetadata:`
- members:
  - `tree_flatten(self)` — [`L56`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/flashattention_backend.py#L56)
  - `tree_unflatten(cls, aux_data, children)` — [`L71`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/flashattention_backend.py#L71)
  - `cu_kv_lens` — [`L49`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/flashattention_backend.py#L49)
  - `cu_q_lens` — [`L48`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/flashattention_backend.py#L48)
  - `custom_mask` — [`L54`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/flashattention_backend.py#L54)
  - `distribution` — [`L53`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/flashattention_backend.py#L53)
  - `page_indices` — [`L50`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/flashattention_backend.py#L50)
  - `seq_lens` — [`L52`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/flashattention_backend.py#L52)
  - `swa_page_indices` — [`L51`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/flashattention_backend.py#L51)
- used by: [`get_eagle_forward_metadata`](flashattention_backend.md#FlashAttention.get_eagle_forward_metadata), [`get_eagle_multi_step_metadata`](flashattention_backend.md#FlashAttention.get_eagle_multi_step_metadata), [`__call__`](flashattention_backend.md#FlashAttention.__call__), [`get_forward_metadata`](flashattention_backend.md#FlashAttention.get_forward_metadata), [`_make_draft_extend_metadata`](../../speculative/draft_extend_fused.md#_make_draft_extend_metadata), [`_make_target_verify_metadata`](../../speculative/draft_extend_fused.md#_make_target_verify_metadata), [`forward_metadata`](flashattention_backend.md#FlashAttention.forward_metadata)

## Functions
- `_per_dp_cumsum(lens, dp_size: int, per_dp_bs: int)` — [`L28`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/flashattention_backend.py#L28) — `(dp*(per_dp_bs+1),)` row-wise cumsum with leading 0 per DP rank.

## Module values
- `logger` — [`L25`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/flashattention_backend.py#L25)


---
title: 'Module: python/sgl_jax/srt/layers/attention/mla_backend.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/layers/attention/mla_backend.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.layers.attention.mla_backend`/
symbols:
  MLAAttentionBackend.get_forward_metadata: MLAAttentionBackend#get_forward_metadata().
  MLAAttentionBackend.__call__: MLAAttentionBackend#__call__().
  MLAAttentionBackend.tree_flatten: MLAAttentionBackend#tree_flatten().
  MLAAttentionBackend._run: MLAAttentionBackend#_run().
  MLAAttentionMetadata.tree_flatten: MLAAttentionMetadata#tree_flatten().
  MLAAttentionMetadata.tree_unflatten: MLAAttentionMetadata#tree_unflatten().
  MLAAttentionBackend.forward_metadata: MLAAttentionBackend#forward_metadata.
  MLAAttentionBackend.page_size: MLAAttentionBackend#page_size.
  MLAAttentionBackend: MLAAttentionBackend#
  MLAAttentionMetadata.cu_q_lens: MLAAttentionMetadata#cu_q_lens.
  MLAAttentionMetadata.cu_kv_lens: MLAAttentionMetadata#cu_kv_lens.
  MLAAttentionMetadata.page_indices: MLAAttentionMetadata#page_indices.
  MLAAttentionMetadata.seq_lens: MLAAttentionMetadata#seq_lens.
  MLAAttentionMetadata.distribution: MLAAttentionMetadata#distribution.
  MLAAttentionBackend.attention_data_partition_axis: MLAAttentionBackend#attention_data_partition_axis.
  MLAAttentionBackend.tree_unflatten: MLAAttentionBackend#tree_unflatten().
  MLAAttentionBackend.get_max_running_reqests: MLAAttentionBackend#get_max_running_reqests().
  MLAAttentionMetadata: MLAAttentionMetadata#
  MLAAttentionBackend.qk_nope_head_dim: MLAAttentionBackend#qk_nope_head_dim.
  MLAAttentionBackend.qk_rope_head_dim: MLAAttentionBackend#qk_rope_head_dim.
  MLAAttentionBackend.mesh: MLAAttentionBackend#mesh.
  MLAAttentionBackend.vmem_limit_bytes: MLAAttentionBackend#vmem_limit_bytes.
  MLAAttentionBackend.num_kv_pages_per_block: MLAAttentionBackend#num_kv_pages_per_block.
  MLAAttentionBackend.num_queries_per_block: MLAAttentionBackend#num_queries_per_block.
  MLAAttentionBackend.decode_batch_size: MLAAttentionBackend#decode_batch_size.
  MLAAttentionBackend.num_heads: MLAAttentionBackend#num_heads.
  MLAAttentionBackend.kv_lora_rank: MLAAttentionBackend#kv_lora_rank.
  MLAAttentionBackend.v_head_dim: MLAAttentionBackend#v_head_dim.
  logger: logger.
  MLAAttentionBackend.__init__: MLAAttentionBackend#__init__().
---
# Module: [`python/sgl_jax/srt/layers/attention/mla_backend.py`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/mla_backend.py)

## Classes
### `MLAAttentionBackend`  ·  implements/extends AttentionBackend
- def: [`python/sgl_jax/srt/layers/attention/mla_backend.py:83`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/mla_backend.py#L83)
- doc: Absorbed-MLA attention backend backed by the v2 Pallas kernel.
- signature: `class MLAAttentionBackend(AttentionBackend):`
- members:
  - `__call__(self, q: jax.Array, k: jax.Array, v: jax.Array, layer: RadixAttention, forward_batch: ForwardBatch, token_to_kv_pool: KVCache, **kwargs)` — [`L251`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/mla_backend.py#L251) — Absorbed-MLA forward, called through ``RadixAttention``.
  - `get_forward_metadata(self, batch: ModelWorkerBatch)` — [`L138`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/mla_backend.py#L138) — Build per-batch metadata, DP-aware.
  - `get_max_running_reqests(max_context_len: int, page_size: int)` — [`L380`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/mla_backend.py#L380)
  - `tree_flatten(self)` — [`L213`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/mla_backend.py#L213)
  - `tree_unflatten(cls, aux_data, children)` — [`L232`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/mla_backend.py#L232)
  - `attention_data_partition_axis` — [`L126`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/mla_backend.py#L126)
  - `decode_batch_size` — [`L134`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/mla_backend.py#L134)
  - `forward_metadata` — [`L136`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/mla_backend.py#L136)
  - `kv_lora_rank` — [`L120`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/mla_backend.py#L120)
  - `mesh` — [`L125`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/mla_backend.py#L125)
  - `num_heads` — [`L119`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/mla_backend.py#L119)
  - `num_kv_pages_per_block` — [`L132`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/mla_backend.py#L132)
  - `num_queries_per_block` — [`L133`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/mla_backend.py#L133)
  - `page_size` — [`L124`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/mla_backend.py#L124)
  - `qk_nope_head_dim` — [`L121`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/mla_backend.py#L121)
  - `qk_rope_head_dim` — [`L122`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/mla_backend.py#L122)
  - `v_head_dim` — [`L123`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/mla_backend.py#L123)
  - `vmem_limit_bytes` — [`L131`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/mla_backend.py#L131)
- protocol/private: `__init__`[`L86`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/mla_backend.py#L86), `_run`[`L327`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/mla_backend.py#L327)
- uses (calls/refs, reference-scoped): [`ForwardMode`](../../model_executor/forward_batch_info.md#ForwardMode), `ModelWorkerBatch`, `seq_lens`, [`ForwardBatch`](../../model_executor/forward_batch_info.md#ForwardBatch), `forward_mode`, `dp_size`, [`device_array`](../../utils/jax_utils.md#device_array), `per_dp_bs_size`, [`named_scope`](../../utils/profiling_utils.md#named_scope), [`KVCache`](../../mem_cache/memory_pool.md#KVCache), [`EXTEND`](../../model_executor/forward_batch_info.md#ForwardMode.EXTEND), [`mla_ragged_paged_attention`](../../kernels/mla/v2/kernel.md#mla_ragged_paged_attention), [`AttentionBackend`](base_attn_backend.md#AttentionBackend), `extend_seq_lens`, [`DECODE`](../../model_executor/forward_batch_info.md#ForwardMode.DECODE), `cache_loc`, [`get_fused_kv_buffer`](../../mem_cache/memory_pool.md#KVCache.get_fused_kv_buffer), [`cdiv`](../../kernels/mla/v2/kernel.md#cdiv), [`RadixAttention`](../radix_attention.md#RadixAttention), [`scaling`](../radix_attention.md#RadixAttention.scaling), [`sliding_window_size`](../radix_attention.md#RadixAttention.sliding_window_size), [`cu_kv_lens`](mla_backend.md#MLAAttentionMetadata.cu_kv_lens), [`cu_q_lens`](mla_backend.md#MLAAttentionMetadata.cu_q_lens), [`distribution`](mla_backend.md#MLAAttentionMetadata.distribution), [`layer_id`](../radix_attention.md#RadixAttention.layer_id), [`page_indices`](mla_backend.md#MLAAttentionMetadata.page_indices), [`seq_lens`](mla_backend.md#MLAAttentionMetadata.seq_lens), [`MLAAttentionMetadata`](mla_backend.md#MLAAttentionMetadata), [`logit_cap`](../radix_attention.md#RadixAttention.logit_cap)
- used by: [`_get_attention_backend`](../../model_executor/model_runner.md#ModelRunner._get_attention_backend), [`AttentionBackend`](base_attn_backend.md#AttentionBackend), [`get_forward_metadata`](base_attn_backend.md#AttentionBackend.get_forward_metadata), [`get_max_running_reqests`](base_attn_backend.md#AttentionBackend.get_max_running_reqests)

### `MLAAttentionMetadata`
- def: [`python/sgl_jax/srt/layers/attention/mla_backend.py:45`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/mla_backend.py#L45)
- doc: Per-forward metadata for the MLA v2 kernel.
- signature: `class MLAAttentionMetadata:`
- members:
  - `tree_flatten(self)` — [`L60`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/mla_backend.py#L60)
  - `tree_unflatten(cls, aux_data, children)` — [`L72`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/mla_backend.py#L72)
  - `cu_kv_lens` — [`L55`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/mla_backend.py#L55)
  - `cu_q_lens` — [`L54`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/mla_backend.py#L54)
  - `distribution` — [`L58`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/mla_backend.py#L58)
  - `page_indices` — [`L56`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/mla_backend.py#L56)
  - `seq_lens` — [`L57`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/mla_backend.py#L57)
- used by: [`get_forward_metadata`](mla_backend.md#MLAAttentionBackend.get_forward_metadata), [`__call__`](mla_backend.md#MLAAttentionBackend.__call__), [`forward_metadata`](mla_backend.md#MLAAttentionBackend.forward_metadata)

## Module values
- `logger` — [`L40`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/mla_backend.py#L40)


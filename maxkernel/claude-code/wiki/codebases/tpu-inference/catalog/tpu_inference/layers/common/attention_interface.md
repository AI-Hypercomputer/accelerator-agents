---
title: 'Module: tpu_inference/layers/common/attention_interface.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/common/attention_interface.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.common.attention_interface`/
symbols:
  mla_attention._mla_ragged_paged_attention: mla_attention()._mla_ragged_paged_attention().
  attention: attention().
  mla_attention: mla_attention().
  sharded_flash_attention: sharded_flash_attention().
  sharded_ragged_paged_attention: sharded_ragged_paged_attention().
  MAX_ALLOWED_PAGE_INDICES_N: MAX_ALLOWED_PAGE_INDICES_N.
  encoder_only_attention: encoder_only_attention().
  logger: logger.
  ragged_paged_attention: ragged_paged_attention.
  ragged_paged_attention_hd64: ragged_paged_attention_hd64.
  sharded_flash_attention._flash_attention_use_ab: sharded_flash_attention()._flash_attention_use_ab().
  sharded_flash_attention._flash_attention: sharded_flash_attention()._flash_attention().
  sharded_encoder_only_attention: sharded_encoder_only_attention().
  sharded_encoder_only_attention._flash_attention: sharded_encoder_only_attention()._flash_attention().
  sharded_paged_attention._paged_attention_fn: sharded_paged_attention()._paged_attention_fn().
  get_kv_cache_shape: get_kv_cache_shape.
  get_kv_cache_shape_hd64: get_kv_cache_shape_hd64.
  sharded_paged_attention: sharded_paged_attention().
  paged_attention_with_guarded_smem: paged_attention_with_guarded_smem().
  sharded_splash_attention: sharded_splash_attention().
  apply_splash: apply_splash().
  sharded_ragged_paged_attention._ragged_paged_attention: sharded_ragged_paged_attention()._ragged_paged_attention().
  update_cache: update_cache().
---
# Module: [`tpu_inference/layers/common/attention_interface.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/attention_interface.py)

## Functions
- `_flash_attention(q, k, v, segment_ids)` — [`L106`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/attention_interface.py#L106)
- `_flash_attention(q, k, v, seq_lens)` — [`L140`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/attention_interface.py#L140)
- `_flash_attention_use_ab(q, k, v, attention_bias, segment_ids)` — [`L85`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/attention_interface.py#L85)
- `_mla_ragged_paged_attention(q, q_rope, k, k_rope, cache, *args)` — [`L584`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/attention_interface.py#L584) — documented in [tpu_inference-kernels-mla-v2-kernel](../../../../concepts/tpu_inference-kernels-mla-v2-kernel.md)
- `_paged_attention_fn(q, k, v, lengths, page_indices)` — [`L176`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/attention_interface.py#L176)
- `_ragged_paged_attention(*args)` — [`L440`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/attention_interface.py#L440)
- `apply_splash(q, k, v, window_size, attn_logits_soft_cap, is_mqa)` — [`L312`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/attention_interface.py#L312)
- `attention(kv_cache: jax.Array, q: jax.Array, k: jax.Array, v: jax.Array, attention_metadata: AttentionMetadata, mesh: Mesh, head_dim_original: int | None = None, sm_scale: float | None = None, attention_chunk_size: int | None = None, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, sinks: jax.Array | None = None, update_kv_cache: bool = True)` — [`L464`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/attention_interface.py#L464)
- `encoder_only_attention(q: jax.Array, k: jax.Array, v: jax.Array, attention_metadata: AttentionMetadata, mesh: Mesh, sm_scale: float | None = None, sliding_window: int | None = None)` — [`L649`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/attention_interface.py#L649)
- `mla_attention(q_NTA: jax.Array, q_rope_TNH: jax.Array, k_SA: jax.Array, k_rope_SH: jax.Array, kv_cache: jax.Array, md: AttentionMetadata, mesh: Mesh, num_attention_heads: int, qk_nope_head_dim: int, query_nth_sharding: Sharding | None = None, query_tnh_sharding: Sharding | None = None, keyvalue_skh_sharding: Sharding | None = None, attn_o_nth_sharding: Sharding | None = None, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, sm_scale: float | None = None)` — [`L523`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/attention_interface.py#L523) — Main shared interface for MLA attention.  Computes the sharded attention
- `paged_attention_with_guarded_smem(paged_attention_kernel: Callable, q: jax.Array, k_pages: jax.Array, v_pages: jax.Array, lengths: jax.Array, page_indices: jax.Array)` — [`L205`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/attention_interface.py#L205)
- `sharded_encoder_only_attention(mesh: Mesh, causal: bool = True, sm_scale: Optional[float] = None, sliding_window: Optional[int] = None, vmem_limit_bytes: int | None = None)` — [`L125`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/attention_interface.py#L125)
- `sharded_flash_attention(mesh: Mesh, causal: bool = True, sm_scale: Optional[float] = None, vmem_limit_bytes: int | None = None, use_attention_bias: bool = False, batch_axis="data", head_axis="model")` — [`L65`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/attention_interface.py#L65)
- `sharded_paged_attention(mesh: Mesh, attn_logits_soft_cap: Optional[float] = None)` — [`L162`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/attention_interface.py#L162) — Shards GQA PagedAttention along KV heads.
- `sharded_ragged_paged_attention(mesh: Mesh, q: jax.Array, k: jax.Array, v: jax.Array, kv_cache: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, distribution: jax.Array, attention_sink: jax.Array | None, sm_scale: float, attention_chunk_size: int | None = None, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, update_kv_cache: bool = True)` — [`L370`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/attention_interface.py#L370) — Shards along KV heads.
- `sharded_splash_attention(mesh: Mesh, window_size: Optional[int] = None, attn_logits_soft_cap: Optional[float] = None, is_mqa: bool = False)` — [`L343`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/attention_interface.py#L343)
- `update_cache(is_prefill, cache, indices, operand, prefill_seq_len=None, sliding_window=None)` — [`L254`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/attention_interface.py#L254)

## Module values
- `MAX_ALLOWED_PAGE_INDICES_N` — [`L44`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/attention_interface.py#L44)
- `get_kv_cache_shape` — [`L59`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/attention_interface.py#L59)
- `get_kv_cache_shape_hd64` — [`L62`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/attention_interface.py#L62)
- `logger` — [`L42`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/attention_interface.py#L42)
- `ragged_paged_attention` — [`L58`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/attention_interface.py#L58)
- `ragged_paged_attention_hd64` — [`L61`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/attention_interface.py#L61)


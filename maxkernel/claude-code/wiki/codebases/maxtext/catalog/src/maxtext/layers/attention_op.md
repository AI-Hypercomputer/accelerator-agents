---
title: 'Module: src/maxtext/layers/attention_op.py'
type: catalog
provenance: extracted
module: src/maxtext/layers/attention_op.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.layers.attention_op`/
symbols:
  AttentionOp.config: AttentionOp#config.
  AttentionOp.use_base2_exp: AttentionOp#use_base2_exp.
  AttentionOp.tpu_flash_attention: AttentionOp#tpu_flash_attention().
  AttentionOp.create_sa_config: AttentionOp#create_sa_config().
  AttentionOp.apply_attention_dot: AttentionOp#apply_attention_dot().
  AttentionOp.cudnn_flash_attention: AttentionOp#cudnn_flash_attention().
  AttentionOp.max_logits: AttentionOp#max_logits.
  AttentionOp.__call__: AttentionOp#__call__().
  AttentionOp._logical_to_mesh_axes: AttentionOp#_logical_to_mesh_axes().
  AttentionOp.__init__: AttentionOp#__init__().
  AttentionOp.AqtEinsum_3: AttentionOp#AqtEinsum_3.
  AttentionOp.attention_type: AttentionOp#attention_type.
  AttentionOp.wrap_flash_attention: AttentionOp#wrap_flash_attention().
  AttentionOp.generate_attention_mask: AttentionOp#generate_attention_mask().
  AttentionOp.apply_attention: AttentionOp#apply_attention().
  AttentionOp.mesh: AttentionOp#mesh.
  AttentionOp._maybe_shard_with_pspec: AttentionOp#_maybe_shard_with_pspec().
  AttentionOp.gpu_ragged_attention: AttentionOp#gpu_ragged_attention().
  AttentionOp.tpu_ragged_attention: AttentionOp#tpu_ragged_attention().
  AttentionOp.qk_product: AttentionOp#qk_product().
  AttentionOp.wv_product: AttentionOp#wv_product().
  AttentionOp.attention_kernel: AttentionOp#attention_kernel.
  AttentionOp.compute_axis_order: AttentionOp#compute_axis_order.
  AttentionOp.dtype: AttentionOp#dtype.
  AttentionOp.kv_quant: AttentionOp#kv_quant.
  AttentionOp.compute_local_attention: AttentionOp#compute_local_attention().
  AttentionOp.sliding_window_size: AttentionOp#sliding_window_size.
  AttentionOp.is_partition_in_decode: AttentionOp#is_partition_in_decode().
  LoadBalancedCausalMask.__init__: LoadBalancedCausalMask#__init__().
  AttentionOp.AqtEinsum_0: AttentionOp#AqtEinsum_0.
  AttentionOp.AqtEinsum_1: AttentionOp#AqtEinsum_1.
  AttentionOp.AqtEinsum_2: AttentionOp#AqtEinsum_2.
  AttentionOp.block_q: AttentionOp#block_q.
  AttentionOp.block_kv: AttentionOp#block_kv.
  AttentionOp.use_fused_bwd_kernel: AttentionOp#use_fused_bwd_kernel.
  ChunkedCausalMask.chunk_size: ChunkedCausalMask#chunk_size.
  AttentionOp.max_target_length: AttentionOp#max_target_length.
  AttentionOp.generate_moba_mask_single_item: AttentionOp#generate_moba_mask_single_item().
  LoadBalancedCausalMask.offset: LoadBalancedCausalMask#offset.
  AttentionOp.block_kv_compute: AttentionOp#block_kv_compute.
  AttentionOp.block_q_dkv: AttentionOp#block_q_dkv.
  AttentionOp.block_kv_dkv: AttentionOp#block_kv_dkv.
  AttentionOp.block_kv_dkv_compute: AttentionOp#block_kv_dkv_compute.
  AttentionOp.q_layout: AttentionOp#q_layout.
  AttentionOp.k_layout: AttentionOp#k_layout.
  AttentionOp.v_layout: AttentionOp#v_layout.
  ChunkedCausalMask.__init__: ChunkedCausalMask#__init__().
  AttentionOp.num_kv_heads: AttentionOp#num_kv_heads.
  LoadBalancedCausalMask.__eq__: LoadBalancedCausalMask#__eq__().
  LoadBalancedCausalMask.__hash__: LoadBalancedCausalMask#__hash__().
  AttentionOp.block_q_dq: AttentionOp#block_q_dq.
  AttentionOp.block_kv_dq: AttentionOp#block_kv_dq.
  AttentionOp.use_splash_scheduler: AttentionOp#use_splash_scheduler.
  AttentionOp.fuse_reciprocal: AttentionOp#fuse_reciprocal.
  AttentionOp.wrap_ragged_attention: AttentionOp#wrap_ragged_attention().
  AttentionOp.attn_logits_soft_cap: AttentionOp#attn_logits_soft_cap.
  AttentionOp.maybe_create_nnx: AttentionOp#maybe_create_nnx().
  ChunkedCausalMask.chunked_causal_mask_function: ChunkedCausalMask#chunked_causal_mask_function().
  _make_bidirectional_block_mask: _make_bidirectional_block_mask().
  AttentionOp.calculate_moba_gate_logic: AttentionOp#calculate_moba_gate_logic().
  AttentionOp._generate_moba_mask: AttentionOp#_generate_moba_mask().
  AttentionOp.cudnn_jax_flash_attention: AttentionOp#cudnn_jax_flash_attention().
  LoadBalancedCausalMask.causal_mask_function: LoadBalancedCausalMask#causal_mask_function().
  validate_gpu_flash_attention: validate_gpu_flash_attention().
  AttentionOp.num_query_heads: AttentionOp#num_query_heads.
  AttentionOp.reshape_q: AttentionOp#reshape_q.
  AttentionOp.dropout_rate: AttentionOp#dropout_rate.
  AttentionOp.chunk_attn_window_size: AttentionOp#chunk_attn_window_size.
  AttentionOp.wrap_splash_kernel: AttentionOp#wrap_splash_kernel().
  LoadBalancedCausalMask.shape: LoadBalancedCausalMask#shape.
  ChunkedCausalMask.__eq__: ChunkedCausalMask#__eq__().
  ChunkedCausalMask.__hash__: ChunkedCausalMask#__hash__().
  attention_op_as_linen: attention_op_as_linen().
  LoadBalancedLocalMask.__init__: LoadBalancedLocalMask#__init__().
  apply_mask_to_logits: apply_mask_to_logits().
  AttentionOp.max_prefill_predict_length: AttentionOp#max_prefill_predict_length.
  AttentionOp.float32_logits: AttentionOp#float32_logits.
  AttentionOp.cache_logical_axis_names: AttentionOp#cache_logical_axis_names.
  AttentionOp.ragged_lengths_names: AttentionOp#ragged_lengths_names.
  AttentionOp.use_ragged_attention: AttentionOp#use_ragged_attention.
  AttentionOp._sequence_descriptor: AttentionOp#_sequence_descriptor().
  _load_balanced_q_sequence: _load_balanced_q_sequence().
  validate_compute_axis_order: validate_compute_axis_order().
  ChunkedCausalMask: ChunkedCausalMask#
  _generate_chunk_attention_mask: _generate_chunk_attention_mask().
  _make_block_mask_indices: _make_block_mask_indices().
  AttentionOp.float32_qk_product: AttentionOp#float32_qk_product.
  AttentionOp.flash_axis_names_q: AttentionOp#flash_axis_names_q.
  AttentionOp.flash_axis_names_kv: AttentionOp#flash_axis_names_kv.
  AttentionOp.flash_axis_names_splash_kernel: AttentionOp#flash_axis_names_splash_kernel.
  AttentionOp.ragged_block_size: AttentionOp#ragged_block_size.
  AttentionOp.rngs: AttentionOp#rngs.
  AttentionOp.check_attention_inputs: AttentionOp#check_attention_inputs().
  AttentionOp.dynamic_mask_splash_kernel: AttentionOp#dynamic_mask_splash_kernel().
  AttentionOp.kernel_fn: AttentionOp#kernel_fn().
  AttentionOp.reverse_transepose: AttentionOp#reverse_transepose().
  AttentionOp.normalize_cudnn_attention: AttentionOp#normalize_cudnn_attention().
  AttentionOp.normalize_attention: AttentionOp#normalize_attention().
  AttentionOp: AttentionOp#
  LoadBalancedCausalMask: LoadBalancedCausalMask#
  LoadBalancedLocalMask: LoadBalancedLocalMask#
  dynamic_vector_slice_in_dim: dynamic_vector_slice_in_dim.
  AttentionOp.prefill_cache_logical_axis_names: AttentionOp#prefill_cache_logical_axis_names.
  AttentionOp.cache_scale_logical_axis_names: AttentionOp#cache_scale_logical_axis_names.
  AttentionOp.ragged_qkv_axis_names: AttentionOp#ragged_qkv_axis_names.
  AttentionOp.key_axis_order: AttentionOp#key_axis_order.
  AttentionOp.quant: AttentionOp#quant.
  LoadBalancedCausalMask.cp_size: LoadBalancedCausalMask#cp_size.
---
# Module: [`src/maxtext/layers/attention_op.py`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py)

## Classes
### `AttentionOp`
- def: [`src/maxtext/layers/attention_op.py:369`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L369)
- members:
  - `__init__(self, config: Config, mesh: Mesh, attention_kernel: str, max_target_length: int, num_query_heads: int, num_kv_heads: int, float32_qk_product: bool = False, max_prefill_predict_length: int = -1, float32_logits: bool = False, flash_axis_names_q: AxisNames = (BATCH_ATTN, HEAD, LENGTH, D_KV), flash_axis_names_kv: AxisNames = (BATCH_ATTN, HEAD, KV_LENGTH, D_KV), flash_axis_names_splash_kernel: AxisNames = (HEAD, LENGTH), prefill_cache_logical_axis_names: AxisNames = (CACHE_BATCH_PREFILL, CACHE_SEQUENCE, CACHE_HEADS, CACHE_KV), cache_logical_axis_names: AxisNames = (CACHE_BATCH, CACHE_SEQUENCE, CACHE_HEADS, CACHE_KV), cache_scale_logical_axis_names: AxisNames = (CACHE_SCALE_BATCH, CACHE_SCALE_SEQUENCE, CACHE_SCALE_HEADS, CACHE_SCALE_KV), ragged_qkv_axis_names: AxisNames = (CACHE_BATCH, CACHE_HEADS, CACHE_SEQUENCE, CACHE_KV), ragged_lengths_names: AxisNames = (CACHE_BATCH,), compute_axis_order: AxisIdxes = (0, 1, 2, 3), key_axis_order: AxisIdxes = (2, 0, 1, 3), reshape_q: bool = False, dropout_rate: float = 0, dtype: DType = jnp.float32, quant: Optional[Quant] = None, kv_quant: Optional[KVQuant] = None, attention_type: AttentionType = AttentionType.GLOBAL, attn_logits_soft_cap: float | None = None, sliding_window_size: int | None = None, chunk_attn_window_size: int | None = None, use_ragged_attention: bool = False, ragged_block_size: int = 256, rngs: nnx.Rngs | None = None)` — [`L372`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L372) — Initializes the AttentionOp module. — documented in [maxtext-layers-attention_op](../../../../concepts/maxtext-layers-attention_op.md)
  - `_generate_moba_mask(self, query: Array, key: Array, q_positions: Array)` — [`L901`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L901) — Builds the token-level MoBA additive mask for the whole batch.
  - `apply_attention(self, query: Array, key: Array | KVTensor, value: Array | KVTensor, decoder_segment_ids: Array | None, segment_positions: Array | None, lengths: Array | None, model_mode: str, use_ragged_attention: bool = False, previous_chunk: Any = None, bidirectional_mask: Any = None, sinks: Array | None = None, indexer_mask: Array | None = None, compressed_mask: Optional[Array] = None, record_max_logits: bool = False, *, qk_product_einsum: Callable[..., Array], wv_product_einsum: Callable[..., Array])` — [`L922`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L922) — Apply attention
  - `apply_attention_dot(self, query: Array, key: Array | KVTensor, value: Array | KVTensor, decoder_segment_ids: Array | None, model_mode: str = MODEL_MODE_TRAIN, previous_chunk: Any = None, bidirectional_mask: Any = None, sinks: Array | None = None, indexer_mask: Array | None = None, compressed_mask: Optional[Array] = None, record_max_logits: bool = False, *, qk_product_einsum: Callable[..., Array], wv_product_einsum: Callable[..., Array])` — [`L1810`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L1810) — Apply Attention. — documented in [maxtext-layers-attention_op](../../../../concepts/maxtext-layers-attention_op.md)
  - `calculate_moba_gate_logic(self, q_item, k_item, q_pos_item)` — [`L789`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L789) — Computes the block-level MoBA gating intermediates for one batch item. — documented in [maxtext-layers-attention_op](../../../../concepts/maxtext-layers-attention_op.md)
  - `check_attention_inputs(self, query: Array, key: Array | KVTensor, value: Array | KVTensor)` — [`L588`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L588) — Check attention inputs.
  - `compute_local_attention(self, attn_weights: Array, value: Array | KVTensor, q_seq_len: int, model_mode: str, wv_product_einsum: Callable[..., Array], sinks: Array | None = None)` — [`L1741`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L1741) — Computes the attention of a local subset of the kv cache.
  - `create_sa_config(config, query, key, attn_logits_soft_cap)` — [`L1219`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L1219) — documented in [maxtext-layers-attention_op](../../../../concepts/maxtext-layers-attention_op.md)
  - `cudnn_flash_attention(self, query: Array, key: Array, value: Array, decoder_segment_ids: Array | None, segment_positions: Array | None, model_mode: str = MODEL_MODE_TRAIN)` — [`L1562`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L1562) — CUDNN Flash Attention with Transformer Engine. — documented in [maxtext-layers-attention_op](../../../../concepts/maxtext-layers-attention_op.md)
  - `cudnn_jax_flash_attention(self, query: Array, key: Array, value: Array, decoder_segment_ids: Array | None, model_mode: str = MODEL_MODE_TRAIN)` — [`L1693`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L1693) — CUDNN Flash Attention with JAX SDPA API.
  - `dynamic_mask_splash_kernel(q, k, v, segment, sinks, indexer_mask)` — [`L1458`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L1458) — documented in [maxtext-layers-attention_op](../../../../concepts/maxtext-layers-attention_op.md)
  - `generate_attention_mask(self, query, key, decoder_segment_ids: Array | None, model_mode: str, previous_chunk: Any = None, bidirectional_mask: Any = None, compressed_mask: Optional[Array] = None)` — [`L609`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L609) — Generates a combined attention mask for Transformer models.
  - `generate_moba_mask_single_item(self, q_item, k_item, q_positions)` — [`L876`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L876) — Generates the token-level MoBA additive mask for a single batch item. — documented in [maxtext-layers-attention_op](../../../../concepts/maxtext-layers-attention_op.md)
  - `gpu_ragged_attention(self, q: Array, k: Array | KVTensor, v: Array | KVTensor, lengths: Array, block_size: int)` — [`L1080`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L1080) — gpu ragged attention
  - `is_partition_in_decode(self, seq_len)` — [`L1807`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L1807) — documented in [maxtext-layers-attention_op](../../../../concepts/maxtext-layers-attention_op.md)
  - `kernel_fn(q, k, v, d, s)` — [`L1486`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L1486) — documented in [maxtext-layers-attention_op](../../../../concepts/maxtext-layers-attention_op.md)
  - `maybe_create_nnx(einsum, *args)` — [`L523`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L523)
  - `normalize_attention(self, local_outs, local_maxes, local_sums)` — [`L2056`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L2056) — Normalize across multiple localized attentions
  - `normalize_cudnn_attention(self, local_outs, local_stats)` — [`L2033`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L2033) — Normalize across two cuDNN attentions
  - `qk_product(self, query: Array, key: Array | KVTensor, q_seq_len: int, model_mode: str, einsum: Callable[..., Array])` — [`L1930`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L1930) — Query-Key product. — documented in [maxtext-layers-attention_op](../../../../concepts/maxtext-layers-attention_op.md)
  - `reverse_transepose(self, transposed_array, transpose_axis_order)` — [`L2030`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L2030)
  - `tpu_flash_attention(self, query: Array, key: Array, value: Array, decoder_segment_ids: Array | None, attn_logits_soft_cap: float | None = None, sinks: Array | None = None, indexer_mask: Array | None = None, record_max_logits: bool = False)` — [`L1179`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L1179) — TPU Flash Attention. — documented in [maxtext-layers-attention_op](../../../../concepts/maxtext-layers-attention_op.md)
  - `tpu_ragged_attention(self, query: Array, key: Array | KVTensor, value: Array | KVTensor, lengths: Array, block_size: int)` — [`L1144`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L1144) — Ragged Attention.
  - `wrap_flash_attention(query, key, value, decoder_segment_ids_q, decoder_segment_ids_kv, sa_config, splash_kernel, cp_size, load_balanced_context_parallel, sinks, indexer_mask)` — [`L1416`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L1416) — documented in [maxtext-layers-attention_op](../../../../concepts/maxtext-layers-attention_op.md)
  - `wrap_ragged_attention(q: Array, k: Array, v: Array, lengths: Array, block_size: int)` — [`L1107`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L1107) — Wraps the GQA function with appropriate sharding.
  - `wrap_splash_kernel(single_head_mask)` — [`L1323`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L1323) — documented in [maxtext-layers-attention_op](../../../../concepts/maxtext-layers-attention_op.md)
  - `wv_product(self, attn_weights: Array, value: Array | KVTensor, model_mode: str, einsum: Callable[..., Array])` — [`L1981`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L1981) — weighted value product. — documented in [maxtext-layers-attention_op](../../../../concepts/maxtext-layers-attention_op.md)
  - `AqtEinsum_0` — [`L553`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L553)
  - `AqtEinsum_1` — [`L559`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L559)
  - `AqtEinsum_2` — [`L566`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L566)
  - `AqtEinsum_3` — [`L572`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L572)
  - `attention_kernel` — [`L459`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L459) — documented in [maxtext-layers-attention_op](../../../../concepts/maxtext-layers-attention_op.md)
  - `attention_type` — [`L481`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L481) — documented in [maxtext-layers-attention_op](../../../../concepts/maxtext-layers-attention_op.md)
  - `attn_logits_soft_cap` — [`L516`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L516)
  - `block_kv` — [`L486`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L486) — documented in [maxtext-layers-attention_op](../../../../concepts/maxtext-layers-attention_op.md)
  - `block_kv_compute` — [`L487`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L487) — documented in [maxtext-layers-attention_op](../../../../concepts/maxtext-layers-attention_op.md)
  - `block_kv_dkv` — [`L489`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L489) — documented in [maxtext-layers-attention_op](../../../../concepts/maxtext-layers-attention_op.md)
  - `block_kv_dkv_compute` — [`L490`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L490) — documented in [maxtext-layers-attention_op](../../../../concepts/maxtext-layers-attention_op.md)
  - `block_kv_dq` — [`L492`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L492) — documented in [maxtext-layers-attention_op](../../../../concepts/maxtext-layers-attention_op.md)
  - `block_q` — [`L485`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L485) — documented in [maxtext-layers-attention_op](../../../../concepts/maxtext-layers-attention_op.md)
  - `block_q_dkv` — [`L488`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L488) — documented in [maxtext-layers-attention_op](../../../../concepts/maxtext-layers-attention_op.md)
  - `block_q_dq` — [`L491`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L491) — documented in [maxtext-layers-attention_op](../../../../concepts/maxtext-layers-attention_op.md)
  - `cache_logical_axis_names` — [`L470`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L470)
  - `cache_scale_logical_axis_names` — [`L471`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L471)
  - `chunk_attn_window_size` — [`L518`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L518) — documented in [maxtext-layers-attention_op](../../../../concepts/maxtext-layers-attention_op.md)
  - `compute_axis_order` — [`L474`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L474)
  - `config` — [`L457`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L457) — documented in [maxtext-layers-attention_op](../../../../concepts/maxtext-layers-attention_op.md)
  - `dropout_rate` — [`L477`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L477)
  - `dtype` — [`L478`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L478)
  - `flash_axis_names_kv` — [`L467`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L467) — documented in [maxtext-layers-attention_op](../../../../concepts/maxtext-layers-attention_op.md)
  - `flash_axis_names_q` — [`L466`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L466) — documented in [maxtext-layers-attention_op](../../../../concepts/maxtext-layers-attention_op.md)
  - `flash_axis_names_splash_kernel` — [`L468`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L468) — documented in [maxtext-layers-attention_op](../../../../concepts/maxtext-layers-attention_op.md)
  - `float32_logits` — [`L465`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L465)
  - `float32_qk_product` — [`L463`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L463)
  - `fuse_reciprocal` — [`L498`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L498) — documented in [maxtext-layers-attention_op](../../../../concepts/maxtext-layers-attention_op.md)
  - `k_layout` — [`L495`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L495) — documented in [maxtext-layers-attention_op](../../../../concepts/maxtext-layers-attention_op.md)
  - `key_axis_order` — [`L475`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L475)
  - `kv_quant` — [`L480`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L480)
  - `max_logits` — [`L960`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L960) — documented in [maxtext-layers-attention_op](../../../../concepts/maxtext-layers-attention_op.md)
  - `max_prefill_predict_length` — [`L464`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L464)
  - `max_target_length` — [`L460`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L460)
  - `mesh` — [`L458`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L458) — documented in [maxtext-layers-attention_op](../../../../concepts/maxtext-layers-attention_op.md)
  - `num_kv_heads` — [`L462`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L462)
  - `num_query_heads` — [`L461`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L461)
  - `prefill_cache_logical_axis_names` — [`L469`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L469)
  - `q_layout` — [`L494`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L494) — documented in [maxtext-layers-attention_op](../../../../concepts/maxtext-layers-attention_op.md)
  - `quant` — [`L479`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L479)
  - `ragged_block_size` — [`L520`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L520)
  - `ragged_lengths_names` — [`L473`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L473)
  - `ragged_qkv_axis_names` — [`L472`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L472)
  - `reshape_q` — [`L476`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L476)
  - `rngs` — [`L521`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L521)
  - `sliding_window_size` — [`L517`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L517) — documented in [maxtext-layers-attention_op](../../../../concepts/maxtext-layers-attention_op.md)
  - `use_base2_exp` — [`L499`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L499) — documented in [maxtext-layers-attention_op](../../../../concepts/maxtext-layers-attention_op.md)
  - `use_fused_bwd_kernel` — [`L493`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L493) — documented in [maxtext-layers-attention_op](../../../../concepts/maxtext-layers-attention_op.md)
  - `use_ragged_attention` — [`L519`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L519)
  - `use_splash_scheduler` — [`L497`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L497) — documented in [maxtext-layers-attention_op](../../../../concepts/maxtext-layers-attention_op.md)
  - `v_layout` — [`L496`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L496) — documented in [maxtext-layers-attention_op](../../../../concepts/maxtext-layers-attention_op.md)
- protocol/private: `__call__`[`L2082`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L2082), `_logical_to_mesh_axes`[`L584`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L584), `_maybe_shard_with_pspec`[`L599`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L599), `_sequence_descriptor`[`L1611`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L1611)
- uses (calls/refs, reference-scoped): [`_make_bidirectional_block_mask`](attention_op.md#_make_bidirectional_block_mask), [`validate_gpu_flash_attention`](attention_op.md#validate_gpu_flash_attention), [`apply_mask_to_logits`](attention_op.md#apply_mask_to_logits), [`ChunkedCausalMask`](attention_op.md#ChunkedCausalMask), [`LoadBalancedCausalMask`](attention_op.md#LoadBalancedCausalMask), [`LoadBalancedLocalMask`](attention_op.md#LoadBalancedLocalMask), [`_generate_chunk_attention_mask`](attention_op.md#_generate_chunk_attention_mask), [`validate_compute_axis_order`](attention_op.md#validate_compute_axis_order)
- used by: [`attention_op_as_linen`](attention_op.md#attention_op_as_linen)

### `ChunkedCausalMask`  ·  implements/extends _ComputableMask
- def: [`src/maxtext/layers/attention_op.py:133`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L133) — documented in [maxtext-layers-attention_op](../../../../concepts/maxtext-layers-attention_op.md)
- doc: Lazy chunked causal mask.
- signature: `class ChunkedCausalMask(splash_attention_mask._ComputableMask):`
- members:
  - `chunked_causal_mask_function(q_ids, kv_ids)` — [`L162`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L162) — Computes the mask logic for the given slice indices.
  - `chunk_size` — [`L149`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L149)
- protocol/private: `__eq__`[`L185`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L185), `__hash__`[`L195`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L195), `__init__`[`L151`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L151)
- used by: [`tpu_flash_attention`](attention_op.md#AttentionOp.tpu_flash_attention)

### `LoadBalancedCausalMask`
- def: [`src/maxtext/layers/attention_op.py:2182`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L2182) — documented in [maxtext-layers-attention_op](../../../../concepts/maxtext-layers-attention_op.md)
- members:
  - `causal_mask_function(q_ids, kv_ids)` — [`L2205`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L2205)
  - `cp_size` — [`L2194`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L2194)
  - `offset` — [`L2192`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L2192)
  - `shape` — [`L2193`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L2193)
- protocol/private: `__eq__`[`L2220`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L2220), `__hash__`[`L2226`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L2226), `__init__`[`L2196`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L2196)
- uses (calls/refs, reference-scoped): [`_load_balanced_q_sequence`](attention_op.md#_load_balanced_q_sequence)
- used by: [`tpu_flash_attention`](attention_op.md#AttentionOp.tpu_flash_attention)

### `LoadBalancedLocalMask`
- def: [`src/maxtext/layers/attention_op.py:2237`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L2237) — documented in [maxtext-layers-attention_op](../../../../concepts/maxtext-layers-attention_op.md)
- protocol/private: `__init__`[`L2240`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L2240)
- uses (calls/refs, reference-scoped): [`_load_balanced_q_sequence`](attention_op.md#_load_balanced_q_sequence)
- used by: [`tpu_flash_attention`](attention_op.md#AttentionOp.tpu_flash_attention)

## Functions
- `_generate_chunk_attention_mask(mask_shape: tuple[int, int], chunk_size: int, q_offset: int = 0)` — [`L206`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L206) — Generates an explicit boolean mask for chunked causal attention.
- `_load_balanced_q_sequence(shape: tuple[int, int], cp_size: int)` — [`L2175`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L2175) — Reorders query positions the same way as load-balanced input tokens.
- `_make_bidirectional_block_mask(bidirectional_mask)` — [`L252`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L252) — Creates bidirectional block mask from bidirectional_mask, where True corresponds to image tokens.
- `_make_block_mask_indices(bidirectional_mask)` — [`L236`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L236) — Creates block mask identifying segments based on a bidirectional mask.
- `apply_mask_to_logits(logits: Array, mask: Array)` — [`L95`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L95) — Applies a floating-point mask to a set of logits.
- `attention_op_as_linen(*, config: Config, mesh: Mesh, attention_kernel: str, max_target_length: int, num_query_heads: int, num_kv_heads: int, float32_qk_product: bool = False, max_prefill_predict_length: int = -1, float32_logits: bool = False, flash_axis_names_q: AxisNames = (BATCH_ATTN, HEAD, LENGTH, D_KV), flash_axis_names_kv: AxisNames = (BATCH_ATTN, HEAD, KV_LENGTH, D_KV), flash_axis_names_splash_kernel: AxisNames = (HEAD, LENGTH), prefill_cache_logical_axis_names: AxisNames = (CACHE_BATCH_PREFILL, CACHE_SEQUENCE, CACHE_HEADS, CACHE_KV), cache_logical_axis_names: AxisNames = (CACHE_BATCH, CACHE_SEQUENCE, CACHE_HEADS, CACHE_KV), cache_scale_logical_axis_names: AxisNames = (CACHE_SCALE_BATCH, CACHE_SCALE_SEQUENCE, CACHE_SCALE_HEADS, CACHE_SCALE_KV), ragged_qkv_axis_names: AxisNames = (CACHE_BATCH, CACHE_HEADS, CACHE_SEQUENCE, CACHE_KV), ragged_lengths_names: AxisNames = (CACHE_BATCH,), compute_axis_order: AxisIdxes = (0, 1, 2, 3), key_axis_order: AxisIdxes = (2, 0, 1, 3), reshape_q: bool = False, dropout_rate: float = 0, dtype: DType = jnp.float32, quant: Optional[Quant] = None, kv_quant: Optional[KVQuant] = None, attention_type: AttentionType = AttentionType.GLOBAL, attn_logits_soft_cap: float | None = None, sliding_window_size: int | None = None, chunk_attn_window_size: int | None = None, use_ragged_attention: bool = False, ragged_block_size: int = 256)` — [`L275`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L275) — A factory function to create an AttentionOp as a Linen module.
- `validate_compute_axis_order(s: AxisIdxes)` — [`L86`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L86)
- `validate_gpu_flash_attention(sinks: Array | None, record_max_logits: bool)` — [`L124`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L124) — Helper function to check for unsupported features with flash attention on GPU.

## Module values
- `dynamic_vector_slice_in_dim` — [`L83`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_op.py#L83)


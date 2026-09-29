---
title: 'Module: src/maxtext/layers/attention_mla.py'
type: catalog
provenance: extracted
module: src/maxtext/layers/attention_mla.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.layers.attention_mla`/
symbols:
  MLA.mla_query_projection: MLA#mla_query_projection().
  Indexer.__call__: Indexer#__call__().
  Indexer.wq_b: Indexer#wq_b.
  Indexer.wk: Indexer#wk.
  MLA.softmax_scale: MLA#softmax_scale.
  MLA.__call__: MLA#__call__().
  Indexer.weights_proj: Indexer#weights_proj.
  MLA.mla_kv_projection: MLA#mla_kv_projection().
  MLA._init_projections: MLA#_init_projections().
  Indexer.config: Indexer#config.
  MLA.qk_head_dim: MLA#qk_head_dim.
  MLA.IndexerKVCache_0: MLA#IndexerKVCache_0.
  MLA.wkv_b: MLA#wkv_b.
  MLA.init_mla_kv_caches: MLA#init_mla_kv_caches().
  MLA.mla_get_key_value: MLA#mla_get_key_value().
  Indexer.apply_partial_rope: Indexer#apply_partial_rope().
  Indexer.k_norm: Indexer#k_norm.
  MLA.kv_lora_rank: MLA#kv_lora_rank.
  MLA.qk_nope_head_dim: MLA#qk_nope_head_dim.
  MLA.wq_b: MLA#wq_b.
  MLA.wkv_a: MLA#wkv_a.
  MLA.update_mla_kv_caches: MLA#update_mla_kv_caches().
  Indexer.head_dim: Indexer#head_dim.
  MLA.q_lora_rank: MLA#q_lora_rank.
  MLA.qk_rope_head_dim: MLA#qk_rope_head_dim.
  DLL: DLL.
  MLA.use_indexer: MLA#use_indexer.
  MLA.indexer: MLA#indexer.
  PLACEHOLDER_SEQ_LEN: PLACEHOLDER_SEQ_LEN.
  Indexer.dtype: Indexer#dtype.
  Indexer.n_heads: Indexer#n_heads.
  Indexer.softmax_scale: Indexer#softmax_scale.
  Indexer.update_indexer_cache: Indexer#update_indexer_cache().
  Indexer.generate_mask: Indexer#generate_mask().
  MLA.MlaKVCache_0: MLA#MlaKVCache_0.
  MLA.init_indexer_cache: MLA#init_indexer_cache().
  MLA.query: MLA#query.
  MLA.wq_a: MLA#wq_a.
  MLA.q_norm: MLA#q_norm.
  MLA.kv_norm: MLA#kv_norm.
  Indexer.kernel_init: Indexer#kernel_init.
  Indexer.rngs: Indexer#rngs.
  Indexer.weight_dtype: Indexer#weight_dtype.
  Indexer.indexer_topk: Indexer#indexer_topk.
  MLA.v_head_dim: MLA#v_head_dim.
  mla_as_linen: mla_as_linen().
  MLA.out_head_dim: MLA#out_head_dim().
  Indexer.quant: Indexer#quant.
  Indexer.emb_dim: Indexer#emb_dim.
  MLA.max_position_embeddings: MLA#max_position_embeddings.
  MLA.original_max_position_embeddings: MLA#original_max_position_embeddings.
  MLA.mscale: MLA#mscale.
  MLA.rope_factor: MLA#rope_factor.
  Indexer: Indexer#
  Indexer.rotary_embedding: Indexer#rotary_embedding.
  Indexer.rope_head_dim: Indexer#rope_head_dim.
  Indexer.q_lora_rank: Indexer#q_lora_rank.
  MLA.calculate_indexer_loss: MLA#calculate_indexer_loss().
  MLA: MLA#
  Layout: Layout.
  Indexer.__init__: Indexer#__init__().
  Indexer.model_mode: Indexer#model_mode.
  Indexer.max_target_length: Indexer#max_target_length.
  MLA.__init__: MLA#__init__().
  MLA.out: MLA#out.
  MLA.indexer_loss: MLA#indexer_loss.
---
# Module: [`src/maxtext/layers/attention_mla.py`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py)

## Classes
### `Indexer`  ·  implements/extends Module
- def: [`src/maxtext/layers/attention_mla.py:81`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L81)
- doc: Indexer for DeepSeek Sparse Attention (DSA).
- signature: `class Indexer(nnx.Module):`
- members:
  - `__call__(self, inputs_q: Array, low_rank_q: Array, inputs_kv: Array, inputs_positions: Optional[Array | None] = None, attention_mask: Optional[Array | None] = None, decoder_segment_ids: Optional[Array | None] = None, previous_chunk: Any = None, kv_cache: Any = None, model_mode: str = MODEL_MODE_TRAIN)` — [`L245`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L245) — Computes the index score to determine the top-k relevant tokens. — documented in [maxtext-layers-attention_mla](../../../../concepts/maxtext-layers-attention_mla.md)
  - `apply_partial_rope(self, inputs: Array, inputs_positions: Optional[Array | None] = None)` — [`L199`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L199) — Applies partial RoPE to the indexer query or key — documented in [maxtext-layers-attention_mla](../../../../concepts/maxtext-layers-attention_mla.md)
  - `generate_mask(self, topk_indices, s)` — [`L224`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L224) — Creates a mask for top-k indices. — documented in [maxtext-layers-attention_mla](../../../../concepts/maxtext-layers-attention_mla.md)
  - `update_indexer_cache(self, kv_cache, k, decoder_segment_ids, model_mode, previous_chunk)` — [`L174`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L174) — Updates Indexer buffers by processing KV cache results. — documented in [maxtext-layers-attention_mla](../../../../concepts/maxtext-layers-attention_mla.md)
  - `config` — [`L106`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L106) — documented in [maxtext-layers-attention_mla](../../../../concepts/maxtext-layers-attention_mla.md)
  - `dtype` — [`L112`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L112)
  - `emb_dim` — [`L119`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L119) — documented in [maxtext-layers-attention_mla](../../../../concepts/maxtext-layers-attention_mla.md)
  - `head_dim` — [`L117`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L117) — documented in [maxtext-layers-attention_mla](../../../../concepts/maxtext-layers-attention_mla.md)
  - `indexer_topk` — [`L118`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L118) — documented in [maxtext-layers-attention_mla](../../../../concepts/maxtext-layers-attention_mla.md)
  - `k_norm` — [`L156`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L156) — documented in [maxtext-layers-attention_mla](../../../../concepts/maxtext-layers-attention_mla.md)
  - `kernel_init` — [`L109`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L109)
  - `max_target_length` — [`L114`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L114)
  - `model_mode` — [`L110`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L110)
  - `n_heads` — [`L116`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L116) — documented in [maxtext-layers-attention_mla](../../../../concepts/maxtext-layers-attention_mla.md)
  - `q_lora_rank` — [`L121`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L121)
  - `quant` — [`L108`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L108)
  - `rngs` — [`L111`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L111)
  - `rope_head_dim` — [`L120`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L120)
  - `rotary_embedding` — [`L107`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L107)
  - `softmax_scale` — [`L123`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L123) — documented in [maxtext-layers-attention_mla](../../../../concepts/maxtext-layers-attention_mla.md)
  - `weight_dtype` — [`L113`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L113)
  - `weights_proj` — [`L160`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L160) — documented in [maxtext-layers-attention_mla](../../../../concepts/maxtext-layers-attention_mla.md)
  - `wk` — [`L141`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L141) — documented in [maxtext-layers-attention_mla](../../../../concepts/maxtext-layers-attention_mla.md)
  - `wq_b` — [`L126`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L126) — documented in [maxtext-layers-attention_mla](../../../../concepts/maxtext-layers-attention_mla.md)
- protocol/private: `__init__`[`L97`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L97)
- used by: [`indexer`](attention_mla.md#MLA.indexer)

### `MLA`
- def: [`src/maxtext/layers/attention_mla.py:526`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L526)
- members:
  - `__call__(self, inputs_q: Array, inputs_kv: Array, inputs_positions: Array | None = None, decoder_segment_ids: Array | None = None, out_sharding: NamedSharding | None = None, *, model_mode: str = MODEL_MODE_TRAIN, deterministic: bool = False, previous_chunk: Any = None, slot: Optional[int] = None, bidirectional_mask: Optional[Any] = None, rope_kwargs: dict | None = None, kv_cache: Optional[Array] = None, attention_metadata: Optional[dict[str, Any]] = None)` — [`L1091`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L1091) — Forward pass for MLA, reusing `AttentionOp` for the actual attention. — documented in [maxtext-layers-attention_mla](../../../../concepts/maxtext-layers-attention_mla.md)
  - `__init__(self, config: Config, num_query_heads: int, num_kv_heads: int, head_dim: int, max_target_length: int, mesh: Mesh, attention_kernel: str, inputs_q_shape: Tuple, inputs_kv_shape: Tuple, dtype: DType = jnp.float32, weight_dtype: DType = jnp.float32, max_prefill_predict_length: int = -1, dropout_rate: float = 0, kernel_init: NdInitializer = nd_dense_init(1, "fan_in", "normal"), float32_qk_product: bool = False, float32_logits: bool = False, quant: Optional[Quant] = None, kv_quant: Optional[KVQuant] = None, attention_type: AttentionType = AttentionType.MLA, attn_logits_soft_cap: float | None = None, sliding_window_size: int | None = None, use_ragged_attention: bool = False, ragged_block_size: int = 256, use_qk_norm: bool = False, query_pre_attn_scalar: float | None = None, use_bias_in_projections: bool = False, temperature_tuning: bool = False, temperature_tuning_scale: float = 0.1, temperature_tuning_floor_scale: float = 8192, prefill_query_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, KV_HEAD, KV_HEAD_DIM), prefill_key_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, KV_HEAD, KV_HEAD_DIM), prefill_value_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, KV_HEAD, KV_HEAD_DIM), query_axis_names: AxisNames = (KV_BATCH, LENGTH, KV_HEAD, KV_HEAD_DIM), key_axis_names: AxisNames = (KV_BATCH, LENGTH, KV_HEAD, KV_HEAD_DIM), value_axis_names: AxisNames = (KV_BATCH, LENGTH, KV_HEAD, KV_HEAD_DIM), input_axis_names: AxisNames = (BATCH_ATTN, LENGTH, EMBED), out_axis_names: AxisNames = (BATCH_ATTN, LENGTH, HEAD, D_KV), prefill_input_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, EMBED), decode_input_axis_names: AxisNames = (DECODE_BATCH, DECODE_LENGTH, EMBED), prefill_out_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, HEAD, D_KV), decode_out_axis_names: AxisNames = (DECODE_BATCH, DECODE_LENGTH, HEAD, D_KV), prefill_cache_axis_order: AxisIdxes = (1, 2, 0, 3), ar_cache_axis_order: AxisIdxes = (1, 2, 0, 3), compute_axis_order: AxisIdxes = (0, 1, 2, 3), reshape_q: bool = False, is_nope_layer: bool = False, is_vision: bool = False, model_mode: str = MODEL_MODE_TRAIN, q_lora_rank: int = 0, kv_lora_rank: int = 512, qk_nope_head_dim: int = 128, qk_rope_head_dim: int = 64, v_head_dim: int = 128, max_position_embeddings: int = 4096 * 4, original_max_position_embeddings: int = 4096, mscale: float = 1, rope_factor: float = 40, name: str | None = None, rngs: Optional[nnx.Rngs] = None)` — [`L529`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L529) — Initializes the MLA module.
  - `_init_projections(self, inputs_q_shape: Tuple, inputs_kv_shape: Tuple)` — [`L721`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L721) — Initializes the MLA-specific projections.
  - `calculate_indexer_loss(self, indexer_score: Array, query: Array, key: Array, attention_mask: Optional[Array | None], indexer_mask: Array, sparse_loss: bool, scaling_factor: float)` — [`L1019`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L1019) — Calculates the indexer KL divergence loss.
  - `init_indexer_cache(self, inputs_kv_shape: Tuple)` — [`L694`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L694) — Initializes Indexer Cache.
  - `init_mla_kv_caches(self, inputs_kv_shape: Tuple)` — [`L906`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L906) — Initializes MlaKVCache.
  - `mla_get_key_value(self, low_rank_main, key_rope, model_mode)` — [`L882`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L882) — get (key,value) pair from mla
  - `mla_kv_projection(self, inputs: Array, inputs_positions: Array, decoder_segment_ids, model_mode, previous_chunk)` — [`L991`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L991) — MLA key/value projection with integrated rotary embedding.
  - `mla_query_projection(self, inputs_q: Array, inputs_positions: Array, model_mode)` — [`L837`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L837) — Query projection for MLA, e.g. includes LoRA if q_lora_rank > 0. — documented in [maxtext-layers-attention_mla](../../../../concepts/maxtext-layers-attention_mla.md)
  - `out_head_dim(self)` — [`L834`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L834)
  - `update_mla_kv_caches(self, low_rank_main, key_rope, decoder_segment_ids, model_mode, previous_chunk=None)` — [`L944`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L944) — Updates the MLA (Multi-Head Latent Attention) KV caches.
  - `IndexerKVCache_0` — [`L686`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L686)
  - `MlaKVCache_0` — [`L692`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L692)
  - `indexer` — [`L678`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L678)
  - `indexer_loss` — [`L1181`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L1181)
  - `kv_lora_rank` — [`L606`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L606)
  - `kv_norm` — [`L800`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L800)
  - `max_position_embeddings` — [`L610`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L610) — documented in [maxtext-layers-attention_mla](../../../../concepts/maxtext-layers-attention_mla.md)
  - `mscale` — [`L612`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L612) — documented in [maxtext-layers-attention_mla](../../../../concepts/maxtext-layers-attention_mla.md)
  - `original_max_position_embeddings` — [`L611`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L611) — documented in [maxtext-layers-attention_mla](../../../../concepts/maxtext-layers-attention_mla.md)
  - `out` — [`L831`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L831)
  - `q_lora_rank` — [`L605`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L605) — documented in [maxtext-layers-attention_mla](../../../../concepts/maxtext-layers-attention_mla.md)
  - `q_norm` — [`L764`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L764) — documented in [maxtext-layers-attention_mla](../../../../concepts/maxtext-layers-attention_mla.md)
  - `qk_head_dim` — [`L615`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L615) — documented in [maxtext-layers-attention_mla](../../../../concepts/maxtext-layers-attention_mla.md)
  - `qk_nope_head_dim` — [`L607`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L607) — documented in [maxtext-layers-attention_mla](../../../../concepts/maxtext-layers-attention_mla.md)
  - `qk_rope_head_dim` — [`L608`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L608) — documented in [maxtext-layers-attention_mla](../../../../concepts/maxtext-layers-attention_mla.md)
  - `query` — [`L736`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L736) — documented in [maxtext-layers-attention_mla](../../../../concepts/maxtext-layers-attention_mla.md)
  - `rope_factor` — [`L613`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L613) — documented in [maxtext-layers-attention_mla](../../../../concepts/maxtext-layers-attention_mla.md)
  - `softmax_scale` — [`L826`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L826) — documented in [maxtext-layers-attention_mla](../../../../concepts/maxtext-layers-attention_mla.md)
  - `use_indexer` — [`L671`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L671)
  - `v_head_dim` — [`L609`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L609)
  - `wkv_a` — [`L787`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L787)
  - `wkv_b` — [`L808`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L808)
  - `wq_a` — [`L751`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L751) — documented in [maxtext-layers-attention_mla](../../../../concepts/maxtext-layers-attention_mla.md)
  - `wq_b` — [`L772`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L772) — documented in [maxtext-layers-attention_mla](../../../../concepts/maxtext-layers-attention_mla.md)
- uses (calls/refs, reference-scoped): [`DLL`](attention_mla.md#DLL), [`PLACEHOLDER_SEQ_LEN`](attention_mla.md#PLACEHOLDER_SEQ_LEN), [`Indexer`](attention_mla.md#Indexer)
- used by: [`mla_as_linen`](attention_mla.md#mla_as_linen)

## Functions
- `mla_as_linen(*, config: Config, num_query_heads: int, num_kv_heads: int, head_dim: int, max_target_length: int, mesh: Mesh, attention_kernel: str, inputs_q_shape: Tuple, inputs_kv_shape: Tuple, dtype: DType = jnp.float32, weight_dtype: DType = jnp.float32, max_prefill_predict_length: int = -1, dropout_rate: float = 0, kernel_init: NdInitializer = nd_dense_init(1, "fan_in", "normal"), float32_qk_product: bool = False, float32_logits: bool = False, quant: Optional[Quant] = None, kv_quant: Optional[KVQuant] = None, attention_type: AttentionType = AttentionType.MLA, attn_logits_soft_cap: float | None = None, sliding_window_size: int | None = None, use_ragged_attention: bool = False, ragged_block_size: int = 256, use_qk_norm: bool = False, query_pre_attn_scalar: float | None = None, use_bias_in_projections: bool = False, temperature_tuning: bool = False, temperature_tuning_scale: float = 0.1, temperature_tuning_floor_scale: float = 8192, prefill_query_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, KV_HEAD, KV_HEAD_DIM), prefill_key_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, KV_HEAD, KV_HEAD_DIM), prefill_value_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, KV_HEAD, KV_HEAD_DIM), query_axis_names: AxisNames = (KV_BATCH, LENGTH, KV_HEAD, KV_HEAD_DIM), key_axis_names: AxisNames = (KV_BATCH, LENGTH, KV_HEAD, KV_HEAD_DIM), value_axis_names: AxisNames = (KV_BATCH, LENGTH, KV_HEAD, KV_HEAD_DIM), input_axis_names: AxisNames = (BATCH_ATTN, LENGTH, EMBED), out_axis_names: AxisNames = (BATCH_ATTN, LENGTH, HEAD, D_KV), prefill_input_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, EMBED), decode_input_axis_names: AxisNames = (DECODE_BATCH, DECODE_LENGTH, EMBED), prefill_out_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, HEAD, D_KV), decode_out_axis_names: AxisNames = (DECODE_BATCH, DECODE_LENGTH, HEAD, D_KV), prefill_cache_axis_order: AxisIdxes = (1, 2, 0, 3), ar_cache_axis_order: AxisIdxes = (1, 2, 0, 3), compute_axis_order: AxisIdxes = (0, 1, 2, 3), reshape_q: bool = False, is_nope_layer: bool = False, is_vision: bool = False, model_mode: str = MODEL_MODE_TRAIN, q_lora_rank: int = 0, kv_lora_rank: int = 512, qk_nope_head_dim: int = 128, qk_rope_head_dim: int = 64, v_head_dim: int = 128, max_position_embeddings: int = 4096 * 4, original_max_position_embeddings: int = 4096, mscale: float = 1, rope_factor: float = 40, name: str | None = None)` — [`L391`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L391) — A factory function to create an MLA as a Linen module.

## Module values
- `DLL` — [`L29`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L29)
- `Layout` — [`L27`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L27)
- `PLACEHOLDER_SEQ_LEN` — [`L78`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_mla.py#L78)


---
title: 'Module: src/maxtext/layers/attentions.py'
type: catalog
provenance: extracted
module: src/maxtext/layers/attentions.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.layers.attentions`/
symbols:
  Attention.config: Attention#config.
  Attention.__call__: Attention#__call__().
  Attention.attention_op: Attention#attention_op.
  Attention.compute_shared_kv: Attention#compute_shared_kv().
  Attention.init_rotary_embedding: Attention#init_rotary_embedding().
  Attention.init_query_w: Attention#init_query_w().
  Attention.init_kv_w: Attention#init_kv_w().
  Attention.init_out_w: Attention#init_out_w().
  Attention.init_kv_caches: Attention#init_kv_caches().
  Attention.init_qkv_w: Attention#init_qkv_w().
  Attention.sinks: Attention#sinks.
  Attention.key_norm: Attention#key_norm.
  Attention.rngs: Attention#rngs.
  Attention.dtype: Attention#dtype.
  Attention.head_dim: Attention#head_dim.
  Attention._maybe_shard_with_logical: Attention#_maybe_shard_with_logical.
  Attention.forward_serve_vllm: Attention#forward_serve_vllm().
  Attention.is_qwen3_hybrid: Attention#is_qwen3_hybrid.
  Attention.value_norm: Attention#value_norm.
  Attention.query_norm: Attention#query_norm.
  Attention.mesh: Attention#mesh.
  Attention.rope_max_timescale: Attention#rope_max_timescale.
  Attention.kv_projection: Attention#kv_projection().
  Attention.num_query_heads: Attention#num_query_heads.
  Attention.num_kv_heads: Attention#num_kv_heads.
  Attention.share_kv_layer: Attention#share_kv_layer.
  Attention.KVCache_0: Attention#KVCache_0.
  Attention.query: Attention#query.
  Attention.key: Attention#key.
  Attention.update_kv_caches: Attention#update_kv_caches().
  Attention.rotary_embedding: Attention#rotary_embedding.
  Attention.apply_rotary_embedding: Attention#apply_rotary_embedding().
  Attention.weight_dtype: Attention#weight_dtype.
  Attention.quant: Attention#quant.
  Attention.use_qk_norm: Attention#use_qk_norm.
  Attention.query_pre_attn_scalar: Attention#query_pre_attn_scalar.
  Attention.query_projection: Attention#query_projection().
  Attention.out_head_dim: Attention#out_head_dim().
  Attention.kernel_init: Attention#kernel_init.
  Attention.attention_type: Attention#attention_type.
  Attention.use_bias_in_projections: Attention#use_bias_in_projections.
  Attention.partial_rotary_factor: Attention#partial_rotary_factor.
  Attention.rope_type: Attention#rope_type.
  Attention.is_qwen2: Attention#is_qwen2.
  Attention._init_projections: Attention#_init_projections().
  Attention.qkv_proj: Attention#qkv_proj.
  Attention.value: Attention#value.
  Attention.out: Attention#out.
  Attention.query_init: Attention#query_init().
  Attention.qkv_projection: Attention#qkv_projection().
  Attention.out_projection: Attention#out_projection().
  L2Norm: L2Norm#
  L2Norm.eps: L2Norm#eps.
  Attention.kv_quant: Attention#kv_quant.
  Attention.share_kv_projections: Attention#share_kv_projections.
  Attention.is_nope_layer: Attention#is_nope_layer.
  Attention.use_v_norm: Attention#use_v_norm.
  Attention.convert_dense_general_inputs_shape: Attention#convert_dense_general_inputs_shape().
  L2Norm.__call__: L2Norm#__call__().
  l2_norm_as_linen: l2_norm_as_linen().
  attention_as_linen: attention_as_linen().
  Attention.max_target_length: Attention#max_target_length.
  Attention.max_prefill_predict_length: Attention#max_prefill_predict_length.
  Attention.use_ragged_attention: Attention#use_ragged_attention.
  Attention.input_axis_names: Attention#input_axis_names.
  Attention.is_vision: Attention#is_vision.
  Attention.model_mode: Attention#model_mode.
  Attention.attention_kernel: Attention#attention_kernel.
  Attention.dropout_rate: Attention#dropout_rate.
  Attention.float32_qk_product: Attention#float32_qk_product.
  Attention.float32_logits: Attention#float32_logits.
  Attention.attn_logits_soft_cap: Attention#attn_logits_soft_cap.
  Attention.sliding_window_size: Attention#sliding_window_size.
  Attention.ragged_block_size: Attention#ragged_block_size.
  Attention.temperature_tuning: Attention#temperature_tuning.
  Attention.temperature_tuning_scale: Attention#temperature_tuning_scale.
  Attention.temperature_tuning_floor_scale: Attention#temperature_tuning_floor_scale.
  Attention.prefill_query_axis_names: Attention#prefill_query_axis_names.
  Attention.prefill_key_axis_names: Attention#prefill_key_axis_names.
  Attention.prefill_value_axis_names: Attention#prefill_value_axis_names.
  Attention.query_axis_names: Attention#query_axis_names.
  Attention.key_axis_names: Attention#key_axis_names.
  Attention.value_axis_names: Attention#value_axis_names.
  Attention.out_axis_names: Attention#out_axis_names.
  Attention.prefill_input_axis_names: Attention#prefill_input_axis_names.
  Attention.decode_input_axis_names: Attention#decode_input_axis_names.
  Attention.prefill_out_axis_names: Attention#prefill_out_axis_names.
  Attention.decode_out_axis_names: Attention#decode_out_axis_names.
  Attention.prefill_cache_axis_order: Attention#prefill_cache_axis_order.
  Attention.ar_cache_axis_order: Attention#ar_cache_axis_order.
  Attention.compute_axis_order: Attention#compute_axis_order.
  Attention.reshape_q: Attention#reshape_q.
  Attention.use_mrope: Attention#use_mrope.
  Attention.mrope_section: Attention#mrope_section.
  Attention: Attention#
  L2Norm.rngs: L2Norm#rngs.
  Attention.__init__: Attention#__init__().
---
# Module: [`src/maxtext/layers/attentions.py`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py)

## Classes
### `Attention`
- def: [`src/maxtext/layers/attentions.py:227`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L227)
- members:
  - `__call__(self, inputs_q: Array, inputs_kv: Array, inputs_positions: Array | None = None, decoder_segment_ids: Array | None = None, out_sharding: NamedSharding | None = None, *, model_mode: str = MODEL_MODE_TRAIN, deterministic: bool = False, previous_chunk: Any = None, slot: Optional[int] = None, bidirectional_mask: Any = None, rope_kwargs: dict | None = None, kv_cache: Optional[Array] = None, attention_metadata: Optional[dict[str, Any]] = None, shared_key: Array | None = None, shared_value: Array | None = None)` — [`L1067`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L1067) — Applies Attention on the input data. — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `__init__(self, config: Config, num_query_heads: int, num_kv_heads: int, head_dim: int, max_target_length: int, mesh: Mesh, attention_kernel: str, inputs_q_shape: Tuple, inputs_kv_shape: Tuple, dtype: DType = jnp.float32, weight_dtype: DType = jnp.float32, max_prefill_predict_length: int = -1, dropout_rate: float = 0, kernel_init: NdInitializer = nd_dense_init(1, "fan_in", "normal"), float32_qk_product: bool = False, float32_logits: bool = False, quant: Optional[Quant] = None, kv_quant: Optional[KVQuant] = None, attention_type: AttentionType = AttentionType.GLOBAL, attn_logits_soft_cap: float | None = None, sliding_window_size: int | None = None, use_ragged_attention: bool = False, ragged_block_size: int = 256, use_qk_norm: bool = False, query_pre_attn_scalar: float | None = None, use_bias_in_projections: bool = False, share_kv_projections: bool = False, temperature_tuning: bool = False, temperature_tuning_scale: float = 0.1, temperature_tuning_floor_scale: float = 8192, prefill_query_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, KV_HEAD, KV_HEAD_DIM), prefill_key_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, KV_HEAD, KV_HEAD_DIM), prefill_value_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, KV_HEAD, KV_HEAD_DIM), query_axis_names: AxisNames = (KV_BATCH, ATTN_LENGTH, KV_HEAD, KV_HEAD_DIM), key_axis_names: AxisNames = (KV_BATCH, ATTN_LENGTH, KV_HEAD, KV_HEAD_DIM), value_axis_names: AxisNames = (KV_BATCH, ATTN_LENGTH, KV_HEAD, KV_HEAD_DIM), input_axis_names: AxisNames = (BATCH_ATTN, ATTN_LENGTH, ATTN_EMBED), out_axis_names: AxisNames = (BATCH_ATTN, ATTN_LENGTH, HEAD, D_KV), prefill_input_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, ATTN_EMBED), decode_input_axis_names: AxisNames = (DECODE_BATCH, DECODE_LENGTH, ATTN_EMBED), prefill_out_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, HEAD, D_KV), decode_out_axis_names: AxisNames = (DECODE_BATCH, DECODE_LENGTH, HEAD, D_KV), prefill_cache_axis_order: AxisIdxes = (1, 2, 0, 3), ar_cache_axis_order: AxisIdxes = (1, 2, 0, 3), compute_axis_order: AxisIdxes = (0, 1, 2, 3), reshape_q: bool = False, is_nope_layer: bool = False, is_vision: bool = False, model_mode: str = MODEL_MODE_TRAIN, base_kv_cache: bool = True, use_mrope: bool = False, mrope_section: tuple[int, int, int] | None = None, name: str | None = None, rope_type: str | None = None, use_v_norm: bool = False, rope_max_timescale: float | None = None, partial_rotary_factor: float | None = None, share_kv_layer: bool = False, rngs: nnx.Rngs | None = None)` — [`L259`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L259) — Initializes the Attention module.
  - `_init_projections(self, inputs_q_shape: Tuple, inputs_kv_shape: Tuple)` — [`L558`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L558) — Initializes the query, key, value, and output projections. — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `apply_rotary_embedding(self, inputs: Array, inputs_positions: Optional[Array | None] = None, rope_kwargs: dict | None = None)` — [`L914`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L914) — Applies rotary embeddings, handling different model types. — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `compute_shared_kv(self, inputs_kv: Array, inputs_positions: Array | None = None, rope_kwargs: dict | None = None)` — [`L735`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L735) — Computes the rotated, normed K / V for this layer. — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `convert_dense_general_inputs_shape(self, inputs_shape: tuple[int, ...] | None = None, axis: Union[Iterable[int], int] = -1)` — [`L768`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L768)
  - `forward_serve_vllm(self, query: Array, key: Array, value: Array, rpa_kv_cache: list[Array] | None = None, rpa_metadata: dict[str, Any] | None = None)` — [`L1004`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L1004) — Forward function for vLLM serving with RPA attention. — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `init_kv_caches(self, inputs_kv_shape: Tuple)` — [`L937`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L937) — Initializes KVCache. — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `init_kv_w(self, inputs_kv_shape: Tuple)` — [`L616`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L616) — Initializes the key or value projection. — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `init_out_w(self, output_dim: int)` — [`L702`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L702) — out projection — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `init_qkv_w(self, inputs_shape: Tuple)` — [`L674`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L674) — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `init_query_w(self, inputs_q_shape: Tuple)` — [`L570`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L570) — Query projection initialization. — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `init_rotary_embedding(self)` — [`L776`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L776) — Initializes the rotary embeddings, handling different model types. — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `kv_projection(self, inputs_kv: Array, proj_name: str, out_sharding: NamedSharding | None = None)` — [`L652`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L652) — Applies the key or value projection. — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `out_head_dim(self)` — [`L699`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L699)
  - `out_projection(self, out: Array, out_sharding: NamedSharding | None = None)` — [`L731`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L731) — out projection — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `qkv_projection(self, inputs: Array, proj_name: str, out_sharding: NamedSharding | None = None)` — [`L690`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L690) — Fused QKV projection — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `query_init(*args)` — [`L583`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L583)
  - `query_projection(self, inputs_q: Array, out_sharding: NamedSharding | None = None)` — [`L611`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L611) — Query projection. — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `update_kv_caches(self, key, value, decoder_segment_ids, model_mode, previous_chunk)` — [`L975`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L975) — Updates the KV caches for prefill and autoregressive modes. — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `KVCache_0` — [`L437`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L437) — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `ar_cache_axis_order` — [`L415`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L415)
  - `attention_kernel` — [`L380`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L380)
  - `attention_op` — [`L445`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L445) — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `attention_type` — [`L390`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L390) — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `attn_logits_soft_cap` — [`L391`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L391)
  - `compute_axis_order` — [`L416`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L416)
  - `config` — [`L374`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L374) — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `decode_input_axis_names` — [`L411`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L411)
  - `decode_out_axis_names` — [`L413`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L413)
  - `dropout_rate` — [`L384`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L384)
  - `dtype` — [`L381`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L381) — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `float32_logits` — [`L387`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L387)
  - `float32_qk_product` — [`L386`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L386)
  - `head_dim` — [`L377`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L377) — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `input_axis_names` — [`L408`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L408)
  - `is_nope_layer` — [`L418`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L418) — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `is_qwen2` — [`L431`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L431)
  - `is_qwen3_hybrid` — [`L432`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L432) — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `is_vision` — [`L419`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L419)
  - `kernel_init` — [`L385`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L385) — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `key` — [`L565`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L565) — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `key_axis_names` — [`L406`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L406)
  - `key_norm` — [`L505`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L505) — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `kv_quant` — [`L389`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L389) — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `max_prefill_predict_length` — [`L383`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L383) — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `max_target_length` — [`L378`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L378) — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `mesh` — [`L379`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L379) — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `model_mode` — [`L420`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L420)
  - `mrope_section` — [`L422`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L422)
  - `num_kv_heads` — [`L376`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L376) — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `num_query_heads` — [`L375`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L375)
  - `out` — [`L568`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L568) — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `out_axis_names` — [`L409`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L409)
  - `partial_rotary_factor` — [`L428`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L428) — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `prefill_cache_axis_order` — [`L414`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L414)
  - `prefill_input_axis_names` — [`L410`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L410)
  - `prefill_key_axis_names` — [`L403`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L403)
  - `prefill_out_axis_names` — [`L412`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L412)
  - `prefill_query_axis_names` — [`L402`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L402)
  - `prefill_value_axis_names` — [`L404`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L404)
  - `qkv_proj` — [`L561`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L561) — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `quant` — [`L388`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L388) — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `query` — [`L563`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L563) — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `query_axis_names` — [`L405`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L405)
  - `query_norm` — [`L494`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L494) — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `query_pre_attn_scalar` — [`L396`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L396) — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `ragged_block_size` — [`L394`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L394)
  - `reshape_q` — [`L417`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L417)
  - `rngs` — [`L423`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L423)
  - `rope_max_timescale` — [`L427`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L427) — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `rope_type` — [`L425`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L425)
  - `rotary_embedding` — [`L443`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L443) — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `share_kv_layer` — [`L429`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L429) — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `share_kv_projections` — [`L398`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L398) — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `sinks` — [`L473`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L473) — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `sliding_window_size` — [`L392`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L392)
  - `temperature_tuning` — [`L399`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L399)
  - `temperature_tuning_floor_scale` — [`L401`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L401)
  - `temperature_tuning_scale` — [`L400`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L400)
  - `use_bias_in_projections` — [`L397`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L397) — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `use_mrope` — [`L421`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L421)
  - `use_qk_norm` — [`L395`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L395) — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `use_ragged_attention` — [`L393`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L393) — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `use_v_norm` — [`L426`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L426) — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `value` — [`L567`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L567) — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `value_axis_names` — [`L407`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L407)
  - `value_norm` — [`L538`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L538) — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `weight_dtype` — [`L382`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L382) — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
- protocol/private: `_maybe_shard_with_logical`[`L551`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L551)
- uses (calls/refs, reference-scoped): [`L2Norm`](attentions.md#L2Norm), [`eps`](attentions.md#L2Norm.eps)
- used by: [`attention_as_linen`](attentions.md#attention_as_linen)

### `L2Norm`  ·  implements/extends Module
- def: [`src/maxtext/layers/attentions.py:77`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L77) — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
- doc: Implementation of L2Norm in JAX.
- signature: `class L2Norm(nnx.Module):`
- members:
  - `eps` — [`L85`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L85) — documented in [maxtext-layers-attentions](../../../../concepts/maxtext-layers-attentions.md)
  - `rngs` — [`L86`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L86)
- protocol/private: `__call__`[`L88`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L88)
- used by: [`__call__`](attentions.md#Attention.__call__), [`compute_shared_kv`](attentions.md#Attention.compute_shared_kv), [`l2_norm_as_linen`](attentions.md#l2_norm_as_linen)

## Functions
- `attention_as_linen(*, config: Config, num_query_heads: int, num_kv_heads: int, head_dim: int, max_target_length: int, mesh: Mesh, attention_kernel: str, inputs_q_shape: Tuple, inputs_kv_shape: Tuple, dtype: DType = jnp.float32, weight_dtype: DType = jnp.float32, max_prefill_predict_length: int = -1, dropout_rate: float = 0, kernel_init: NdInitializer = nd_dense_init(1, "fan_in", "normal"), float32_qk_product: bool = False, float32_logits: bool = False, quant: Optional[Quant] = None, kv_quant: Optional[KVQuant] = None, attention_type: AttentionType = AttentionType.GLOBAL, attn_logits_soft_cap: float | None = None, sliding_window_size: int | None = None, use_ragged_attention: bool = False, ragged_block_size: int = 256, use_qk_norm: bool = False, query_pre_attn_scalar: float | None = None, use_bias_in_projections: bool = False, share_kv_projections: bool = False, temperature_tuning: bool = False, temperature_tuning_scale: float = 0.1, temperature_tuning_floor_scale: float = 8192, prefill_query_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, KV_HEAD, KV_HEAD_DIM), prefill_key_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, KV_HEAD, KV_HEAD_DIM), prefill_value_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, KV_HEAD, KV_HEAD_DIM), query_axis_names: AxisNames = (KV_BATCH, ATTN_LENGTH, KV_HEAD, KV_HEAD_DIM), key_axis_names: AxisNames = (KV_BATCH, ATTN_LENGTH, KV_HEAD, KV_HEAD_DIM), value_axis_names: AxisNames = (KV_BATCH, ATTN_LENGTH, KV_HEAD, KV_HEAD_DIM), input_axis_names: AxisNames = (BATCH_ATTN, ATTN_LENGTH, ATTN_EMBED), out_axis_names: AxisNames = (BATCH_ATTN, ATTN_LENGTH, HEAD, D_KV), prefill_input_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, ATTN_EMBED), decode_input_axis_names: AxisNames = (DECODE_BATCH, DECODE_LENGTH, ATTN_EMBED), prefill_out_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, HEAD, D_KV), decode_out_axis_names: AxisNames = (DECODE_BATCH, DECODE_LENGTH, HEAD, D_KV), prefill_cache_axis_order: AxisIdxes = (1, 2, 0, 3), ar_cache_axis_order: AxisIdxes = (1, 2, 0, 3), compute_axis_order: AxisIdxes = (0, 1, 2, 3), reshape_q: bool = False, is_nope_layer: bool = False, is_vision: bool = False, model_mode: str = MODEL_MODE_TRAIN, use_mrope: bool = False, mrope_section: tuple[int, int, int] | None = None, name: str | None = None, rope_type: str | None = None)` — [`L102`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L102) — A factory function to create an Attention as a Linen module.
- `l2_norm_as_linen(self, eps: float = 0.000001)` — [`L92`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attentions.py#L92) — Initializes the L2Norm module and returns it as a Linen module.


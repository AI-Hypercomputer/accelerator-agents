---
title: 'Module: tpu_inference/layers/jax/attention/llama4_attention.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/jax/attention/llama4_attention.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.jax.attention.llama4_attention`/
symbols:
  Llama4VisionAttention.__call__: Llama4VisionAttention#__call__().
  Llama4Attention.__call__: Llama4Attention#__call__().
  Llama4Attention.apply_temperature_tuning: Llama4Attention#apply_temperature_tuning().
  Llama4VisionAttention.dtype: Llama4VisionAttention#dtype.
  Llama4VisionAttention.__post_init__: Llama4VisionAttention#__post_init__().
  Llama4VisionAttention.kernel_q_proj_DNH: Llama4VisionAttention#kernel_q_proj_DNH.
  Llama4VisionAttention.kernel_k_proj_DKH: Llama4VisionAttention#kernel_k_proj_DKH.
  Llama4VisionAttention.kernel_v_proj_DKH: Llama4VisionAttention#kernel_v_proj_DKH.
  Llama4VisionAttention.kernel_o_proj_NHD: Llama4VisionAttention#kernel_o_proj_NHD.
  Llama4VisionAttention.bias_q_proj_NH: Llama4VisionAttention#bias_q_proj_NH.
  Llama4VisionAttention.bias_k_proj_KH: Llama4VisionAttention#bias_k_proj_KH.
  Llama4VisionAttention.bias_v_proj_KH: Llama4VisionAttention#bias_v_proj_KH.
  Llama4VisionAttention.bias_o_proj_D: Llama4VisionAttention#bias_o_proj_D.
  Llama4Attention: Llama4Attention#
  Llama4Attention.use_qk_norm: Llama4Attention#use_qk_norm.
  Llama4VisionAttention.head_dim: Llama4VisionAttention#head_dim.
  Llama4VisionAttention.dkh_sharding: Llama4VisionAttention#dkh_sharding.
  Llama4Attention.temperature_tuning: Llama4Attention#temperature_tuning.
  Llama4Attention.temperature_tuning_floor_scale: Llama4Attention#temperature_tuning_floor_scale.
  Llama4Attention.temperature_tuning_scale: Llama4Attention#temperature_tuning_scale.
  Llama4Attention.activation_attention_td: Llama4Attention#activation_attention_td.
  Llama4Attention.activation_attention_out_td: Llama4Attention#activation_attention_out_td.
  Llama4VisionAttention.rope_input_ordering: Llama4VisionAttention#rope_input_ordering.
  logger: logger.
  L2Norm.__call__: L2Norm#__call__().
  Llama4VisionAttention: Llama4VisionAttention#
  Llama4VisionAttention.hidden_size: Llama4VisionAttention#hidden_size.
  Llama4VisionAttention.num_attention_heads: Llama4VisionAttention#num_attention_heads.
  Llama4VisionAttention.num_key_value_heads: Llama4VisionAttention#num_key_value_heads.
  Llama4VisionAttention.rope_theta: Llama4VisionAttention#rope_theta.
  Llama4VisionAttention.rope_scaling: Llama4VisionAttention#rope_scaling.
  Llama4VisionAttention.mesh: Llama4VisionAttention#mesh.
  Llama4VisionAttention.activation_attention_td: Llama4VisionAttention#activation_attention_td.
  Llama4VisionAttention.activation_attention_out_td: Llama4VisionAttention#activation_attention_out_td.
  Llama4VisionAttention.is_causal: Llama4VisionAttention#is_causal.
  Llama4VisionAttention.kv_cache_quantized_dtype: Llama4VisionAttention#kv_cache_quantized_dtype.
  Llama4VisionAttention.dnh_sharding: Llama4VisionAttention#dnh_sharding.
  Llama4VisionAttention.nhd_sharding: Llama4VisionAttention#nhd_sharding.
  Llama4VisionAttention.activation_q_td: Llama4VisionAttention#activation_q_td.
  Llama4VisionAttention.keyvalue_skh: Llama4VisionAttention#keyvalue_skh.
  L2Norm: L2Norm#
  L2Norm.eps: L2Norm#eps.
  SegmentIds: SegmentIds#
  SegmentIds.q: SegmentIds#q.
  SegmentIds.kv: SegmentIds#kv.
  Llama4VisionAttention.use_qk_norm: Llama4VisionAttention#use_qk_norm.
  Llama4VisionAttention.temperature_tuning: Llama4VisionAttention#temperature_tuning.
  Llama4VisionAttention.temperature_tuning_floor_scale: Llama4VisionAttention#temperature_tuning_floor_scale.
  Llama4VisionAttention.temperature_tuning_scale: Llama4VisionAttention#temperature_tuning_scale.
  Llama4VisionAttention.rngs: Llama4VisionAttention#rngs.
  Llama4VisionAttention.query_tnh: Llama4VisionAttention#query_tnh.
  Llama4VisionAttention._k_scale: Llama4VisionAttention#_k_scale.
  Llama4VisionAttention._v_scale: Llama4VisionAttention#_v_scale.
  L2Norm.__init__: L2Norm#__init__().
  Llama4VisionAttention._q_scale: Llama4VisionAttention#_q_scale.
---
# Module: [`tpu_inference/layers/jax/attention/llama4_attention.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py)

## Classes
### `L2Norm`  ·  implements/extends Module
- def: [`tpu_inference/layers/jax/attention/llama4_attention.py:38`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L38)
- doc: Implementation of L2 Norm in JAX (taken from MaxText repo - maxtext/MaxText/layers/attentions.py).
- signature: `class L2Norm(nnx.Module):`
- members:
  - `eps` — [`L47`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L47)
- protocol/private: `__call__`[`L49`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L49), `__init__`[`L46`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L46)
- used by: [`__call__`](llama4_attention.md#Llama4Attention.__call__)

### `Llama4Attention`  ·  implements/extends Attention
- def: [`tpu_inference/layers/jax/attention/llama4_attention.py:55`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L55)
- signature: `class Llama4Attention(Attention):`
- members:
  - `__call__(self, x, is_prefill, kv_cache: KVCache, attention_metadata: AttentionMetadata, use_attention_rope: bool = True, **kwargs)` — [`L63`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L63) — Performs the forward pass of the attention module. — documented in [tpu_inference-layers-jax-attention](../../../../../concepts/tpu_inference-layers-jax-attention.md)
  - `apply_temperature_tuning(self, md: AttentionMetadata, input_arr_TNH: jax.Array)` — [`L162`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L162) — Applies temperature tuning to the input array of shape (T, N, H).
  - `activation_attention_out_td` — [`L61`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L61)
  - `activation_attention_td` — [`L60`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L60)
  - `temperature_tuning` — [`L57`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L57)
  - `temperature_tuning_floor_scale` — [`L58`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L58)
  - `temperature_tuning_scale` — [`L59`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L59)
  - `use_qk_norm` — [`L56`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L56)
- uses (calls/refs, reference-scoped): [`AttentionMetadata`](../../common/attention_metadata.md#AttentionMetadata), [`input_positions`](../../common/attention_metadata.md#AttentionMetadata.input_positions), [`attention`](attention.md#Attention.attention), [`apply_rope`](../rope_interface.md#apply_rope), [`quantize_kv`](../../common/__init__.md#quantize_kv), [`kernel_o_proj_NHD`](attention.md#Attention.kernel_o_proj_NHD), [`kernel_k_proj_DKH`](attention.md#Attention.kernel_k_proj_DKH), [`kernel_q_proj_DNH`](attention.md#Attention.kernel_q_proj_DNH), [`kernel_v_proj_DKH`](attention.md#Attention.kernel_v_proj_DKH), [`keyvalue_skh`](attention.md#Attention.keyvalue_skh), [`KVCache`](attention.md#KVCache), [`dtype`](attention.md#Attention.dtype), [`query_tnh`](attention.md#Attention.query_tnh), [`mesh`](attention.md#Attention.mesh), [`activation_q_td`](attention.md#Attention.activation_q_td), [`rope_input_ordering`](attention.md#Attention.rope_input_ordering), [`head_dim`](attention.md#Attention.head_dim), [`kv_cache_quantized_dtype`](attention.md#Attention.kv_cache_quantized_dtype), [`rope_scaling`](attention.md#Attention.rope_scaling), [`rope_theta`](attention.md#Attention.rope_theta), [`Attention`](attention.md#Attention), [`_k_scale`](attention.md#Attention._k_scale), [`_v_scale`](attention.md#Attention._v_scale), [`L2Norm`](llama4_attention.md#L2Norm)
- used by: [`end_layer`](../../../models/jax/llama4.md#Llama4ForCausalLM.end_layer), [`end_layer`](../../../models/jax/llama_guard_4.md#LlamaGuard4ForCausalLM.end_layer), [`Attention`](attention.md#Attention)

### `Llama4VisionAttention`  ·  implements/extends Module
- def: [`tpu_inference/layers/jax/attention/llama4_attention.py:194`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L194)
- signature: `class Llama4VisionAttention(nnx.Module):`
- members:
  - `__post_init__(self, rngs: nnx.Rngs)` — [`L225`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L225) — Initializes the weight kernels for Q, K, V, and O projections.
  - `activation_attention_out_td` — [`L208`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L208)
  - `activation_attention_td` — [`L207`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L207)
  - `activation_q_td` — [`L216`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L216)
  - `bias_k_proj_KH` — [`L254`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L254)
  - `bias_o_proj_D` — [`L262`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L262)
  - `bias_q_proj_NH` — [`L250`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L250)
  - `bias_v_proj_KH` — [`L258`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L258)
  - `dkh_sharding` — [`L214`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L214)
  - `dnh_sharding` — [`L213`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L213)
  - `dtype` — [`L201`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L201)
  - `head_dim` — [`L198`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L198)
  - `hidden_size` — [`L195`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L195)
  - `is_causal` — [`L209`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L209)
  - `kernel_k_proj_DKH` — [`L237`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L237)
  - `kernel_o_proj_NHD` — [`L245`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L245)
  - `kernel_q_proj_DNH` — [`L233`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L233)
  - `kernel_v_proj_DKH` — [`L241`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L241)
  - `keyvalue_skh` — [`L218`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L218)
  - `kv_cache_quantized_dtype` — [`L210`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L210)
  - `mesh` — [`L202`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L202)
  - `nhd_sharding` — [`L215`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L215)
  - `num_attention_heads` — [`L196`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L196)
  - `num_key_value_heads` — [`L197`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L197)
  - `query_tnh` — [`L217`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L217)
  - `rngs` — [`L211`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L211)
  - `rope_input_ordering` — [`L219`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L219)
  - `rope_scaling` — [`L200`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L200)
  - `rope_theta` — [`L199`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L199)
  - `temperature_tuning` — [`L204`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L204)
  - `temperature_tuning_floor_scale` — [`L205`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L205)
  - `temperature_tuning_scale` — [`L206`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L206)
  - `use_qk_norm` — [`L203`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L203)
- protocol/private: `__call__`[`L267`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L267), `_k_scale`[`L222`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L222), `_q_scale`[`L221`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L221), `_v_scale`[`L223`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L223)
- uses (calls/refs, reference-scoped): [`AttentionMetadata`](../../common/attention_metadata.md#AttentionMetadata), [`create_param`](../base.md#create_param), [`apply_rope`](../rope_interface.md#apply_rope), [`quantize_kv`](../../common/__init__.md#quantize_kv), [`sharded_flash_attention`](../../common/attention_interface.md#sharded_flash_attention), [`KVCache`](attention.md#KVCache), [`SegmentIds`](llama4_attention.md#SegmentIds), [`kv`](llama4_attention.md#SegmentIds.kv), [`q`](llama4_attention.md#SegmentIds.q)
- used by: [`self_attn`](../../../models/jax/llama4.md#JAXLlama4VisionEncoderLayer.self_attn)

### `SegmentIds`  ·  implements/extends NamedTuple
- def: [`tpu_inference/layers/jax/attention/llama4_attention.py:176`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L176)
- doc: SegmentIds for Q and KV sequences.
- signature: `class SegmentIds(NamedTuple):`
- members:
  - `kv` — [`L190`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L190)
  - `q` — [`L189`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L189)
- used by: [`__call__`](llama4_attention.md#Llama4VisionAttention.__call__)

## Module values
- `logger` — [`L35`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/llama4_attention.py#L35)


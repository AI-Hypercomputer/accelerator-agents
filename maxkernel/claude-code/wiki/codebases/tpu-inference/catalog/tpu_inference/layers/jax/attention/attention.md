---
title: 'Module: tpu_inference/layers/jax/attention/attention.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/jax/attention/attention.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.jax.attention.attention`/
symbols:
  Attention.__call__: Attention#__call__().
  Attention.attention: Attention#attention().
  Attention.kernel_o_proj_NHD: Attention#kernel_o_proj_NHD.
  Attention.kernel_q_proj_DNH: Attention#kernel_q_proj_DNH.
  Attention.kernel_k_proj_DKH: Attention#kernel_k_proj_DKH.
  Attention.kernel_v_proj_DKH: Attention#kernel_v_proj_DKH.
  Attention.__post_init__: Attention#__post_init__().
  Attention.keyvalue_skh: Attention#keyvalue_skh.
  KVCache: KVCache.
  Attention.dtype: Attention#dtype.
  Attention.query_tnh: Attention#query_tnh.
  Attention.mesh: Attention#mesh.
  Attention.activation_q_td: Attention#activation_q_td.
  Attention.random_init: Attention#random_init.
  Attention.rope_input_ordering: Attention#rope_input_ordering.
  Attention.attn_o_tnh: Attention#attn_o_tnh.
  Attention.head_dim: Attention#head_dim.
  Attention.rope_theta: Attention#rope_theta.
  Attention.rope_scaling: Attention#rope_scaling.
  Attention.kv_cache_quantized_dtype: Attention#kv_cache_quantized_dtype.
  Attention: Attention#
  Attention.kv_cache_dtype: Attention#kv_cache_dtype.
  Attention.dkh_sharding: Attention#dkh_sharding.
  Attention._ragged_paged_attention: Attention#_ragged_paged_attention().
  Attention.hidden_size: Attention#hidden_size.
  Attention.num_attention_heads: Attention#num_attention_heads.
  Attention.num_key_value_heads: Attention#num_key_value_heads.
  Attention.dnh_sharding: Attention#dnh_sharding.
  Attention.nhd_sharding: Attention#nhd_sharding.
  Attention.rngs: Attention#rngs.
  Attention.attention_chunk_size: Attention#attention_chunk_size.
  Attention._k_scale: Attention#_k_scale.
  Attention._v_scale: Attention#_v_scale.
  Attention._q_scale: Attention#_q_scale.
---
# Module: [`tpu_inference/layers/jax/attention/attention.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/attention.py)

## Classes
### `Attention`  ·  implements/extends Module
- def: [`tpu_inference/layers/jax/attention/attention.py:39`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/attention.py#L39)
- doc: An implementation of attention.
- signature: `class Attention(nnx.Module):`
- members:
  - `__call__(self, x, is_prefill, kv_cache: KVCache, attention_metadata: AttentionMetadata, use_attention_rope: bool = True)` — [`L110`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/attention.py#L110) — Performs the forward pass of the attention module. — documented in [tpu_inference-layers-common-attention_metadata](../../../../../concepts/tpu_inference-layers-common-attention_metadata.md)
  - `__post_init__(self, rngs: nnx.Rngs)` — [`L82`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/attention.py#L82) — Initializes the weight kernels for Q, K, V, and O projections.
  - `attention(self, is_prefill: bool, kv_cache: KVCache, q_TNH: jax.Array, k_SKH: jax.Array, v_SKH: jax.Array, attention_metadata: AttentionMetadata, mesh: Mesh, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None)` — [`L189`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/attention.py#L189) — Performs scaled dot-product attention and updates the KV cache. — documented in [tpu_inference-layers-common-attention_metadata](../../../../../concepts/tpu_inference-layers-common-attention_metadata.md)
  - `activation_q_td` — [`L65`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/attention.py#L65)
  - `attention_chunk_size` — [`L73`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/attention.py#L73)
  - `attn_o_tnh` — [`L69`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/attention.py#L69) — documented in [tpu_inference-layers-jax-attention](../../../../../concepts/tpu_inference-layers-jax-attention.md)
  - `dkh_sharding` — [`L62`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/attention.py#L62)
  - `dnh_sharding` — [`L61`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/attention.py#L61)
  - `dtype` — [`L57`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/attention.py#L57)
  - `head_dim` — [`L54`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/attention.py#L54)
  - `hidden_size` — [`L51`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/attention.py#L51)
  - `kernel_k_proj_DKH` — [`L93`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/attention.py#L93)
  - `kernel_o_proj_NHD` — [`L101`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/attention.py#L101) — documented in [tpu_inference-layers-jax-attention](../../../../../concepts/tpu_inference-layers-jax-attention.md)
  - `kernel_q_proj_DNH` — [`L89`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/attention.py#L89)
  - `kernel_v_proj_DKH` — [`L97`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/attention.py#L97) — documented in [tpu_inference-layers-jax-attention](../../../../../concepts/tpu_inference-layers-jax-attention.md)
  - `keyvalue_skh` — [`L67`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/attention.py#L67)
  - `kv_cache_dtype` — [`L59`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/attention.py#L59)
  - `kv_cache_quantized_dtype` — [`L80`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/attention.py#L80) — documented in [tpu_inference-layers-jax-attention](../../../../../concepts/tpu_inference-layers-jax-attention.md)
  - `mesh` — [`L58`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/attention.py#L58)
  - `nhd_sharding` — [`L63`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/attention.py#L63)
  - `num_attention_heads` — [`L52`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/attention.py#L52)
  - `num_key_value_heads` — [`L53`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/attention.py#L53)
  - `query_tnh` — [`L66`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/attention.py#L66)
  - `random_init` — [`L72`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/attention.py#L72)
  - `rngs` — [`L70`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/attention.py#L70)
  - `rope_input_ordering` — [`L74`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/attention.py#L74)
  - `rope_scaling` — [`L56`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/attention.py#L56)
  - `rope_theta` — [`L55`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/attention.py#L55)
- protocol/private: `_k_scale`[`L77`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/attention.py#L77), `_q_scale`[`L76`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/attention.py#L76), `_ragged_paged_attention`[`L243`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/attention.py#L243), `_v_scale`[`L78`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/attention.py#L78)
- uses (calls/refs, reference-scoped): [`ShardingAxisName`](../../common/sharding.md#ShardingAxisName), [`AttentionMetadata`](../../common/attention_metadata.md#AttentionMetadata), [`create_param`](../base.md#create_param), [`input_positions`](../../common/attention_metadata.md#AttentionMetadata.input_positions), [`ragged_paged_attention`](../../../kernels/ragged_paged_attention/v3/kernel.md#ragged_paged_attention), [`apply_rope`](../rope_interface.md#apply_rope), [`seq_lens`](../../common/attention_metadata.md#AttentionMetadata.seq_lens), [`quantize_kv`](../../common/__init__.md#quantize_kv), [`block_tables`](../../common/attention_metadata.md#AttentionMetadata.block_tables), [`query_start_loc`](../../common/attention_metadata.md#AttentionMetadata.query_start_loc), [`request_distribution`](../../common/attention_metadata.md#AttentionMetadata.request_distribution), [`KVCache`](attention.md#KVCache), [`Llama4Attention`](llama4_attention.md#Llama4Attention)
- used by: [`end_layer`](../../../models/jax/llama4.md#Llama4ForCausalLM.end_layer), [`end_layer`](../../../models/jax/llama_guard_4.md#LlamaGuard4ForCausalLM.end_layer), [`__call__`](llama4_attention.md#Llama4Attention.__call__), [`apply_temperature_tuning`](llama4_attention.md#Llama4Attention.apply_temperature_tuning), [`Llama4Attention`](llama4_attention.md#Llama4Attention)

## Module values
- `KVCache` — [`L35`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/attention.py#L35) — documented in [tpu_inference-layers-jax-attention](../../../../../concepts/tpu_inference-layers-jax-attention.md)


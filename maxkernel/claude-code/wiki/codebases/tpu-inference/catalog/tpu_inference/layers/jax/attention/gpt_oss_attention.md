---
title: 'Module: tpu_inference/layers/jax/attention/gpt_oss_attention.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/jax/attention/gpt_oss_attention.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.jax.attention.gpt_oss_attention`/
symbols:
  GptOssAttention.__call__: GptOssAttention#__call__().
  GptOssAttention.rope: GptOssAttention#rope.
  GptOssAttention.attention: GptOssAttention#attention().
  GptOssAttention.kernel_q_DNH: GptOssAttention#kernel_q_DNH.
  GptOssAttention.kernel_k_DKH: GptOssAttention#kernel_k_DKH.
  GptOssAttention.kernel_v_DKH: GptOssAttention#kernel_v_DKH.
  GptOssAttention.kernel_o_proj_NHD: GptOssAttention#kernel_o_proj_NHD.
  GptOssAttention.bias_q_NH: GptOssAttention#bias_q_NH.
  GptOssAttention.bias_k_KH: GptOssAttention#bias_k_KH.
  GptOssAttention.bias_v_KH: GptOssAttention#bias_v_KH.
  GptOssAttention.bias_o_D: GptOssAttention#bias_o_D.
  GptOssAttention.head_dim: GptOssAttention#head_dim.
  GptOssAttention.dtype: GptOssAttention#dtype.
  GptOssAttention.sinks_N: GptOssAttention#sinks_N.
  GptOssAttention.random_init: GptOssAttention#random_init.
  GptOssAttention.hidden_size: GptOssAttention#hidden_size.
  GptOssAttention._ragged_paged_attention_wrapper: GptOssAttention#_ragged_paged_attention_wrapper().
  GptOssAttention.num_attention_heads: GptOssAttention#num_attention_heads.
  GptOssAttention.num_key_value_heads: GptOssAttention#num_key_value_heads.
  GptOssAttention.sm_scale: GptOssAttention#sm_scale.
  KVCache: KVCache.
  GptOssAttention.kv_cache_dtype: GptOssAttention#kv_cache_dtype.
  GptOssAttention.keyvalue_skh: GptOssAttention#keyvalue_skh.
  GptOssAttention.dkh_sharding: GptOssAttention#dkh_sharding.
  GptOssAttention.kv_cache_quantized_dtype: GptOssAttention#kv_cache_quantized_dtype.
  GptOssAttention: GptOssAttention#
  GptOssAttention.rope_theta: GptOssAttention#rope_theta.
  GptOssAttention.initial_context_length: GptOssAttention#initial_context_length.
  GptOssAttention.rope_scaling_factor: GptOssAttention#rope_scaling_factor.
  GptOssAttention.rope_ntk_alpha: GptOssAttention#rope_ntk_alpha.
  GptOssAttention.rope_ntk_beta: GptOssAttention#rope_ntk_beta.
  GptOssAttention.query_tnh: GptOssAttention#query_tnh.
  GptOssAttention.attn_o_tnh: GptOssAttention#attn_o_tnh.
  GptOssAttention.dnh_sharding: GptOssAttention#dnh_sharding.
  GptOssAttention.nhd_sharding: GptOssAttention#nhd_sharding.
  GptOssAttention.kh_sharding: GptOssAttention#kh_sharding.
  GptOssAttention.mesh: GptOssAttention#mesh.
  GptOssAttention.rngs: GptOssAttention#rngs.
  GptOssAttention.n_sharding: GptOssAttention#n_sharding.
  GptOssAttention.nh_sharding: GptOssAttention#nh_sharding.
  GptOssAttention.d_sharding: GptOssAttention#d_sharding.
  GptOssAttention._k_scale: GptOssAttention#_k_scale.
  GptOssAttention._v_scale: GptOssAttention#_v_scale.
  GptOssAttention._q_scale: GptOssAttention#_q_scale.
  GptOssAttention.__post_init__: GptOssAttention#__post_init__().
---
# Module: [`tpu_inference/layers/jax/attention/gpt_oss_attention.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/gpt_oss_attention.py)

## Classes
### `GptOssAttention`  ·  implements/extends Module
- def: [`tpu_inference/layers/jax/attention/gpt_oss_attention.py:37`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/gpt_oss_attention.py#L37)
- doc: JAX implementation of the GPT-OSS Attention block
- signature: `class GptOssAttention(nnx.Module):`
- members:
  - `__call__(self, x_TD, is_prefill, kv_cache: KVCache, attention_metadata: AttentionMetadata, use_attention_rope: bool = True)` — [`L221`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/gpt_oss_attention.py#L221) — Forward pass for the Attention module using 3D kernels.
  - `__post_init__(self, rngs: nnx.Rngs)` — [`L74`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/gpt_oss_attention.py#L74) — Initializes weights, biases, and RoPE module.
  - `attention(self, kv_cache: KVCache, q_TNH: jax.Array, k_SKH: jax.Array, v_SKH: jax.Array, sinks: jax.Array, attention_metadata: AttentionMetadata, mesh: Mesh, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None)` — [`L160`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/gpt_oss_attention.py#L160) — Performs scaled dot-product attention by calling the ragged_paged_attention kernel.
  - `attn_o_tnh` — [`L57`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/gpt_oss_attention.py#L57)
  - `bias_k_KH` — [`L109`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/gpt_oss_attention.py#L109)
  - `bias_o_D` — [`L138`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/gpt_oss_attention.py#L138)
  - `bias_q_NH` — [`L95`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/gpt_oss_attention.py#L95)
  - `bias_v_KH` — [`L123`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/gpt_oss_attention.py#L123)
  - `d_sharding` — [`L64`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/gpt_oss_attention.py#L64)
  - `dkh_sharding` — [`L59`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/gpt_oss_attention.py#L59)
  - `dnh_sharding` — [`L58`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/gpt_oss_attention.py#L58)
  - `dtype` — [`L45`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/gpt_oss_attention.py#L45)
  - `head_dim` — [`L44`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/gpt_oss_attention.py#L44)
  - `hidden_size` — [`L41`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/gpt_oss_attention.py#L41)
  - `initial_context_length` — [`L49`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/gpt_oss_attention.py#L49)
  - `kernel_k_DKH` — [`L102`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/gpt_oss_attention.py#L102)
  - `kernel_o_proj_NHD` — [`L131`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/gpt_oss_attention.py#L131)
  - `kernel_q_DNH` — [`L88`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/gpt_oss_attention.py#L88)
  - `kernel_v_DKH` — [`L116`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/gpt_oss_attention.py#L116)
  - `keyvalue_skh` — [`L56`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/gpt_oss_attention.py#L56)
  - `kh_sharding` — [`L63`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/gpt_oss_attention.py#L63)
  - `kv_cache_dtype` — [`L53`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/gpt_oss_attention.py#L53)
  - `kv_cache_quantized_dtype` — [`L72`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/gpt_oss_attention.py#L72)
  - `mesh` — [`L67`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/gpt_oss_attention.py#L67)
  - `n_sharding` — [`L61`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/gpt_oss_attention.py#L61)
  - `nh_sharding` — [`L62`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/gpt_oss_attention.py#L62)
  - `nhd_sharding` — [`L60`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/gpt_oss_attention.py#L60)
  - `num_attention_heads` — [`L42`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/gpt_oss_attention.py#L42)
  - `num_key_value_heads` — [`L43`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/gpt_oss_attention.py#L43)
  - `query_tnh` — [`L55`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/gpt_oss_attention.py#L55)
  - `random_init` — [`L66`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/gpt_oss_attention.py#L66)
  - `rngs` — [`L46`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/gpt_oss_attention.py#L46)
  - `rope` — [`L147`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/gpt_oss_attention.py#L147)
  - `rope_ntk_alpha` — [`L51`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/gpt_oss_attention.py#L51)
  - `rope_ntk_beta` — [`L52`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/gpt_oss_attention.py#L52)
  - `rope_scaling_factor` — [`L50`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/gpt_oss_attention.py#L50)
  - `rope_theta` — [`L48`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/gpt_oss_attention.py#L48)
  - `sinks_N` — [`L79`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/gpt_oss_attention.py#L79)
  - `sm_scale` — [`L77`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/gpt_oss_attention.py#L77)
- protocol/private: `_k_scale`[`L70`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/gpt_oss_attention.py#L70), `_q_scale`[`L69`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/gpt_oss_attention.py#L69), `_ragged_paged_attention_wrapper`[`L190`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/gpt_oss_attention.py#L190), `_v_scale`[`L71`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/gpt_oss_attention.py#L71)
- uses (calls/refs, reference-scoped): [`AttentionMetadata`](../../common/attention_metadata.md#AttentionMetadata), [`create_param`](../base.md#create_param), [`input_positions`](../../common/attention_metadata.md#AttentionMetadata.input_positions), [`seq_lens`](../../common/attention_metadata.md#AttentionMetadata.seq_lens), [`ragged_paged_attention_hd64`](../../../kernels/ragged_paged_attention/v3/kernel_hd64.md#ragged_paged_attention_hd64), [`quantize_kv`](../../common/__init__.md#quantize_kv), [`block_tables`](../../common/attention_metadata.md#AttentionMetadata.block_tables), [`query_start_loc`](../../common/attention_metadata.md#AttentionMetadata.query_start_loc), [`request_distribution`](../../common/attention_metadata.md#AttentionMetadata.request_distribution), [`head_dim`](../rope.md#GptOssRotaryEmbedding.head_dim), [`rope_scaling_factor`](../rope.md#GptOssRotaryEmbedding.rope_scaling_factor), [`rope_theta`](../rope.md#GptOssRotaryEmbedding.rope_theta), [`KVCache`](gpt_oss_attention.md#KVCache), [`initial_context_length`](../rope.md#GptOssRotaryEmbedding.initial_context_length), [`GptOssRotaryEmbedding`](../rope.md#GptOssRotaryEmbedding), [`dtype`](../rope.md#GptOssRotaryEmbedding.dtype), [`rope_ntk_alpha`](../rope.md#GptOssRotaryEmbedding.rope_ntk_alpha), [`rope_ntk_beta`](../rope.md#GptOssRotaryEmbedding.rope_ntk_beta)
- used by: [`layers`](../../../models/jax/gpt_oss.md#GptOss.layers)

## Module values
- `KVCache` — [`L33`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/attention/gpt_oss_attention.py#L33)


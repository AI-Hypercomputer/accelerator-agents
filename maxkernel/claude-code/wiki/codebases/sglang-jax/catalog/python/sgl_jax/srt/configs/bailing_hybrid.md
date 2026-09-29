---
title: 'Module: python/sgl_jax/srt/configs/bailing_hybrid.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/configs/bailing_hybrid.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.configs.bailing_hybrid`/
symbols:
  BailingHybridConfig.linear_state_params: BailingHybridConfig#linear_state_params().
  BailingHybridConfig.layers_block_type: BailingHybridConfig#layers_block_type().
  BailingHybridConfig: BailingHybridConfig#
  get_bailing_hybrid_config: get_bailing_hybrid_config().
  BailingHybridConfig.linear_attn_config: BailingHybridConfig#linear_attn_config().
  BailingHybridConfig.linear_layer_ids: BailingHybridConfig#linear_layer_ids().
  BailingHybridConfig.layer_group_size: BailingHybridConfig#layer_group_size.
  BailingHybridConfig._layers_block_type: BailingHybridConfig#_layers_block_type.
  BailingHybridConfig.use_linear_silu: BailingHybridConfig#use_linear_silu.
  BailingHybridConfig.full_attention_layer_ids: BailingHybridConfig#full_attention_layer_ids().
  BailingHybridConfig.num_hidden_layers: BailingHybridConfig#num_hidden_layers.
  BailingHybridConfig.num_attention_heads: BailingHybridConfig#num_attention_heads.
  BailingHybridConfig.head_dim: BailingHybridConfig#head_dim.
  BailingHybridConfig.linear_silu: BailingHybridConfig#linear_silu.
  BailingHybridConfig.num_linear_key_value_heads: BailingHybridConfig#num_linear_key_value_heads.
  _is_bailing_hybrid_config: _is_bailing_hybrid_config().
  _get_layer_ids: _get_layer_ids().
  BailingHybridConfig.model_type: BailingHybridConfig#model_type.
  BailingHybridConfig.keys_to_ignore_at_inference: BailingHybridConfig#keys_to_ignore_at_inference.
  BailingHybridConfig.__init__: BailingHybridConfig#__init__().
  BailingHybridConfig.vocab_size: BailingHybridConfig#vocab_size.
  BailingHybridConfig.hidden_size: BailingHybridConfig#hidden_size.
  BailingHybridConfig.intermediate_size: BailingHybridConfig#intermediate_size.
  BailingHybridConfig.num_key_value_heads: BailingHybridConfig#num_key_value_heads.
  BailingHybridConfig.hidden_act: BailingHybridConfig#hidden_act.
  BailingHybridConfig.use_qkv_bias: BailingHybridConfig#use_qkv_bias.
  BailingHybridConfig.use_bias: BailingHybridConfig#use_bias.
  BailingHybridConfig.rms_norm_eps: BailingHybridConfig#rms_norm_eps.
  BailingHybridConfig.max_position_embeddings: BailingHybridConfig#max_position_embeddings.
  BailingHybridConfig.rope_theta: BailingHybridConfig#rope_theta.
  BailingHybridConfig.rope_scaling: BailingHybridConfig#rope_scaling.
  BailingHybridConfig.use_qk_norm: BailingHybridConfig#use_qk_norm.
  BailingHybridConfig.num_experts: BailingHybridConfig#num_experts.
  BailingHybridConfig.num_shared_experts: BailingHybridConfig#num_shared_experts.
  BailingHybridConfig.num_experts_per_tok: BailingHybridConfig#num_experts_per_tok.
  BailingHybridConfig.n_group: BailingHybridConfig#n_group.
  BailingHybridConfig.topk_group: BailingHybridConfig#topk_group.
  BailingHybridConfig.moe_intermediate_size: BailingHybridConfig#moe_intermediate_size.
  BailingHybridConfig.first_k_dense_replace: BailingHybridConfig#first_k_dense_replace.
  BailingHybridConfig.moe_router_enable_expert_bias: BailingHybridConfig#moe_router_enable_expert_bias.
  BailingHybridConfig.norm_topk_prob: BailingHybridConfig#norm_topk_prob.
  BailingHybridConfig.routed_scaling_factor: BailingHybridConfig#routed_scaling_factor.
  BailingHybridConfig.score_function: BailingHybridConfig#score_function.
  BailingHybridConfig.router_dtype: BailingHybridConfig#router_dtype.
  BailingHybridConfig.group_norm_size: BailingHybridConfig#group_norm_size.
  BailingHybridConfig.linear_rope: BailingHybridConfig#linear_rope.
  BailingHybridConfig.full_attention_type: BailingHybridConfig#full_attention_type.
  BailingHybridConfig.kv_lora_rank: BailingHybridConfig#kv_lora_rank.
  BailingHybridConfig.q_lora_rank: BailingHybridConfig#q_lora_rank.
  BailingHybridConfig.qk_rope_head_dim: BailingHybridConfig#qk_rope_head_dim.
  BailingHybridConfig.qk_nope_head_dim: BailingHybridConfig#qk_nope_head_dim.
  BailingHybridConfig.qk_head_dim: BailingHybridConfig#qk_head_dim.
  BailingHybridConfig.v_head_dim: BailingHybridConfig#v_head_dim.
  BailingHybridConfig.rope_interleave: BailingHybridConfig#rope_interleave.
  BailingHybridConfig.num_nextn_predict_layers: BailingHybridConfig#num_nextn_predict_layers.
  BailingHybridConfig.mtp_loss_scaling_factor: BailingHybridConfig#mtp_loss_scaling_factor.
  BailingHybridConfig.quantization_config: BailingHybridConfig#quantization_config.
  __all__: __all__.
---
# Module: [`python/sgl_jax/srt/configs/bailing_hybrid.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py)

## Classes
### `BailingHybridConfig`  ·  implements/extends PretrainedConfig
- def: [`python/sgl_jax/srt/configs/bailing_hybrid.py:8`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L8)
- doc: Minimal Bailing hybrid config for Ling/Ring 2.5 linear-attention models.
- signature: `class BailingHybridConfig(PretrainedConfig):`
- members:
  - `full_attention_layer_ids(self)` — [`L154`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L154)
  - `layers_block_type(self)` — [`L131`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L131)
  - `linear_attn_config(self)` — [`L177`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L177)
  - `linear_layer_ids(self)` — [`L146`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L146)
  - `linear_state_params(self)` — [`L162`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L162)
  - `first_k_dense_replace` — [`L86`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L86)
  - `full_attention_type` — [`L100`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L100)
  - `group_norm_size` — [`L95`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L95)
  - `head_dim` — [`L77`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L77)
  - `hidden_act` — [`L70`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L70)
  - `hidden_size` — [`L65`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L65)
  - `intermediate_size` — [`L66`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L66)
  - `keys_to_ignore_at_inference` — [`L12`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L12)
  - `kv_lora_rank` — [`L102`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L102)
  - `layer_group_size` — [`L93`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L93)
  - `linear_rope` — [`L98`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L98)
  - `linear_silu` — [`L96`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L96)
  - `max_position_embeddings` — [`L74`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L74)
  - `model_type` — [`L11`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L11)
  - `moe_intermediate_size` — [`L85`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L85)
  - `moe_router_enable_expert_bias` — [`L87`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L87)
  - `mtp_loss_scaling_factor` — [`L114`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L114)
  - `n_group` — [`L83`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L83)
  - `norm_topk_prob` — [`L88`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L88)
  - `num_attention_heads` — [`L68`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L68)
  - `num_experts` — [`L80`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L80)
  - `num_experts_per_tok` — [`L82`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L82)
  - `num_hidden_layers` — [`L67`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L67)
  - `num_key_value_heads` — [`L69`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L69)
  - `num_linear_key_value_heads` — [`L99`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L99)
  - `num_nextn_predict_layers` — [`L113`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L113)
  - `num_shared_experts` — [`L81`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L81)
  - `q_lora_rank` — [`L103`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L103)
  - `qk_head_dim` — [`L106`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L106)
  - `qk_nope_head_dim` — [`L105`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L105)
  - `qk_rope_head_dim` — [`L104`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L104)
  - `quantization_config` — [`L121`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L121)
  - `rms_norm_eps` — [`L73`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L73)
  - `rope_interleave` — [`L108`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L108)
  - `rope_scaling` — [`L76`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L76)
  - `rope_theta` — [`L75`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L75)
  - `routed_scaling_factor` — [`L89`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L89)
  - `router_dtype` — [`L91`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L91)
  - `score_function` — [`L90`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L90)
  - `topk_group` — [`L84`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L84)
  - `use_bias` — [`L72`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L72)
  - `use_linear_silu` — [`L97`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L97)
  - `use_qk_norm` — [`L78`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L78)
  - `use_qkv_bias` — [`L71`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L71)
  - `v_head_dim` — [`L107`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L107)
  - `vocab_size` — [`L64`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L64)
- protocol/private: `__init__`[`L14`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L14), `_layers_block_type`[`L94`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L94)
- uses (calls/refs, reference-scoped): [`recurrent_state_dtype`](../mem_cache/recurrent_state_pool.md#recurrent_state_dtype), [`dtype`](../mem_cache/recurrent_state_pool.md#LinearRecurrentStateParams.dtype), [`LinearRecurrentStateParams`](../mem_cache/recurrent_state_pool.md#LinearRecurrentStateParams), [`conv_kernel_size`](../mem_cache/recurrent_state_pool.md#LinearRecurrentStateParams.conv_kernel_size), [`head_dim`](../mem_cache/recurrent_state_pool.md#LinearRecurrentStateParams.head_dim), [`layers`](../mem_cache/recurrent_state_pool.md#LinearRecurrentStateParams.layers), [`num_heads`](../mem_cache/recurrent_state_pool.md#LinearRecurrentStateParams.num_heads)
- used by: [`attn_backend_wrapper`](../layers/attention/hybrid_linear_attn_backend.md#attn_backend_wrapper), [`_CONFIG_REGISTRY`](../hf_transformers_utils.md#_CONFIG_REGISTRY._CONFIG_REGISTRY), [`get_bailing_hybrid_config`](bailing_hybrid.md#get_bailing_hybrid_config)

## Functions
- `_get_layer_ids(hf_config: Any, num_hidden_layers: int)` — [`L210`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L210)
- `_is_bailing_hybrid_config(hf_config: Any)` — [`L203`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L203)
- `get_bailing_hybrid_config(hf_config: Any)` — [`L186`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L186)

## Module values
- `__all__` — [`L244`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/bailing_hybrid.py#L244)


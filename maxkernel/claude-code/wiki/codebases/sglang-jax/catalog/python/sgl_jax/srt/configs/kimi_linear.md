---
title: 'Module: python/sgl_jax/srt/configs/kimi_linear.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/configs/kimi_linear.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.configs.kimi_linear`/
symbols:
  KimiLinearConfig.is_mla: KimiLinearConfig#is_mla().
  KimiLinearConfig.full_attention_layer_ids: KimiLinearConfig#full_attention_layer_ids().
  get_kimi_linear_config: get_kimi_linear_config().
  KimiLinearConfig.linear_attn_config: KimiLinearConfig#linear_attn_config.
  KimiLinearConfig: KimiLinearConfig#
  KimiLinearConfig.linear_layer_ids: KimiLinearConfig#linear_layer_ids().
  KimiLinearConfig.is_kda_layer: KimiLinearConfig#is_kda_layer().
  KimiLinearConfig.moe_router_activation_func: KimiLinearConfig#moe_router_activation_func.
  KimiLinearConfig.is_moe: KimiLinearConfig#is_moe().
  KimiLinearConfig.is_linear_attn: KimiLinearConfig#is_linear_attn().
  KimiLinearConfig.num_hidden_layers: KimiLinearConfig#num_hidden_layers.
  KimiLinearConfig.q_lora_rank: KimiLinearConfig#q_lora_rank.
  KimiLinearConfig.kv_lora_rank: KimiLinearConfig#kv_lora_rank.
  KimiLinearConfig.qk_nope_head_dim: KimiLinearConfig#qk_nope_head_dim.
  KimiLinearConfig.qk_rope_head_dim: KimiLinearConfig#qk_rope_head_dim.
  KimiLinearConfig.v_head_dim: KimiLinearConfig#v_head_dim.
  KimiLinearConfig.mla_use_nope: KimiLinearConfig#mla_use_nope.
  KimiLinearConfig.num_experts: KimiLinearConfig#num_experts.
  _is_kimi_linear_config: _is_kimi_linear_config().
  KimiLinearConfig.model_type: KimiLinearConfig#model_type.
  KimiLinearConfig.keys_to_ignore_at_inference: KimiLinearConfig#keys_to_ignore_at_inference.
  KimiLinearConfig.__init__: KimiLinearConfig#__init__().
  KimiLinearConfig.vocab_size: KimiLinearConfig#vocab_size.
  KimiLinearConfig.hidden_size: KimiLinearConfig#hidden_size.
  KimiLinearConfig.head_dim: KimiLinearConfig#head_dim.
  KimiLinearConfig.intermediate_size: KimiLinearConfig#intermediate_size.
  KimiLinearConfig.num_attention_heads: KimiLinearConfig#num_attention_heads.
  KimiLinearConfig.num_key_value_heads: KimiLinearConfig#num_key_value_heads.
  KimiLinearConfig.hidden_act: KimiLinearConfig#hidden_act.
  KimiLinearConfig.initializer_range: KimiLinearConfig#initializer_range.
  KimiLinearConfig.rms_norm_eps: KimiLinearConfig#rms_norm_eps.
  KimiLinearConfig.use_cache: KimiLinearConfig#use_cache.
  KimiLinearConfig.rope_theta: KimiLinearConfig#rope_theta.
  KimiLinearConfig.rope_scaling: KimiLinearConfig#rope_scaling.
  KimiLinearConfig.n_routed_experts: KimiLinearConfig#n_routed_experts.
  KimiLinearConfig.num_experts_per_token: KimiLinearConfig#num_experts_per_token.
  KimiLinearConfig.moe_renormalize: KimiLinearConfig#moe_renormalize.
  KimiLinearConfig.num_shared_experts: KimiLinearConfig#num_shared_experts.
  KimiLinearConfig.routed_scaling_factor: KimiLinearConfig#routed_scaling_factor.
  KimiLinearConfig.moe_intermediate_size: KimiLinearConfig#moe_intermediate_size.
  KimiLinearConfig.first_k_dense_replace: KimiLinearConfig#first_k_dense_replace.
  KimiLinearConfig.moe_layer_freq: KimiLinearConfig#moe_layer_freq.
  KimiLinearConfig.use_grouped_topk: KimiLinearConfig#use_grouped_topk.
  KimiLinearConfig.num_expert_group: KimiLinearConfig#num_expert_group.
  KimiLinearConfig.topk_group: KimiLinearConfig#topk_group.
  KimiLinearConfig.num_nextn_predict_layers: KimiLinearConfig#num_nextn_predict_layers.
---
# Module: [`python/sgl_jax/srt/configs/kimi_linear.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/kimi_linear.py)

## Classes
### `KimiLinearConfig`  ·  implements/extends PretrainedConfig
- def: [`python/sgl_jax/srt/configs/kimi_linear.py:10`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/kimi_linear.py#L10)
- signature: `class KimiLinearConfig(PretrainedConfig):`
- members:
  - `full_attention_layer_ids(self)` — [`L148`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/kimi_linear.py#L148)
  - `is_kda_layer(self, layer_idx: int)` — [`L137`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/kimi_linear.py#L137)
  - `is_linear_attn(self)` — [`L127`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/kimi_linear.py#L127)
  - `is_mla(self)` — [`L112`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/kimi_linear.py#L112)
  - `is_moe(self)` — [`L123`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/kimi_linear.py#L123)
  - `linear_layer_ids(self)` — [`L144`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/kimi_linear.py#L144)
  - `first_k_dense_replace` — [`L91`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/kimi_linear.py#L91)
  - `head_dim` — [`L59`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/kimi_linear.py#L59)
  - `hidden_act` — [`L69`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/kimi_linear.py#L69)
  - `hidden_size` — [`L58`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/kimi_linear.py#L58)
  - `initializer_range` — [`L70`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/kimi_linear.py#L70)
  - `intermediate_size` — [`L60`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/kimi_linear.py#L60)
  - `keys_to_ignore_at_inference` — [`L12`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/kimi_linear.py#L12)
  - `kv_lora_rank` — [`L77`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/kimi_linear.py#L77)
  - `linear_attn_config` — [`L101`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/kimi_linear.py#L101)
  - `mla_use_nope` — [`L81`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/kimi_linear.py#L81)
  - `model_type` — [`L11`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/kimi_linear.py#L11)
  - `moe_intermediate_size` — [`L90`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/kimi_linear.py#L90)
  - `moe_layer_freq` — [`L92`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/kimi_linear.py#L92)
  - `moe_renormalize` — [`L85`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/kimi_linear.py#L85)
  - `moe_router_activation_func` — [`L88`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/kimi_linear.py#L88)
  - `n_routed_experts` — [`L83`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/kimi_linear.py#L83)
  - `num_attention_heads` — [`L62`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/kimi_linear.py#L62)
  - `num_expert_group` — [`L94`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/kimi_linear.py#L94)
  - `num_experts` — [`L83`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/kimi_linear.py#L83)
  - `num_experts_per_token` — [`L84`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/kimi_linear.py#L84)
  - `num_hidden_layers` — [`L61`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/kimi_linear.py#L61)
  - `num_key_value_heads` — [`L68`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/kimi_linear.py#L68)
  - `num_nextn_predict_layers` — [`L96`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/kimi_linear.py#L96)
  - `num_shared_experts` — [`L86`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/kimi_linear.py#L86)
  - `q_lora_rank` — [`L76`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/kimi_linear.py#L76)
  - `qk_nope_head_dim` — [`L78`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/kimi_linear.py#L78)
  - `qk_rope_head_dim` — [`L79`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/kimi_linear.py#L79)
  - `rms_norm_eps` — [`L71`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/kimi_linear.py#L71)
  - `rope_scaling` — [`L74`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/kimi_linear.py#L74)
  - `rope_theta` — [`L73`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/kimi_linear.py#L73)
  - `routed_scaling_factor` — [`L87`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/kimi_linear.py#L87)
  - `topk_group` — [`L95`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/kimi_linear.py#L95)
  - `use_cache` — [`L72`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/kimi_linear.py#L72)
  - `use_grouped_topk` — [`L93`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/kimi_linear.py#L93)
  - `v_head_dim` — [`L80`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/kimi_linear.py#L80)
  - `vocab_size` — [`L57`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/kimi_linear.py#L57)
- protocol/private: `__init__`[`L14`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/kimi_linear.py#L14)
- used by: [`attn_backend_wrapper`](../layers/attention/hybrid_linear_attn_backend.md#attn_backend_wrapper), [`_maybe_wrap_hybrid_kv_pool`](../model_executor/model_runner_kv_cache_mixin.md#ModelRunnerKVCacheMixin._maybe_wrap_hybrid_kv_pool), [`_CONFIG_REGISTRY`](../hf_transformers_utils.md#_CONFIG_REGISTRY._CONFIG_REGISTRY), [`_kv_pool_layer_count`](../model_executor/model_runner_kv_cache_mixin.md#ModelRunnerKVCacheMixin._kv_pool_layer_count), [`get_kimi_linear_config`](kimi_linear.md#get_kimi_linear_config)

## Functions
- `_is_kimi_linear_config(hf_config: Any)` — [`L152`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/kimi_linear.py#L152)
- `get_kimi_linear_config(hf_config: Any)` — [`L159`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/kimi_linear.py#L159) — Return a KimiLinearConfig if hf_config describes a Kimi-Linear model, else None.


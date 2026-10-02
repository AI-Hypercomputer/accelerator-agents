---
title: 'Module: tpu_inference/models/jax/deepseek_v3.py'
type: catalog
provenance: extracted
module: tpu_inference/models/jax/deepseek_v3.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.models.jax.deepseek_v3`/
symbols:
  DeepseekV2Moe.experts: DeepseekV2Moe#experts.
  DeepSeekV3.get_decoder_layer: DeepSeekV3#get_decoder_layer().
  DeepseekV2Moe.shared_experts: DeepseekV2Moe#shared_experts.
  DeepseekV3Attention.compute_attention: DeepseekV3Attention#compute_attention().
  DeepSeekV3._create_deepseek_attention: DeepSeekV3#_create_deepseek_attention().
  DeepseekV3MLA.compute_attention: DeepseekV3MLA#compute_attention().
  DeepSeekV3.rope_emb: DeepSeekV3#rope_emb.
  DeepseekV3BaseAttention.__call__: DeepseekV3BaseAttention#__call__().
  DeepseekV3Attention.compute_kv_projection: DeepseekV3Attention#compute_kv_projection().
  DeepseekV3BaseAttention.kv_b_proj: DeepseekV3BaseAttention#kv_b_proj.
  MLAEinsum.load_weights: MLAEinsum#load_weights().
  DeepseekV3Attention.kv_b_proj: DeepseekV3Attention#kv_b_proj.
  DeepseekV3Attention.compute_q_projection: DeepseekV3Attention#compute_q_projection().
  DeepseekV3MLA.kv_b_proj: DeepseekV3MLA#kv_b_proj.
  DeepseekV3MLA.compute_q_projection: DeepseekV3MLA#compute_q_projection().
  DeepSeekV3Router.__call__: DeepSeekV3Router#__call__().
  DeepseekV3BaseAttention.q_b_proj: DeepseekV3BaseAttention#q_b_proj.
  DeepseekV3BaseAttention.kv_a_proj_with_mqa: DeepseekV3BaseAttention#kv_a_proj_with_mqa.
  DeepseekV3BaseAttention.o_proj: DeepseekV3BaseAttention#o_proj.
  DeepseekV2Moe.gate: DeepseekV2Moe#gate.
  DeepSeekV3Router.moe_backend: DeepSeekV3Router#moe_backend.
  DeepseekV3MLP.__call__: DeepseekV3MLP#__call__().
  DeepseekV3ForCausalLM.load_weights: DeepseekV3ForCausalLM#load_weights().
  DeepseekV3BaseAttention.qk_head_dim: DeepseekV3BaseAttention#qk_head_dim.
  DeepseekV3BaseAttention.q_a_proj: DeepseekV3BaseAttention#q_a_proj.
  DeepseekV3BaseAttention.q_a_layernorm: DeepseekV3BaseAttention#q_a_layernorm.
  DeepseekV3BaseAttention.kv_a_layernorm: DeepseekV3BaseAttention#kv_a_layernorm.
  DeepSeekV3.embed_tokens: DeepSeekV3#embed_tokens.
  DeepseekV3ForCausalLM.lm_head: DeepseekV3ForCausalLM#lm_head.
  DeepSeekV3.__call__: DeepSeekV3#__call__().
  DeepseekV3MLA.compute_kv_projection: DeepseekV3MLA#compute_kv_projection().
  DeepSeekV3.moe_backend: DeepSeekV3#moe_backend.
  DeepSeekV3.norm: DeepSeekV3#norm.
  DeepSeekV3Router.get_topk_indices: DeepSeekV3Router#get_topk_indices().
  DeepseekV3BaseAttention.dtype: DeepseekV3BaseAttention#dtype.
  DeepseekV3BaseAttention.kv_cache_quantized_dtype: DeepseekV3BaseAttention#kv_cache_quantized_dtype.
  DeepseekV3BaseAttention.N: DeepseekV3BaseAttention#N.
  DeepseekV3DecoderLayer.__init__: DeepseekV3DecoderLayer#__init__().
  DeepseekV3DecoderLayer.__call__: DeepseekV3DecoderLayer#__call__().
  DeepSeekV3.use_mla_kernel: DeepSeekV3#use_mla_kernel.
  DeepseekV3BaseAttention.qk_nope_head_dim: DeepseekV3BaseAttention#qk_nope_head_dim.
  DeepSeekV3.num_expert_parallelism: DeepSeekV3#num_expert_parallelism.
  DeepseekV3BaseAttention.quant_config: DeepseekV3BaseAttention#quant_config.
  DeepseekV3ForCausalLM.model: DeepseekV3ForCausalLM#model.
  DeepseekV3MLP.gate_proj: DeepseekV3MLP#gate_proj.
  DeepseekV3MLP.up_proj: DeepseekV3MLP#up_proj.
  DeepseekV3MLP.down_proj: DeepseekV3MLP#down_proj.
  DeepSeekV3Router.e_score_correction_bias: DeepSeekV3Router#e_score_correction_bias.
  DeepseekV3MLP.__post_init__: DeepseekV3MLP#__post_init__().
  DeepseekV3ForCausalLM.__call__: DeepseekV3ForCausalLM#__call__().
  DeepseekV3BaseAttention: DeepseekV3BaseAttention#
  DeepseekV3BaseAttention.scale: DeepseekV3BaseAttention#scale.
  DeepSeekV3.is_last_rank: DeepSeekV3#is_last_rank.
  DeepSeekV3.layers: DeepSeekV3#layers.
  DeepseekV3BaseAttention.kv_lora_rank: DeepseekV3BaseAttention#kv_lora_rank.
  DeepseekV3BaseAttention.v_head_dim: DeepseekV3BaseAttention#v_head_dim.
  DeepseekV3BaseAttention.prefix: DeepseekV3BaseAttention#prefix.
  DeepseekV3BaseAttention.rope: DeepseekV3BaseAttention#rope.
  DeepseekV3BaseAttention.compute_attention: DeepseekV3BaseAttention#compute_attention().
  hidden_size.hidden_size: hidden_size.hidden_size.
  _weight_init: _weight_init().
  DeepseekV3Attention.__post_init__: DeepseekV3Attention#__post_init__().
  DeepseekV3MLA.__post_init__: DeepseekV3MLA#__post_init__().
  DeepseekV3MLP.quant_config: DeepseekV3MLP#quant_config.
  SharedFusedMoe.__call__: SharedFusedMoe#__call__().
  DeepSeekV3Router.__init__: DeepSeekV3Router#__init__().
  KVCache: KVCache.
  rope_scaling: rope_scaling.
  expert_axis_name: expert_axis_name.
  DeepseekV3BaseAttention.keyvalue_skh: DeepseekV3BaseAttention#keyvalue_skh.
  MLAEinsum.mla_layer: MLAEinsum#mla_layer.
  DeepseekV3ForCausalLM: DeepseekV3ForCausalLM#
  DeepseekV3MLP: DeepseekV3MLP#
  DeepseekV3BaseAttention.compute_q_projection: DeepseekV3BaseAttention#compute_q_projection().
  DeepseekV3BaseAttention.compute_kv_projection: DeepseekV3BaseAttention#compute_kv_projection().
  DeepseekV3Attention._ragged_paged_attention: DeepseekV3Attention#_ragged_paged_attention().
  DeepseekV3MLP.dtype: DeepseekV3MLP#dtype.
  DeepSeekV3.is_first_rank: DeepSeekV3#is_first_rank.
  DeepSeekV3.initialize_cache: DeepSeekV3#initialize_cache().
  logger: logger.
  DeepseekV3BaseAttention.D: DeepseekV3BaseAttention#D.
  DeepseekV3Attention: DeepseekV3Attention#
  DeepseekV3MLA: DeepseekV3MLA#
  SharedFusedMoe.shared_experts: SharedFusedMoe#shared_experts.
  MLAEinsum.__init__: MLAEinsum#__init__().
  SharedFusedMoe: SharedFusedMoe#
  DeepseekV3BaseAttention.ap_sharding: DeepseekV3BaseAttention#ap_sharding.
  MLAEinsum.loaded: MLAEinsum#loaded.
  DeepseekV3MLP.df_sharding: DeepseekV3MLP#df_sharding.
  DeepSeekV3Router.n_groups: DeepSeekV3Router#n_groups.
  DeepSeekV3.vllm_config: DeepSeekV3#vllm_config.
  rms_norm_eps.rms_norm_eps: rms_norm_eps.rms_norm_eps.
  modeling_flax_utils: modeling_flax_utils.
  DeepseekV3BaseAttention.process_output: DeepseekV3BaseAttention#process_output().
  MLAEinsum: MLAEinsum#
  DeepseekV2Moe: DeepseekV2Moe#
  DeepseekV3DecoderLayer: DeepseekV3DecoderLayer#
  DeepSeekV3: DeepSeekV3#
  DeepSeekV3.enable_return_routed_experts: DeepSeekV3#enable_return_routed_experts.
  DeepSeekV3.use_ep: DeepSeekV3#use_ep.
  DeepSeekV3Router: DeepSeekV3Router#
  DeepseekV3BaseAttention.q_lora_rank: DeepseekV3BaseAttention#q_lora_rank.
  DeepseekV3BaseAttention.qk_rope_head_dim: DeepseekV3BaseAttention#qk_rope_head_dim.
  DeepseekV3BaseAttention.query_tnh: DeepseekV3BaseAttention#query_tnh.
  DeepseekV3BaseAttention.random_init: DeepseekV3BaseAttention#random_init.
  MLAEinsum.quant_config: MLAEinsum#quant_config.
  DeepseekV3MLP.hidden_act: DeepseekV3MLP#hidden_act.
  DeepseekV3MLP.hidden_size: DeepseekV3MLP#hidden_size.
  DeepseekV3MLP.intermediate_size: DeepseekV3MLP#intermediate_size.
  DeepseekV3MLP.fd_sharding: DeepseekV3MLP#fd_sharding.
  DeepseekV3MLP.activation_ffw_td: DeepseekV3MLP#activation_ffw_td.
  DeepSeekV3Router.num_experts: DeepSeekV3Router#num_experts.
  DeepSeekV3Router.dtype: DeepSeekV3Router#dtype.
  DeepSeekV3.mesh: DeepSeekV3#mesh.
  hidden_act.hidden_act: hidden_act.hidden_act.
  DeepseekV3BaseAttention.K: DeepseekV3BaseAttention#K.
  DeepseekV3MLA.process_output: DeepseekV3MLA#process_output().
  DeepseekV2Moe.__call__: DeepseekV2Moe#__call__().
  DeepSeekV3.apply: DeepSeekV3#apply().
  DeepseekV3ForCausalLM.compute_logits: DeepseekV3ForCausalLM#compute_logits().
  qk_rope_head_dim: qk_rope_head_dim.
  v_head_dim: v_head_dim.
  DeepseekV3BaseAttention.num_attention_heads: DeepseekV3BaseAttention#num_attention_heads.
  DeepseekV3BaseAttention.kv_cache_dtype: DeepseekV3BaseAttention#kv_cache_dtype.
  DeepseekV3BaseAttention.mesh: DeepseekV3BaseAttention#mesh.
  DeepseekV3BaseAttention.rms_norm_eps: DeepseekV3BaseAttention#rms_norm_eps.
  DeepseekV3BaseAttention.query_nth: DeepseekV3BaseAttention#query_nth.
  DeepseekV3BaseAttention.attn_o_nth: DeepseekV3BaseAttention#attn_o_nth.
  DeepseekV3BaseAttention._q_scale: DeepseekV3BaseAttention#_q_scale.
  DeepseekV3BaseAttention._k_scale: DeepseekV3BaseAttention#_k_scale.
  DeepseekV3BaseAttention.__post_init__: DeepseekV3BaseAttention#__post_init__().
  DeepseekV3MLP.rngs: DeepseekV3MLP#rngs.
  SharedFusedMoe.routed_scaling_factor: SharedFusedMoe#routed_scaling_factor.
  DeepSeekV3Router.random_init: DeepSeekV3Router#random_init.
  DeepSeekV3Router.scoring_func: DeepSeekV3Router#scoring_func.
  num_local_experts.num_local_experts: num_local_experts.num_local_experts.
  num_attention_heads.num_attention_heads: num_attention_heads.num_attention_heads.
  num_key_value_heads.num_key_value_heads: num_key_value_heads.num_key_value_heads.
  moe_intermediate_size.moe_intermediate_size: moe_intermediate_size.moe_intermediate_size.
  num_experts_per_token.num_experts_per_token: num_experts_per_token.num_experts_per_token.
  routed_scaling_factor.routed_scaling_factor: routed_scaling_factor.routed_scaling_factor.
  num_shared_experts: num_shared_experts.
  rope_theta: rope_theta.
  q_lora_rank: q_lora_rank.
  kv_lora_rank: kv_lora_rank.
  qk_nope_head_dim: qk_nope_head_dim.
  DeepseekV3BaseAttention.hidden_size: DeepseekV3BaseAttention#hidden_size.
  DeepseekV3BaseAttention.num_key_value_heads: DeepseekV3BaseAttention#num_key_value_heads.
  DeepseekV3BaseAttention.rd_sharding: DeepseekV3BaseAttention#rd_sharding.
  DeepseekV3BaseAttention.q_da_sharding: DeepseekV3BaseAttention#q_da_sharding.
  DeepseekV3BaseAttention.kv_da_sharding: DeepseekV3BaseAttention#kv_da_sharding.
  DeepseekV3BaseAttention.activation_attention_td: DeepseekV3BaseAttention#activation_attention_td.
  DeepseekV3BaseAttention.activation_q_td: DeepseekV3BaseAttention#activation_q_td.
  DeepseekV3BaseAttention.rope_mscale_all_dim: DeepseekV3BaseAttention#rope_mscale_all_dim.
  DeepseekV3BaseAttention._v_scale: DeepseekV3BaseAttention#_v_scale.
  DeepseekV3MLP.random_init: DeepseekV3MLP#random_init.
  DeepseekV3DecoderLayer.input_layernorm: DeepseekV3DecoderLayer#input_layernorm.
  DeepseekV3DecoderLayer.post_attention_layernorm: DeepseekV3DecoderLayer#post_attention_layernorm.
  DeepseekV3DecoderLayer.self_attn: DeepseekV3DecoderLayer#self_attn.
  DeepseekV3DecoderLayer.mlp: DeepseekV3DecoderLayer#mlp.
  DeepSeekV3Router.hidden_size: DeepSeekV3Router#hidden_size.
  DeepSeekV3Router.num_experts_per_tok: DeepSeekV3Router#num_experts_per_tok.
  DeepSeekV3Router.topk_groups: DeepSeekV3Router#topk_groups.
  DeepSeekV3Router.norm_topk_prob: DeepSeekV3Router#norm_topk_prob.
  DeepSeekV3Router.activation_ffw_td: DeepSeekV3Router#activation_ffw_td.
  DeepSeekV3Router.ed_sharding: DeepSeekV3Router#ed_sharding.
  DeepSeekV3Router.e_sharding: DeepSeekV3Router#e_sharding.
  DeepSeekV3Router.router_bias_dtype: DeepSeekV3Router#router_bias_dtype.
  DeepSeekV3.start_layer: DeepSeekV3#start_layer.
  DeepSeekV3.end_layer: DeepSeekV3#end_layer.
  vocab_size.vocab_size: vocab_size.vocab_size.
  ffw_intermediate_size.ffw_intermediate_size: ffw_intermediate_size.ffw_intermediate_size.
  n_group.n_group: n_group.n_group.
  interleave_moe_layer_step.interleave_moe_layer_step: interleave_moe_layer_step.interleave_moe_layer_step.
  first_k_dense_replace.first_k_dense_replace: first_k_dense_replace.first_k_dense_replace.
  DeepseekV3BaseAttention.head_dim: DeepseekV3BaseAttention#head_dim.
  DeepseekV3BaseAttention.activation_attention_out_td: DeepseekV3BaseAttention#activation_attention_out_td.
  DeepseekV3BaseAttention.rngs: DeepseekV3BaseAttention#rngs.
  MLAEinsum.named_children: MLAEinsum#named_children().
  DeepseekV3MLA.anh_sharding: DeepseekV3MLA#anh_sharding.
  DeepseekV2Moe.__init__: DeepseekV2Moe#__init__().
  DeepSeekV3Router.routed_scaling_factor: DeepSeekV3Router#routed_scaling_factor.
  DeepSeekV3Router.quant_config: DeepSeekV3Router#quant_config.
  DeepSeekV3.__init__: DeepSeekV3#__init__().
  DeepseekV3ForCausalLM.__init__: DeepseekV3ForCausalLM#__init__().
  DeepseekV3ForCausalLM.vllm_config: DeepseekV3ForCausalLM#vllm_config.
  DeepseekV3ForCausalLM.mesh: DeepseekV3ForCausalLM#mesh.
---
# Module: [`tpu_inference/models/jax/deepseek_v3.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py)

## Classes
### `DeepSeekV3`  ·  implements/extends JaxModule
- def: [`tpu_inference/models/jax/deepseek_v3.py:1123`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L1123)
- signature: `class DeepSeekV3(JaxModule):`
- members:
  - `apply(self, variables, *args, **kwargs)` — [`L1336`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L1336)
  - `get_decoder_layer(layer_index: int)` — [`L1253`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L1253)
  - `initialize_cache(self)` — [`L1339`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L1339)
  - `embed_tokens` — [`L1161`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L1161)
  - `enable_return_routed_experts` — [`L1132`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L1132)
  - `end_layer` — [`L1317`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L1317)
  - `is_first_rank` — [`L1154`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L1154)
  - `is_last_rank` — [`L1155`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L1155)
  - `layers` — [`L1317`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L1317)
  - `mesh` — [`L1138`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L1138)
  - `moe_backend` — [`L1145`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L1145)
  - `norm` — [`L1321`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L1321)
  - `num_expert_parallelism` — [`L1140`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L1140)
  - `rope_emb` — [`L1175`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L1175)
  - `start_layer` — [`L1317`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L1317)
  - `use_ep` — [`L1144`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L1144)
  - `use_mla_kernel` — [`L1134`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L1134)
  - `vllm_config` — [`L1131`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L1131)
- protocol/private: `__call__`[`L1343`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L1343), `__init__`[`L1125`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L1125), `_create_deepseek_attention`[`L1188`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L1188)
- uses (calls/refs, reference-scoped): [`JaxModule`](../../layers/__init__.md#JaxModule), [`AttentionMetadata`](../../layers/common/attention_metadata.md#AttentionMetadata), [`MoEBackend`](../../layers/common/moe.md#MoEBackend), [`JaxRmsNorm`](../../layers/jax/norm.md#JaxRmsNorm), [`PPMissingLayer`](../../layers/jax/pp_utils.md#PPMissingLayer), [`get_pp_group`](../../distributed/jax_parallel_state.md#get_pp_group), [`ShardingAxisNameBase`](../../layers/common/sharding.md#ShardingAxisNameBase), [`make_layers`](../../layers/jax/pp_utils.md#make_layers), [`select_moe_backend`](../../layers/jax/moe/utils.md#select_moe_backend), [`JaxEmbed`](../../layers/jax/embed.md#JaxEmbed), [`fused_moe_backends`](../../layers/common/moe.md#MoEBackend.fused_moe_backends), [`is_first_rank`](../../distributed/jax_parallel_state.md#GroupCoordinator.is_first_rank), [`is_last_rank`](../../distributed/jax_parallel_state.md#GroupCoordinator.is_last_rank), [`initialize_cache`](../../layers/jax/rope.md#DeepseekScalingRotaryEmbedding.initialize_cache), [`ATTN_HEAD`](../../layers/common/sharding.md#ShardingAxisNameBase.ATTN_HEAD), [`_init_fn`](../../layers/jax/base.md#_init_fn), [`rotary_dim`](../../layers/jax/rope.md#RotaryEmbedding.rotary_dim), [`hidden_size`](deepseek_v3.md#hidden_size.hidden_size), [`quant_config`](deepseek_v3.md#DeepseekV3MLP.quant_config), [`ATTN_DATA`](../../layers/common/sharding.md#ShardingAxisNameBase.ATTN_DATA), [`MLP_DATA`](../../layers/common/sharding.md#ShardingAxisNameBase.MLP_DATA), [`expert_axis_name`](deepseek_v3.md#expert_axis_name), [`rope_scaling`](deepseek_v3.md#rope_scaling), [`scaling_factor`](../../layers/jax/rope.md#DeepseekScalingRotaryEmbedding.scaling_factor), [`DeepseekV3MLP`](deepseek_v3.md#DeepseekV3MLP), [`dtype`](deepseek_v3.md#DeepseekV3MLP.dtype), [`DeepseekScalingRotaryEmbedding`](../../layers/jax/rope.md#DeepseekScalingRotaryEmbedding), [`DeepseekV3Attention`](deepseek_v3.md#DeepseekV3Attention), [`DeepseekV3MLA`](deepseek_v3.md#DeepseekV3MLA), [`logger`](deepseek_v3.md#logger), [`MLP_TENSOR`](../../layers/common/sharding.md#ShardingAxisNameBase.MLP_TENSOR), [`rope_theta`](../../layers/jax/rope.md#RotaryEmbedding.rope_theta), [`df_sharding`](deepseek_v3.md#DeepseekV3MLP.df_sharding), [`get_expert_parallelism`](../../layers/jax/moe/utils.md#get_expert_parallelism), [`original_max_position_embeddings`](../../layers/jax/rope.md#RotaryEmbedding.original_max_position_embeddings), [`rms_norm_eps`](deepseek_v3.md#rms_norm_eps.rms_norm_eps), [`DeepseekV2Moe`](deepseek_v3.md#DeepseekV2Moe), [`DeepseekV3DecoderLayer`](deepseek_v3.md#DeepseekV3DecoderLayer), [`activation_ffw_td`](deepseek_v3.md#DeepseekV3MLP.activation_ffw_td), [`dtype`](../../layers/jax/rope.md#RotaryEmbedding.dtype)  (+22 more)
- used by: [`JaxModule`](../../layers/__init__.md#JaxModule), [`load_weights`](deepseek_v3.md#DeepseekV3ForCausalLM.load_weights), [`model`](deepseek_v3.md#DeepseekV3ForCausalLM.model)

### `DeepSeekV3Router`
- def: [`tpu_inference/models/jax/deepseek_v3.py:983`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L983)
- members:
  - `__call__(self, x_TD: Float)` — [`L1080`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L1080) — Routes tokens to top k experts.
  - `get_topk_indices(self, scores_TE: Float)` — [`L1046`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L1046) — Get the topk indices of the scores.
  - `activation_ffw_td` — [`L1017`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L1017)
  - `dtype` — [`L1016`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L1016)
  - `e_score_correction_bias` — [`L1039`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L1039)
  - `e_sharding` — [`L1019`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L1019)
  - `ed_sharding` — [`L1018`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L1018)
  - `hidden_size` — [`L1009`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L1009)
  - `moe_backend` — [`L1024`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L1024) — ---
  - `n_groups` — [`L1012`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L1012)
  - `norm_topk_prob` — [`L1014`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L1014)
  - `num_experts` — [`L1010`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L1010)
  - `num_experts_per_tok` — [`L1011`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L1011)
  - `quant_config` — [`L1021`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L1021)
  - `random_init` — [`L1020`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L1020)
  - `routed_scaling_factor` — [`L1015`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L1015)
  - `router_bias_dtype` — [`L1022`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L1022)
  - `scoring_func` — [`L1023`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L1023)
  - `topk_groups` — [`L1013`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L1013)
- protocol/private: `__init__`[`L989`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L989)
- uses (calls/refs, reference-scoped): [`MoEBackend`](../../layers/common/moe.md#MoEBackend), [`JaxEinsum`](../../layers/jax/linear.md#JaxEinsum), [`create_param`](../../layers/jax/base.md#create_param), [`QuantizationConfig`](../../layers/jax/quantization/configs.md#QuantizationConfig), [`__call__`](../../layers/jax/linear.md#JaxEinsum.__call__), [`fused_moe_backends`](../../layers/common/moe.md#MoEBackend.fused_moe_backends), [`_weight_init`](deepseek_v3.md#_weight_init), [`DENSE_MAT`](../../layers/common/moe.md#MoEBackend.DENSE_MAT), [`__init__`](../../layers/jax/linear.md#JaxEinsum.__init__)
- used by: [`gate`](deepseek_v3.md#DeepseekV2Moe.gate)

### `DeepseekV2Moe`  ·  implements/extends JaxModule
- def: [`tpu_inference/models/jax/deepseek_v3.py:840`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L840)
- doc: Jax implementation of Deepseek MoE layer
- signature: `class DeepseekV2Moe(JaxModule):`
- members:
  - `experts` — [`L904`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L904)
  - `gate` — [`L858`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L858)
  - `shared_experts` — [`L876`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L876)
- protocol/private: `__call__`[`L933`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L933), `__init__`[`L846`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L846)
- uses (calls/refs, reference-scoped): [`JaxModule`](../../layers/__init__.md#JaxModule), [`MoEBackend`](../../layers/common/moe.md#MoEBackend), [`ShardingAxisNameBase`](../../layers/common/sharding.md#ShardingAxisNameBase), [`moe_backend`](../../layers/jax/moe/moe.md#JaxMoE.moe_backend), [`GMM_TP`](../../layers/common/moe.md#MoEBackend.GMM_TP), [`ATTN_HEAD`](../../layers/common/sharding.md#ShardingAxisNameBase.ATTN_HEAD), [`edf_sharding`](../../layers/jax/moe/moe.md#JaxMoE.edf_sharding), [`quant_config`](../../layers/jax/moe/moe.md#JaxMoE.quant_config), [`hidden_size`](deepseek_v3.md#hidden_size.hidden_size), [`quant_config`](deepseek_v3.md#DeepseekV3MLP.quant_config), [`MLP_DATA`](../../layers/common/sharding.md#ShardingAxisNameBase.MLP_DATA), [`activation_ffw_td`](../../layers/jax/moe/moe.md#JaxMoE.activation_ffw_td), [`dtype`](../../layers/jax/moe/moe.md#JaxMoE.dtype), [`efd_sharding`](../../layers/jax/moe/moe.md#JaxMoE.efd_sharding), [`expert_axis_name`](deepseek_v3.md#expert_axis_name), [`prefix`](../../layers/jax/moe/moe.md#JaxMoE.prefix), [`DeepseekV3MLP`](deepseek_v3.md#DeepseekV3MLP), [`dtype`](deepseek_v3.md#DeepseekV3MLP.dtype), [`mesh`](../../layers/jax/moe/moe.md#JaxMoE.mesh), [`shared_experts`](deepseek_v3.md#SharedFusedMoe.shared_experts), [`ATTN_DATA_EXPERT`](../../layers/common/sharding.md#ShardingAxisNameBase.ATTN_DATA_EXPERT), [`MOE_TENSOR`](../../layers/common/sharding.md#ShardingAxisNameBase.MOE_TENSOR), [`expert_axis_name`](../../layers/jax/moe/moe.md#JaxMoE.expert_axis_name), [`qwix_quantized_weight_dtype`](../../layers/jax/moe/moe.md#JaxMoE.qwix_quantized_weight_dtype), [`router`](../../layers/jax/moe/moe.md#JaxMoE.router), [`SharedFusedMoe`](deepseek_v3.md#SharedFusedMoe), [`activation_ffw_ted`](../../layers/jax/moe/moe.md#JaxMoE.activation_ffw_ted), [`apply_expert_weight_before_computation`](../../layers/jax/moe/moe.md#JaxMoE.apply_expert_weight_before_computation), [`df_sharding`](deepseek_v3.md#DeepseekV3MLP.df_sharding), [`enable_return_routed_experts`](../../layers/jax/moe/moe.md#JaxMoE.enable_return_routed_experts), [`hidden_act`](../../layers/jax/moe/moe.md#JaxMoE.hidden_act), [`hidden_size`](../../layers/jax/moe/moe.md#JaxMoE.hidden_size), [`intermediate_size_moe`](../../layers/jax/moe/moe.md#JaxMoE.intermediate_size_moe), [`num_expert_parallelism`](../../layers/jax/moe/moe.md#JaxMoE.num_expert_parallelism), [`num_local_experts`](../../layers/jax/moe/moe.md#JaxMoE.num_local_experts), [`DeepSeekV3Router`](deepseek_v3.md#DeepSeekV3Router), [`activation_ffw_td`](deepseek_v3.md#DeepseekV3MLP.activation_ffw_td), [`fd_sharding`](deepseek_v3.md#DeepseekV3MLP.fd_sharding), [`hidden_act`](deepseek_v3.md#DeepseekV3MLP.hidden_act), [`hidden_act`](deepseek_v3.md#hidden_act.hidden_act)  (+13 more)
- used by: [`JaxModule`](../../layers/__init__.md#JaxModule), [`get_decoder_layer`](deepseek_v3.md#DeepSeekV3.get_decoder_layer)

### `DeepseekV3Attention`  ·  implements/extends DeepseekV3BaseAttention
- def: [`tpu_inference/models/jax/deepseek_v3.py:317`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L317)
- doc: Standard Multi-Head Attention (MHA) for DeepSeek models.
- signature: `class DeepseekV3Attention(DeepseekV3BaseAttention):`
- members:
  - `compute_attention(self, q_data: jax.Array, kv_data: Tuple[jax.Array, jax.Array], kv_cache: KVCache, md: AttentionMetadata)` — [`L402`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L402) — Computes self-attention for MHA.
  - `compute_kv_projection(self, x_SD: jax.Array, input_positions: jax.Array)` — [`L359`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L359) — Computes the key-value projection for MHA.
  - `compute_q_projection(self, x_q_TD: jax.Array, input_positions: jax.Array)` — [`L335`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L335) — Computes the query projection for MHA.
  - `kv_b_proj` — [`L324`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L324)
- protocol/private: `__post_init__`[`L320`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L320), `_ragged_paged_attention`[`L437`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L437)
- uses (calls/refs, reference-scoped): [`AttentionMetadata`](../../layers/common/attention_metadata.md#AttentionMetadata), [`JaxEinsum`](../../layers/jax/linear.md#JaxEinsum), [`ShardingAxisNameBase`](../../layers/common/sharding.md#ShardingAxisNameBase), [`ragged_paged_attention`](../../kernels/ragged_paged_attention/v3/kernel.md#ragged_paged_attention), [`seq_lens`](../../layers/common/attention_metadata.md#AttentionMetadata.seq_lens), [`kv_a_proj_with_mqa`](deepseek_v3.md#DeepseekV3BaseAttention.kv_a_proj_with_mqa), [`q_b_proj`](deepseek_v3.md#DeepseekV3BaseAttention.q_b_proj), [`quantize_kv`](../../layers/common/__init__.md#quantize_kv), [`block_tables`](../../layers/common/attention_metadata.md#AttentionMetadata.block_tables), [`query_start_loc`](../../layers/common/attention_metadata.md#AttentionMetadata.query_start_loc), [`request_distribution`](../../layers/common/attention_metadata.md#AttentionMetadata.request_distribution), [`kv_a_layernorm`](deepseek_v3.md#DeepseekV3BaseAttention.kv_a_layernorm), [`q_a_layernorm`](deepseek_v3.md#DeepseekV3BaseAttention.q_a_layernorm), [`q_a_proj`](deepseek_v3.md#DeepseekV3BaseAttention.q_a_proj), [`qk_head_dim`](deepseek_v3.md#DeepseekV3BaseAttention.qk_head_dim), [`dtype`](deepseek_v3.md#DeepseekV3BaseAttention.dtype), [`kv_cache_quantized_dtype`](deepseek_v3.md#DeepseekV3BaseAttention.kv_cache_quantized_dtype), [`N`](deepseek_v3.md#DeepseekV3BaseAttention.N), [`qk_nope_head_dim`](deepseek_v3.md#DeepseekV3BaseAttention.qk_nope_head_dim), [`apply_rope`](../../layers/jax/rope.md#DeepseekScalingRotaryEmbedding.apply_rope), [`quant_config`](deepseek_v3.md#DeepseekV3BaseAttention.quant_config), [`ATTN_HEAD`](../../layers/common/sharding.md#ShardingAxisNameBase.ATTN_HEAD), [`DeepseekV3BaseAttention`](deepseek_v3.md#DeepseekV3BaseAttention), [`scale`](deepseek_v3.md#DeepseekV3BaseAttention.scale), [`kv_lora_rank`](deepseek_v3.md#DeepseekV3BaseAttention.kv_lora_rank), [`prefix`](deepseek_v3.md#DeepseekV3BaseAttention.prefix), [`v_head_dim`](deepseek_v3.md#DeepseekV3BaseAttention.v_head_dim), [`rope`](deepseek_v3.md#DeepseekV3BaseAttention.rope), [`_weight_init`](deepseek_v3.md#_weight_init), [`KVCache`](deepseek_v3.md#KVCache), [`keyvalue_skh`](deepseek_v3.md#DeepseekV3BaseAttention.keyvalue_skh), [`ap_sharding`](deepseek_v3.md#DeepseekV3BaseAttention.ap_sharding), [`qk_rope_head_dim`](deepseek_v3.md#DeepseekV3BaseAttention.qk_rope_head_dim), [`query_tnh`](deepseek_v3.md#DeepseekV3BaseAttention.query_tnh), [`random_init`](deepseek_v3.md#DeepseekV3BaseAttention.random_init), [`__post_init__`](deepseek_v3.md#DeepseekV3BaseAttention.__post_init__), [`_k_scale`](deepseek_v3.md#DeepseekV3BaseAttention._k_scale), [`attn_o_nth`](deepseek_v3.md#DeepseekV3BaseAttention.attn_o_nth), [`mesh`](deepseek_v3.md#DeepseekV3BaseAttention.mesh), [`BATCH`](../../layers/common/sharding.md#ShardingAxisNameBase.BATCH)  (+1 more)
- used by: [`_create_deepseek_attention`](deepseek_v3.md#DeepSeekV3._create_deepseek_attention), [`__init__`](deepseek_v3.md#DeepseekV3DecoderLayer.__init__), [`DeepseekV3BaseAttention`](deepseek_v3.md#DeepseekV3BaseAttention), [`compute_attention`](deepseek_v3.md#DeepseekV3BaseAttention.compute_attention), [`compute_kv_projection`](deepseek_v3.md#DeepseekV3BaseAttention.compute_kv_projection), [`compute_q_projection`](deepseek_v3.md#DeepseekV3BaseAttention.compute_q_projection)

### `DeepseekV3BaseAttention`  ·  implements/extends JaxModule
- def: [`tpu_inference/models/jax/deepseek_v3.py:112`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L112)
- doc: Base class containing shared logic for DeepSeek Attention mechanisms.
- signature: `class DeepseekV3BaseAttention(JaxModule):`
- members:
  - `__call__(self, x: jax.Array, kv_cache: KVCache, attention_metadata: AttentionMetadata)` — [`L268`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L268) — Performs the forward pass of the attention module.  Expects that the
  - `compute_attention(self, *args)` — [`L262`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L262)
  - `compute_kv_projection(self, *args)` — [`L258`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L258)
  - `compute_q_projection(self, *args)` — [`L254`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L254)
  - `process_output(self, outputs_TNH)` — [`L265`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L265)
  - `D` — [`L166`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L166)
  - `K` — [`L165`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L165)
  - `N` — [`L164`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L164)
  - `activation_attention_out_td` — [`L146`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L146)
  - `activation_attention_td` — [`L140`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L140)
  - `activation_q_td` — [`L141`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L141)
  - `ap_sharding` — [`L138`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L138)
  - `attn_o_nth` — [`L145`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L145)
  - `dtype` — [`L123`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L123)
  - `head_dim` — [`L121`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L121)
  - `hidden_size` — [`L118`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L118)
  - `keyvalue_skh` — [`L144`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L144)
  - `kv_a_layernorm` — [`L227`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L227)
  - `kv_a_proj_with_mqa` — [`L198`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L198)
  - `kv_b_proj` — [`L242`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L242)
  - `kv_cache_dtype` — [`L124`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L124)
  - `kv_cache_quantized_dtype` — [`L237`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L237)
  - `kv_da_sharding` — [`L139`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L139)
  - `kv_lora_rank` — [`L129`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L129)
  - `mesh` — [`L125`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L125)
  - `num_attention_heads` — [`L119`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L119)
  - `num_key_value_heads` — [`L120`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L120)
  - `o_proj` — [`L208`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L208)
  - `prefix` — [`L161`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L161)
  - `q_a_layernorm` — [`L217`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L217)
  - `q_a_proj` — [`L179`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L179)
  - `q_b_proj` — [`L189`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L189)
  - `q_da_sharding` — [`L137`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L137)
  - `q_lora_rank` — [`L128`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L128)
  - `qk_head_dim` — [`L167`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L167)
  - `qk_nope_head_dim` — [`L130`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L130)
  - `qk_rope_head_dim` — [`L131`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L131)
  - `quant_config` — [`L154`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L154)
  - `query_nth` — [`L142`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L142)
  - `query_tnh` — [`L143`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L143)
  - `random_init` — [`L148`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L148)
  - `rd_sharding` — [`L136`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L136)
  - `rms_norm_eps` — [`L133`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L133)
  - `rngs` — [`L152`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L152)
  - `rope` — [`L122`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L122)
  - `rope_mscale_all_dim` — [`L149`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L149)
  - `scale` — [`L175`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L175)
  - `v_head_dim` — [`L132`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L132)
- protocol/private: `__post_init__`[`L163`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L163), `_k_scale`[`L158`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L158), `_q_scale`[`L157`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L157), `_v_scale`[`L159`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L159)
- uses (calls/refs, reference-scoped): [`JaxModule`](../../layers/__init__.md#JaxModule), [`AttentionMetadata`](../../layers/common/attention_metadata.md#AttentionMetadata), [`JaxEinsum`](../../layers/jax/linear.md#JaxEinsum), [`JaxRmsNorm`](../../layers/jax/norm.md#JaxRmsNorm), [`compute_attention`](deepseek_v3.md#DeepseekV3Attention.compute_attention), [`compute_attention`](deepseek_v3.md#DeepseekV3MLA.compute_attention), [`input_positions`](../../layers/common/attention_metadata.md#AttentionMetadata.input_positions), [`compute_kv_projection`](deepseek_v3.md#DeepseekV3Attention.compute_kv_projection), [`QuantizationConfig`](../../layers/jax/quantization/configs.md#QuantizationConfig), [`compute_q_projection`](deepseek_v3.md#DeepseekV3Attention.compute_q_projection), [`compute_q_projection`](deepseek_v3.md#DeepseekV3MLA.compute_q_projection), [`compute_kv_projection`](deepseek_v3.md#DeepseekV3MLA.compute_kv_projection), [`_init_fn`](../../layers/jax/base.md#_init_fn), [`_weight_init`](deepseek_v3.md#_weight_init), [`KVCache`](deepseek_v3.md#KVCache), [`scaling_factor`](../../layers/jax/rope.md#DeepseekScalingRotaryEmbedding.scaling_factor), [`DeepseekScalingRotaryEmbedding`](../../layers/jax/rope.md#DeepseekScalingRotaryEmbedding), [`DeepseekV3Attention`](deepseek_v3.md#DeepseekV3Attention), [`DeepseekV3MLA`](deepseek_v3.md#DeepseekV3MLA), [`process_output`](deepseek_v3.md#DeepseekV3MLA.process_output)
- used by: [`JaxModule`](../../layers/__init__.md#JaxModule), [`compute_attention`](deepseek_v3.md#DeepseekV3Attention.compute_attention), [`compute_attention`](deepseek_v3.md#DeepseekV3MLA.compute_attention), [`compute_kv_projection`](deepseek_v3.md#DeepseekV3Attention.compute_kv_projection), [`kv_b_proj`](deepseek_v3.md#DeepseekV3Attention.kv_b_proj), [`compute_q_projection`](deepseek_v3.md#DeepseekV3Attention.compute_q_projection), [`compute_q_projection`](deepseek_v3.md#DeepseekV3MLA.compute_q_projection), [`kv_b_proj`](deepseek_v3.md#DeepseekV3MLA.kv_b_proj), [`compute_kv_projection`](deepseek_v3.md#DeepseekV3MLA.compute_kv_projection), [`__post_init__`](deepseek_v3.md#DeepseekV3Attention.__post_init__), [`__post_init__`](deepseek_v3.md#DeepseekV3MLA.__post_init__), [`_ragged_paged_attention`](deepseek_v3.md#DeepseekV3Attention._ragged_paged_attention), [`DeepseekV3Attention`](deepseek_v3.md#DeepseekV3Attention), [`DeepseekV3MLA`](deepseek_v3.md#DeepseekV3MLA), [`process_output`](deepseek_v3.md#DeepseekV3MLA.process_output)

### `DeepseekV3DecoderLayer`  ·  implements/extends JaxModule
- def: [`tpu_inference/models/jax/deepseek_v3.py:937`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L937)
- doc: Implementats the DecoderLayer for DeepseekV3.
- signature: `class DeepseekV3DecoderLayer(JaxModule):`
- members:
  - `input_layernorm` — [`L951`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L951)
  - `mlp` — [`L954`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L954)
  - `post_attention_layernorm` — [`L952`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L952)
  - `self_attn` — [`L953`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L953)
- protocol/private: `__call__`[`L956`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L956), `__init__`[`L942`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L942)
- uses (calls/refs, reference-scoped): [`JaxModule`](../../layers/__init__.md#JaxModule), [`AttentionMetadata`](../../layers/common/attention_metadata.md#AttentionMetadata), [`JaxRmsNorm`](../../layers/jax/norm.md#JaxRmsNorm), [`DeepseekV3MLP`](deepseek_v3.md#DeepseekV3MLP), [`DeepseekV3Attention`](deepseek_v3.md#DeepseekV3Attention), [`DeepseekV3MLA`](deepseek_v3.md#DeepseekV3MLA), [`SharedFusedMoe`](deepseek_v3.md#SharedFusedMoe)
- used by: [`JaxModule`](../../layers/__init__.md#JaxModule), [`get_decoder_layer`](deepseek_v3.md#DeepSeekV3.get_decoder_layer)

### `DeepseekV3ForCausalLM`  ·  implements/extends JaxModule, LoadableWithIterator
- def: [`tpu_inference/models/jax/deepseek_v3.py:1374`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L1374)
- signature: `class DeepseekV3ForCausalLM(JaxModule, LoadableWithIterator):`
- members:
  - `compute_logits(self, hidden_states: jax.Array)` — [`L1438`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L1438)
  - `load_weights(self, weights: Iterable)` — [`L1441`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L1441)
  - `lm_head` — [`L1394`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L1394)
  - `mesh` — [`L1380`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L1380)
  - `model` — [`L1382`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L1382)
  - `vllm_config` — [`L1378`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L1378)
- protocol/private: `__call__`[`L1407`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L1407), `__init__`[`L1376`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L1376)
- uses (calls/refs, reference-scoped): [`JaxModule`](../../layers/__init__.md#JaxModule), [`AttentionMetadata`](../../layers/common/attention_metadata.md#AttentionMetadata), [`JaxIntermediateTensors`](jax_intermediate_tensor.md#JaxIntermediateTensors), [`PPMissingLayer`](../../layers/jax/pp_utils.md#PPMissingLayer), [`ShardingAxisNameBase`](../../layers/common/sharding.md#ShardingAxisNameBase), [`LoadableWithIterator`](utils/weight_utils.md#LoadableWithIterator), [`JaxLmHead`](../../layers/jax/linear.md#JaxLmHead), [`load_weights`](utils/weight_utils.md#LoadableWithIterator.load_weights), [`tensors`](jax_intermediate_tensor.md#JaxIntermediateTensors.tensors), [`named_parameters`](../../layers/__init__.md#JaxModule.named_parameters), [`_init_fn`](../../layers/jax/base.md#_init_fn), [`is_last_rank`](deepseek_v3.md#DeepSeekV3.is_last_rank), [`layers`](deepseek_v3.md#DeepSeekV3.layers), [`load_weights`](utils/weight_utils.md#JaxAutoWeightsLoader.load_weights), [`initialize_cache`](deepseek_v3.md#DeepSeekV3.initialize_cache), [`logger`](deepseek_v3.md#logger), [`JaxAutoWeightsLoader`](utils/weight_utils.md#JaxAutoWeightsLoader), [`MLP_TENSOR`](../../layers/common/sharding.md#ShardingAxisNameBase.MLP_TENSOR), [`DeepSeekV3`](deepseek_v3.md#DeepSeekV3)
- used by: [`JaxModule`](../../layers/__init__.md#JaxModule), [`_get_model_architecture`](../common/model_loader.md#_get_model_architecture), [`LoadableWithIterator`](utils/weight_utils.md#LoadableWithIterator), [`load_weights`](utils/weight_utils.md#LoadableWithIterator.load_weights)

### `DeepseekV3MLA`  ·  implements/extends DeepseekV3BaseAttention
- def: [`tpu_inference/models/jax/deepseek_v3.py:587`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L587)
- doc: Multi-Head Latent Attention (MLA) for DeepSeek V3.
- signature: `class DeepseekV3MLA(DeepseekV3BaseAttention):`
- members:
  - `compute_attention(self, q_data: Tuple[jax.Array, jax.Array], kv_data: Tuple[jax.Array, jax.Array], kv_cache: KVCache, md: AttentionMetadata)` — [`L663`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L663) — Computes the attention for MLA.
  - `compute_kv_projection(self, x_SD: jax.Array, input_positions: jax.Array)` — [`L635`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L635) — Computes the key-value projection for MLA.
  - `compute_q_projection(self, x_q_TD: jax.Array, input_positions: jax.Array)` — [`L607`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L607) — Computes the query projection for MLA.
  - `process_output(self, outputs_NTA: jax.Array)` — [`L720`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L720) — Processes output for MLA specifically.
  - `anh_sharding` — [`L589`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L589)
  - `kv_b_proj` — [`L595`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L595)
- protocol/private: `__post_init__`[`L591`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L591)
- uses (calls/refs, reference-scoped): [`AttentionMetadata`](../../layers/common/attention_metadata.md#AttentionMetadata), [`kv_a_proj_with_mqa`](deepseek_v3.md#DeepseekV3BaseAttention.kv_a_proj_with_mqa), [`q_b_proj`](deepseek_v3.md#DeepseekV3BaseAttention.q_b_proj), [`quantize_kv`](../../layers/common/__init__.md#quantize_kv), [`kv_a_layernorm`](deepseek_v3.md#DeepseekV3BaseAttention.kv_a_layernorm), [`mla_attention`](../../layers/common/attention_interface.md#mla_attention), [`q_a_layernorm`](deepseek_v3.md#DeepseekV3BaseAttention.q_a_layernorm), [`q_a_proj`](deepseek_v3.md#DeepseekV3BaseAttention.q_a_proj), [`qk_head_dim`](deepseek_v3.md#DeepseekV3BaseAttention.qk_head_dim), [`dtype`](deepseek_v3.md#DeepseekV3BaseAttention.dtype), [`kv_cache_quantized_dtype`](deepseek_v3.md#DeepseekV3BaseAttention.kv_cache_quantized_dtype), [`N`](deepseek_v3.md#DeepseekV3BaseAttention.N), [`qk_nope_head_dim`](deepseek_v3.md#DeepseekV3BaseAttention.qk_nope_head_dim), [`apply_rope`](../../layers/jax/rope.md#DeepseekScalingRotaryEmbedding.apply_rope), [`quant_config`](deepseek_v3.md#DeepseekV3BaseAttention.quant_config), [`DeepseekV3BaseAttention`](deepseek_v3.md#DeepseekV3BaseAttention), [`scale`](deepseek_v3.md#DeepseekV3BaseAttention.scale), [`kv_lora_rank`](deepseek_v3.md#DeepseekV3BaseAttention.kv_lora_rank), [`prefix`](deepseek_v3.md#DeepseekV3BaseAttention.prefix), [`v_head_dim`](deepseek_v3.md#DeepseekV3BaseAttention.v_head_dim), [`rope`](deepseek_v3.md#DeepseekV3BaseAttention.rope), [`_weight_init`](deepseek_v3.md#_weight_init), [`KVCache`](deepseek_v3.md#KVCache), [`keyvalue_skh`](deepseek_v3.md#DeepseekV3BaseAttention.keyvalue_skh), [`static_per_tensor_quantize_tensor`](../../layers/common/__init__.md#static_per_tensor_quantize_tensor), [`ap_sharding`](deepseek_v3.md#DeepseekV3BaseAttention.ap_sharding), [`MLAEinsum`](deepseek_v3.md#MLAEinsum), [`query_tnh`](deepseek_v3.md#DeepseekV3BaseAttention.query_tnh), [`random_init`](deepseek_v3.md#DeepseekV3BaseAttention.random_init), [`__post_init__`](deepseek_v3.md#DeepseekV3BaseAttention.__post_init__), [`_k_scale`](deepseek_v3.md#DeepseekV3BaseAttention._k_scale), [`_q_scale`](deepseek_v3.md#DeepseekV3BaseAttention._q_scale), [`attn_o_nth`](deepseek_v3.md#DeepseekV3BaseAttention.attn_o_nth), [`mesh`](deepseek_v3.md#DeepseekV3BaseAttention.mesh), [`num_attention_heads`](deepseek_v3.md#DeepseekV3BaseAttention.num_attention_heads), [`query_nth`](deepseek_v3.md#DeepseekV3BaseAttention.query_nth)
- used by: [`_create_deepseek_attention`](deepseek_v3.md#DeepSeekV3._create_deepseek_attention), [`__init__`](deepseek_v3.md#DeepseekV3DecoderLayer.__init__), [`DeepseekV3BaseAttention`](deepseek_v3.md#DeepseekV3BaseAttention), [`compute_attention`](deepseek_v3.md#DeepseekV3BaseAttention.compute_attention), [`compute_kv_projection`](deepseek_v3.md#DeepseekV3BaseAttention.compute_kv_projection), [`compute_q_projection`](deepseek_v3.md#DeepseekV3BaseAttention.compute_q_projection), [`process_output`](deepseek_v3.md#DeepseekV3BaseAttention.process_output)

### `DeepseekV3MLP`  ·  implements/extends JaxModule
- def: [`tpu_inference/models/jax/deepseek_v3.py:738`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L738)
- doc: A Gated Feed-Forward Network (FFN) layer.
- signature: `class DeepseekV3MLP(JaxModule):`
- members:
  - `__call__(self, x_TD)` — [`L760`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L760) — Performs the forward pass of the FFW layer.
  - `activation_ffw_td` — [`L754`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L754)
  - `df_sharding` — [`L752`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L752)
  - `down_proj` — [`L804`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L804)
  - `dtype` — [`L748`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L748)
  - `fd_sharding` — [`L753`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L753)
  - `gate_proj` — [`L788`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L788)
  - `hidden_act` — [`L749`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L749)
  - `hidden_size` — [`L750`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L750)
  - `intermediate_size` — [`L751`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L751)
  - `quant_config` — [`L756`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L756)
  - `random_init` — [`L755`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L755)
  - `rngs` — [`L758`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L758)
  - `up_proj` — [`L796`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L796)
- protocol/private: `__post_init__`[`L783`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L783)
- uses (calls/refs, reference-scoped): [`JaxModule`](../../layers/__init__.md#JaxModule), [`JaxEinsum`](../../layers/jax/linear.md#JaxEinsum), [`QuantizationConfig`](../../layers/jax/quantization/configs.md#QuantizationConfig), [`ACT2FN`](../../layers/jax/layers.md#FlaxUtils.ACT2FN), [`_weight_init`](deepseek_v3.md#_weight_init), [`modeling_flax_utils`](deepseek_v3.md#modeling_flax_utils)
- used by: [`JaxModule`](../../layers/__init__.md#JaxModule), [`get_decoder_layer`](deepseek_v3.md#DeepSeekV3.get_decoder_layer), [`shared_experts`](deepseek_v3.md#DeepseekV2Moe.shared_experts), [`__init__`](deepseek_v3.md#DeepseekV3DecoderLayer.__init__), [`shared_experts`](deepseek_v3.md#SharedFusedMoe.shared_experts)

### `MLAEinsum`  ·  implements/extends JaxEinsum
- def: [`tpu_inference/models/jax/deepseek_v3.py:480`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L480)
- doc: Extending JaxEinsum to handle MLA.
- signature: `class MLAEinsum(JaxEinsum):`
- members:
  - `load_weights(self, weights)` — [`L512`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L512)
  - `named_children(self)` — [`L508`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L508)
  - `loaded` — [`L504`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L504)
  - `mla_layer` — [`L505`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L505)
  - `quant_config` — [`L506`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L506)
- protocol/private: `__init__`[`L488`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L488)
- uses (calls/refs, reference-scoped): [`JaxEinsum`](../../layers/jax/linear.md#JaxEinsum), [`weight`](../../layers/jax/linear.md#JaxEinsum.weight), [`QuantizationConfig`](../../layers/jax/quantization/configs.md#QuantizationConfig), [`cpu_mesh_context`](../../layers/common/utils.md#cpu_mesh_context), [`shard_put`](utils/weight_utils.md#shard_put), [`named_parameters`](../../layers/__init__.md#JaxModule.named_parameters), [`dequantize_tensor`](../../layers/common/__init__.md#dequantize_tensor), [`quantize_tensor`](../../kernels/quantized_matmul/util.md#quantize_tensor), [`__init__`](../../layers/jax/linear.md#JaxEinsum.__init__)
- used by: [`JaxEinsum`](../../layers/jax/linear.md#JaxEinsum), [`kv_b_proj`](deepseek_v3.md#DeepseekV3MLA.kv_b_proj), [`named_children`](../../layers/__init__.md#JaxModule.named_children)

### `SharedFusedMoe`  ·  implements/extends JaxMoE
- def: [`tpu_inference/models/jax/deepseek_v3.py:815`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L815)
- doc: Corresponds to vLLM's SharedFusedMoe.
- signature: `class SharedFusedMoe(JaxMoE):`
- members:
  - `routed_scaling_factor` — [`L824`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L824)
  - `shared_experts` — [`L822`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L822)
- protocol/private: `__call__`[`L826`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L826)
- uses (calls/refs, reference-scoped): [`JaxMoE`](../../layers/jax/moe/moe.md#JaxMoE), [`__call__`](../../layers/jax/moe/moe.md#JaxMoE.__call__), [`DeepseekV3MLP`](deepseek_v3.md#DeepseekV3MLP)
- used by: [`experts`](deepseek_v3.md#DeepseekV2Moe.experts), [`JaxMoE`](../../layers/jax/moe/moe.md#JaxMoE), [`__init__`](deepseek_v3.md#DeepseekV3DecoderLayer.__init__)

## Functions
- `_weight_init(random_init: bool)` — [`L70`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L70)

## Module values
- `KVCache` — [`L65`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L65)
- `expert_axis_name` — [`L108`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L108)
- `ffw_intermediate_size` — [`L82`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L82)
- `first_k_dense_replace` — [`L90`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L90)
- `hidden_act` — [`L87`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L87)
- `hidden_size` — [`L79`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L79)
- `interleave_moe_layer_step` — [`L86`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L86)
- `kv_lora_rank` — [`L104`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L104)
- `logger` — [`L67`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L67)
- `modeling_flax_utils` — [`L74`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L74)
- `moe_intermediate_size` — [`L83`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L83)
- `n_group` — [`L85`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L85)
- `num_attention_heads` — [`L80`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L80)
- `num_experts_per_token` — [`L84`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L84)
- `num_key_value_heads` — [`L81`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L81)
- `num_local_experts` — [`L77`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L77)
- `num_shared_experts` — [`L92`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L92)
- `q_lora_rank` — [`L103`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L103)
- `qk_nope_head_dim` — [`L105`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L105)
- `qk_rope_head_dim` — [`L106`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L106)
- `rms_norm_eps` — [`L88`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L88)
- `rope_scaling` — [`L94`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L94)
- `rope_theta` — [`L93`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L93)
- `routed_scaling_factor` — [`L89`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L89)
- `v_head_dim` — [`L107`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L107)
- `vocab_size` — [`L78`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/deepseek_v3.py#L78)


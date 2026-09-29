---
title: 'Module: simply/model_lib.py'
type: catalog
provenance: extracted
module: simply/model_lib.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.model_lib`/
symbols:
  LMInterface.generate: LMInterface#generate().
  Array: Array.
  run_experiment: run_experiment().
  Attention.apply: Attention#apply().
  MoEFeedForward.moe_ffn: MoEFeedForward#moe_ffn().
  PyTree: PyTree.
  Attention.init_decode_state: Attention#init_decode_state().
  TransformerBlock.attn: TransformerBlock#attn.
  TransformerBlock.ffn: TransformerBlock#ffn.
  TransformerLM.apply: TransformerLM#apply().
  LMInterface.score: LMInterface#score().
  TransformerBlock.apply: TransformerBlock#apply().
  continue_decode.body_fn: continue_decode().body_fn().
  MoEFeedForward.apply: MoEFeedForward#apply().
  LMInterface.score_tokens: LMInterface#score_tokens().
  TransformerBlock.init: TransformerBlock#init().
  TransformerLM: TransformerLM#
  LayerNorm.init: LayerNorm#init().
  Attention.per_dim_scale: Attention#per_dim_scale.
  LayerNorm.apply: LayerNorm#apply().
  FeedForward.apply: FeedForward#apply().
  MoEFeedForward.compute_block: MoEFeedForward#compute_block().
  MoEFeedForward._apply_dense_moe: MoEFeedForward#_apply_dense_moe().
  LMInterface.input_processor: LMInterface#input_processor.
  Attention.init: Attention#init().
  FeedForward.ffn_0: FeedForward#ffn_0.
  TransformerBlock.post_skip_ln_1: TransformerBlock#post_skip_ln_1.
  SamplingParams: SamplingParams.
  MoEFeedForward._apply_sparse_moe: MoEFeedForward#_apply_sparse_moe().
  MoEFeedForward._apply_sparse_moe_v2: MoEFeedForward#_apply_sparse_moe_v2().
  Attention._scale_qk: Attention#_scale_qk().
  SamplingState.pad_to: SamplingState#pad_to().
  TransformerLM.config: TransformerLM#config.
  TransformerLM.init: TransformerLM#init().
  train_one_step: train_one_step().
  SamplingOutput: SamplingOutput#
  FeedForward.ffn_0_gate: FeedForward#ffn_0_gate.
  FeedForward.ffn_1: FeedForward#ffn_1.
  TransformerBlock.pre_ln_1: TransformerBlock#pre_ln_1.
  TransformerBlock.post_ln_1: TransformerBlock#post_ln_1.
  LMInterface: LMInterface#
  LayerNorm: LayerNorm#
  TransformerBlock.pre_ln_0: TransformerBlock#pre_ln_0.
  TransformerBlock.post_ln_0: TransformerBlock#post_ln_0.
  TransformerBlock.post_skip_ln_0: TransformerBlock#post_skip_ln_0.
  SamplingState.has_ended: SamplingState#has_ended().
  SamplingState.position: SamplingState#position.
  PerDimScale.init: PerDimScale#init().
  FeedForward.init: FeedForward#init().
  MoEFeedForward.init: MoEFeedForward#init().
  Attention.attn_activation_partition: Attention#attn_activation_partition.
  get_init_state: get_init_state().
  pad_to_along_axis: pad_to_along_axis().
  TransformerLM.blocks: TransformerLM#blocks.
  SamplingState.reached_eos: SamplingState#reached_eos().
  continue_decode: continue_decode().
  MoEFeedForward.num_experts: MoEFeedForward#num_experts.
  ScoringParams.from_sampling_params: ScoringParams#from_sampling_params().
  PerDimScale.apply: PerDimScale#apply().
  TransformerBlock.activation_dtype: TransformerBlock#activation_dtype.
  LMInterface.__init__: LMInterface#__init__().
  Attention.v_proj: Attention#v_proj.
  updated_decode_state: updated_decode_state().
  Attention.window_size: Attention#window_size.
  InputEncoderInterface.apply: InputEncoderInterface#apply().
  continue_decode.cond_fn: continue_decode().cond_fn().
  PartitionAnnotation: PartitionAnnotation.
  get_partition_axis: get_partition_axis.
  EinsumLinear: EinsumLinear.
  FeedForward.expand_dim: FeedForward#expand_dim().
  MoEFeedForward: MoEFeedForward#
  MoEFeedForward.all_gather_if_sharded: MoEFeedForward#all_gather_if_sharded().
  TransformerBlock.sharding_config: TransformerBlock#sharding_config.
  TransformerLM.sharding_config: TransformerLM#sharding_config.
  TransformerLM.input_encoders: TransformerLM#input_encoders.
  SamplingState.input_tokens: SamplingState#input_tokens().
  TransformerBlock.model_dim: TransformerBlock#model_dim.
  ScoringParams: ScoringParams#
  LMInterface.count_num_tokens: LMInterface#count_num_tokens().
  FeedForward: FeedForward#
  MoEFeedForward.ffn_0: MoEFeedForward#ffn_0.
  MoEFeedForward.ffn_0_gate: MoEFeedForward#ffn_0_gate.
  MoEFeedForward.ffn_1: MoEFeedForward#ffn_1.
  Attention.activation_dtype: Attention#activation_dtype.
  TransformerBlock: TransformerBlock#
  build_global_array_from_replicated: build_global_array_from_replicated().
  SamplingState.tokens: SamplingState#tokens.
  SamplingState.next_position_is_output: SamplingState#next_position_is_output().
  pad_decode_state_to: pad_decode_state_to().
  quantize_tfm_params: quantize_tfm_params().
  PRNGKey: PRNGKey.
  attn: attn().
  MoEFeedForward.ffn0_partition: MoEFeedForward#ffn0_partition.
  Attention.n_kv_heads: Attention#n_kv_heads.
  create_model: create_model().
  get_raw_arrays: get_raw_arrays.
  round_up_to_base: round_up_to_base.
  PerDimScale: PerDimScale#
  create_mask: create_mask().
  FeedForward.sharding_config: FeedForward#sharding_config.
  Attention: Attention#
  Attention.qk_norm: Attention#qk_norm.
  TransformerLM._process_per_repeat: TransformerLM#_process_per_repeat().
  build_global_batch_from_sharded: build_global_batch_from_sharded().
  SamplingState.input_lens: SamplingState#input_lens.
  SamplingState.next_tokens: SamplingState#next_tokens().
  SamplingState.updated_tokens: SamplingState#updated_tokens().
  SamplingState.updated_token_logprobs: SamplingState#updated_token_logprobs().
  SamplingState.updated_token_scores: SamplingState#updated_token_scores().
  LMInterface.prefill_fn: LMInterface#prefill_fn().
  LMInterface.decode_fn: LMInterface#decode_fn.
  DTypeLike: DTypeLike.
  FeedForward.ffn_weight_init: FeedForward#ffn_weight_init.
  Attention.flash_attention_block_size: Attention#flash_attention_block_size.
  Attention.position_encoding: Attention#position_encoding.
  TransformerLM.embed_linear: TransformerLM#embed_linear.
  SamplingState: SamplingState#
  gelu: gelu().
  squared_relu: squared_relu().
  LayerNorm.activation_dtype: LayerNorm#activation_dtype.
  FeedForward.activation_dtype: FeedForward#activation_dtype.
  TransformerBlock.expand_dim: TransformerBlock#expand_dim().
  TransformerLM.predict_probs: TransformerLM#predict_probs().
  TransformerLM.init_decode_state: TransformerLM#init_decode_state().
  SamplingState.token_scores: SamplingState#token_scores.
  SamplingState.__post_init__: SamplingState#__post_init__().
  SimplyConfig: SimplyConfig.
  chunked_local_attn: chunked_local_attn().
  MoEFeedForward.num_experts_per_token: MoEFeedForward#num_experts_per_token.
  MoEFeedForward.ep_method: MoEFeedForward#ep_method.
  MoEFeedForward.router: MoEFeedForward#router.
  Attention.per_head_dim: Attention#per_head_dim.
  Attention.q_proj: Attention#q_proj.
  Attention.k_proj: Attention#k_proj.
  Attention.o_proj: Attention#o_proj.
  TransformerBlock.use_rmsnorm: TransformerBlock#use_rmsnorm.
  TransformerBlock.norm_scale_plus_one: TransformerBlock#norm_scale_plus_one.
  TransformerBlock.rms_norm_epsilon: TransformerBlock#rms_norm_epsilon.
  TransformerLM.final_ln: TransformerLM#final_ln.
  compute_loss: compute_loss().
  train_one_step._compute_grad: train_one_step()._compute_grad().
  ScoringParams.temperature: ScoringParams#temperature.
  ScoringParams.top_k: ScoringParams#top_k.
  ScoringParams.top_p: ScoringParams#top_p.
  permute: permute().
  Attention.qkv_partition: Attention#qkv_partition.
  Attention.attn_mask_value: Attention#attn_mask_value.
  TrainLoopRegistry: TrainLoopRegistry#
  SamplingState.token_logprobs: SamplingState#token_logprobs.
  LMInterface.default_sampling_params: LMInterface#default_sampling_params.
  Attention.n_heads: Attention#n_heads.
  Attention.weight_init: Attention#weight_init.
  Attention.attn_soft_cap: Attention#attn_soft_cap.
  TransformerBlock.attn_weight_init: TransformerBlock#attn_weight_init.
  TransformerBlock.ffn_weight_init: TransformerBlock#ffn_weight_init.
  TransformerBlock.position_encoding: TransformerBlock#position_encoding.
  TransformerLM._create_transformer_block: TransformerLM#_create_transformer_block().
  run_eval: run_eval().
  SamplingState.is_pad_seq: SamplingState#is_pad_seq().
  SamplingState.all_has_ended: SamplingState#all_has_ended().
  LMInterface.pad_state_to_fn: LMInterface#pad_state_to_fn.
  continue_decode.body_fn._score_fn: continue_decode().body_fn()._score_fn().
  get_scaling_info: get_scaling_info().
  AnnotatedArray: AnnotatedArray.
  neg_inf: neg_inf.
  LayerNorm.weight_dtype: LayerNorm#weight_dtype.
  SamplingState.decode_state: SamplingState#decode_state.
  TransformerBlock.init_decode_state: TransformerBlock#init_decode_state().
  TransformerLM.setup: TransformerLM#setup().
  TrainLoopRegistry.namespace: TrainLoopRegistry#namespace.
  SamplingOutput.input_text: SamplingOutput#input_text().
  SamplingOutput.output_text: SamplingOutput#output_text().
  SamplingOutput.sum_output_logprob: SamplingOutput#sum_output_logprob().
  SamplingOutput.sum_input_score: SamplingOutput#sum_input_score().
  SamplingOutput.sum_output_score: SamplingOutput#sum_output_score().
  ScoringOutput.input_text: ScoringOutput#input_text().
  ScoringOutput.output_text: ScoringOutput#output_text().
  LMInterface.eos_ids: LMInterface#eos_ids().
  soft_cap: soft_cap().
  Attention.weight_dtype: Attention#weight_dtype.
  InputEncoderInterface: InputEncoderInterface#
  TransformerLM._prepare_stack_list: TransformerLM#_prepare_stack_list().
  collect_tag: collect_tag().
  clip_tree_fn: clip_tree_fn().
  compute_batch_stats_info: compute_batch_stats_info().
  SamplingState.prng_key: SamplingState#prng_key.
  SamplingState.max_decode_steps: SamplingState#max_decode_steps.
  SamplingState.eos_ids: SamplingState#eos_ids.
  SamplingState.decode_state_length: SamplingState#decode_state_length().
  SamplingState.batch_size: SamplingState#batch_size().
  ScoringOutput.input_chunks: ScoringOutput#input_chunks.
  ScoringOutput.output_chunks: ScoringOutput#output_chunks.
  pad_along_axis: pad_along_axis().
  LayerNorm.use_bias: LayerNorm#use_bias.
  FeedForward.model_dim: FeedForward#model_dim.
  MoEFeedForward.ep_capacity_factor: MoEFeedForward#ep_capacity_factor.
  TransformerBlock.use_pre_ln: TransformerBlock#use_pre_ln.
  TransformerBlock.use_post_ln: TransformerBlock#use_post_ln.
  TransformerBlock.use_post_skip_ln: TransformerBlock#use_post_skip_ln.
  TransformerBlock.ffn_expand_dim: TransformerBlock#ffn_expand_dim.
  TransformerLM.activation_dtype: TransformerLM#activation_dtype.
  ScoringOutput.input_token_scores: ScoringOutput#input_token_scores.
  ScoringOutput.output_token_scores: ScoringOutput#output_token_scores.
  ExperimentHelper: ExperimentHelper.
  create_lr_schedule: create_lr_schedule.
  LayerNorm.scale_partition: LayerNorm#scale_partition.
  LayerNorm.bias_partition: LayerNorm#bias_partition.
  PerDimScale.weight_dtype: PerDimScale#weight_dtype.
  PerDimScale.activation_dtype: PerDimScale#activation_dtype.
  updated_decode_state._windowized_array: updated_decode_state()._windowized_array().
  _custom_permute_fwd: _custom_permute_fwd().
  Attention.o_partition: Attention#o_partition.
  Attention.output_partition: Attention#output_partition.
  Attention._preprocess_flash_qkv: Attention#_preprocess_flash_qkv().
  InputEncoderInterface.Output: InputEncoderInterface#Output#
  InputEncoderInterface.Output.embeddings: InputEncoderInterface#Output#embeddings.
  TransformerLM._replace_embeddings: TransformerLM#_replace_embeddings().
  collect_loss_and_metric: collect_loss_and_metric().
  compute_train_loss: compute_train_loss().
  compute_eval_loss: compute_eval_loss().
  compute_distill_loss: compute_distill_loss().
  train_one_step.grad_accum_step_fn: train_one_step().grad_accum_step_fn().
  run_experiment.train_one_step_fn: run_experiment().train_one_step_fn().
  flatten_dict: flatten_dict().
  SamplingOutput.input_chunks: SamplingOutput#input_chunks.
  SamplingOutput.output_chunks: SamplingOutput#output_chunks.
  ScoringOutput.params: ScoringOutput#params.
  LayerNorm.dim: LayerNorm#dim.
  LayerNorm.use_scale: LayerNorm#use_scale.
  FeedForward.use_gated_activation_in_ffn: FeedForward#use_gated_activation_in_ffn.
  FeedForward.ffn_use_bias: FeedForward#ffn_use_bias.
  _custom_permute: _custom_permute().
  MoEFeedForward.ffn1_partition: MoEFeedForward#ffn1_partition.
  Attention.model_dim: Attention#model_dim.
  Attention.use_per_dim_scale: Attention#use_per_dim_scale.
  TransformerBlock.expand_factor: TransformerBlock#expand_factor.
  InputEncoderInterface.name: InputEncoderInterface#name.
  InputEncoderInterface.extra_input_keys: InputEncoderInterface#extra_input_keys.
  tree_norm: tree_norm().
  LMInterface.model_params: LMInterface#model_params.
  SimplyModule: SimplyModule.
  RawT: RawT.
  get_default_mesh: get_default_mesh.
  maybe_dequantize_array: maybe_dequantize_array.
  mesh_sharding: mesh_sharding.
  Attention.setup: Attention#setup().
  TransformerBlock.attn_mask_value: TransformerBlock#attn_mask_value.
  TransformerBlock.setup: TransformerBlock#setup().
  SamplingOutput.processed_input: SamplingOutput#processed_input.
  SamplingOutput.avg_output_logprob: SamplingOutput#avg_output_logprob().
  SamplingOutput.avg_input_score: SamplingOutput#avg_input_score().
  SamplingOutput.avg_output_score: SamplingOutput#avg_output_score().
  ScoringOutput.sum_input_score: ScoringOutput#sum_input_score().
  ScoringOutput.avg_input_score: ScoringOutput#avg_input_score().
  ScoringOutput.sum_output_score: ScoringOutput#sum_output_score().
  ScoringOutput.avg_output_score: ScoringOutput#avg_output_score().
  Batch: Batch.
  LayerNorm.axis: LayerNorm#axis.
  LayerNorm.scale_plus_one: LayerNorm#scale_plus_one.
  PerDimScale.dim: PerDimScale#dim.
  gmm: gmm().
  FeedForward.ffn_expand_dim: FeedForward#ffn_expand_dim.
  MoEFeedForward.router_z_loss_weight: MoEFeedForward#router_z_loss_weight.
  MoEFeedForward.lbl_loss_weight: MoEFeedForward#lbl_loss_weight.
  MoEFeedForward.tile_batch_seq: MoEFeedForward#tile_batch_seq.
  MoEFeedForward.tile_model_dim: MoEFeedForward#tile_model_dim.
  MoEFeedForward.tile_expand_dim: MoEFeedForward#tile_expand_dim.
  MoEFeedForward.gmm_impl: MoEFeedForward#gmm_impl.
  MoEFeedForward.get_global_input_output_offsets: MoEFeedForward#get_global_input_output_offsets().
  Attention.query_scale: Attention#query_scale.
  Attention.total_num_pages: Attention#total_num_pages.
  Attention.page_size: Attention#page_size.
  TransformerBlock.per_head_dim: TransformerBlock#per_head_dim.
  TransformerBlock.use_gated_activation_in_ffn: TransformerBlock#use_gated_activation_in_ffn.
  TransformerBlock.use_moe: TransformerBlock#use_moe.
  TransformerBlock.ffn_use_bias: TransformerBlock#ffn_use_bias.
  TransformerBlock.ffn_activation: TransformerBlock#ffn_activation.
  tree_rms: tree_rms().
  safe_clip: safe_clip().
  compute_tree_info_fn: compute_tree_info_fn().
  SamplingOutput.output_token_logprobs: SamplingOutput#output_token_logprobs.
  SamplingOutput.input_token_scores: SamplingOutput#input_token_scores.
  SamplingOutput.output_token_scores: SamplingOutput#output_token_scores.
  ScoringOutput: ScoringOutput#
  LayerNorm.epsilon: LayerNorm#epsilon.
  updated_decode_state._update_kv: updated_decode_state()._update_kv().
  FeedForward.expand_factor: FeedForward#expand_factor.
  FeedForward.ffn_activation: FeedForward#ffn_activation.
  _custom_permute_bwd: _custom_permute_bwd().
  MoEFeedForward.ep_pipeline_stages: MoEFeedForward#ep_pipeline_stages.
  MoEFeedForward.ep_pipeline_comms: MoEFeedForward#ep_pipeline_comms.
  MoEFeedForward.ep_pipeline_fine_grained_ra2a: MoEFeedForward#ep_pipeline_fine_grained_ra2a.
  MoEFeedForward.ep_pipeline_use_opt_barriers: MoEFeedForward#ep_pipeline_use_opt_barriers.
  Attention.add_extra_output: Attention#add_extra_output.
  Attention.use_flash_attention: Attention#use_flash_attention.
  Attention.use_window_chunk: Attention#use_window_chunk.
  Attention.qkv_use_bias: Attention#qkv_use_bias.
  Attention.o_use_bias: Attention#o_use_bias.
  Attention._postprocess_flash_output: Attention#_postprocess_flash_output().
  Attention.flash_attention_fn: Attention#flash_attention_fn().
  TransformerBlock.n_heads: TransformerBlock#n_heads.
  TransformerBlock.use_qk_norm: TransformerBlock#use_qk_norm.
  TransformerBlock.use_per_dim_scale: TransformerBlock#use_per_dim_scale.
  TransformerBlock.num_experts: TransformerBlock#num_experts.
  TransformerBlock.num_experts_per_token: TransformerBlock#num_experts_per_token.
  TransformerBlock.ep_capacity_factor: TransformerBlock#ep_capacity_factor.
  TransformerBlock.lbl_loss_weight: TransformerBlock#lbl_loss_weight.
  TransformerBlock.router_z_loss_weight: TransformerBlock#router_z_loss_weight.
  TransformerBlock.use_flash_attention: TransformerBlock#use_flash_attention.
  TransformerBlock.flash_attention_block_size: TransformerBlock#flash_attention_block_size.
  TransformerBlock.window_size: TransformerBlock#window_size.
  TransformerBlock.use_window_chunk: TransformerBlock#use_window_chunk.
  TransformerBlock.n_kv_heads: TransformerBlock#n_kv_heads.
  TransformerBlock.qkv_use_bias: TransformerBlock#qkv_use_bias.
  TransformerBlock.o_use_bias: TransformerBlock#o_use_bias.
  TransformerBlock.attn_soft_cap: TransformerBlock#attn_soft_cap.
  TransformerBlock.query_scale: TransformerBlock#query_scale.
  TransformerBlock.tile_batch_seq: TransformerBlock#tile_batch_seq.
  TransformerBlock.tile_model_dim: TransformerBlock#tile_model_dim.
  TransformerBlock.tile_expand_dim: TransformerBlock#tile_expand_dim.
  TransformerBlock.gmm_impl: TransformerBlock#gmm_impl.
  TransformerBlock.total_num_pages: TransformerBlock#total_num_pages.
  TransformerBlock.page_size: TransformerBlock#page_size.
  InputEncoderInterface.Output.embedding_mask: InputEncoderInterface#Output#embedding_mask.
  TransformerLM.substitute_embeddings: TransformerLM#substitute_embeddings().
  TransformerLM.convert_to_lower_bits: TransformerLM#convert_to_lower_bits().
  build_global_batch_from_sharded._build_global_array_from_sharded: build_global_batch_from_sharded()._build_global_array_from_sharded().
  ScoringOutput.input_token_ids: ScoringOutput#input_token_ids.
  ScoringOutput.output_token_ids: ScoringOutput#output_token_ids.
  LMInterface.model: LMInterface#model.
  PerDimScale.axis: PerDimScale#axis.
  FeedForward.setup: FeedForward#setup().
  MoEFeedForward.setup: MoEFeedForward#setup().
  Attention.use_causal: Attention#use_causal.
  Attention.update_kv_cache_in_place: Attention#update_kv_cache_in_place.
  SamplingOutput.input_token_ids: SamplingOutput#input_token_ids.
  SamplingOutput.output_token_ids: SamplingOutput#output_token_ids.
  SamplingOutput.is_truncated: SamplingOutput#is_truncated.
---
# Module: [`simply/model_lib.py`](../../../../../raw/code/simply/simply/model_lib.py)

## Classes
### `Attention`
- def: [`simply/model_lib.py:1416`](../../../../../raw/code/simply/simply/model_lib.py#L1416)
- doc: Standard Multi-head Attention layer.
- signature: `class Attention(module.SimplyModule):`
- members:
  - `_postprocess_flash_output(self, output, shard_count)` — [`L1483`](../../../../../raw/code/simply/simply/model_lib.py#L1483) — Post-process flash attention output before axis swap.
  - `_preprocess_flash_qkv(self, q, segment_ids, q_seq_len, kv_seq_len, shard_count)` — [`L1454`](../../../../../raw/code/simply/simply/model_lib.py#L1454) — Prepare Q, segment_ids, and mask for flash attention.
  - `_scale_qk(self, q: Array, k: Array, segment_positions: Array, params: PyTree)` — [`L1497`](../../../../../raw/code/simply/simply/model_lib.py#L1497) — Scales query and key.
  - `apply(self, params: PyTree, x: Array, *, segment_ids: Array, segment_positions: Array, extra_inputs: PyTree = None, decode_state: PyTree = None)` — [`L1601`](../../../../../raw/code/simply/simply/model_lib.py#L1601) — documented in [simply-model_lib](../../concepts/simply-model_lib.md)
  - `flash_attention_fn(kernel, query, key, value, q_segment_ids, kv_segment_ids)` — [`L1742`](../../../../../raw/code/simply/simply/model_lib.py#L1742)
  - `init(self, prng_key: PRNGKey)` — [`L1584`](../../../../../raw/code/simply/simply/model_lib.py#L1584)
  - `init_decode_state(self, batch_size: int, max_seq_len: int)` — [`L1831`](../../../../../raw/code/simply/simply/model_lib.py#L1831)
  - `setup(self)` — [`L1532`](../../../../../raw/code/simply/simply/model_lib.py#L1532)
  - `activation_dtype` — [`L1428`](../../../../../raw/code/simply/simply/model_lib.py#L1428)
  - `add_extra_output` — [`L1423`](../../../../../raw/code/simply/simply/model_lib.py#L1423)
  - `attn_activation_partition` — [`L1433`](../../../../../raw/code/simply/simply/model_lib.py#L1433)
  - `attn_mask_value` — [`L1446`](../../../../../raw/code/simply/simply/model_lib.py#L1446)
  - `attn_soft_cap` — [`L1445`](../../../../../raw/code/simply/simply/model_lib.py#L1445)
  - `flash_attention_block_size` — [`L1439`](../../../../../raw/code/simply/simply/model_lib.py#L1439)
  - `k_proj` — [`L1562`](../../../../../raw/code/simply/simply/model_lib.py#L1562)
  - `model_dim` — [`L1419`](../../../../../raw/code/simply/simply/model_lib.py#L1419)
  - `n_heads` — [`L1420`](../../../../../raw/code/simply/simply/model_lib.py#L1420)
  - `n_kv_heads` — [`L1442`](../../../../../raw/code/simply/simply/model_lib.py#L1442)
  - `o_partition` — [`L1432`](../../../../../raw/code/simply/simply/model_lib.py#L1432)
  - `o_proj` — [`L1577`](../../../../../raw/code/simply/simply/model_lib.py#L1577)
  - `o_use_bias` — [`L1444`](../../../../../raw/code/simply/simply/model_lib.py#L1444)
  - `output_partition` — [`L1434`](../../../../../raw/code/simply/simply/model_lib.py#L1434)
  - `page_size` — [`L1450`](../../../../../raw/code/simply/simply/model_lib.py#L1450)
  - `per_dim_scale` — [`L1534`](../../../../../raw/code/simply/simply/model_lib.py#L1534)
  - `per_head_dim` — [`L1421`](../../../../../raw/code/simply/simply/model_lib.py#L1421)
  - `position_encoding` — [`L1452`](../../../../../raw/code/simply/simply/model_lib.py#L1452)
  - `q_proj` — [`L1556`](../../../../../raw/code/simply/simply/model_lib.py#L1556)
  - `qk_norm` — [`L1424`](../../../../../raw/code/simply/simply/model_lib.py#L1424)
  - `qkv_partition` — [`L1431`](../../../../../raw/code/simply/simply/model_lib.py#L1431)
  - `qkv_use_bias` — [`L1443`](../../../../../raw/code/simply/simply/model_lib.py#L1443)
  - `query_scale` — [`L1447`](../../../../../raw/code/simply/simply/model_lib.py#L1447)
  - `total_num_pages` — [`L1449`](../../../../../raw/code/simply/simply/model_lib.py#L1449)
  - `update_kv_cache_in_place` — [`L1436`](../../../../../raw/code/simply/simply/model_lib.py#L1436)
  - `use_causal` — [`L1422`](../../../../../raw/code/simply/simply/model_lib.py#L1422)
  - `use_flash_attention` — [`L1438`](../../../../../raw/code/simply/simply/model_lib.py#L1438)
  - `use_per_dim_scale` — [`L1425`](../../../../../raw/code/simply/simply/model_lib.py#L1425)
  - `use_window_chunk` — [`L1441`](../../../../../raw/code/simply/simply/model_lib.py#L1441)
  - `v_proj` — [`L1568`](../../../../../raw/code/simply/simply/model_lib.py#L1568)
  - `weight_dtype` — [`L1429`](../../../../../raw/code/simply/simply/model_lib.py#L1429)
  - `weight_init` — [`L1426`](../../../../../raw/code/simply/simply/model_lib.py#L1426)
  - `window_size` — [`L1440`](../../../../../raw/code/simply/simply/model_lib.py#L1440)
- uses (calls/refs, reference-scoped): [`register`](utils/registry.md#RootRegistry.register), [`Array`](model_lib.md#Array), [`init`](utils/ragged_paged_attention.md#DecodeStateConfig.init), [`update_decode_state_and_compute_attn`](utils/ragged_paged_attention.md#DecodeState.update_decode_state_and_compute_attn), [`with_sharding_constraint`](utils/sharding.md#with_sharding_constraint), [`PyTree`](model_lib.md#PyTree), [`RoPE`](utils/position_encoding.md#RoPE), [`EinsumLinear`](utils/module.md#EinsumLinear), [`RaggedArray`](utils/common.md#RaggedArray), [`SimplyModule`](utils/module.md#SimplyModule), [`ModuleRegistry`](utils/module.md#ModuleRegistry), [`LayerNorm`](model_lib.md#LayerNorm), [`Initializer`](utils/initializer.md#Initializer), [`neg_inf`](utils/common.md#neg_inf), [`XavierUniformInit`](utils/initializer.md#XavierUniformInit), [`updated_decode_state`](model_lib.md#updated_decode_state), [`PartitionAnnotation`](model_lib.md#PartitionAnnotation), [`get_partition_size`](utils/sharding.md#get_partition_size), [`DecodeState`](utils/ragged_paged_attention.md#DecodeState), [`PRNGKey`](model_lib.md#PRNGKey), [`attn`](model_lib.md#attn), [`batch_size`](utils/ragged_paged_attention.md#DecodeStateConfig.batch_size), [`dtype`](utils/ragged_paged_attention.md#DecodeStateConfig.dtype), [`max_seq_len`](utils/ragged_paged_attention.md#DecodeStateConfig.max_seq_len), [`page_size`](utils/ragged_paged_attention.md#DecodeStateConfig.page_size), [`total_num_pages`](utils/ragged_paged_attention.md#DecodeStateConfig.total_num_pages), [`PerDimScale`](model_lib.md#PerDimScale), [`create_mask`](model_lib.md#create_mask), [`get_raw_arrays`](model_lib.md#get_raw_arrays), [`DTypeLike`](model_lib.md#DTypeLike), [`n_kv_heads`](utils/ragged_paged_attention.md#DecodeStateConfig.n_kv_heads), [`named_sharding`](utils/sharding.md#named_sharding), [`per_head_dim`](utils/ragged_paged_attention.md#DecodeStateConfig.per_head_dim), [`get_partition_axis`](utils/sharding.md#get_partition_axis), [`DecodeStateConfig`](utils/ragged_paged_attention.md#DecodeStateConfig), [`chunked_local_attn`](model_lib.md#chunked_local_attn), [`seq_partition`](utils/ragged_paged_attention.md#DecodeStateConfig.seq_partition), [`PositionEncodingConfig`](utils/position_encoding.md#PositionEncodingConfig), [`head_partition`](utils/ragged_paged_attention.md#DecodeStateConfig.head_partition), [`window_size`](utils/ragged_paged_attention.md#DecodeStateConfig.window_size)
- used by: [`attn`](model_lib.md#TransformerBlock.attn), [`apply`](utils/module.md#SimplyModule.apply), [`init`](utils/module.md#SimplyModule.init), [`setup`](utils/module.md#SimplyModule.setup)

### `FeedForward`
- def: [`simply/model_lib.py:529`](../../../../../raw/code/simply/simply/model_lib.py#L529)
- doc: The FeedForward block in Transformer.
- signature: `class FeedForward(module.SimplyModule):`
- members:
  - `apply(self, params: PyTree, x: Array, inputs_mask: Array | None = None)` — [`L595`](../../../../../raw/code/simply/simply/model_lib.py#L595) — documented in [simply-utils-module](../../concepts/simply-utils-module.md)
  - `expand_dim(self)` — [`L545`](../../../../../raw/code/simply/simply/model_lib.py#L545)
  - `init(self, prng_key: PRNGKey)` — [`L586`](../../../../../raw/code/simply/simply/model_lib.py#L586)
  - `setup(self)` — [`L550`](../../../../../raw/code/simply/simply/model_lib.py#L550)
  - `activation_dtype` — [`L537`](../../../../../raw/code/simply/simply/model_lib.py#L537)
  - `expand_factor` — [`L533`](../../../../../raw/code/simply/simply/model_lib.py#L533)
  - `ffn_0` — [`L551`](../../../../../raw/code/simply/simply/model_lib.py#L551)
  - `ffn_0_gate` — [`L563`](../../../../../raw/code/simply/simply/model_lib.py#L563)
  - `ffn_1` — [`L574`](../../../../../raw/code/simply/simply/model_lib.py#L574)
  - `ffn_activation` — [`L541`](../../../../../raw/code/simply/simply/model_lib.py#L541)
  - `ffn_expand_dim` — [`L539`](../../../../../raw/code/simply/simply/model_lib.py#L539)
  - `ffn_use_bias` — [`L540`](../../../../../raw/code/simply/simply/model_lib.py#L540)
  - `ffn_weight_init` — [`L542`](../../../../../raw/code/simply/simply/model_lib.py#L542)
  - `model_dim` — [`L532`](../../../../../raw/code/simply/simply/model_lib.py#L532)
  - `sharding_config` — [`L534`](../../../../../raw/code/simply/simply/model_lib.py#L534)
  - `use_gated_activation_in_ffn` — [`L535`](../../../../../raw/code/simply/simply/model_lib.py#L535)
- uses (calls/refs, reference-scoped): [`register`](utils/registry.md#RootRegistry.register), [`Array`](model_lib.md#Array), [`PyTree`](model_lib.md#PyTree), [`get`](utils/registry.md#RootRegistry.get), [`SimplyModule`](utils/module.md#SimplyModule), [`ModuleRegistry`](utils/module.md#ModuleRegistry), [`Initializer`](utils/initializer.md#Initializer), [`FunctionRegistry`](utils/registry.md#FunctionRegistry), [`XavierUniformInit`](utils/initializer.md#XavierUniformInit), [`EinsumLinear`](model_lib.md#EinsumLinear), [`PRNGKey`](model_lib.md#PRNGKey), [`DTypeLike`](model_lib.md#DTypeLike), [`SimplyConfig`](model_lib.md#SimplyConfig)
- used by: [`ffn`](model_lib.md#TransformerBlock.ffn), [`apply`](utils/module.md#SimplyModule.apply), [`init`](utils/module.md#SimplyModule.init), [`setup`](utils/module.md#SimplyModule.setup), [`MoEFeedForward`](model_lib.md#MoEFeedForward)

### `InputEncoderInterface`  ·  implements/extends SimplyModule
- def: [`simply/model_lib.py:2143`](../../../../../raw/code/simply/simply/model_lib.py#L2143)
- doc: Interface for custom input encoding for TransformerLM.
- signature: `class InputEncoderInterface(module.SimplyModule):`
- members:
  - `apply(self, params: common.PyTree, x: common.Array, **kwargs: Mapping[str, Any])` — [`L2170`](../../../../../raw/code/simply/simply/model_lib.py#L2170)
  - `extra_input_keys` — [`L2168`](../../../../../raw/code/simply/simply/model_lib.py#L2168)
  - `name` — [`L2166`](../../../../../raw/code/simply/simply/model_lib.py#L2166)
- uses (calls/refs, reference-scoped): [`PyTree`](utils/common.md#PyTree.PyTree), [`Array`](utils/common.md#Array.Array), [`SimplyModule`](utils/module.md#SimplyModule), [`Output`](model_lib.md#InputEncoderInterface.Output)
- used by: [`apply`](model_lib.md#TransformerLM.apply), [`apply`](utils/module.md#SimplyModule.apply), [`SimplyModule`](utils/module.md#SimplyModule), [`init`](model_lib.md#TransformerLM.init), [`input_encoders`](model_lib.md#TransformerLM.input_encoders)

### `LMInterface`
- def: [`simply/model_lib.py:3640`](../../../../../raw/code/simply/simply/model_lib.py#L3640) — documented in [simply-serving-vanilla_server](../../concepts/simply-serving-vanilla_server.md)
- signature: `class LMInterface:`
- members:
  - `__init__(self, model: module.SimplyModule, params: PyTree, vocab: tokenization.SimplyVocab[str] | None = None, input_processor: sampling_lib.InputProcessorInterface | None = None, default_sampling_params: SamplingParams | None = None, bos_id: int | None = None, pad_id: int | None = None, extra_eos_ids: Sequence[int] | None = None, extra_eos_tokens: Sequence[str] | None = None)` — [`L3642`](../../../../../raw/code/simply/simply/model_lib.py#L3642) — An interface to interact with a language model.
  - `count_num_tokens(self, text: sampling_lib.SamplingInput)` — [`L4099`](../../../../../raw/code/simply/simply/model_lib.py#L4099)
  - `eos_ids(self)` — [`L3720`](../../../../../raw/code/simply/simply/model_lib.py#L3720)
  - `generate(self, input_text: (sampling_lib.SamplingInput | Sequence[sampling_lib.SamplingInput]), prng_key: int | PRNGKey | None = None, params: PyTree = None, prefill_size: int = -1, sampling_params: SamplingParams | None = None, scoring_params: ScoringParams | None = None, include_eos_in_output_text: bool = False, scoring_inputs: bool = True, batch_size: int | None = None)` — [`L3723`](../../../../../raw/code/simply/simply/model_lib.py#L3723) — Generate samples from a given input text. — documented in [simply-model_lib](../../concepts/simply-model_lib.md)
  - `prefill_fn(params: PyTree, inputs: Array, extra_inputs: Mapping[str, Array], position: int, return_logits: bool = True)` — [`L3686`](../../../../../raw/code/simply/simply/model_lib.py#L3686)
  - `score(self, input_text: sampling_lib.SamplingInput, output_text: sampling_lib.SamplingInput, params: PyTree | None = None, scoring_params: ScoringParams | None = None)` — [`L4005`](../../../../../raw/code/simply/simply/model_lib.py#L4005) — Decode on given texts to compute their token scores (loglikelihood).
  - `score_tokens(self, tokens: Sequence[int], extra_inputs: Mapping[str, Array] | None = None, scoring_params: ScoringParams | None = None, params: PyTree | None = None)` — [`L4056`](../../../../../raw/code/simply/simply/model_lib.py#L4056) — Compute the token scores (loglikelihood) of a list of tokens. — documented in [simply-model_lib](../../concepts/simply-model_lib.md)
  - `decode_fn` — [`L3703`](../../../../../raw/code/simply/simply/model_lib.py#L3703)
  - `default_sampling_params` — [`L3684`](../../../../../raw/code/simply/simply/model_lib.py#L3684)
  - `input_processor` — [`L3674`](../../../../../raw/code/simply/simply/model_lib.py#L3674) — documented in [simply-model_lib](../../concepts/simply-model_lib.md)
  - `model` — [`L3672`](../../../../../raw/code/simply/simply/model_lib.py#L3672)
  - `model_params` — [`L3717`](../../../../../raw/code/simply/simply/model_lib.py#L3717)
  - `pad_state_to_fn` — [`L3712`](../../../../../raw/code/simply/simply/model_lib.py#L3712)
  - `prefill_fn` — [`L3701`](../../../../../raw/code/simply/simply/model_lib.py#L3701)
- uses (calls/refs, reference-scoped): [`Array`](model_lib.md#Array), [`with_sharding_constraint`](utils/sharding.md#with_sharding_constraint), [`PyTree`](model_lib.md#PyTree), [`apply`](utils/module.md#SimplyModule.apply), [`get_decoding_schedule`](utils/sampling_lib.md#SamplingParams.get_decoding_schedule), [`SimplyModule`](utils/module.md#SimplyModule), [`SamplingParams`](model_lib.md#SamplingParams), [`pad_to`](model_lib.md#SamplingState.pad_to), [`Chunk`](utils/sampling_lib.md#Chunk), [`SamplingOutput`](model_lib.md#SamplingOutput), [`max_decode_steps`](utils/sampling_lib.md#SamplingParams.max_decode_steps), [`input_as_chunks`](utils/sampling_lib.md#input_as_chunks), [`SimplyVocab`](utils/tokenization.md#SimplyVocab), [`compute_log_likelihood`](utils/sampling_lib.md#compute_log_likelihood), [`get_next_length`](utils/sampling_lib.md#DecodingSchedule.get_next_length), [`SamplingInput`](utils/sampling_lib.md#SamplingInput), [`position`](model_lib.md#SamplingState.position), [`temperature`](utils/sampling_lib.md#SamplingParams.temperature), [`encode`](utils/sampling_lib.md#InputProcessorInterface.encode), [`top_k`](utils/sampling_lib.md#SamplingParams.top_k), [`top_p`](utils/sampling_lib.md#SamplingParams.top_p), [`continue_decode`](model_lib.md#continue_decode), [`from_sampling_params`](model_lib.md#ScoringParams.from_sampling_params), [`max_seq_len`](utils/sampling_lib.md#SamplingParams.max_seq_len), [`eos_ids`](utils/sampling_lib.md#InputProcessorInterface.eos_ids), [`BasicTextInputProcessor`](utils/sampling_lib.md#BasicTextInputProcessor), [`ScoringParams`](model_lib.md#ScoringParams), [`num_samples`](utils/sampling_lib.md#SamplingParams.num_samples), [`tokens`](model_lib.md#SamplingState.tokens), [`PRNGKey`](model_lib.md#PRNGKey), [`decode`](utils/sampling_lib.md#InputProcessorInterface.decode), [`prefill_size`](utils/sampling_lib.md#DecodingSchedule.prefill_size), [`input_lens`](model_lib.md#SamplingState.input_lens), [`ProcessedInputBatch`](utils/sampling_lib.md#ProcessedInputBatch), [`SamplingState`](model_lib.md#SamplingState), [`token_scores`](model_lib.md#SamplingState.token_scores), [`temperature`](model_lib.md#ScoringParams.temperature), [`top_k`](model_lib.md#ScoringParams.top_k), [`top_p`](model_lib.md#ScoringParams.top_p), [`token_logprobs`](model_lib.md#SamplingState.token_logprobs)  (+20 more)
- used by: [`run_experiment`](rl_lib.md#run_experiment), [`main`](eval/decode_eval.md#main), [`sample_with_tool`](tool_lib.md#ToolExecutor.sample_with_tool), [`loop`](serving/vanilla_server.md#Batcher.loop), [`test_batch_sampling_with_max_seq_len`](model_lib_test.md#ModelLibTest.test_batch_sampling_with_max_seq_len), [`test_sampling_token_scores`](model_lib_test.md#ModelLibTest.test_sampling_token_scores), [`test_batch_sampling`](model_lib_test.md#ModelLibTest.test_batch_sampling), [`test_lm_interface_generate_with_local_state`](model_lib_test.md#ModelLibTest.test_lm_interface_generate_with_local_state), [`test_continue_decoding`](model_lib_test.md#ModelLibTest.test_continue_decoding), [`test_lm_interface_generate_without_scoring`](model_lib_test.md#ModelLibTest.test_lm_interface_generate_without_scoring), [`test_sampling_max_decode_steps_equals_prefill_size`](model_lib_test.md#ModelLibTest.test_sampling_max_decode_steps_equals_prefill_size), [`test_sampling_output_logprobs`](model_lib_test.md#ModelLibTest.test_sampling_output_logprobs), [`lm_interface`](serving/vanilla_server.md#Batcher.lm_interface), [`test_lm_interface_batch`](model_lib_test.md#ModelLibTest.test_lm_interface_batch), [`test_lm_interface_generate`](model_lib_test.md#ModelLibTest.test_lm_interface_generate), [`test_lm_interface_score`](model_lib_test.md#ModelLibTest.test_lm_interface_score), [`test_lm_interface_score_tokens`](model_lib_test.md#ModelLibTest.test_lm_interface_score_tokens), [`_encode`](tool_lib.md#ToolExecutor._encode), [`execute_tool_batch_with_sharding`](tool_lib.md#ToolExecutor.execute_tool_batch_with_sharding)

### `LayerNorm`
- def: [`simply/model_lib.py:111`](../../../../../raw/code/simply/simply/model_lib.py#L111)
- doc: Layer normalization layer (can be also configured as RMSNorm).
- signature: `class LayerNorm(module.SimplyModule):`
- members:
  - `apply(self, params: PyTree, x: Array)` — [`L151`](../../../../../raw/code/simply/simply/model_lib.py#L151)
  - `init(self, prng_key: PRNGKey | None = None)` — [`L127`](../../../../../raw/code/simply/simply/model_lib.py#L127)
  - `activation_dtype` — [`L119`](../../../../../raw/code/simply/simply/model_lib.py#L119)
  - `axis` — [`L114`](../../../../../raw/code/simply/simply/model_lib.py#L114)
  - `bias_partition` — [`L123`](../../../../../raw/code/simply/simply/model_lib.py#L123)
  - `dim` — [`L113`](../../../../../raw/code/simply/simply/model_lib.py#L113)
  - `epsilon` — [`L125`](../../../../../raw/code/simply/simply/model_lib.py#L125)
  - `scale_partition` — [`L122`](../../../../../raw/code/simply/simply/model_lib.py#L122)
  - `scale_plus_one` — [`L120`](../../../../../raw/code/simply/simply/model_lib.py#L120)
  - `use_bias` — [`L115`](../../../../../raw/code/simply/simply/model_lib.py#L115)
  - `use_scale` — [`L116`](../../../../../raw/code/simply/simply/model_lib.py#L116)
  - `weight_dtype` — [`L118`](../../../../../raw/code/simply/simply/model_lib.py#L118)
- uses (calls/refs, reference-scoped): [`register`](utils/registry.md#RootRegistry.register), [`Array`](model_lib.md#Array), [`with_sharding_constraint`](utils/sharding.md#with_sharding_constraint), [`PyTree`](model_lib.md#PyTree), [`SimplyModule`](utils/module.md#SimplyModule), [`ModuleRegistry`](utils/module.md#ModuleRegistry), [`convert_or_dequantize`](utils/common.md#convert_or_dequantize), [`create`](utils/common.md#AnnotatedArray.create), [`PartitionAnnotation`](model_lib.md#PartitionAnnotation), [`PRNGKey`](model_lib.md#PRNGKey), [`get_raw_arrays`](model_lib.md#get_raw_arrays), [`DTypeLike`](model_lib.md#DTypeLike), [`AnnotatedArray`](model_lib.md#AnnotatedArray)
- used by: [`apply`](utils/module.md#SimplyModule.apply), [`init`](utils/module.md#SimplyModule.init), [`post_skip_ln_1`](model_lib.md#TransformerBlock.post_skip_ln_1), [`post_ln_1`](model_lib.md#TransformerBlock.post_ln_1), [`pre_ln_1`](model_lib.md#TransformerBlock.pre_ln_1), [`post_ln_0`](model_lib.md#TransformerBlock.post_ln_0), [`post_skip_ln_0`](model_lib.md#TransformerBlock.post_skip_ln_0), [`pre_ln_0`](model_lib.md#TransformerBlock.pre_ln_0), [`qk_norm`](model_lib.md#Attention.qk_norm), [`final_ln`](model_lib.md#TransformerLM.final_ln), [`post_downsample_norm`](utils/vision_encoder.md#VisionTransformer.post_downsample_norm), [`pre_downsample_norm`](utils/vision_encoder.md#VisionTransformer.pre_downsample_norm)

### `MoEFeedForward`
- def: [`simply/model_lib.py:643`](../../../../../raw/code/simply/simply/model_lib.py#L643)
- doc: A Mixture-of-Experts FeedForward block.
- signature: `class MoEFeedForward(FeedForward):`
- members:
  - `_apply_sparse_moe(self, params: PyTree, inputs: Array, selected_indices: Array, selected_weights: Array, inputs_mask: Array | None = None)` — [`L838`](../../../../../raw/code/simply/simply/model_lib.py#L838) — Apply sparse MoE. — documented in [simply-model_lib](../../concepts/simply-model_lib.md)
  - `_apply_sparse_moe_v2(self, params: PyTree, inputs: Array, selected_indices: Array, selected_weights: Array, inputs_mask: Array | None = None)` — [`L1130`](../../../../../raw/code/simply/simply/model_lib.py#L1130) — Apply sparse MoE. — documented in [simply-model_lib](../../concepts/simply-model_lib.md)
  - `all_gather_if_sharded(w, partition, axis)` — [`L909`](../../../../../raw/code/simply/simply/model_lib.py#L909)
  - `apply(self, params: PyTree, x: Array, inputs_mask: Array | None = None)` — [`L732`](../../../../../raw/code/simply/simply/model_lib.py#L732) — documented in [simply-model_lib](../../concepts/simply-model_lib.md)
  - `compute_block(sorted_inputs, local_group_sizes, ffn0_w, ffn0_gate_w, ffn1_w)` — [`L1195`](../../../../../raw/code/simply/simply/model_lib.py#L1195)
  - `get_global_input_output_offsets(global_send_sizes)` — [`L983`](../../../../../raw/code/simply/simply/model_lib.py#L983)
  - `init(self, prng_key: PRNGKey)` — [`L720`](../../../../../raw/code/simply/simply/model_lib.py#L720)
  - `moe_ffn(inputs, ffn0_w, ffn0_gate_w, ffn1_w, selected_indices, selected_weights)` — [`L905`](../../../../../raw/code/simply/simply/model_lib.py#L905)
  - `setup(self)` — [`L664`](../../../../../raw/code/simply/simply/model_lib.py#L664)
  - `ep_capacity_factor` — [`L656`](../../../../../raw/code/simply/simply/model_lib.py#L656)
  - `ep_method` — [`L657`](../../../../../raw/code/simply/simply/model_lib.py#L657)
  - `ep_pipeline_comms` — [`L660`](../../../../../raw/code/simply/simply/model_lib.py#L660)
  - `ep_pipeline_fine_grained_ra2a` — [`L661`](../../../../../raw/code/simply/simply/model_lib.py#L661)
  - `ep_pipeline_stages` — [`L659`](../../../../../raw/code/simply/simply/model_lib.py#L659)
  - `ep_pipeline_use_opt_barriers` — [`L662`](../../../../../raw/code/simply/simply/model_lib.py#L662)
  - `ffn0_partition` — [`L686`](../../../../../raw/code/simply/simply/model_lib.py#L686)
  - `ffn1_partition` — [`L697`](../../../../../raw/code/simply/simply/model_lib.py#L697)
  - `ffn_0` — [`L687`](../../../../../raw/code/simply/simply/model_lib.py#L687)
  - `ffn_0_gate` — [`L699`](../../../../../raw/code/simply/simply/model_lib.py#L699)
  - `ffn_1` — [`L709`](../../../../../raw/code/simply/simply/model_lib.py#L709)
  - `gmm_impl` — [`L652`](../../../../../raw/code/simply/simply/model_lib.py#L652)
  - `lbl_loss_weight` — [`L648`](../../../../../raw/code/simply/simply/model_lib.py#L648)
  - `num_experts` — [`L645`](../../../../../raw/code/simply/simply/model_lib.py#L645)
  - `num_experts_per_token` — [`L646`](../../../../../raw/code/simply/simply/model_lib.py#L646)
  - `router` — [`L675`](../../../../../raw/code/simply/simply/model_lib.py#L675)
  - `router_z_loss_weight` — [`L647`](../../../../../raw/code/simply/simply/model_lib.py#L647)
  - `tile_batch_seq` — [`L649`](../../../../../raw/code/simply/simply/model_lib.py#L649)
  - `tile_expand_dim` — [`L651`](../../../../../raw/code/simply/simply/model_lib.py#L651)
  - `tile_model_dim` — [`L650`](../../../../../raw/code/simply/simply/model_lib.py#L650)
- protocol/private: `_apply_dense_moe`[`L1324`](../../../../../raw/code/simply/simply/model_lib.py#L1324)
- uses (calls/refs, reference-scoped): [`register`](utils/registry.md#RootRegistry.register), [`Array`](model_lib.md#Array), [`with_sharding_constraint`](utils/sharding.md#with_sharding_constraint), [`PyTree`](model_lib.md#PyTree), [`get`](utils/registry.md#RootRegistry.get), [`run_moe_pipelined_shard_map`](utils/moe_lib.md#run_moe_pipelined_shard_map), [`ModuleRegistry`](utils/module.md#ModuleRegistry), [`convert_or_dequantize`](utils/common.md#convert_or_dequantize), [`FunctionRegistry`](utils/registry.md#FunctionRegistry), [`EinsumLinear`](model_lib.md#EinsumLinear), [`get_partition_axis`](model_lib.md#get_partition_axis), [`partition_spec`](utils/sharding.md#partition_spec), [`PipelinedMoEConfig`](utils/moe_lib.md#PipelinedMoEConfig), [`FeedForward`](model_lib.md#FeedForward), [`PRNGKey`](model_lib.md#PRNGKey), [`get_raw_arrays`](model_lib.md#get_raw_arrays), [`round_up_to_base`](model_lib.md#round_up_to_base), [`ep_method`](utils/moe_lib.md#PipelinedMoEConfig.ep_method), [`gathers`](utils/moe_lib.md#PipelinedMoEConfig.gathers), [`permute`](model_lib.md#permute), [`get_default_mesh`](utils/sharding.md#get_default_mesh), [`ra2a`](utils/moe_lib.md#PipelinedMoEConfig.ra2a), [`fine_grained_ra2a`](utils/moe_lib.md#PipelinedMoEConfig.fine_grained_ra2a), [`use_pipelined_ra2a_barriers`](utils/moe_lib.md#PipelinedMoEConfig.use_pipelined_ra2a_barriers), [`safety_factor`](utils/moe_lib.md#PipelinedMoEConfig.safety_factor), [`gmm`](model_lib.md#gmm), [`use_scheduling_groups`](utils/moe_lib.md#PipelinedMoEConfig.use_scheduling_groups)
- used by: [`ffn`](model_lib.md#TransformerBlock.ffn), [`test_pipelined_moe_feed_forward_equivalence`](model_lib_test.md#MoETest.test_pipelined_moe_feed_forward_equivalence), [`test_moe_feed_forward_equivalence`](model_lib_test.md#MoETest.test_moe_feed_forward_equivalence)

### `Output`
- def: [`simply/model_lib.py:2160`](../../../../../raw/code/simply/simply/model_lib.py#L2160)
- signature: `class Output:`
- members:
  - `embedding_mask` — [`L2162`](../../../../../raw/code/simply/simply/model_lib.py#L2162)
  - `embeddings` — [`L2161`](../../../../../raw/code/simply/simply/model_lib.py#L2161)
- uses (calls/refs, reference-scoped): [`Array`](utils/common.md#Array.Array)
- used by: [`apply`](model_lib.md#TransformerLM.apply), [`apply`](model_lib.md#InputEncoderInterface.apply)

### `PerDimScale`
- def: [`simply/model_lib.py:180`](../../../../../raw/code/simply/simply/model_lib.py#L180)
- doc: Layer to scale individual dims of the input.
- signature: `class PerDimScale(module.SimplyModule):`
- members:
  - `apply(self, params: PyTree, x: Array)` — [`L195`](../../../../../raw/code/simply/simply/model_lib.py#L195)
  - `init(self, prng_key: PRNGKey | None = None)` — [`L188`](../../../../../raw/code/simply/simply/model_lib.py#L188)
  - `activation_dtype` — [`L186`](../../../../../raw/code/simply/simply/model_lib.py#L186)
  - `axis` — [`L183`](../../../../../raw/code/simply/simply/model_lib.py#L183)
  - `dim` — [`L182`](../../../../../raw/code/simply/simply/model_lib.py#L182)
  - `weight_dtype` — [`L185`](../../../../../raw/code/simply/simply/model_lib.py#L185)
- uses (calls/refs, reference-scoped): [`register`](utils/registry.md#RootRegistry.register), [`Array`](model_lib.md#Array), [`PyTree`](model_lib.md#PyTree), [`SimplyModule`](utils/module.md#SimplyModule), [`ModuleRegistry`](utils/module.md#ModuleRegistry), [`create`](utils/common.md#AnnotatedArray.create), [`PRNGKey`](model_lib.md#PRNGKey), [`get_raw_arrays`](model_lib.md#get_raw_arrays), [`DTypeLike`](model_lib.md#DTypeLike), [`AnnotatedArray`](model_lib.md#AnnotatedArray)
- used by: [`apply`](utils/module.md#SimplyModule.apply), [`init`](utils/module.md#SimplyModule.init), [`per_dim_scale`](model_lib.md#Attention.per_dim_scale)

### `SamplingOutput`
- def: [`simply/model_lib.py:3522`](../../../../../raw/code/simply/simply/model_lib.py#L3522) — documented in [simply-model_lib](../../concepts/simply-model_lib.md)
- signature: `class SamplingOutput:`
- members:
  - `avg_input_score(self)` — [`L3568`](../../../../../raw/code/simply/simply/model_lib.py#L3568)
  - `avg_output_logprob(self)` — [`L3558`](../../../../../raw/code/simply/simply/model_lib.py#L3558)
  - `avg_output_score(self)` — [`L3578`](../../../../../raw/code/simply/simply/model_lib.py#L3578)
  - `input_text(self)` — [`L3544`](../../../../../raw/code/simply/simply/model_lib.py#L3544)
  - `output_text(self)` — [`L3548`](../../../../../raw/code/simply/simply/model_lib.py#L3548)
  - `sum_input_score(self)` — [`L3562`](../../../../../raw/code/simply/simply/model_lib.py#L3562)
  - `sum_output_logprob(self)` — [`L3552`](../../../../../raw/code/simply/simply/model_lib.py#L3552)
  - `sum_output_score(self)` — [`L3572`](../../../../../raw/code/simply/simply/model_lib.py#L3572)
  - `input_chunks` — [`L3523`](../../../../../raw/code/simply/simply/model_lib.py#L3523)
  - `input_token_ids` — [`L3524`](../../../../../raw/code/simply/simply/model_lib.py#L3524)
  - `input_token_scores` — [`L3538`](../../../../../raw/code/simply/simply/model_lib.py#L3538)
  - `is_truncated` — [`L3532`](../../../../../raw/code/simply/simply/model_lib.py#L3532)
  - `output_chunks` — [`L3526`](../../../../../raw/code/simply/simply/model_lib.py#L3526)
  - `output_token_ids` — [`L3527`](../../../../../raw/code/simply/simply/model_lib.py#L3527)
  - `output_token_logprobs` — [`L3530`](../../../../../raw/code/simply/simply/model_lib.py#L3530)
  - `output_token_scores` — [`L3541`](../../../../../raw/code/simply/simply/model_lib.py#L3541)
  - `processed_input` — [`L3534`](../../../../../raw/code/simply/simply/model_lib.py#L3534)
- uses (calls/refs, reference-scoped): [`register`](utils/registry.md#RootRegistry.register), [`ProcessedInput`](utils/sampling_lib.md#ProcessedInput), [`ChunkSequence`](utils/sampling_lib.md#ChunkSequence), [`chunks_as_text`](utils/sampling_lib.md#chunks_as_text), [`SamplingRegistry`](utils/sampling_lib.md#SamplingRegistry), [`neg_inf`](model_lib.md#neg_inf)
- used by: [`generate`](model_lib.md#LMInterface.generate), [`test_batch_sampling_with_max_seq_len`](model_lib_test.md#ModelLibTest.test_batch_sampling_with_max_seq_len), [`test_sampling_token_scores`](model_lib_test.md#ModelLibTest.test_sampling_token_scores), [`test_batch_sampling`](model_lib_test.md#ModelLibTest.test_batch_sampling), [`test_lm_interface_generate_with_local_state`](model_lib_test.md#ModelLibTest.test_lm_interface_generate_with_local_state), [`test_continue_decoding`](model_lib_test.md#ModelLibTest.test_continue_decoding), [`test_lm_interface_generate_without_scoring`](model_lib_test.md#ModelLibTest.test_lm_interface_generate_without_scoring), [`test_sampling_max_decode_steps_equals_prefill_size`](model_lib_test.md#ModelLibTest.test_sampling_max_decode_steps_equals_prefill_size), [`test_sampling_output_logprobs`](model_lib_test.md#ModelLibTest.test_sampling_output_logprobs), [`test_lm_interface_generate`](model_lib_test.md#ModelLibTest.test_lm_interface_generate), [`sampling_output`](rl_lib.md#RewardedSample.sampling_output), [`SamplingOutput`](tool_lib.md#SamplingOutput)

### `SamplingState`
- def: [`simply/model_lib.py:3413`](../../../../../raw/code/simply/simply/model_lib.py#L3413)
- doc: Sampling state.
- signature: `class SamplingState:`
- members:
  - `all_has_ended(self)` — [`L3467`](../../../../../raw/code/simply/simply/model_lib.py#L3467) — Returns whether all sequences in the batch are done with generation.
  - `batch_size(self)` — [`L3501`](../../../../../raw/code/simply/simply/model_lib.py#L3501)
  - `decode_state_length(self)` — [`L3497`](../../../../../raw/code/simply/simply/model_lib.py#L3497)
  - `has_ended(self)` — [`L3457`](../../../../../raw/code/simply/simply/model_lib.py#L3457) — Returns whether each sequence in the batch is done with generation.
  - `input_tokens(self)` — [`L3443`](../../../../../raw/code/simply/simply/model_lib.py#L3443)
  - `is_pad_seq(self)` — [`L3438`](../../../../../raw/code/simply/simply/model_lib.py#L3438) — This sequence is a padding sequence, in [batch, 1].
  - `next_position_is_output(self)` — [`L3472`](../../../../../raw/code/simply/simply/model_lib.py#L3472)
  - `next_tokens(self)` — [`L3476`](../../../../../raw/code/simply/simply/model_lib.py#L3476)
  - `pad_to(self, length: int)` — [`L3504`](../../../../../raw/code/simply/simply/model_lib.py#L3504)
  - `reached_eos(self)` — [`L3447`](../../../../../raw/code/simply/simply/model_lib.py#L3447) — This position is output and eos, in [batch, 1].
  - `updated_token_logprobs(self, output_logprobs: Array)` — [`L3486`](../../../../../raw/code/simply/simply/model_lib.py#L3486)
  - `updated_token_scores(self, output_scores: Array)` — [`L3491`](../../../../../raw/code/simply/simply/model_lib.py#L3491)
  - `updated_tokens(self, output_tokens: Array)` — [`L3481`](../../../../../raw/code/simply/simply/model_lib.py#L3481)
  - `decode_state` — [`L3423`](../../../../../raw/code/simply/simply/model_lib.py#L3423)
  - `eos_ids` — [`L3431`](../../../../../raw/code/simply/simply/model_lib.py#L3431)
  - `input_lens` — [`L3429`](../../../../../raw/code/simply/simply/model_lib.py#L3429)
  - `max_decode_steps` — [`L3430`](../../../../../raw/code/simply/simply/model_lib.py#L3430)
  - `position` — [`L3427`](../../../../../raw/code/simply/simply/model_lib.py#L3427)
  - `prng_key` — [`L3422`](../../../../../raw/code/simply/simply/model_lib.py#L3422)
  - `token_logprobs` — [`L3425`](../../../../../raw/code/simply/simply/model_lib.py#L3425)
  - `token_scores` — [`L3426`](../../../../../raw/code/simply/simply/model_lib.py#L3426)
  - `tokens` — [`L3424`](../../../../../raw/code/simply/simply/model_lib.py#L3424)
- protocol/private: `__post_init__`[`L3433`](../../../../../raw/code/simply/simply/model_lib.py#L3433)
- uses (calls/refs, reference-scoped): [`Array`](model_lib.md#Array), [`PyTree`](model_lib.md#PyTree), [`pad_to_along_axis`](model_lib.md#pad_to_along_axis), [`pad_decode_state_to`](model_lib.md#pad_decode_state_to), [`PRNGKey`](model_lib.md#PRNGKey)
- used by: [`generate`](model_lib.md#LMInterface.generate), [`body_fn`](model_lib.md#continue_decode.body_fn), [`continue_decode`](model_lib.md#continue_decode), [`cond_fn`](model_lib.md#continue_decode.cond_fn), [`pad_state_to_fn`](model_lib.md#LMInterface.pad_state_to_fn)

### `ScoringOutput`
- def: [`simply/model_lib.py:3598`](../../../../../raw/code/simply/simply/model_lib.py#L3598)
- signature: `class ScoringOutput:`
- members:
  - `avg_input_score(self)` — [`L3628`](../../../../../raw/code/simply/simply/model_lib.py#L3628)
  - `avg_output_score(self)` — [`L3636`](../../../../../raw/code/simply/simply/model_lib.py#L3636)
  - `input_text(self)` — [`L3616`](../../../../../raw/code/simply/simply/model_lib.py#L3616)
  - `output_text(self)` — [`L3620`](../../../../../raw/code/simply/simply/model_lib.py#L3620)
  - `sum_input_score(self)` — [`L3624`](../../../../../raw/code/simply/simply/model_lib.py#L3624)
  - `sum_output_score(self)` — [`L3632`](../../../../../raw/code/simply/simply/model_lib.py#L3632)
  - `input_chunks` — [`L3601`](../../../../../raw/code/simply/simply/model_lib.py#L3601)
  - `input_token_ids` — [`L3602`](../../../../../raw/code/simply/simply/model_lib.py#L3602)
  - `input_token_scores` — [`L3609`](../../../../../raw/code/simply/simply/model_lib.py#L3609)
  - `output_chunks` — [`L3604`](../../../../../raw/code/simply/simply/model_lib.py#L3604)
  - `output_token_ids` — [`L3605`](../../../../../raw/code/simply/simply/model_lib.py#L3605)
  - `output_token_scores` — [`L3613`](../../../../../raw/code/simply/simply/model_lib.py#L3613)
  - `params` — [`L3599`](../../../../../raw/code/simply/simply/model_lib.py#L3599)
- uses (calls/refs, reference-scoped): [`ChunkSequence`](utils/sampling_lib.md#ChunkSequence), [`chunks_as_text`](utils/sampling_lib.md#chunks_as_text), [`ScoringParams`](model_lib.md#ScoringParams)
- used by: [`score`](model_lib.md#LMInterface.score), [`test_lm_interface_score`](model_lib_test.md#ModelLibTest.test_lm_interface_score)

### `ScoringParams`
- def: [`simply/model_lib.py:3583`](../../../../../raw/code/simply/simply/model_lib.py#L3583)
- signature: `class ScoringParams:`
- members:
  - `from_sampling_params(cls, sampling_params: SamplingParams)` — [`L3589`](../../../../../raw/code/simply/simply/model_lib.py#L3589)
  - `temperature` — [`L3584`](../../../../../raw/code/simply/simply/model_lib.py#L3584)
  - `top_k` — [`L3585`](../../../../../raw/code/simply/simply/model_lib.py#L3585)
  - `top_p` — [`L3586`](../../../../../raw/code/simply/simply/model_lib.py#L3586)
- uses (calls/refs, reference-scoped): [`SamplingParams`](model_lib.md#SamplingParams), [`temperature`](utils/sampling_lib.md#SamplingParams.temperature), [`top_k`](utils/sampling_lib.md#SamplingParams.top_k), [`top_p`](utils/sampling_lib.md#SamplingParams.top_p)
- used by: [`generate`](model_lib.md#LMInterface.generate), [`score`](model_lib.md#LMInterface.score), [`test_sampling_token_scores`](model_lib_test.md#ModelLibTest.test_sampling_token_scores), [`score_tokens`](model_lib.md#LMInterface.score_tokens), [`test_lm_interface_score`](model_lib_test.md#ModelLibTest.test_lm_interface_score), [`test_lm_interface_score_tokens`](model_lib_test.md#ModelLibTest.test_lm_interface_score_tokens), [`params`](model_lib.md#ScoringOutput.params)

### `TrainLoopRegistry`  ·  implements/extends RootRegistry
- def: [`simply/model_lib.py:2957`](../../../../../raw/code/simply/simply/model_lib.py#L2957)
- doc: Registry for train loop functions.
- signature: `class TrainLoopRegistry(registry.RootRegistry):`
- members:
  - `namespace` — [`L2959`](../../../../../raw/code/simply/simply/model_lib.py#L2959)
- uses (calls/refs, reference-scoped): [`register`](utils/registry.md#RootRegistry.register), [`RootRegistry`](utils/registry.md#RootRegistry)
- used by: [`run_experiment`](rl_lib.md#run_experiment), [`RootRegistry`](utils/registry.md#RootRegistry), [`main`](main.md#main), [`TrainLoopRegistry`](rl_lib.md#TrainLoopRegistry)

### `TransformerBlock`
- def: [`simply/model_lib.py:1884`](../../../../../raw/code/simply/simply/model_lib.py#L1884)
- doc: A single transformer block.
- signature: `class TransformerBlock(module.SimplyModule):`
- members:
  - `apply(self, params: PyTree, x: Array, *, segment_ids: Array, segment_positions: Array, extra_inputs: PyTree | None = None, decode_state: PyTree = None)` — [`L2087`](../../../../../raw/code/simply/simply/model_lib.py#L2087) — documented in [simply-model_lib](../../concepts/simply-model_lib.md)
  - `expand_dim(self)` — [`L1939`](../../../../../raw/code/simply/simply/model_lib.py#L1939)
  - `init(self, prng_key: PRNGKey)` — [`L2070`](../../../../../raw/code/simply/simply/model_lib.py#L2070)
  - `init_decode_state(self, batch_size: int, max_seq_len: int)` — [`L2138`](../../../../../raw/code/simply/simply/model_lib.py#L2138)
  - `setup(self)` — [`L1944`](../../../../../raw/code/simply/simply/model_lib.py#L1944)
  - `activation_dtype` — [`L1907`](../../../../../raw/code/simply/simply/model_lib.py#L1907)
  - `attn` — [`L2001`](../../../../../raw/code/simply/simply/model_lib.py#L2001) — documented in [simply-model_lib](../../concepts/simply-model_lib.md)
  - `attn_mask_value` — [`L1923`](../../../../../raw/code/simply/simply/model_lib.py#L1923)
  - `attn_soft_cap` — [`L1922`](../../../../../raw/code/simply/simply/model_lib.py#L1922)
  - `attn_weight_init` — [`L1909`](../../../../../raw/code/simply/simply/model_lib.py#L1909)
  - `ep_capacity_factor` — [`L1903`](../../../../../raw/code/simply/simply/model_lib.py#L1903)
  - `expand_factor` — [`L1890`](../../../../../raw/code/simply/simply/model_lib.py#L1890)
  - `ffn` — [`L2030`](../../../../../raw/code/simply/simply/model_lib.py#L2030) — documented in [simply-model_lib](../../concepts/simply-model_lib.md)
  - `ffn_activation` — [`L1920`](../../../../../raw/code/simply/simply/model_lib.py#L1920)
  - `ffn_expand_dim` — [`L1911`](../../../../../raw/code/simply/simply/model_lib.py#L1911)
  - `ffn_use_bias` — [`L1917`](../../../../../raw/code/simply/simply/model_lib.py#L1917)
  - `ffn_weight_init` — [`L1910`](../../../../../raw/code/simply/simply/model_lib.py#L1910)
  - `flash_attention_block_size` — [`L1913`](../../../../../raw/code/simply/simply/model_lib.py#L1913)
  - `gmm_impl` — [`L1933`](../../../../../raw/code/simply/simply/model_lib.py#L1933)
  - `lbl_loss_weight` — [`L1904`](../../../../../raw/code/simply/simply/model_lib.py#L1904)
  - `model_dim` — [`L1887`](../../../../../raw/code/simply/simply/model_lib.py#L1887)
  - `n_heads` — [`L1888`](../../../../../raw/code/simply/simply/model_lib.py#L1888)
  - `n_kv_heads` — [`L1916`](../../../../../raw/code/simply/simply/model_lib.py#L1916)
  - `norm_scale_plus_one` — [`L1921`](../../../../../raw/code/simply/simply/model_lib.py#L1921)
  - `num_experts` — [`L1901`](../../../../../raw/code/simply/simply/model_lib.py#L1901)
  - `num_experts_per_token` — [`L1902`](../../../../../raw/code/simply/simply/model_lib.py#L1902)
  - `o_use_bias` — [`L1919`](../../../../../raw/code/simply/simply/model_lib.py#L1919)
  - `page_size` — [`L1936`](../../../../../raw/code/simply/simply/model_lib.py#L1936)
  - `per_head_dim` — [`L1889`](../../../../../raw/code/simply/simply/model_lib.py#L1889)
  - `position_encoding` — [`L1926`](../../../../../raw/code/simply/simply/model_lib.py#L1926)
  - `post_ln_0` — [`L1961`](../../../../../raw/code/simply/simply/model_lib.py#L1961)
  - `post_ln_1` — [`L1968`](../../../../../raw/code/simply/simply/model_lib.py#L1968)
  - `post_skip_ln_0` — [`L1976`](../../../../../raw/code/simply/simply/model_lib.py#L1976)
  - `post_skip_ln_1` — [`L1983`](../../../../../raw/code/simply/simply/model_lib.py#L1983)
  - `pre_ln_0` — [`L1946`](../../../../../raw/code/simply/simply/model_lib.py#L1946)
  - `pre_ln_1` — [`L1953`](../../../../../raw/code/simply/simply/model_lib.py#L1953)
  - `qkv_use_bias` — [`L1918`](../../../../../raw/code/simply/simply/model_lib.py#L1918)
  - `query_scale` — [`L1927`](../../../../../raw/code/simply/simply/model_lib.py#L1927)
  - `rms_norm_epsilon` — [`L1924`](../../../../../raw/code/simply/simply/model_lib.py#L1924)
  - `router_z_loss_weight` — [`L1905`](../../../../../raw/code/simply/simply/model_lib.py#L1905)
  - `sharding_config` — [`L1891`](../../../../../raw/code/simply/simply/model_lib.py#L1891)
  - `tile_batch_seq` — [`L1929`](../../../../../raw/code/simply/simply/model_lib.py#L1929)
  - `tile_expand_dim` — [`L1931`](../../../../../raw/code/simply/simply/model_lib.py#L1931)
  - `tile_model_dim` — [`L1930`](../../../../../raw/code/simply/simply/model_lib.py#L1930)
  - `total_num_pages` — [`L1935`](../../../../../raw/code/simply/simply/model_lib.py#L1935)
  - `use_flash_attention` — [`L1912`](../../../../../raw/code/simply/simply/model_lib.py#L1912)
  - `use_gated_activation_in_ffn` — [`L1897`](../../../../../raw/code/simply/simply/model_lib.py#L1897)
  - `use_moe` — [`L1900`](../../../../../raw/code/simply/simply/model_lib.py#L1900)
  - `use_per_dim_scale` — [`L1898`](../../../../../raw/code/simply/simply/model_lib.py#L1898)
  - `use_post_ln` — [`L1894`](../../../../../raw/code/simply/simply/model_lib.py#L1894)
  - `use_post_skip_ln` — [`L1895`](../../../../../raw/code/simply/simply/model_lib.py#L1895)
  - `use_pre_ln` — [`L1893`](../../../../../raw/code/simply/simply/model_lib.py#L1893)
  - `use_qk_norm` — [`L1896`](../../../../../raw/code/simply/simply/model_lib.py#L1896)
  - `use_rmsnorm` — [`L1892`](../../../../../raw/code/simply/simply/model_lib.py#L1892)
  - `use_window_chunk` — [`L1915`](../../../../../raw/code/simply/simply/model_lib.py#L1915)
  - `window_size` — [`L1914`](../../../../../raw/code/simply/simply/model_lib.py#L1914)
- uses (calls/refs, reference-scoped): [`register`](utils/registry.md#RootRegistry.register), [`Array`](model_lib.md#Array), [`with_sharding_constraint`](utils/sharding.md#with_sharding_constraint), [`PyTree`](model_lib.md#PyTree), [`RoPE`](utils/position_encoding.md#RoPE), [`SimplyModule`](utils/module.md#SimplyModule), [`ModuleRegistry`](utils/module.md#ModuleRegistry), [`LayerNorm`](model_lib.md#LayerNorm), [`Initializer`](utils/initializer.md#Initializer), [`neg_inf`](utils/common.md#neg_inf), [`XavierUniformInit`](utils/initializer.md#XavierUniformInit), [`MoEFeedForward`](model_lib.md#MoEFeedForward), [`FeedForward`](model_lib.md#FeedForward), [`PRNGKey`](model_lib.md#PRNGKey), [`Attention`](model_lib.md#Attention), [`DTypeLike`](model_lib.md#DTypeLike), [`SimplyConfig`](model_lib.md#SimplyConfig), [`PositionEncodingConfig`](utils/position_encoding.md#PositionEncodingConfig)
- used by: [`apply`](utils/module.md#SimplyModule.apply), [`init`](utils/module.md#SimplyModule.init), [`setup`](utils/module.md#SimplyModule.setup), [`transformer_blocks`](utils/vision_encoder.md#VisionTransformer.transformer_blocks), [`_create_transformer_block`](model_lib.md#TransformerLM._create_transformer_block)

### `TransformerLM`
- def: [`simply/model_lib.py:2178`](../../../../../raw/code/simply/simply/model_lib.py#L2178)
- doc: A decoder-only Transformer.
- signature: `class TransformerLM(module.SimplyModule):`
- members:
  - `_replace_embeddings(self, orig_embeddings, replacement_embeddings, replacement_mask)` — [`L2298`](../../../../../raw/code/simply/simply/model_lib.py#L2298) — Replaces a sequence of embeddings at certain positions.
  - `apply(self, params: PyTree, x: Array, *, segment_ids: Array | None = None, segment_positions: Array | None = None, extra_inputs: PyTree = None, decode_state: PyTree = None)` — [`L2329`](../../../../../raw/code/simply/simply/model_lib.py#L2329) — Transformer forward pass. — documented in [simply-model_lib](../../concepts/simply-model_lib.md)
  - `convert_to_lower_bits(x, activation_dtype)` — [`L2380`](../../../../../raw/code/simply/simply/model_lib.py#L2380)
  - `init(self, prng_key: PRNGKey)` — [`L2278`](../../../../../raw/code/simply/simply/model_lib.py#L2278)
  - `init_decode_state(self, max_seq_len: int)` — [`L2583`](../../../../../raw/code/simply/simply/model_lib.py#L2583)
  - `predict_probs(self, params: PyTree, x: Array, temperature: float = 1)` — [`L2575`](../../../../../raw/code/simply/simply/model_lib.py#L2575)
  - `setup(self)` — [`L2184`](../../../../../raw/code/simply/simply/model_lib.py#L2184)
  - `substitute_embeddings(x, y, mask)` — [`L2318`](../../../../../raw/code/simply/simply/model_lib.py#L2318)
  - `activation_dtype` — [`L2189`](../../../../../raw/code/simply/simply/model_lib.py#L2189)
  - `blocks` — [`L2263`](../../../../../raw/code/simply/simply/model_lib.py#L2263)
  - `config` — [`L2181`](../../../../../raw/code/simply/simply/model_lib.py#L2181)
  - `embed_linear` — [`L2191`](../../../../../raw/code/simply/simply/model_lib.py#L2191)
  - `final_ln` — [`L2270`](../../../../../raw/code/simply/simply/model_lib.py#L2270)
  - `input_encoders` — [`L2200`](../../../../../raw/code/simply/simply/model_lib.py#L2200)
  - `sharding_config` — [`L2182`](../../../../../raw/code/simply/simply/model_lib.py#L2182)
- protocol/private: `_create_transformer_block`[`L2206`](../../../../../raw/code/simply/simply/model_lib.py#L2206), `_prepare_stack_list`[`L2424`](../../../../../raw/code/simply/simply/model_lib.py#L2424), `_process_per_repeat`[`L2463`](../../../../../raw/code/simply/simply/model_lib.py#L2463)
- uses (calls/refs, reference-scoped): [`register`](utils/registry.md#RootRegistry.register), [`Array`](model_lib.md#Array), [`with_sharding_constraint`](utils/sharding.md#with_sharding_constraint), [`PyTree`](model_lib.md#PyTree), [`init`](utils/module.md#SimplyModule.init), [`SimplyModule`](utils/module.md#SimplyModule), [`ModuleRegistry`](utils/module.md#ModuleRegistry), [`LayerNorm`](model_lib.md#LayerNorm), [`EmbeddingLinear`](utils/module.md#EmbeddingLinear), [`apply`](model_lib.md#InputEncoderInterface.apply), [`TransformerBlock`](model_lib.md#TransformerBlock), [`PRNGKey`](model_lib.md#PRNGKey), [`SimplyConfig`](model_lib.md#SimplyConfig), [`InputEncoderInterface`](model_lib.md#InputEncoderInterface), [`soft_cap`](model_lib.md#soft_cap), [`embeddings`](model_lib.md#InputEncoderInterface.Output.embeddings), [`extra_input_keys`](model_lib.md#InputEncoderInterface.extra_input_keys), [`name`](model_lib.md#InputEncoderInterface.name), [`embedding_mask`](model_lib.md#InputEncoderInterface.Output.embedding_mask)
- used by: [`main`](eval/decode_eval.md#main), [`apply`](utils/module.md#SimplyModule.apply), [`tfm_lm`](model_lib_test.md#ModelLibTest.tfm_lm), [`init`](utils/module.md#SimplyModule.init), [`test_lm_interface_generate_with_local_state`](model_lib_test.md#ModelLibTest.test_lm_interface_generate_with_local_state), [`test_restore_qwen3_moe_format`](utils/checkpoint_lib_test.md#QwenFormatTest.test_restore_qwen3_moe_format), [`setup`](utils/module.md#SimplyModule.setup), [`test_dump_load_baseline_config`](config_lib_test.md#ConfigLibTest.test_dump_load_baseline_config), [`test_restore_qwen2_format`](utils/checkpoint_lib_test.md#QwenFormatTest.test_restore_qwen2_format), [`test_grad_accumulation`](model_lib_test.md#ModelLibTest.test_grad_accumulation), [`expected_state`](utils/checkpoint_lib_test.md#CheckpointFormatTest.expected_state), [`test_backward_pass`](model_lib_test.md#ModelLibTest.test_backward_pass), [`test_expand_dim_in_ffn`](model_lib_test.md#ModelLibTest.test_expand_dim_in_ffn), [`test_quantization`](model_lib_test.md#ModelLibTest.test_quantization), [`model`](serving/page_batcher.md#Batcher.model), [`expected_state`](utils/checkpoint_lib_test.md#QwenFormatTest.expected_state), [`test_dim_annotation`](model_lib_test.md#ModelLibTest.test_dim_annotation), [`test_logits_with_kv_cache`](model_lib_test.md#ModelLibTest.test_logits_with_kv_cache), [`test_mixed_precision`](model_lib_test.md#ModelLibTest.test_mixed_precision), [`test_n_blocks`](model_lib_test.md#ModelLibTest.test_n_blocks), [`model`](serving/vanilla_server.md#Batcher.model), [`test_forward_pass`](model_lib_test.md#ModelLibTest.test_forward_pass), [`test_identical_inputs_run_to_run_consistency`](model_lib_test.md#ModelLibTest.test_identical_inputs_run_to_run_consistency), [`test_identitical_inputs_batch_consistency`](model_lib_test.md#ModelLibTest.test_identitical_inputs_batch_consistency)

## Functions
- `_build_global_array_from_sharded(array: np.ndarray)` — [`L3235`](../../../../../raw/code/simply/simply/model_lib.py#L3235)
- `_compute_grad(batch)` — [`L2787`](../../../../../raw/code/simply/simply/model_lib.py#L2787)
- `_custom_permute(x, permute_indices)` — [`L624`](../../../../../raw/code/simply/simply/model_lib.py#L624)
- `_custom_permute_bwd(res, g)` — [`L632`](../../../../../raw/code/simply/simply/model_lib.py#L632)
- `_custom_permute_fwd(x, permute_indices)` — [`L628`](../../../../../raw/code/simply/simply/model_lib.py#L628)
- `_score_fn(logits: Array, tokens: Array)` — [`L4205`](../../../../../raw/code/simply/simply/model_lib.py#L4205)
- `_update_kv(cache_state, input_state)` — [`L285`](../../../../../raw/code/simply/simply/model_lib.py#L285)
- `_windowized_array(x: Array)` — [`L321`](../../../../../raw/code/simply/simply/model_lib.py#L321)
- `attn(q: jax.Array, k: jax.Array, v: jax.Array, mask: jax.Array, *, attn_soft_cap: float = 50, attn_mask_value: float = common.neg_inf('float32'), dtype: jax.typing.DTypeLike = 'bfloat16')` — [`L497`](../../../../../raw/code/simply/simply/model_lib.py#L497)
- `body_fn(sampling_state: SamplingState)` — [`L4167`](../../../../../raw/code/simply/simply/model_lib.py#L4167)
- `build_global_array_from_replicated(batch: PyTree, data_partition: PartitionAnnotation = None)` — [`L3217`](../../../../../raw/code/simply/simply/model_lib.py#L3217)
- `build_global_batch_from_sharded(batch: PyTree, data_partition: PartitionAnnotation = None)` — [`L3227`](../../../../../raw/code/simply/simply/model_lib.py#L3227)
- `chunked_local_attn(q: jax.Array, k: jax.Array, v: jax.Array, mask: jax.Array, window_size: int, *, attn_soft_cap: float = 50, attn_mask_value: float = common.neg_inf('float32'), dtype: jax.typing.DTypeLike = 'bfloat16')` — [`L404`](../../../../../raw/code/simply/simply/model_lib.py#L404) — Chunked local attention.
- `clip_tree_fn(tree, name, threshold, fn, fn_name, clip_local=False, add_log_info=False)` — [`L2921`](../../../../../raw/code/simply/simply/model_lib.py#L2921)
- `collect_loss_and_metric(extra_outputs)` — [`L2664`](../../../../../raw/code/simply/simply/model_lib.py#L2664)
- `collect_tag(extra_outputs, tag)` — [`L2643`](../../../../../raw/code/simply/simply/model_lib.py#L2643)
- `compute_batch_stats_info(batch: Batch, pad_id: int = 0)` — [`L3369`](../../../../../raw/code/simply/simply/model_lib.py#L3369)
- `compute_distill_loss(model, params, teacher_model, teacher_params, batch, temperature=1, alpha: float = 1)` — [`L2679`](../../../../../raw/code/simply/simply/model_lib.py#L2679) — Computes the distillation loss between a student and teacher model.
- `compute_eval_loss(model, params, batch)` — [`L2675`](../../../../../raw/code/simply/simply/model_lib.py#L2675)
- `compute_loss(model, params, batch, add_extra_loss=True)` — [`L2597`](../../../../../raw/code/simply/simply/model_lib.py#L2597) — The base method for loss computation.
- `compute_train_loss(model, params, batch)` — [`L2671`](../../../../../raw/code/simply/simply/model_lib.py#L2671)
- `compute_tree_info_fn(tree, name, fn, fn_name)` — [`L2946`](../../../../../raw/code/simply/simply/model_lib.py#L2946)
- `cond_fn(sampling_state: SamplingState)` — [`L4242`](../../../../../raw/code/simply/simply/model_lib.py#L4242)
- `continue_decode(apply_fn: Callable[..., Array], params: PyTree, init_sampling_state: SamplingState, extra_inputs: Mapping[str, PyTree] | None = None, temperature: float = 1, top_k: int = -1, top_p: float = 1, scoring_temperature: float = 1, scoring_top_k: int = -1, scoring_top_p: float = 1)` — [`L4154`](../../../../../raw/code/simply/simply/model_lib.py#L4154)
- `create_mask(segment_positions: Array, kv_segment_positions: Array, segment_ids: Array, kv_segment_ids: Array, window_size: int = 0)` — [`L355`](../../../../../raw/code/simply/simply/model_lib.py#L355) — Create a mask for attention.
- `create_model(config, sharding_config=None)` — [`L3203`](../../../../../raw/code/simply/simply/model_lib.py#L3203)
- `flatten_dict(d: dict[str, Any])` — [`L3361`](../../../../../raw/code/simply/simply/model_lib.py#L3361)
- `gelu(x: Array)` — [`L91`](../../../../../raw/code/simply/simply/model_lib.py#L91)
- `get_init_state(config, sharding_config, ckpt_mngr, ckpt_dir)` — [`L3247`](../../../../../raw/code/simply/simply/model_lib.py#L3247)
- `get_scaling_info(config, also_print=False, add_attn_flops=False)` — [`L4257`](../../../../../raw/code/simply/simply/model_lib.py#L4257)
- `gmm(lhs, rhs, group_sizes, *, tiling, gmm_impl, activation_dtype)` — [`L206`](../../../../../raw/code/simply/simply/model_lib.py#L206)
- `grad_accum_step_fn(accum_info, minibatch)` — [`L2820`](../../../../../raw/code/simply/simply/model_lib.py#L2820)
- `pad_along_axis(x: Array, pad_widths: tuple[int, int], axis: int, **kwargs: Any)` — [`L4106`](../../../../../raw/code/simply/simply/model_lib.py#L4106) — Pads the given array along the given axis.
- `pad_decode_state_to(d: PyTree, length_to_pad: int)` — [`L4127`](../../../../../raw/code/simply/simply/model_lib.py#L4127) — Pads the given decode state to the given length.
- `pad_to_along_axis(x: Array, pad_widths_to: int, axis: int, **kwargs: Any)` — [`L4116`](../../../../../raw/code/simply/simply/model_lib.py#L4116) — Pads the given array along the given axis to the given length.
- `permute(x, permute_indices, use_custom_vjp=True)` — [`L614`](../../../../../raw/code/simply/simply/model_lib.py#L614)
- `quantize_tfm_params(params, symmetric=False)` — [`L4311`](../../../../../raw/code/simply/simply/model_lib.py#L4311)
- `run_eval(eval_set, num_eval_steps, loss_fn, state)` — [`L3311`](../../../../../raw/code/simply/simply/model_lib.py#L3311)
- `run_experiment(config, experiment_dir='', mesh_shape=None, dcn_mesh_shape=None, decoding_mesh_shape=None, sharding_config=None, create_dataset=None)` — [`L2963`](../../../../../raw/code/simply/simply/model_lib.py#L2963) — documented in [simply-model_lib](../../concepts/simply-model_lib.md)
- `safe_clip(x, val, threshold)` — [`L2914`](../../../../../raw/code/simply/simply/model_lib.py#L2914)
- `soft_cap(x: Array, cap: float)` — [`L104`](../../../../../raw/code/simply/simply/model_lib.py#L104)
- `squared_relu(x: Array)` — [`L97`](../../../../../raw/code/simply/simply/model_lib.py#L97)
- `train_one_step(state, batch, model, opt, teacher_model=None, lr=0.0001, grad_accum_steps=-1, clip_grad_norm=-1, clip_update_norm=-1, clip_update_rms=-1, clip_local_update_rms=-1, weight_decay=-1, custom_loss_fn=None, add_log_info=False, distill_temperature: float = 1, distill_alpha: float = 1)` — [`L2754`](../../../../../raw/code/simply/simply/model_lib.py#L2754)
- `train_one_step_fn(state, batch, lr, add_log_info=False)` — [`L3020`](../../../../../raw/code/simply/simply/model_lib.py#L3020)
- `tree_norm(tree)` — [`L2900`](../../../../../raw/code/simply/simply/model_lib.py#L2900)
- `tree_rms(tree)` — [`L2906`](../../../../../raw/code/simply/simply/model_lib.py#L2906)
- `updated_decode_state(k: Array, v: Array, segment_positions: Array, segment_ids: Array, decode_state: PyTree, window_size: int = 0, update_kv_cache: bool = True)` — [`L229`](../../../../../raw/code/simply/simply/model_lib.py#L229) — Updates decode state when decode_state is not None.

## Module values
- `AnnotatedArray` — [`L79`](../../../../../raw/code/simply/simply/model_lib.py#L79)
- `Array` — [`L59`](../../../../../raw/code/simply/simply/model_lib.py#L59) — documented in [simply-model_lib](../../concepts/simply-model_lib.md)
- `Batch` — [`L52`](../../../../../raw/code/simply/simply/model_lib.py#L52)
- `DTypeLike` — [`L53`](../../../../../raw/code/simply/simply/model_lib.py#L53)
- `EinsumLinear` — [`L68`](../../../../../raw/code/simply/simply/model_lib.py#L68)
- `ExperimentHelper` — [`L65`](../../../../../raw/code/simply/simply/model_lib.py#L65)
- `PRNGKey` — [`L54`](../../../../../raw/code/simply/simply/model_lib.py#L54)
- `PartitionAnnotation` — [`L55`](../../../../../raw/code/simply/simply/model_lib.py#L55)
- `PyTree` — [`L56`](../../../../../raw/code/simply/simply/model_lib.py#L56) — documented in [simply-model_lib](../../concepts/simply-model_lib.md)
- `RawT` — [`L60`](../../../../../raw/code/simply/simply/model_lib.py#L60)
- `SamplingParams` — [`L67`](../../../../../raw/code/simply/simply/model_lib.py#L67) — documented in [simply-model_lib](../../concepts/simply-model_lib.md)
- `SimplyConfig` — [`L57`](../../../../../raw/code/simply/simply/model_lib.py#L57)
- `SimplyModule` — [`L58`](../../../../../raw/code/simply/simply/model_lib.py#L58)
- `create_lr_schedule` — [`L66`](../../../../../raw/code/simply/simply/model_lib.py#L66)
- `get_default_mesh` — [`L61`](../../../../../raw/code/simply/simply/model_lib.py#L61)
- `get_partition_axis` — [`L62`](../../../../../raw/code/simply/simply/model_lib.py#L62)
- `get_raw_arrays` — [`L82`](../../../../../raw/code/simply/simply/model_lib.py#L82)
- `maybe_dequantize_array` — [`L63`](../../../../../raw/code/simply/simply/model_lib.py#L63)
- `mesh_sharding` — [`L64`](../../../../../raw/code/simply/simply/model_lib.py#L64)
- `neg_inf` — [`L83`](../../../../../raw/code/simply/simply/model_lib.py#L83)
- `round_up_to_base` — [`L84`](../../../../../raw/code/simply/simply/model_lib.py#L84)


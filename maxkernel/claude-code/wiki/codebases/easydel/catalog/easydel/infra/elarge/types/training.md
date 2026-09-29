---
title: 'Module: easydel/infra/elarge/types/training.py'
type: catalog
provenance: extracted
module: easydel/infra/elarge/types/training.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.infra.elarge.types.training`/
symbols:
  TrainerConfig: TrainerConfig#
  BaseTrainerCfg: BaseTrainerCfg#
  GRPOTrainerCfg: GRPOTrainerCfg#
  normalize_trainer_config: normalize_trainer_config().
  DistillationTrainerCfg: DistillationTrainerCfg#
  get_trainer_defaults: get_trainer_defaults().
  SFTTrainerCfg: SFTTrainerCfg#
  get_trainer_class: get_trainer_class().
  get_training_arguments_class: get_training_arguments_class().
  register_trainer_defaults: register_trainer_defaults().
  _normalize_trainer_type: _normalize_trainer_type().
  DPOTrainerCfg: DPOTrainerCfg#
  ORPOTrainerCfg: ORPOTrainerCfg#
  AgenticMoshPitTrainerCfg: AgenticMoshPitTrainerCfg#
  RLVRTrainerCfg: RLVRTrainerCfg#
  SDPOTrainerCfg: SDPOTrainerCfg#
  PPOTrainerCfg: PPOTrainerCfg#
  RewardTrainerCfg: RewardTrainerCfg#
  OnPolicyDistillationTrainerCfg: OnPolicyDistillationTrainerCfg#
  SeqKDTrainerCfg: SeqKDTrainerCfg#
  SparseDistillationTrainerCfg: SparseDistillationTrainerCfg#
  KTOTrainerCfg: KTOTrainerCfg#
  BCOTrainerCfg: BCOTrainerCfg#
  CPOTrainerCfg: CPOTrainerCfg#
  GKDTrainerCfg: GKDTrainerCfg#
  NashMDTrainerCfg: NashMDTrainerCfg#
  XPOTrainerCfg: XPOTrainerCfg#
  TRAINER_SPECIFIC_DEFAULTS.TRAINER_SPECIFIC_DEFAULTS: TRAINER_SPECIFIC_DEFAULTS.TRAINER_SPECIFIC_DEFAULTS.
  EmbeddingTrainerCfg: EmbeddingTrainerCfg#
  BASE_TRAINER_DEFAULTS.BASE_TRAINER_DEFAULTS: BASE_TRAINER_DEFAULTS.BASE_TRAINER_DEFAULTS.
  BaseTrainerCfg.loss_config: BaseTrainerCfg#loss_config.
  BaseTrainerCfg.benchmarks: BaseTrainerCfg#benchmarks.
  LossConfig: LossConfig#
  _TRAINERS_WITH_COMPLETION_LENGTH: _TRAINERS_WITH_COMPLETION_LENGTH.
  _TRAINER_TYPE_ALIASES._TRAINER_TYPE_ALIASES: _TRAINER_TYPE_ALIASES._TRAINER_TYPE_ALIASES.
  LossConfig.ignore_index: LossConfig#ignore_index.
  LossConfig.label_smoothing: LossConfig#label_smoothing.
  LossConfig.z_loss: LossConfig#z_loss.
  LossConfig.loss_normalizing_factor: LossConfig#loss_normalizing_factor.
  LossConfig.num_labels: LossConfig#num_labels.
  LossConfig.problem_type: LossConfig#problem_type.
  LossConfig.divide_weight_sum: LossConfig#divide_weight_sum.
  LossConfig.shift_tokens: LossConfig#shift_tokens.
  LossConfig.break_on_nan: LossConfig#break_on_nan.
  LossConfig.reduction: LossConfig#reduction.
  LossConfig.num_classification_labels: LossConfig#num_classification_labels.
  LossConfig.classification_problem_type: LossConfig#classification_problem_type.
  LossConfig.chunk_vocab_size: LossConfig#chunk_vocab_size.
  LossConfig.chunk_token_size: LossConfig#chunk_token_size.
  LossConfig.chunk_block_size: LossConfig#chunk_block_size.
  LossConfig.compute_dtype: LossConfig#compute_dtype.
  BaseTrainerCfg.trainer_type: BaseTrainerCfg#trainer_type.
  BaseTrainerCfg.learning_rate: BaseTrainerCfg#learning_rate.
  BaseTrainerCfg.learning_rate_end: BaseTrainerCfg#learning_rate_end.
  BaseTrainerCfg.num_train_epochs: BaseTrainerCfg#num_train_epochs.
  BaseTrainerCfg.max_training_steps: BaseTrainerCfg#max_training_steps.
  BaseTrainerCfg.per_epoch_training_steps: BaseTrainerCfg#per_epoch_training_steps.
  BaseTrainerCfg.per_epoch_evaluation_steps: BaseTrainerCfg#per_epoch_evaluation_steps.
  BaseTrainerCfg.total_batch_size: BaseTrainerCfg#total_batch_size.
  BaseTrainerCfg.eval_batch_size: BaseTrainerCfg#eval_batch_size.
  BaseTrainerCfg.gradient_accumulation_steps: BaseTrainerCfg#gradient_accumulation_steps.
  BaseTrainerCfg.optimizer: BaseTrainerCfg#optimizer.
  BaseTrainerCfg.scheduler: BaseTrainerCfg#scheduler.
  BaseTrainerCfg.warmup_steps: BaseTrainerCfg#warmup_steps.
  BaseTrainerCfg.weight_decay: BaseTrainerCfg#weight_decay.
  BaseTrainerCfg.clip_grad: BaseTrainerCfg#clip_grad.
  BaseTrainerCfg.extra_optimizer_kwargs: BaseTrainerCfg#extra_optimizer_kwargs.
  BaseTrainerCfg.custom_scheduler: BaseTrainerCfg#custom_scheduler.
  BaseTrainerCfg.dataloader_num_workers: BaseTrainerCfg#dataloader_num_workers.
  BaseTrainerCfg.dataloader_pin_memory: BaseTrainerCfg#dataloader_pin_memory.
  BaseTrainerCfg.remove_unused_columns: BaseTrainerCfg#remove_unused_columns.
  BaseTrainerCfg.ids_to_pop_from_dataset: BaseTrainerCfg#ids_to_pop_from_dataset.
  BaseTrainerCfg.shuffle_train_dataset: BaseTrainerCfg#shuffle_train_dataset.
  BaseTrainerCfg.shuffle_seed_train: BaseTrainerCfg#shuffle_seed_train.
  BaseTrainerCfg.use_data_collator: BaseTrainerCfg#use_data_collator.
  BaseTrainerCfg.use_grain: BaseTrainerCfg#use_grain.
  BaseTrainerCfg.grain_shard_index: BaseTrainerCfg#grain_shard_index.
  BaseTrainerCfg.grain_shard_count: BaseTrainerCfg#grain_shard_count.
  BaseTrainerCfg.offload_dataset: BaseTrainerCfg#offload_dataset.
  BaseTrainerCfg.offload_device_type: BaseTrainerCfg#offload_device_type.
  BaseTrainerCfg.offload_device_index: BaseTrainerCfg#offload_device_index.
  BaseTrainerCfg.do_train: BaseTrainerCfg#do_train.
  BaseTrainerCfg.do_eval: BaseTrainerCfg#do_eval.
  BaseTrainerCfg.do_last_save: BaseTrainerCfg#do_last_save.
  BaseTrainerCfg.is_fine_tuning: BaseTrainerCfg#is_fine_tuning.
  BaseTrainerCfg.init_tx: BaseTrainerCfg#init_tx.
  BaseTrainerCfg.train_on_inputs: BaseTrainerCfg#train_on_inputs.
  BaseTrainerCfg.aux_loss_enabled: BaseTrainerCfg#aux_loss_enabled.
  BaseTrainerCfg.training_time_limit: BaseTrainerCfg#training_time_limit.
  BaseTrainerCfg.step_start_point: BaseTrainerCfg#step_start_point.
  BaseTrainerCfg.force_step_start_point: BaseTrainerCfg#force_step_start_point.
  BaseTrainerCfg.resume_if_possible: BaseTrainerCfg#resume_if_possible.
  BaseTrainerCfg.truncation_mode: BaseTrainerCfg#truncation_mode.
  BaseTrainerCfg.max_length: BaseTrainerCfg#max_length.
  BaseTrainerCfg.max_sequence_length: BaseTrainerCfg#max_sequence_length.
  BaseTrainerCfg.save_interval_minutes: BaseTrainerCfg#save_interval_minutes.
  BaseTrainerCfg.save_steps: BaseTrainerCfg#save_steps.
  BaseTrainerCfg.save_total_limit: BaseTrainerCfg#save_total_limit.
  BaseTrainerCfg.save_directory: BaseTrainerCfg#save_directory.
  BaseTrainerCfg.save_optimizer_state: BaseTrainerCfg#save_optimizer_state.
  BaseTrainerCfg.merge_lora_before_save: BaseTrainerCfg#merge_lora_before_save.
  BaseTrainerCfg.merge_lora_before_tpu_preemption_save: BaseTrainerCfg#merge_lora_before_tpu_preemption_save.
  BaseTrainerCfg.save_tpu_preemption_checkpoints: BaseTrainerCfg#save_tpu_preemption_checkpoints.
  BaseTrainerCfg.remove_ckpt_after_load: BaseTrainerCfg#remove_ckpt_after_load.
  BaseTrainerCfg.evaluation_steps: BaseTrainerCfg#evaluation_steps.
  BaseTrainerCfg.max_evaluation_steps: BaseTrainerCfg#max_evaluation_steps.
  BaseTrainerCfg.log_steps: BaseTrainerCfg#log_steps.
  BaseTrainerCfg.report_steps: BaseTrainerCfg#report_steps.
  BaseTrainerCfg.log_all_workers: BaseTrainerCfg#log_all_workers.
  BaseTrainerCfg.log_grad_norms: BaseTrainerCfg#log_grad_norms.
  BaseTrainerCfg.report_metrics: BaseTrainerCfg#report_metrics.
  BaseTrainerCfg.metrics_to_show_in_rich_pbar: BaseTrainerCfg#metrics_to_show_in_rich_pbar.
  BaseTrainerCfg.progress_bar_type: BaseTrainerCfg#progress_bar_type.
  BaseTrainerCfg.watchers: BaseTrainerCfg#watchers.
  BaseTrainerCfg.weight_distribution_pattern: BaseTrainerCfg#weight_distribution_pattern.
  BaseTrainerCfg.weight_distribution_log_steps: BaseTrainerCfg#weight_distribution_log_steps.
  BaseTrainerCfg.verbose: BaseTrainerCfg#verbose.
  BaseTrainerCfg.process_zero_is_admin: BaseTrainerCfg#process_zero_is_admin.
  BaseTrainerCfg.use_wandb: BaseTrainerCfg#use_wandb.
  BaseTrainerCfg.wandb_entity: BaseTrainerCfg#wandb_entity.
  BaseTrainerCfg.wandb_name: BaseTrainerCfg#wandb_name.
  BaseTrainerCfg.trainer_prefix: BaseTrainerCfg#trainer_prefix.
  BaseTrainerCfg.backend: BaseTrainerCfg#backend.
  BaseTrainerCfg.auto_shard_states: BaseTrainerCfg#auto_shard_states.
  BaseTrainerCfg.performance_mode: BaseTrainerCfg#performance_mode.
  BaseTrainerCfg.track_memory: BaseTrainerCfg#track_memory.
  BaseTrainerCfg.low_mem_usage: BaseTrainerCfg#low_mem_usage.
  BaseTrainerCfg.quantization_mode: BaseTrainerCfg#quantization_mode.
  BaseTrainerCfg.quantization_group_size: BaseTrainerCfg#quantization_group_size.
  BaseTrainerCfg.quantization_bits: BaseTrainerCfg#quantization_bits.
  BaseTrainerCfg.tensor_straight_through: BaseTrainerCfg#tensor_straight_through.
  BaseTrainerCfg.straight_through_emulator: BaseTrainerCfg#straight_through_emulator.
  BaseTrainerCfg.model_name: BaseTrainerCfg#model_name.
  BaseTrainerCfg.model_parameters: BaseTrainerCfg#model_parameters.
  BaseTrainerCfg.frozen_parameters: BaseTrainerCfg#frozen_parameters.
  BaseTrainerCfg.lmhead_chunksize: BaseTrainerCfg#lmhead_chunksize.
  BaseTrainerCfg.jax_distributed_config: BaseTrainerCfg#jax_distributed_config.
  BaseTrainerCfg.step_partition_spec: BaseTrainerCfg#step_partition_spec.
  BaseTrainerCfg.state_apply_fn_kwarguments_to_model: BaseTrainerCfg#state_apply_fn_kwarguments_to_model.
  BaseTrainerCfg.sparsify_module: BaseTrainerCfg#sparsify_module.
  BaseTrainerCfg.sparse_module_type: BaseTrainerCfg#sparse_module_type.
  BaseTrainerCfg.pruning_module: BaseTrainerCfg#pruning_module.
  BaseTrainerCfg.tx_mu_dtype: BaseTrainerCfg#tx_mu_dtype.
  BaseTrainerCfg.generation_top_p: BaseTrainerCfg#generation_top_p.
  BaseTrainerCfg.generation_top_k: BaseTrainerCfg#generation_top_k.
  BaseTrainerCfg.generation_presence_penalty: BaseTrainerCfg#generation_presence_penalty.
  BaseTrainerCfg.generation_frequency_penalty: BaseTrainerCfg#generation_frequency_penalty.
  BaseTrainerCfg.generation_repetition_penalty: BaseTrainerCfg#generation_repetition_penalty.
  BaseTrainerCfg.generation_temperature: BaseTrainerCfg#generation_temperature.
  BaseTrainerCfg.generation_do_sample: BaseTrainerCfg#generation_do_sample.
  BaseTrainerCfg.generation_num_return_sequences: BaseTrainerCfg#generation_num_return_sequences.
  BaseTrainerCfg.generation_max_new_tokens: BaseTrainerCfg#generation_max_new_tokens.
  BaseTrainerCfg.generation_shard_inputs: BaseTrainerCfg#generation_shard_inputs.
  BaseTrainerCfg.generation_interval: BaseTrainerCfg#generation_interval.
  BaseTrainerCfg.generation_prompts: BaseTrainerCfg#generation_prompts.
  BaseTrainerCfg.generation_use_train_prompts: BaseTrainerCfg#generation_use_train_prompts.
  BaseTrainerCfg.generation_num_prompts: BaseTrainerCfg#generation_num_prompts.
  BaseTrainerCfg.generation_dataset_prompt_field: BaseTrainerCfg#generation_dataset_prompt_field.
  BaseTrainerCfg.generation_extra_kwargs: BaseTrainerCfg#generation_extra_kwargs.
  BaseTrainerCfg.generation_config_overrides: BaseTrainerCfg#generation_config_overrides.
  BaseTrainerCfg.generation_seed: BaseTrainerCfg#generation_seed.
  BaseTrainerCfg.generation_preview_print: BaseTrainerCfg#generation_preview_print.
  BaseTrainerCfg.generation_log_to_wandb: BaseTrainerCfg#generation_log_to_wandb.
  BaseTrainerCfg.log_training_generations_to_wandb: BaseTrainerCfg#log_training_generations_to_wandb.
  BaseTrainerCfg.benchmark_interval: BaseTrainerCfg#benchmark_interval.
  BaseTrainerCfg.use_esurge_generation: BaseTrainerCfg#use_esurge_generation.
  BaseTrainerCfg.esurge_use_tqdm: BaseTrainerCfg#esurge_use_tqdm.
  BaseTrainerCfg.esurge_hbm_utilization: BaseTrainerCfg#esurge_hbm_utilization.
  BaseTrainerCfg.esurge_max_num_seqs: BaseTrainerCfg#esurge_max_num_seqs.
  BaseTrainerCfg.esurge_min_input_pad: BaseTrainerCfg#esurge_min_input_pad.
  BaseTrainerCfg.esurge_page_size: BaseTrainerCfg#esurge_page_size.
  BaseTrainerCfg.esurge_silent_mode: BaseTrainerCfg#esurge_silent_mode.
  BaseTrainerCfg.esurge_runner_verbose: BaseTrainerCfg#esurge_runner_verbose.
  BaseTrainerCfg.esurge_max_num_batched_tokens: BaseTrainerCfg#esurge_max_num_batched_tokens.
  BaseTrainerCfg.esurge_enable_prefix_caching: BaseTrainerCfg#esurge_enable_prefix_caching.
  BaseTrainerCfg.esurge_data_parallelism_axis: BaseTrainerCfg#esurge_data_parallelism_axis.
  BaseTrainerCfg.esurge_max_num_seq_buckets: BaseTrainerCfg#esurge_max_num_seq_buckets.
  DPOTrainerCfg.beta: DPOTrainerCfg#beta.
  DPOTrainerCfg.label_smoothing: DPOTrainerCfg#label_smoothing.
  DPOTrainerCfg.loss_type: DPOTrainerCfg#loss_type.
  DPOTrainerCfg.use_weighting: DPOTrainerCfg#use_weighting.
  DPOTrainerCfg.label_pad_token_id: DPOTrainerCfg#label_pad_token_id.
  DPOTrainerCfg.padding_value: DPOTrainerCfg#padding_value.
  DPOTrainerCfg.max_length: DPOTrainerCfg#max_length.
  DPOTrainerCfg.max_prompt_length: DPOTrainerCfg#max_prompt_length.
  DPOTrainerCfg.max_completion_length: DPOTrainerCfg#max_completion_length.
  DPOTrainerCfg.is_encoder_decoder: DPOTrainerCfg#is_encoder_decoder.
  DPOTrainerCfg.disable_dropout: DPOTrainerCfg#disable_dropout.
  DPOTrainerCfg.precompute_ref_log_probs: DPOTrainerCfg#precompute_ref_log_probs.
  DPOTrainerCfg.dataset_num_proc: DPOTrainerCfg#dataset_num_proc.
  DPOTrainerCfg.reference_free: DPOTrainerCfg#reference_free.
  DPOTrainerCfg.force_use_ref_model: DPOTrainerCfg#force_use_ref_model.
  DPOTrainerCfg.sync_ref_model: DPOTrainerCfg#sync_ref_model.
  DPOTrainerCfg.ref_model_mixup_alpha: DPOTrainerCfg#ref_model_mixup_alpha.
  DPOTrainerCfg.ref_model_sync_steps: DPOTrainerCfg#ref_model_sync_steps.
  DPOTrainerCfg.rpo_alpha: DPOTrainerCfg#rpo_alpha.
  DPOTrainerCfg.logprob_vocab_chunk_size: DPOTrainerCfg#logprob_vocab_chunk_size.
  DPOTrainerCfg.tools: DPOTrainerCfg#tools.
  ORPOTrainerCfg.beta: ORPOTrainerCfg#beta.
  ORPOTrainerCfg.max_length: ORPOTrainerCfg#max_length.
  ORPOTrainerCfg.max_prompt_length: ORPOTrainerCfg#max_prompt_length.
  ORPOTrainerCfg.max_completion_length: ORPOTrainerCfg#max_completion_length.
  ORPOTrainerCfg.logprob_vocab_chunk_size: ORPOTrainerCfg#logprob_vocab_chunk_size.
  ORPOTrainerCfg.disable_dropout: ORPOTrainerCfg#disable_dropout.
  ORPOTrainerCfg.label_pad_token_id: ORPOTrainerCfg#label_pad_token_id.
  ORPOTrainerCfg.padding_value: ORPOTrainerCfg#padding_value.
  ORPOTrainerCfg.generate_during_eval: ORPOTrainerCfg#generate_during_eval.
  ORPOTrainerCfg.is_encoder_decoder: ORPOTrainerCfg#is_encoder_decoder.
  ORPOTrainerCfg.dataset_num_proc: ORPOTrainerCfg#dataset_num_proc.
  GRPOTrainerCfg.beta: GRPOTrainerCfg#beta.
  GRPOTrainerCfg.max_prompt_length: GRPOTrainerCfg#max_prompt_length.
  GRPOTrainerCfg.max_completion_length: GRPOTrainerCfg#max_completion_length.
  GRPOTrainerCfg.dataset_num_proc: GRPOTrainerCfg#dataset_num_proc.
  GRPOTrainerCfg.sync_ref_model: GRPOTrainerCfg#sync_ref_model.
  GRPOTrainerCfg.ref_model_mixup_alpha: GRPOTrainerCfg#ref_model_mixup_alpha.
  GRPOTrainerCfg.ref_model_sync_steps: GRPOTrainerCfg#ref_model_sync_steps.
  GRPOTrainerCfg.tools: GRPOTrainerCfg#tools.
  GRPOTrainerCfg.skip_apply_chat_template: GRPOTrainerCfg#skip_apply_chat_template.
  GRPOTrainerCfg.num_return_sequences: GRPOTrainerCfg#num_return_sequences.
  GRPOTrainerCfg.ref_logps_chunk_size: GRPOTrainerCfg#ref_logps_chunk_size.
  GRPOTrainerCfg.completion_chunk_size: GRPOTrainerCfg#completion_chunk_size.
  GRPOTrainerCfg.max_loss_completion_tokens: GRPOTrainerCfg#max_loss_completion_tokens.
  GRPOTrainerCfg.logprob_vocab_chunk_size: GRPOTrainerCfg#logprob_vocab_chunk_size.
  GRPOTrainerCfg.top_p: GRPOTrainerCfg#top_p.
  GRPOTrainerCfg.top_k: GRPOTrainerCfg#top_k.
  GRPOTrainerCfg.presence_penalty: GRPOTrainerCfg#presence_penalty.
  GRPOTrainerCfg.frequency_penalty: GRPOTrainerCfg#frequency_penalty.
  GRPOTrainerCfg.repetition_penalty: GRPOTrainerCfg#repetition_penalty.
  GRPOTrainerCfg.temperature: GRPOTrainerCfg#temperature.
  AgenticMoshPitTrainerCfg.max_steps: AgenticMoshPitTrainerCfg#max_steps.
  AgenticMoshPitTrainerCfg.group_size: AgenticMoshPitTrainerCfg#group_size.
  AgenticMoshPitTrainerCfg.num_env_groups: AgenticMoshPitTrainerCfg#num_env_groups.
  AgenticMoshPitTrainerCfg.reward_mode: AgenticMoshPitTrainerCfg#reward_mode.
  AgenticMoshPitTrainerCfg.advantage_estimator: AgenticMoshPitTrainerCfg#advantage_estimator.
  AgenticMoshPitTrainerCfg.step_reward_gamma: AgenticMoshPitTrainerCfg#step_reward_gamma.
  AgenticMoshPitTrainerCfg.episode_reward_weight: AgenticMoshPitTrainerCfg#episode_reward_weight.
  AgenticMoshPitTrainerCfg.step_reward_weight: AgenticMoshPitTrainerCfg#step_reward_weight.
  AgenticMoshPitTrainerCfg.system_prompt: AgenticMoshPitTrainerCfg#system_prompt.
  AgenticMoshPitTrainerCfg.tool_names: AgenticMoshPitTrainerCfg#tool_names.
  AgenticMoshPitTrainerCfg.tool_caller: AgenticMoshPitTrainerCfg#tool_caller.
  AgenticMoshPitTrainerCfg.max_tool_calls_per_step: AgenticMoshPitTrainerCfg#max_tool_calls_per_step.
  AgenticMoshPitTrainerCfg.reasoning_parser: AgenticMoshPitTrainerCfg#reasoning_parser.
  RLVRTrainerCfg.answer_key: RLVRTrainerCfg#answer_key.
  RLVRTrainerCfg.test_key: RLVRTrainerCfg#test_key.
  RLVRTrainerCfg.format_pattern: RLVRTrainerCfg#format_pattern.
  RLVRTrainerCfg.format_reward_weight: RLVRTrainerCfg#format_reward_weight.
  RLVRTrainerCfg.length_penalty_target: RLVRTrainerCfg#length_penalty_target.
  RLVRTrainerCfg.length_penalty_weight: RLVRTrainerCfg#length_penalty_weight.
  RLVRTrainerCfg.max_len_mask: RLVRTrainerCfg#max_len_mask.
  RLVRTrainerCfg.reward_clip_range: RLVRTrainerCfg#reward_clip_range.
  RLVRTrainerCfg.difficulty_key: RLVRTrainerCfg#difficulty_key.
  RLVRTrainerCfg.difficulty_loss_weight: RLVRTrainerCfg#difficulty_loss_weight.
  SDPOTrainerCfg.max_feedback_length: SDPOTrainerCfg#max_feedback_length.
  SDPOTrainerCfg.distillation_type: SDPOTrainerCfg#distillation_type.
  SDPOTrainerCfg.beta: SDPOTrainerCfg#beta.
  PPOTrainerCfg.max_prompt_length: PPOTrainerCfg#max_prompt_length.
  PPOTrainerCfg.max_completion_length: PPOTrainerCfg#max_completion_length.
  PPOTrainerCfg.dataset_num_proc: PPOTrainerCfg#dataset_num_proc.
  PPOTrainerCfg.reward_weights: PPOTrainerCfg#reward_weights.
  PPOTrainerCfg.kl_coef: PPOTrainerCfg#kl_coef.
  PPOTrainerCfg.kl_estimator: PPOTrainerCfg#kl_estimator.
  PPOTrainerCfg.cliprange: PPOTrainerCfg#cliprange.
  PPOTrainerCfg.vf_coef: PPOTrainerCfg#vf_coef.
  PPOTrainerCfg.cliprange_value: PPOTrainerCfg#cliprange_value.
  PPOTrainerCfg.gamma: PPOTrainerCfg#gamma.
  PPOTrainerCfg.lam: PPOTrainerCfg#lam.
  PPOTrainerCfg.whiten_rewards: PPOTrainerCfg#whiten_rewards.
  PPOTrainerCfg.whiten_advantages: PPOTrainerCfg#whiten_advantages.
  PPOTrainerCfg.entropy_coef: PPOTrainerCfg#entropy_coef.
  PPOTrainerCfg.missing_eos_penalty: PPOTrainerCfg#missing_eos_penalty.
  PPOTrainerCfg.tools: PPOTrainerCfg#tools.
  PPOTrainerCfg.skip_apply_chat_template: PPOTrainerCfg#skip_apply_chat_template.
  PPOTrainerCfg.num_return_sequences: PPOTrainerCfg#num_return_sequences.
  PPOTrainerCfg.logprob_vocab_chunk_size: PPOTrainerCfg#logprob_vocab_chunk_size.
  PPOTrainerCfg.top_p: PPOTrainerCfg#top_p.
  PPOTrainerCfg.top_k: PPOTrainerCfg#top_k.
  PPOTrainerCfg.presence_penalty: PPOTrainerCfg#presence_penalty.
  PPOTrainerCfg.frequency_penalty: PPOTrainerCfg#frequency_penalty.
  PPOTrainerCfg.repetition_penalty: PPOTrainerCfg#repetition_penalty.
  PPOTrainerCfg.temperature: PPOTrainerCfg#temperature.
  SFTTrainerCfg.dataset_text_field: SFTTrainerCfg#dataset_text_field.
  SFTTrainerCfg.add_special_tokens: SFTTrainerCfg#add_special_tokens.
  SFTTrainerCfg.packing: SFTTrainerCfg#packing.
  SFTTrainerCfg.dataset_num_proc: SFTTrainerCfg#dataset_num_proc.
  SFTTrainerCfg.dataset_batch_size: SFTTrainerCfg#dataset_batch_size.
  SFTTrainerCfg.dataset_kwargs: SFTTrainerCfg#dataset_kwargs.
  SFTTrainerCfg.eval_packing: SFTTrainerCfg#eval_packing.
  SFTTrainerCfg.num_of_sequences: SFTTrainerCfg#num_of_sequences.
  RewardTrainerCfg.max_length: RewardTrainerCfg#max_length.
  RewardTrainerCfg.disable_dropout: RewardTrainerCfg#disable_dropout.
  RewardTrainerCfg.dataset_num_proc: RewardTrainerCfg#dataset_num_proc.
  RewardTrainerCfg.center_rewards_coefficient: RewardTrainerCfg#center_rewards_coefficient.
  DistillationTrainerCfg.temperature: DistillationTrainerCfg#temperature.
  DistillationTrainerCfg.alpha: DistillationTrainerCfg#alpha.
  DistillationTrainerCfg.logits_chunk_size: DistillationTrainerCfg#logits_chunk_size.
  OnPolicyDistillationTrainerCfg.max_prompt_length: OnPolicyDistillationTrainerCfg#max_prompt_length.
  OnPolicyDistillationTrainerCfg.max_completion_length: OnPolicyDistillationTrainerCfg#max_completion_length.
  OnPolicyDistillationTrainerCfg.num_generations_per_prompt: OnPolicyDistillationTrainerCfg#num_generations_per_prompt.
  OnPolicyDistillationTrainerCfg.temperature_sampling: OnPolicyDistillationTrainerCfg#temperature_sampling.
  OnPolicyDistillationTrainerCfg.top_k: OnPolicyDistillationTrainerCfg#top_k.
  OnPolicyDistillationTrainerCfg.top_p: OnPolicyDistillationTrainerCfg#top_p.
  OnPolicyDistillationTrainerCfg.presence_penalty: OnPolicyDistillationTrainerCfg#presence_penalty.
  OnPolicyDistillationTrainerCfg.frequency_penalty: OnPolicyDistillationTrainerCfg#frequency_penalty.
  OnPolicyDistillationTrainerCfg.repetition_penalty: OnPolicyDistillationTrainerCfg#repetition_penalty.
  OnPolicyDistillationTrainerCfg.generate_with_teacher: OnPolicyDistillationTrainerCfg#generate_with_teacher.
  SeqKDTrainerCfg.max_prompt_length: SeqKDTrainerCfg#max_prompt_length.
  SeqKDTrainerCfg.max_completion_length: SeqKDTrainerCfg#max_completion_length.
  SeqKDTrainerCfg.num_generations_per_prompt: SeqKDTrainerCfg#num_generations_per_prompt.
  SeqKDTrainerCfg.temperature_sampling: SeqKDTrainerCfg#temperature_sampling.
  SeqKDTrainerCfg.top_k: SeqKDTrainerCfg#top_k.
  SeqKDTrainerCfg.top_p: SeqKDTrainerCfg#top_p.
  SeqKDTrainerCfg.presence_penalty: SeqKDTrainerCfg#presence_penalty.
  SeqKDTrainerCfg.frequency_penalty: SeqKDTrainerCfg#frequency_penalty.
  SeqKDTrainerCfg.repetition_penalty: SeqKDTrainerCfg#repetition_penalty.
  SparseDistillationTrainerCfg.top_k_teacher: SparseDistillationTrainerCfg#top_k_teacher.
  SparseDistillationTrainerCfg.max_prompt_length: SparseDistillationTrainerCfg#max_prompt_length.
  SparseDistillationTrainerCfg.max_completion_length: SparseDistillationTrainerCfg#max_completion_length.
  SparseDistillationTrainerCfg.num_generations_per_prompt: SparseDistillationTrainerCfg#num_generations_per_prompt.
  SparseDistillationTrainerCfg.temperature_sampling: SparseDistillationTrainerCfg#temperature_sampling.
  SparseDistillationTrainerCfg.top_k: SparseDistillationTrainerCfg#top_k.
  SparseDistillationTrainerCfg.top_p: SparseDistillationTrainerCfg#top_p.
  SparseDistillationTrainerCfg.presence_penalty: SparseDistillationTrainerCfg#presence_penalty.
  SparseDistillationTrainerCfg.frequency_penalty: SparseDistillationTrainerCfg#frequency_penalty.
  SparseDistillationTrainerCfg.repetition_penalty: SparseDistillationTrainerCfg#repetition_penalty.
  KTOTrainerCfg.beta: KTOTrainerCfg#beta.
  KTOTrainerCfg.desirable_weight: KTOTrainerCfg#desirable_weight.
  KTOTrainerCfg.undesirable_weight: KTOTrainerCfg#undesirable_weight.
  KTOTrainerCfg.loss_type: KTOTrainerCfg#loss_type.
  KTOTrainerCfg.label_pad_token_id: KTOTrainerCfg#label_pad_token_id.
  KTOTrainerCfg.padding_value: KTOTrainerCfg#padding_value.
  KTOTrainerCfg.max_length: KTOTrainerCfg#max_length.
  KTOTrainerCfg.max_prompt_length: KTOTrainerCfg#max_prompt_length.
  KTOTrainerCfg.max_completion_length: KTOTrainerCfg#max_completion_length.
  KTOTrainerCfg.logprob_vocab_chunk_size: KTOTrainerCfg#logprob_vocab_chunk_size.
  KTOTrainerCfg.is_encoder_decoder: KTOTrainerCfg#is_encoder_decoder.
  KTOTrainerCfg.disable_dropout: KTOTrainerCfg#disable_dropout.
  KTOTrainerCfg.dataset_num_proc: KTOTrainerCfg#dataset_num_proc.
  KTOTrainerCfg.precompute_ref_log_probs: KTOTrainerCfg#precompute_ref_log_probs.
  BCOTrainerCfg.beta: BCOTrainerCfg#beta.
  BCOTrainerCfg.label_pad_token_id: BCOTrainerCfg#label_pad_token_id.
  BCOTrainerCfg.padding_value: BCOTrainerCfg#padding_value.
  BCOTrainerCfg.max_length: BCOTrainerCfg#max_length.
  BCOTrainerCfg.max_prompt_length: BCOTrainerCfg#max_prompt_length.
  BCOTrainerCfg.max_completion_length: BCOTrainerCfg#max_completion_length.
  BCOTrainerCfg.logprob_vocab_chunk_size: BCOTrainerCfg#logprob_vocab_chunk_size.
  BCOTrainerCfg.disable_dropout: BCOTrainerCfg#disable_dropout.
  BCOTrainerCfg.generate_during_eval: BCOTrainerCfg#generate_during_eval.
  BCOTrainerCfg.is_encoder_decoder: BCOTrainerCfg#is_encoder_decoder.
  BCOTrainerCfg.precompute_ref_log_probs: BCOTrainerCfg#precompute_ref_log_probs.
  BCOTrainerCfg.model_init_kwargs: BCOTrainerCfg#model_init_kwargs.
  BCOTrainerCfg.ref_model_init_kwargs: BCOTrainerCfg#ref_model_init_kwargs.
  BCOTrainerCfg.dataset_num_proc: BCOTrainerCfg#dataset_num_proc.
  BCOTrainerCfg.prompt_sample_size: BCOTrainerCfg#prompt_sample_size.
  BCOTrainerCfg.min_density_ratio: BCOTrainerCfg#min_density_ratio.
  BCOTrainerCfg.max_density_ratio: BCOTrainerCfg#max_density_ratio.
  CPOTrainerCfg.beta: CPOTrainerCfg#beta.
  CPOTrainerCfg.label_smoothing: CPOTrainerCfg#label_smoothing.
  CPOTrainerCfg.loss_type: CPOTrainerCfg#loss_type.
  CPOTrainerCfg.disable_dropout: CPOTrainerCfg#disable_dropout.
  CPOTrainerCfg.cpo_alpha: CPOTrainerCfg#cpo_alpha.
  CPOTrainerCfg.simpo_gamma: CPOTrainerCfg#simpo_gamma.
  CPOTrainerCfg.alpha: CPOTrainerCfg#alpha.
  CPOTrainerCfg.label_pad_token_id: CPOTrainerCfg#label_pad_token_id.
  CPOTrainerCfg.padding_value: CPOTrainerCfg#padding_value.
  CPOTrainerCfg.max_length: CPOTrainerCfg#max_length.
  CPOTrainerCfg.max_prompt_length: CPOTrainerCfg#max_prompt_length.
  CPOTrainerCfg.max_completion_length: CPOTrainerCfg#max_completion_length.
  CPOTrainerCfg.logprob_vocab_chunk_size: CPOTrainerCfg#logprob_vocab_chunk_size.
  CPOTrainerCfg.is_encoder_decoder: CPOTrainerCfg#is_encoder_decoder.
  CPOTrainerCfg.dataset_num_proc: CPOTrainerCfg#dataset_num_proc.
  GKDTrainerCfg.temperature: GKDTrainerCfg#temperature.
  GKDTrainerCfg.lmbda: GKDTrainerCfg#lmbda.
  GKDTrainerCfg.beta: GKDTrainerCfg#beta.
  GKDTrainerCfg.max_new_tokens: GKDTrainerCfg#max_new_tokens.
  GKDTrainerCfg.disable_dropout: GKDTrainerCfg#disable_dropout.
  GKDTrainerCfg.seq_kd: GKDTrainerCfg#seq_kd.
  NashMDTrainerCfg.beta: NashMDTrainerCfg#beta.
  NashMDTrainerCfg.mixture_coef: NashMDTrainerCfg#mixture_coef.
  NashMDTrainerCfg.missing_eos_penalty: NashMDTrainerCfg#missing_eos_penalty.
  XPOTrainerCfg.loss_type: XPOTrainerCfg#loss_type.
  XPOTrainerCfg.beta: XPOTrainerCfg#beta.
  XPOTrainerCfg.alpha: XPOTrainerCfg#alpha.
  XPOTrainerCfg.missing_eos_penalty: XPOTrainerCfg#missing_eos_penalty.
  EmbeddingTrainerCfg.loss_type: EmbeddingTrainerCfg#loss_type.
  EmbeddingTrainerCfg.temperature: EmbeddingTrainerCfg#temperature.
  EmbeddingTrainerCfg.margin: EmbeddingTrainerCfg#margin.
  EmbeddingTrainerCfg.query_field: EmbeddingTrainerCfg#query_field.
  EmbeddingTrainerCfg.positive_field: EmbeddingTrainerCfg#positive_field.
  EmbeddingTrainerCfg.negative_field: EmbeddingTrainerCfg#negative_field.
  EmbeddingTrainerCfg.matryoshka_dims: EmbeddingTrainerCfg#matryoshka_dims.
  EmbeddingTrainerCfg.normalize_embeddings: EmbeddingTrainerCfg#normalize_embeddings.
  EmbeddingTrainerCfg.pooling_strategy: EmbeddingTrainerCfg#pooling_strategy.
  EmbeddingTrainerCfg.dataset_num_proc: EmbeddingTrainerCfg#dataset_num_proc.
---
# Module: [`easydel/infra/elarge/types/training.py`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py)

## Classes
### `AgenticMoshPitTrainerCfg`  ·  implements/extends GRPOTrainerCfg
- def: [`easydel/infra/elarge/types/training.py:763`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L763)
- doc: Configuration for Agentic MoshPit trainer (AgenticMoshPitConfig).
- signature: `class AgenticMoshPitTrainerCfg(GRPOTrainerCfg):`
- members:
  - `advantage_estimator` — [`L800`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L800)
  - `episode_reward_weight` — [`L802`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L802)
  - `group_size` — [`L797`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L797)
  - `max_steps` — [`L796`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L796)
  - `max_tool_calls_per_step` — [`L807`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L807)
  - `num_env_groups` — [`L798`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L798)
  - `reasoning_parser` — [`L808`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L808)
  - `reward_mode` — [`L799`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L799)
  - `step_reward_gamma` — [`L801`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L801)
  - `step_reward_weight` — [`L803`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L803)
  - `system_prompt` — [`L804`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L804)
  - `tool_caller` — [`L806`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L806)
  - `tool_names` — [`L805`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L805)
- uses (calls/refs, reference-scoped): [`TrainerConfig`](training.md#TrainerConfig), [`GRPOTrainerCfg`](training.md#GRPOTrainerCfg)
- used by: [`TrainerConfig`](training.md#TrainerConfig), [`GRPOTrainerCfg`](training.md#GRPOTrainerCfg)

### `BCOTrainerCfg`  ·  implements/extends BaseTrainerCfg
- def: [`easydel/infra/elarge/types/training.py:1209`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1209)
- doc: Configuration for Binary Classifier Optimization trainer (BCOConfig).
- signature: `class BCOTrainerCfg(BaseTrainerCfg):`
- members:
  - `beta` — [`L1248`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1248)
  - `dataset_num_proc` — [`L1261`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1261)
  - `disable_dropout` — [`L1255`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1255)
  - `generate_during_eval` — [`L1256`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1256)
  - `is_encoder_decoder` — [`L1257`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1257)
  - `label_pad_token_id` — [`L1249`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1249)
  - `logprob_vocab_chunk_size` — [`L1254`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1254)
  - `max_completion_length` — [`L1253`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1253)
  - `max_density_ratio` — [`L1264`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1264)
  - `max_length` — [`L1251`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1251)
  - `max_prompt_length` — [`L1252`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1252)
  - `min_density_ratio` — [`L1263`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1263)
  - `model_init_kwargs` — [`L1259`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1259)
  - `padding_value` — [`L1250`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1250)
  - `precompute_ref_log_probs` — [`L1258`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1258)
  - `prompt_sample_size` — [`L1262`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1262)
  - `ref_model_init_kwargs` — [`L1260`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1260)
- uses (calls/refs, reference-scoped): [`TrainerConfig`](training.md#TrainerConfig), [`BaseTrainerCfg`](training.md#BaseTrainerCfg)
- used by: [`TrainerConfig`](training.md#TrainerConfig), [`BaseTrainerCfg`](training.md#BaseTrainerCfg)

### `BaseTrainerCfg`  ·  implements/extends _TypedDict
- def: [`easydel/infra/elarge/types/training.py:201`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L201)
- doc: Base configuration shared by all trainers (TrainingArguments).
- signature: `class BaseTrainerCfg(TypedDict, total=False):`
- members:
  - `auto_shard_states` — [`L476`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L476)
  - `aux_loss_enabled` — [`L436`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L436)
  - `backend` — [`L475`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L475)
  - `benchmark_interval` — [`L524`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L524)
  - `benchmarks` — [`L525`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L525)
  - `clip_grad` — [`L412`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L412)
  - `custom_scheduler` — [`L414`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L414)
  - `dataloader_num_workers` — [`L416`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L416)
  - `dataloader_pin_memory` — [`L417`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L417)
  - `do_eval` — [`L431`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L431)
  - `do_last_save` — [`L432`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L432)
  - `do_train` — [`L430`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L430)
  - `esurge_data_parallelism_axis` — [`L538`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L538)
  - `esurge_enable_prefix_caching` — [`L537`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L537)
  - `esurge_hbm_utilization` — [`L530`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L530)
  - `esurge_max_num_batched_tokens` — [`L536`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L536)
  - `esurge_max_num_seq_buckets` — [`L539`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L539)
  - `esurge_max_num_seqs` — [`L531`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L531)
  - `esurge_min_input_pad` — [`L532`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L532)
  - `esurge_page_size` — [`L533`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L533)
  - `esurge_runner_verbose` — [`L535`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L535)
  - `esurge_silent_mode` — [`L534`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L534)
  - `esurge_use_tqdm` — [`L529`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L529)
  - `eval_batch_size` — [`L405`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L405)
  - `evaluation_steps` — [`L455`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L455)
  - `extra_optimizer_kwargs` — [`L413`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L413)
  - `force_step_start_point` — [`L439`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L439)
  - `frozen_parameters` — [`L488`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L488)
  - `generation_config_overrides` — [`L519`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L519)
  - `generation_dataset_prompt_field` — [`L517`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L517)
  - `generation_do_sample` — [`L509`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L509)
  - `generation_extra_kwargs` — [`L518`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L518)
  - `generation_frequency_penalty` — [`L506`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L506)
  - `generation_interval` — [`L513`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L513)
  - `generation_log_to_wandb` — [`L522`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L522)
  - `generation_max_new_tokens` — [`L511`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L511)
  - `generation_num_prompts` — [`L516`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L516)
  - `generation_num_return_sequences` — [`L510`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L510)
  - `generation_presence_penalty` — [`L505`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L505)
  - `generation_preview_print` — [`L521`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L521)
  - `generation_prompts` — [`L514`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L514)
  - `generation_repetition_penalty` — [`L507`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L507)
  - `generation_seed` — [`L520`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L520)
  - `generation_shard_inputs` — [`L512`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L512)
  - `generation_temperature` — [`L508`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L508)
  - `generation_top_k` — [`L504`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L504)
  - `generation_top_p` — [`L503`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L503)
  - `generation_use_train_prompts` — [`L515`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L515)
  - `gradient_accumulation_steps` — [`L406`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L406)
  - `grain_shard_count` — [`L425`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L425)
  - `grain_shard_index` — [`L424`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L424)
  - `ids_to_pop_from_dataset` — [`L419`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L419)
  - `init_tx` — [`L434`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L434)
  - `is_fine_tuning` — [`L433`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L433)
  - `jax_distributed_config` — [`L492`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L492)
  - `learning_rate` — [`L398`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L398)
  - `learning_rate_end` — [`L399`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L399)
  - `lmhead_chunksize` — [`L491`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L491)
  - `log_all_workers` — [`L460`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L460)
  - `log_grad_norms` — [`L461`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L461)
  - `log_steps` — [`L458`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L458)
  - `log_training_generations_to_wandb` — [`L523`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L523)
  - `loss_config` — [`L490`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L490)
  - `low_mem_usage` — [`L479`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L479)
  - `max_evaluation_steps` — [`L456`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L456)
  - `max_length` — [`L442`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L442)
  - `max_sequence_length` — [`L444`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L444)
  - `max_training_steps` — [`L401`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L401)
  - `merge_lora_before_save` — [`L450`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L450)
  - `merge_lora_before_tpu_preemption_save` — [`L451`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L451)
  - `metrics_to_show_in_rich_pbar` — [`L463`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L463)
  - `model_name` — [`L486`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L486)
  - `model_parameters` — [`L487`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L487)
  - `num_train_epochs` — [`L400`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L400)
  - `offload_dataset` — [`L426`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L426)
  - `offload_device_index` — [`L428`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L428)
  - `offload_device_type` — [`L427`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L427)
  - `optimizer` — [`L408`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L408)
  - `per_epoch_evaluation_steps` — [`L403`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L403)
  - `per_epoch_training_steps` — [`L402`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L402)
  - `performance_mode` — [`L477`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L477)
  - `process_zero_is_admin` — [`L469`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L469)
  - `progress_bar_type` — [`L464`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L464)
  - `pruning_module` — [`L498`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L498)
  - `quantization_bits` — [`L482`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L482)
  - `quantization_group_size` — [`L481`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L481)
  - `quantization_mode` — [`L480`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L480)
  - `remove_ckpt_after_load` — [`L453`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L453)
  - `remove_unused_columns` — [`L418`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L418)
  - `report_metrics` — [`L462`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L462)
  - `report_steps` — [`L459`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L459)
  - `resume_if_possible` — [`L440`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L440)
  - `save_directory` — [`L448`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L448)
  - `save_interval_minutes` — [`L445`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L445)
  - `save_optimizer_state` — [`L449`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L449)
  - `save_steps` — [`L446`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L446)
  - `save_total_limit` — [`L447`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L447)
  - `save_tpu_preemption_checkpoints` — [`L452`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L452)
  - `scheduler` — [`L409`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L409)
  - `shuffle_seed_train` — [`L421`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L421)
  - `shuffle_train_dataset` — [`L420`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L420)
  - `sparse_module_type` — [`L497`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L497)
  - `sparsify_module` — [`L496`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L496)
  - `state_apply_fn_kwarguments_to_model` — [`L494`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L494)
  - `step_partition_spec` — [`L493`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L493)
  - `step_start_point` — [`L438`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L438)
  - `straight_through_emulator` — [`L484`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L484)
  - `tensor_straight_through` — [`L483`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L483)
  - `total_batch_size` — [`L404`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L404)
  - `track_memory` — [`L478`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L478)
  - `train_on_inputs` — [`L435`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L435)
  - `trainer_prefix` — [`L473`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L473)
  - `trainer_type` — [`L369`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L369)
  - `training_time_limit` — [`L437`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L437)
  - `truncation_mode` — [`L441`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L441)
  - `tx_mu_dtype` — [`L500`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L500)
  - `use_data_collator` — [`L422`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L422)
  - `use_esurge_generation` — [`L528`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L528)
  - `use_grain` — [`L423`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L423)
  - `use_wandb` — [`L470`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L470)
  - `verbose` — [`L468`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L468)
  - `wandb_entity` — [`L471`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L471)
  - `wandb_name` — [`L472`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L472)
  - `warmup_steps` — [`L410`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L410)
  - `watchers` — [`L465`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L465)
  - `weight_decay` — [`L411`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L411)
  - `weight_distribution_log_steps` — [`L467`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L467)
  - `weight_distribution_pattern` — [`L466`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L466)
- uses (calls/refs, reference-scoped): [`TrainerConfig`](training.md#TrainerConfig), [`GRPOTrainerCfg`](training.md#GRPOTrainerCfg), [`BenchmarkConfig`](eval.md#BenchmarkConfig), [`DistillationTrainerCfg`](training.md#DistillationTrainerCfg), [`SFTTrainerCfg`](training.md#SFTTrainerCfg), [`BCOTrainerCfg`](training.md#BCOTrainerCfg), [`CPOTrainerCfg`](training.md#CPOTrainerCfg), [`DPOTrainerCfg`](training.md#DPOTrainerCfg), [`KTOTrainerCfg`](training.md#KTOTrainerCfg), [`ORPOTrainerCfg`](training.md#ORPOTrainerCfg), [`PPOTrainerCfg`](training.md#PPOTrainerCfg), [`RewardTrainerCfg`](training.md#RewardTrainerCfg), [`SeqKDTrainerCfg`](training.md#SeqKDTrainerCfg), [`EmbeddingTrainerCfg`](training.md#EmbeddingTrainerCfg), [`LossConfig`](training.md#LossConfig)
- used by: [`TrainerConfig`](training.md#TrainerConfig), [`GRPOTrainerCfg`](training.md#GRPOTrainerCfg), [`DistillationTrainerCfg`](training.md#DistillationTrainerCfg), [`SFTTrainerCfg`](training.md#SFTTrainerCfg), [`BCOTrainerCfg`](training.md#BCOTrainerCfg), [`CPOTrainerCfg`](training.md#CPOTrainerCfg), [`DPOTrainerCfg`](training.md#DPOTrainerCfg), [`KTOTrainerCfg`](training.md#KTOTrainerCfg), [`ORPOTrainerCfg`](training.md#ORPOTrainerCfg), [`PPOTrainerCfg`](training.md#PPOTrainerCfg), [`RewardTrainerCfg`](training.md#RewardTrainerCfg), [`SeqKDTrainerCfg`](training.md#SeqKDTrainerCfg), [`EmbeddingTrainerCfg`](training.md#EmbeddingTrainerCfg), [`BASE_TRAINER_DEFAULTS`](training.md#BASE_TRAINER_DEFAULTS.BASE_TRAINER_DEFAULTS)

### `CPOTrainerCfg`  ·  implements/extends BaseTrainerCfg
- def: [`easydel/infra/elarge/types/training.py:1267`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1267)
- doc: Configuration for Contrastive Preference Optimization trainer (CPOConfig).
- signature: `class CPOTrainerCfg(BaseTrainerCfg):`
- members:
  - `alpha` — [`L1314`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1314)
  - `beta` — [`L1308`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1308)
  - `cpo_alpha` — [`L1312`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1312)
  - `dataset_num_proc` — [`L1322`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1322)
  - `disable_dropout` — [`L1311`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1311)
  - `is_encoder_decoder` — [`L1321`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1321)
  - `label_pad_token_id` — [`L1315`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1315)
  - `label_smoothing` — [`L1309`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1309)
  - `logprob_vocab_chunk_size` — [`L1320`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1320)
  - `loss_type` — [`L1310`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1310)
  - `max_completion_length` — [`L1319`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1319)
  - `max_length` — [`L1317`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1317)
  - `max_prompt_length` — [`L1318`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1318)
  - `padding_value` — [`L1316`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1316)
  - `simpo_gamma` — [`L1313`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1313)
- uses (calls/refs, reference-scoped): [`TrainerConfig`](training.md#TrainerConfig), [`BaseTrainerCfg`](training.md#BaseTrainerCfg)
- used by: [`TrainerConfig`](training.md#TrainerConfig), [`BaseTrainerCfg`](training.md#BaseTrainerCfg)

### `DPOTrainerCfg`  ·  implements/extends BaseTrainerCfg
- def: [`easydel/infra/elarge/types/training.py:542`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L542)
- doc: Configuration for Direct Preference Optimization trainer (DPOConfig).
- signature: `class DPOTrainerCfg(BaseTrainerCfg):`
- members:
  - `beta` — [`L608`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L608)
  - `dataset_num_proc` — [`L635`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L635)
  - `disable_dropout` — [`L633`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L633)
  - `force_use_ref_model` — [`L637`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L637)
  - `is_encoder_decoder` — [`L632`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L632)
  - `label_pad_token_id` — [`L627`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L627)
  - `label_smoothing` — [`L609`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L609)
  - `logprob_vocab_chunk_size` — [`L642`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L642)
  - `loss_type` — [`L610`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L610)
  - `max_completion_length` — [`L631`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L631)
  - `max_length` — [`L629`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L629)
  - `max_prompt_length` — [`L630`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L630)
  - `padding_value` — [`L628`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L628)
  - `precompute_ref_log_probs` — [`L634`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L634)
  - `ref_model_mixup_alpha` — [`L639`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L639)
  - `ref_model_sync_steps` — [`L640`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L640)
  - `reference_free` — [`L636`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L636)
  - `rpo_alpha` — [`L641`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L641)
  - `sync_ref_model` — [`L638`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L638)
  - `tools` — [`L643`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L643)
  - `use_weighting` — [`L626`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L626)
- uses (calls/refs, reference-scoped): [`TrainerConfig`](training.md#TrainerConfig), [`BaseTrainerCfg`](training.md#BaseTrainerCfg)
- used by: [`TrainerConfig`](training.md#TrainerConfig), [`BaseTrainerCfg`](training.md#BaseTrainerCfg)

### `DistillationTrainerCfg`  ·  implements/extends BaseTrainerCfg
- def: [`easydel/infra/elarge/types/training.py:1017`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1017)
- doc: Configuration for Knowledge Distillation trainer (DistillationConfig).
- signature: `class DistillationTrainerCfg(BaseTrainerCfg):`
- members:
  - `alpha` — [`L1044`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1044)
  - `logits_chunk_size` — [`L1045`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1045)
  - `temperature` — [`L1043`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1043)
- uses (calls/refs, reference-scoped): [`TrainerConfig`](training.md#TrainerConfig), [`BaseTrainerCfg`](training.md#BaseTrainerCfg), [`OnPolicyDistillationTrainerCfg`](training.md#OnPolicyDistillationTrainerCfg), [`SparseDistillationTrainerCfg`](training.md#SparseDistillationTrainerCfg)
- used by: [`TrainerConfig`](training.md#TrainerConfig), [`BaseTrainerCfg`](training.md#BaseTrainerCfg), [`OnPolicyDistillationTrainerCfg`](training.md#OnPolicyDistillationTrainerCfg), [`SparseDistillationTrainerCfg`](training.md#SparseDistillationTrainerCfg)

### `EmbeddingTrainerCfg`  ·  implements/extends BaseTrainerCfg
- def: [`easydel/infra/elarge/types/training.py:1427`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1427)
- doc: Configuration for contrastive Embedding trainer (EmbeddingConfig).
- signature: `class EmbeddingTrainerCfg(BaseTrainerCfg):`
- members:
  - `dataset_num_proc` — [`L1464`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1464)
  - `loss_type` — [`L1455`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1455)
  - `margin` — [`L1457`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1457)
  - `matryoshka_dims` — [`L1461`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1461)
  - `negative_field` — [`L1460`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1460)
  - `normalize_embeddings` — [`L1462`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1462)
  - `pooling_strategy` — [`L1463`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1463)
  - `positive_field` — [`L1459`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1459)
  - `query_field` — [`L1458`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1458)
  - `temperature` — [`L1456`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1456)
- uses (calls/refs, reference-scoped): [`TrainerConfig`](training.md#TrainerConfig), [`BaseTrainerCfg`](training.md#BaseTrainerCfg)
- used by: [`TrainerConfig`](training.md#TrainerConfig), [`BaseTrainerCfg`](training.md#BaseTrainerCfg)

### `GKDTrainerCfg`  ·  implements/extends SFTTrainerCfg
- def: [`easydel/infra/elarge/types/training.py:1325`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1325)
- doc: Configuration for Generalized Knowledge Distillation trainer (GKDConfig).
- signature: `class GKDTrainerCfg(SFTTrainerCfg):`
- members:
  - `beta` — [`L1357`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1357)
  - `disable_dropout` — [`L1359`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1359)
  - `lmbda` — [`L1356`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1356)
  - `max_new_tokens` — [`L1358`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1358)
  - `seq_kd` — [`L1360`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1360)
  - `temperature` — [`L1355`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1355)
- uses (calls/refs, reference-scoped): [`TrainerConfig`](training.md#TrainerConfig), [`SFTTrainerCfg`](training.md#SFTTrainerCfg)
- used by: [`TrainerConfig`](training.md#TrainerConfig), [`SFTTrainerCfg`](training.md#SFTTrainerCfg)

### `GRPOTrainerCfg`  ·  implements/extends BaseTrainerCfg
- def: [`easydel/infra/elarge/types/training.py:694`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L694)
- doc: Configuration for Group Relative Policy Optimization trainer (GRPOConfig).
- signature: `class GRPOTrainerCfg(BaseTrainerCfg):`
- members:
  - `beta` — [`L741`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L741)
  - `completion_chunk_size` — [`L752`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L752)
  - `dataset_num_proc` — [`L744`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L744)
  - `frequency_penalty` — [`L758`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L758)
  - `logprob_vocab_chunk_size` — [`L754`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L754)
  - `max_completion_length` — [`L743`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L743)
  - `max_loss_completion_tokens` — [`L753`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L753)
  - `max_prompt_length` — [`L742`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L742)
  - `num_return_sequences` — [`L750`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L750)
  - `presence_penalty` — [`L757`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L757)
  - `ref_logps_chunk_size` — [`L751`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L751)
  - `ref_model_mixup_alpha` — [`L746`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L746)
  - `ref_model_sync_steps` — [`L747`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L747)
  - `repetition_penalty` — [`L759`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L759)
  - `skip_apply_chat_template` — [`L749`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L749)
  - `sync_ref_model` — [`L745`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L745)
  - `temperature` — [`L760`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L760)
  - `tools` — [`L748`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L748)
  - `top_k` — [`L756`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L756)
  - `top_p` — [`L755`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L755)
- uses (calls/refs, reference-scoped): [`TrainerConfig`](training.md#TrainerConfig), [`BaseTrainerCfg`](training.md#BaseTrainerCfg), [`AgenticMoshPitTrainerCfg`](training.md#AgenticMoshPitTrainerCfg), [`NashMDTrainerCfg`](training.md#NashMDTrainerCfg), [`RLVRTrainerCfg`](training.md#RLVRTrainerCfg), [`SDPOTrainerCfg`](training.md#SDPOTrainerCfg), [`XPOTrainerCfg`](training.md#XPOTrainerCfg)
- used by: [`TrainerConfig`](training.md#TrainerConfig), [`BaseTrainerCfg`](training.md#BaseTrainerCfg), [`AgenticMoshPitTrainerCfg`](training.md#AgenticMoshPitTrainerCfg), [`NashMDTrainerCfg`](training.md#NashMDTrainerCfg), [`RLVRTrainerCfg`](training.md#RLVRTrainerCfg), [`SDPOTrainerCfg`](training.md#SDPOTrainerCfg), [`XPOTrainerCfg`](training.md#XPOTrainerCfg)

### `KTOTrainerCfg`  ·  implements/extends BaseTrainerCfg
- def: [`easydel/infra/elarge/types/training.py:1154`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1154)
- doc: Configuration for Kahneman-Tversky Optimization trainer (KTOConfig).
- signature: `class KTOTrainerCfg(BaseTrainerCfg):`
- members:
  - `beta` — [`L1193`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1193)
  - `dataset_num_proc` — [`L1205`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1205)
  - `desirable_weight` — [`L1194`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1194)
  - `disable_dropout` — [`L1204`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1204)
  - `is_encoder_decoder` — [`L1203`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1203)
  - `label_pad_token_id` — [`L1197`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1197)
  - `logprob_vocab_chunk_size` — [`L1202`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1202)
  - `loss_type` — [`L1196`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1196)
  - `max_completion_length` — [`L1201`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1201)
  - `max_length` — [`L1199`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1199)
  - `max_prompt_length` — [`L1200`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1200)
  - `padding_value` — [`L1198`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1198)
  - `precompute_ref_log_probs` — [`L1206`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1206)
  - `undesirable_weight` — [`L1195`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1195)
- uses (calls/refs, reference-scoped): [`TrainerConfig`](training.md#TrainerConfig), [`BaseTrainerCfg`](training.md#BaseTrainerCfg)
- used by: [`TrainerConfig`](training.md#TrainerConfig), [`BaseTrainerCfg`](training.md#BaseTrainerCfg)

### `LossConfig`  ·  implements/extends _TypedDict
- def: [`easydel/infra/elarge/types/training.py:106`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L106)
- doc: Configuration for loss computation in training.
- signature: `class LossConfig(TypedDict, total=False):`
- members:
  - `break_on_nan` — [`L184`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L184)
  - `chunk_block_size` — [`L197`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L197)
  - `chunk_token_size` — [`L196`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L196)
  - `chunk_vocab_size` — [`L195`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L195)
  - `classification_problem_type` — [`L187`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L187)
  - `compute_dtype` — [`L198`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L198)
  - `divide_weight_sum` — [`L182`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L182)
  - `ignore_index` — [`L169`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L169)
  - `label_smoothing` — [`L170`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L170)
  - `loss_normalizing_factor` — [`L172`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L172)
  - `num_classification_labels` — [`L186`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L186)
  - `num_labels` — [`L180`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L180)
  - `problem_type` — [`L181`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L181)
  - `reduction` — [`L185`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L185)
  - `shift_tokens` — [`L183`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L183)
  - `z_loss` — [`L171`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L171)
- used by: [`loss_config`](training.md#BaseTrainerCfg.loss_config)

### `NashMDTrainerCfg`  ·  implements/extends GRPOTrainerCfg
- def: [`easydel/infra/elarge/types/training.py:1363`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1363)
- doc: Configuration for Nash Mixture-of-Decoders trainer (NashMDConfig).
- signature: `class NashMDTrainerCfg(GRPOTrainerCfg):`
- members:
  - `beta` — [`L1389`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1389)
  - `missing_eos_penalty` — [`L1391`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1391)
  - `mixture_coef` — [`L1390`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1390)
- uses (calls/refs, reference-scoped): [`TrainerConfig`](training.md#TrainerConfig), [`GRPOTrainerCfg`](training.md#GRPOTrainerCfg)
- used by: [`TrainerConfig`](training.md#TrainerConfig), [`GRPOTrainerCfg`](training.md#GRPOTrainerCfg)

### `ORPOTrainerCfg`  ·  implements/extends BaseTrainerCfg
- def: [`easydel/infra/elarge/types/training.py:646`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L646)
- doc: Configuration for Odds Ratio Preference Optimization trainer (ORPOConfig).
- signature: `class ORPOTrainerCfg(BaseTrainerCfg):`
- members:
  - `beta` — [`L681`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L681)
  - `dataset_num_proc` — [`L691`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L691)
  - `disable_dropout` — [`L686`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L686)
  - `generate_during_eval` — [`L689`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L689)
  - `is_encoder_decoder` — [`L690`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L690)
  - `label_pad_token_id` — [`L687`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L687)
  - `logprob_vocab_chunk_size` — [`L685`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L685)
  - `max_completion_length` — [`L684`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L684)
  - `max_length` — [`L682`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L682)
  - `max_prompt_length` — [`L683`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L683)
  - `padding_value` — [`L688`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L688)
- uses (calls/refs, reference-scoped): [`TrainerConfig`](training.md#TrainerConfig), [`BaseTrainerCfg`](training.md#BaseTrainerCfg)
- used by: [`TrainerConfig`](training.md#TrainerConfig), [`BaseTrainerCfg`](training.md#BaseTrainerCfg)

### `OnPolicyDistillationTrainerCfg`  ·  implements/extends DistillationTrainerCfg
- def: [`easydel/infra/elarge/types/training.py:1048`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1048)
- doc: Configuration for On-Policy Knowledge Distillation trainer.
- signature: `class OnPolicyDistillationTrainerCfg(DistillationTrainerCfg):`
- members:
  - `frequency_penalty` — [`L1087`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1087)
  - `generate_with_teacher` — [`L1089`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1089)
  - `max_completion_length` — [`L1081`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1081)
  - `max_prompt_length` — [`L1080`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1080)
  - `num_generations_per_prompt` — [`L1082`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1082)
  - `presence_penalty` — [`L1086`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1086)
  - `repetition_penalty` — [`L1088`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1088)
  - `temperature_sampling` — [`L1083`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1083)
  - `top_k` — [`L1084`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1084)
  - `top_p` — [`L1085`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1085)
- uses (calls/refs, reference-scoped): [`TrainerConfig`](training.md#TrainerConfig), [`DistillationTrainerCfg`](training.md#DistillationTrainerCfg)
- used by: [`TrainerConfig`](training.md#TrainerConfig), [`DistillationTrainerCfg`](training.md#DistillationTrainerCfg)

### `PPOTrainerCfg`  ·  implements/extends BaseTrainerCfg
- def: [`easydel/infra/elarge/types/training.py:867`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L867)
- doc: Configuration for Proximal Policy Optimization trainer (PPOConfig).
- signature: `class PPOTrainerCfg(BaseTrainerCfg):`
- members:
  - `cliprange` — [`L921`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L921)
  - `cliprange_value` — [`L923`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L923)
  - `dataset_num_proc` — [`L917`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L917)
  - `entropy_coef` — [`L928`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L928)
  - `frequency_penalty` — [`L937`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L937)
  - `gamma` — [`L924`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L924)
  - `kl_coef` — [`L919`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L919)
  - `kl_estimator` — [`L920`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L920)
  - `lam` — [`L925`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L925)
  - `logprob_vocab_chunk_size` — [`L933`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L933)
  - `max_completion_length` — [`L916`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L916)
  - `max_prompt_length` — [`L915`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L915)
  - `missing_eos_penalty` — [`L929`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L929)
  - `num_return_sequences` — [`L932`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L932)
  - `presence_penalty` — [`L936`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L936)
  - `repetition_penalty` — [`L938`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L938)
  - `reward_weights` — [`L918`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L918)
  - `skip_apply_chat_template` — [`L931`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L931)
  - `temperature` — [`L939`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L939)
  - `tools` — [`L930`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L930)
  - `top_k` — [`L935`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L935)
  - `top_p` — [`L934`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L934)
  - `vf_coef` — [`L922`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L922)
  - `whiten_advantages` — [`L927`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L927)
  - `whiten_rewards` — [`L926`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L926)
- uses (calls/refs, reference-scoped): [`TrainerConfig`](training.md#TrainerConfig), [`BaseTrainerCfg`](training.md#BaseTrainerCfg)
- used by: [`TrainerConfig`](training.md#TrainerConfig), [`BaseTrainerCfg`](training.md#BaseTrainerCfg)

### `RLVRTrainerCfg`  ·  implements/extends GRPOTrainerCfg
- def: [`easydel/infra/elarge/types/training.py:811`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L811)
- doc: Configuration for Reinforcement Learning with Verifiable Rewards (RLVRConfig).
- signature: `class RLVRTrainerCfg(GRPOTrainerCfg):`
- members:
  - `answer_key` — [`L838`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L838)
  - `difficulty_key` — [`L846`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L846)
  - `difficulty_loss_weight` — [`L847`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L847)
  - `format_pattern` — [`L840`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L840)
  - `format_reward_weight` — [`L841`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L841)
  - `length_penalty_target` — [`L842`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L842)
  - `length_penalty_weight` — [`L843`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L843)
  - `max_len_mask` — [`L844`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L844)
  - `reward_clip_range` — [`L845`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L845)
  - `test_key` — [`L839`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L839)
- uses (calls/refs, reference-scoped): [`TrainerConfig`](training.md#TrainerConfig), [`GRPOTrainerCfg`](training.md#GRPOTrainerCfg)
- used by: [`TrainerConfig`](training.md#TrainerConfig), [`GRPOTrainerCfg`](training.md#GRPOTrainerCfg)

### `RewardTrainerCfg`  ·  implements/extends BaseTrainerCfg
- def: [`easydel/infra/elarge/types/training.py:984`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L984)
- doc: Configuration for Reward Model trainer (RewardConfig).
- signature: `class RewardTrainerCfg(BaseTrainerCfg):`
- members:
  - `center_rewards_coefficient` — [`L1014`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1014)
  - `dataset_num_proc` — [`L1013`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1013)
  - `disable_dropout` — [`L1012`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1012)
  - `max_length` — [`L1011`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1011)
- uses (calls/refs, reference-scoped): [`TrainerConfig`](training.md#TrainerConfig), [`BaseTrainerCfg`](training.md#BaseTrainerCfg)
- used by: [`TrainerConfig`](training.md#TrainerConfig), [`BaseTrainerCfg`](training.md#BaseTrainerCfg)

### `SDPOTrainerCfg`  ·  implements/extends GRPOTrainerCfg
- def: [`easydel/infra/elarge/types/training.py:850`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L850)
- doc: Configuration for Self-Distillation Policy Optimization trainer (SDPOConfig).
- signature: `class SDPOTrainerCfg(GRPOTrainerCfg):`
- members:
  - `beta` — [`L864`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L864)
  - `distillation_type` — [`L863`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L863)
  - `max_feedback_length` — [`L862`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L862)
- uses (calls/refs, reference-scoped): [`TrainerConfig`](training.md#TrainerConfig), [`GRPOTrainerCfg`](training.md#GRPOTrainerCfg)
- used by: [`TrainerConfig`](training.md#TrainerConfig), [`GRPOTrainerCfg`](training.md#GRPOTrainerCfg)

### `SFTTrainerCfg`  ·  implements/extends BaseTrainerCfg
- def: [`easydel/infra/elarge/types/training.py:942`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L942)
- doc: Configuration for Supervised Fine-Tuning trainer (SFTConfig).
- signature: `class SFTTrainerCfg(BaseTrainerCfg):`
- members:
  - `add_special_tokens` — [`L975`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L975)
  - `dataset_batch_size` — [`L978`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L978)
  - `dataset_kwargs` — [`L979`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L979)
  - `dataset_num_proc` — [`L977`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L977)
  - `dataset_text_field` — [`L974`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L974)
  - `eval_packing` — [`L980`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L980)
  - `num_of_sequences` — [`L981`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L981)
  - `packing` — [`L976`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L976)
- uses (calls/refs, reference-scoped): [`TrainerConfig`](training.md#TrainerConfig), [`BaseTrainerCfg`](training.md#BaseTrainerCfg), [`GKDTrainerCfg`](training.md#GKDTrainerCfg)
- used by: [`TrainerConfig`](training.md#TrainerConfig), [`BaseTrainerCfg`](training.md#BaseTrainerCfg), [`GKDTrainerCfg`](training.md#GKDTrainerCfg)

### `SeqKDTrainerCfg`  ·  implements/extends BaseTrainerCfg
- def: [`easydel/infra/elarge/types/training.py:1092`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1092)
- doc: Configuration for Sequence-level Knowledge Distillation (SeqKD) trainer.
- signature: `class SeqKDTrainerCfg(BaseTrainerCfg):`
- members:
  - `frequency_penalty` — [`L1118`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1118)
  - `max_completion_length` — [`L1112`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1112)
  - `max_prompt_length` — [`L1111`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1111)
  - `num_generations_per_prompt` — [`L1113`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1113)
  - `presence_penalty` — [`L1117`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1117)
  - `repetition_penalty` — [`L1119`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1119)
  - `temperature_sampling` — [`L1114`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1114)
  - `top_k` — [`L1115`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1115)
  - `top_p` — [`L1116`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1116)
- uses (calls/refs, reference-scoped): [`TrainerConfig`](training.md#TrainerConfig), [`BaseTrainerCfg`](training.md#BaseTrainerCfg)
- used by: [`TrainerConfig`](training.md#TrainerConfig), [`BaseTrainerCfg`](training.md#BaseTrainerCfg)

### `SparseDistillationTrainerCfg`  ·  implements/extends DistillationTrainerCfg
- def: [`easydel/infra/elarge/types/training.py:1122`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1122)
- doc: Configuration for Sparse (Gray-Box) Distillation trainer.
- signature: `class SparseDistillationTrainerCfg(DistillationTrainerCfg):`
- members:
  - `frequency_penalty` — [`L1150`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1150)
  - `max_completion_length` — [`L1144`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1144)
  - `max_prompt_length` — [`L1143`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1143)
  - `num_generations_per_prompt` — [`L1145`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1145)
  - `presence_penalty` — [`L1149`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1149)
  - `repetition_penalty` — [`L1151`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1151)
  - `temperature_sampling` — [`L1146`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1146)
  - `top_k` — [`L1147`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1147)
  - `top_k_teacher` — [`L1142`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1142)
  - `top_p` — [`L1148`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1148)
- uses (calls/refs, reference-scoped): [`TrainerConfig`](training.md#TrainerConfig), [`DistillationTrainerCfg`](training.md#DistillationTrainerCfg)
- used by: [`TrainerConfig`](training.md#TrainerConfig), [`DistillationTrainerCfg`](training.md#DistillationTrainerCfg)

### `TrainerConfig`  ·  implements/extends AgenticMoshPitTrainerCfg, BCOTrainerCfg, BaseTrainerCfg, CPOTrainerCfg, DPOTrainerCfg, DistillationTrainerCfg, EmbeddingTrainerCfg, GKDTrainerCfg, GRPOTrainerCfg, KTOTrainerCfg, NashMDTrainerCfg, ORPOTrainerCfg, OnPolicyDistillationTrainerCfg, PPOTrainerCfg, RLVRTrainerCfg, RewardTrainerCfg, SDPOTrainerCfg, SFTTrainerCfg, SeqKDTrainerCfg, SparseDistillationTrainerCfg, XPOTrainerCfg
- def: [`easydel/infra/elarge/types/training.py:1467`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1467)
- doc: Unified trainer configuration combining all trainer-specific configs.
- signature: `class TrainerConfig(ORPOTrainerCfg, GRPOTrainerCfg, AgenticMoshPitTrainerCfg, RLVRTrainerCfg, SDPOTrainerCfg, PPOTrainerCfg, SFTTrainerCfg, RewardTrainerCfg, DistillationTrainerCfg, OnPolicyDistillationTrainerCfg, SeqKDTrainerCfg, SparseDistillationTrainerCfg, KTOTrainerCfg, BCOTrainerCfg, CPOTrainerCfg, GKDTrainerCfg, NashMDTrainerCfg, XPOTrainerCfg, EmbeddingTrainerCfg, BaseTrainerCfg, DPOTrainerCfg):`
- uses (calls/refs, reference-scoped): [`BaseTrainerCfg`](training.md#BaseTrainerCfg), [`GRPOTrainerCfg`](training.md#GRPOTrainerCfg), [`DistillationTrainerCfg`](training.md#DistillationTrainerCfg), [`SFTTrainerCfg`](training.md#SFTTrainerCfg), [`AgenticMoshPitTrainerCfg`](training.md#AgenticMoshPitTrainerCfg), [`BCOTrainerCfg`](training.md#BCOTrainerCfg), [`CPOTrainerCfg`](training.md#CPOTrainerCfg), [`DPOTrainerCfg`](training.md#DPOTrainerCfg), [`GKDTrainerCfg`](training.md#GKDTrainerCfg), [`KTOTrainerCfg`](training.md#KTOTrainerCfg), [`NashMDTrainerCfg`](training.md#NashMDTrainerCfg), [`ORPOTrainerCfg`](training.md#ORPOTrainerCfg), [`OnPolicyDistillationTrainerCfg`](training.md#OnPolicyDistillationTrainerCfg), [`PPOTrainerCfg`](training.md#PPOTrainerCfg), [`RLVRTrainerCfg`](training.md#RLVRTrainerCfg), [`RewardTrainerCfg`](training.md#RewardTrainerCfg), [`SDPOTrainerCfg`](training.md#SDPOTrainerCfg), [`SeqKDTrainerCfg`](training.md#SeqKDTrainerCfg), [`SparseDistillationTrainerCfg`](training.md#SparseDistillationTrainerCfg), [`XPOTrainerCfg`](training.md#XPOTrainerCfg), [`EmbeddingTrainerCfg`](training.md#EmbeddingTrainerCfg)
- used by: [`BaseTrainerCfg`](training.md#BaseTrainerCfg), [`GRPOTrainerCfg`](training.md#GRPOTrainerCfg), [`normalize_trainer_config`](training.md#normalize_trainer_config), [`DistillationTrainerCfg`](training.md#DistillationTrainerCfg), [`get_trainer_defaults`](training.md#get_trainer_defaults), [`SFTTrainerCfg`](training.md#SFTTrainerCfg), [`register_trainer_defaults`](training.md#register_trainer_defaults), [`AgenticMoshPitTrainerCfg`](training.md#AgenticMoshPitTrainerCfg), [`BCOTrainerCfg`](training.md#BCOTrainerCfg), [`CPOTrainerCfg`](training.md#CPOTrainerCfg), [`DPOTrainerCfg`](training.md#DPOTrainerCfg), [`GKDTrainerCfg`](training.md#GKDTrainerCfg), [`KTOTrainerCfg`](training.md#KTOTrainerCfg), [`NashMDTrainerCfg`](training.md#NashMDTrainerCfg), [`ORPOTrainerCfg`](training.md#ORPOTrainerCfg), [`OnPolicyDistillationTrainerCfg`](training.md#OnPolicyDistillationTrainerCfg), [`PPOTrainerCfg`](training.md#PPOTrainerCfg), [`RLVRTrainerCfg`](training.md#RLVRTrainerCfg), [`RewardTrainerCfg`](training.md#RewardTrainerCfg), [`SDPOTrainerCfg`](training.md#SDPOTrainerCfg), [`SeqKDTrainerCfg`](training.md#SeqKDTrainerCfg), [`SparseDistillationTrainerCfg`](training.md#SparseDistillationTrainerCfg), [`XPOTrainerCfg`](training.md#XPOTrainerCfg), [`TRAINER_SPECIFIC_DEFAULTS`](training.md#TRAINER_SPECIFIC_DEFAULTS.TRAINER_SPECIFIC_DEFAULTS), [`EmbeddingTrainerCfg`](training.md#EmbeddingTrainerCfg), [`trainer`](root.md#eLMConfig.trainer)

### `XPOTrainerCfg`  ·  implements/extends GRPOTrainerCfg
- def: [`easydel/infra/elarge/types/training.py:1394`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1394)
- doc: Configuration for Exploratory Preference Optimization trainer (XPOConfig).
- signature: `class XPOTrainerCfg(GRPOTrainerCfg):`
- members:
  - `alpha` — [`L1423`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1423)
  - `beta` — [`L1422`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1422)
  - `loss_type` — [`L1421`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1421)
  - `missing_eos_penalty` — [`L1424`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1424)
- uses (calls/refs, reference-scoped): [`TrainerConfig`](training.md#TrainerConfig), [`GRPOTrainerCfg`](training.md#GRPOTrainerCfg)
- used by: [`TrainerConfig`](training.md#TrainerConfig), [`GRPOTrainerCfg`](training.md#GRPOTrainerCfg)

## Functions
- `_normalize_trainer_type(trainer_type: str)` — [`L79`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L79) — Normalize trainer type string to canonical form.
- `get_trainer_class(trainer_type: str)` — [`L2176`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L2176) — Get the appropriate trainer class based on type.
- `get_trainer_defaults(trainer_type: str)` — [`L2053`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L2053) — Get merged defaults for a trainer type.
- `get_training_arguments_class(trainer_type: str)` — [`L2212`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L2212) — Get the appropriate TrainingArguments class based on trainer type.
- `normalize_trainer_config(config: dict[str, Any])` — [`L2084`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L2084) — Normalize and validate trainer configuration.
- `register_trainer_defaults(trainer_type: str, defaults: TrainerConfig)` — [`L2020`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L2020) — Register default configuration for a trainer type.

## Module values
- `BASE_TRAINER_DEFAULTS` — [`L1517`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1517)
- `TRAINER_SPECIFIC_DEFAULTS` — [`L1605`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L1605)
- `_TRAINERS_WITH_COMPLETION_LENGTH` — [`L2011`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L2011)
- `_TRAINER_TYPE_ALIASES` — [`L67`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/training.py#L67)


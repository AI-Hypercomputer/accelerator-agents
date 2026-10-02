---
title: 'Module: easydel/trainers/training_configurations.py'
type: catalog
provenance: extracted
module: easydel/trainers/training_configurations.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.training_configurations`/
symbols:
  TrainingArguments._ensure_variables: TrainingArguments#_ensure_variables().
  TrainingArguments: TrainingArguments#
  TrainingArguments.max_length: TrainingArguments#max_length.
  TrainingArguments.step_partition_spec: TrainingArguments#step_partition_spec.
  TrainingArguments.loss_config: TrainingArguments#loss_config.
  TrainingArguments.gradient_accumulation_steps: TrainingArguments#gradient_accumulation_steps.
  TrainingArguments.quantization_mode: TrainingArguments#quantization_mode.
  TrainingArguments.log_weight_distribution: TrainingArguments#log_weight_distribution().
  TrainingArguments.quantization_group_size: TrainingArguments#quantization_group_size.
  TrainingArguments.quantization_bits: TrainingArguments#quantization_bits.
  TrainingArguments.__post_init__: TrainingArguments#__post_init__().
  TrainingArguments.get_streaming_checkpointer: TrainingArguments#get_streaming_checkpointer().
  TrainingArguments.optimizer_kwargs: TrainingArguments#optimizer_kwargs.
  MetricsType: MetricsType.
  TrainingArguments._validate_config: TrainingArguments#_validate_config().
  TrainingArguments.get_wandb_init: TrainingArguments#get_wandb_init().
  TrainingArguments.log_metrics: TrainingArguments#log_metrics().
  TrainingArguments.ensure_checkpoint_path: TrainingArguments#ensure_checkpoint_path().
  TrainingArguments.can_log_metrics: TrainingArguments#can_log_metrics().
  TrainingArguments.tensor_straight_through: TrainingArguments#tensor_straight_through.
  TrainingArguments.straight_through_emulator: TrainingArguments#straight_through_emulator.
  TrainingArguments.num_train_epochs: TrainingArguments#num_train_epochs.
  TrainingArguments.benchmarks: TrainingArguments#benchmarks.
  TrainingArguments._setup_logging: TrainingArguments#_setup_logging().
  TrainingArguments.build_wandb_run_name: TrainingArguments#build_wandb_run_name().
  TrainingArguments.get_optimizer_and_scheduler: TrainingArguments#get_optimizer_and_scheduler().
  TrainingArguments._handle_deprecated_max_sequence_length: TrainingArguments#_handle_deprecated_max_sequence_length().
  TrainingArguments.log_watchers: TrainingArguments#log_watchers().
  TrainingArguments._get_save_directory: TrainingArguments#_get_save_directory().
  TrainingArguments.optimizer: TrainingArguments#optimizer.
  TrainingArguments.model_name: TrainingArguments#model_name.
  TrainingArguments.truncation_mode: TrainingArguments#truncation_mode.
  TrainingArguments.use_wandb: TrainingArguments#use_wandb.
  TrainingArguments._coerce_float: TrainingArguments#_coerce_float().
  TrainingArguments._inherit_generation_attr: TrainingArguments#_inherit_generation_attr().
  TrainingArguments.scheduler: TrainingArguments#scheduler.
  TrainingArguments._setup_distributed: TrainingArguments#_setup_distributed().
  TrainingArguments.lmhead_chunksize: TrainingArguments#lmhead_chunksize.
  TrainingArguments.generation_temperature: TrainingArguments#generation_temperature.
  TrainingArguments._coerce_int: TrainingArguments#_coerce_int().
  TrainingArguments.get_path: TrainingArguments#get_path().
  TrainingArguments.ensure_training_time_limit: TrainingArguments#ensure_training_time_limit().
  TrainingArguments.clip_grad: TrainingArguments#clip_grad.
  TrainingArguments.learning_rate_end: TrainingArguments#learning_rate_end.
  TrainingArguments.generation_extra_kwargs: TrainingArguments#generation_extra_kwargs.
  TrainingArguments.total_batch_size: TrainingArguments#total_batch_size.
  TrainingArguments.training_time_seconds: TrainingArguments#training_time_seconds().
  AVAILABLE_BACKENDS.AVAILABLE_BACKENDS: AVAILABLE_BACKENDS.AVAILABLE_BACKENDS.
  _normalize_partition_spec_entry: _normalize_partition_spec_entry().
  logger: logger.
  TrainingArguments.learning_rate: TrainingArguments#learning_rate.
  TrainingArguments.generation_num_return_sequences: TrainingArguments#generation_num_return_sequences.
  TrainingArguments.esurge_data_parallelism_axis: TrainingArguments#esurge_data_parallelism_axis.
  TrainingArguments.warmup_steps: TrainingArguments#warmup_steps.
  TrainingArguments.offload_device: TrainingArguments#offload_device().
  TrainingArguments.get_tx_template: TrainingArguments#get_tx_template().
  TrainingArguments._log_to_wandb: TrainingArguments#_log_to_wandb().
  _get_max_sequence_length: _get_max_sequence_length().
  _set_max_sequence_length: _set_max_sequence_length().
  _get_quantization_block: _get_quantization_block().
  _set_quantization_block: _set_quantization_block().
  QuantizationMode: QuantizationMode.
  TrainingArguments.eval_batch_size: TrainingArguments#eval_batch_size.
  TrainingArguments.generation_do_sample: TrainingArguments#generation_do_sample.
  TrainingArguments.generation_shard_inputs: TrainingArguments#generation_shard_inputs.
  TrainingArguments.esurge_max_num_seq_buckets: TrainingArguments#esurge_max_num_seq_buckets.
  TrainingArguments.load_arguments: TrainingArguments#load_arguments().
  STE_QAT_QUANTIZATION_MODES_DOC: STE_QAT_QUANTIZATION_MODES_DOC.
  TrainingArguments.watchers: TrainingArguments#watchers.
  TrainingArguments.get_tensorboard: TrainingArguments#get_tensorboard().
  STE_QAT_QUANTIZATION_MODES.STE_QAT_QUANTIZATION_MODES: STE_QAT_QUANTIZATION_MODES.STE_QAT_QUANTIZATION_MODES.
  FIXED_QUANTIZATION_BITS_BY_MODE.FIXED_QUANTIZATION_BITS_BY_MODE: FIXED_QUANTIZATION_BITS_BY_MODE.FIXED_QUANTIZATION_BITS_BY_MODE.
  TrainingArguments.log_all_workers: TrainingArguments#log_all_workers.
  TrainingArguments.generation_top_p: TrainingArguments#generation_top_p.
  TrainingArguments.generation_top_k: TrainingArguments#generation_top_k.
  TrainingArguments.generation_max_new_tokens: TrainingArguments#generation_max_new_tokens.
  TrainingArguments.generation_interval: TrainingArguments#generation_interval.
  TrainingArguments.generation_num_prompts: TrainingArguments#generation_num_prompts.
  TrainingArguments.benchmark_interval: TrainingArguments#benchmark_interval.
  TrainingArguments.use_esurge_generation: TrainingArguments#use_esurge_generation.
  TrainingArguments.shuffle_train_dataset: TrainingArguments#shuffle_train_dataset.
  TrainingArguments.training_time_limit: TrainingArguments#training_time_limit.
  TrainingArguments._can_log_metrics: TrainingArguments#_can_log_metrics.
  TrainingArguments._im_a_hidden_checkpoint_manager: TrainingArguments#_im_a_hidden_checkpoint_manager.
  _parse_partition_spec: _parse_partition_spec().
  TrainingArguments._handle_deprecated_quantization_block: TrainingArguments#_handle_deprecated_quantization_block().
  TrainingArguments._setup_optimizer: TrainingArguments#_setup_optimizer().
  TrainingArguments.get_checkpoint_policies: TrainingArguments#get_checkpoint_policies().
  TrainingArguments.get_save_interval_timedelta: TrainingArguments#get_save_interval_timedelta().
  TrainingArguments._tensorboard: TrainingArguments#_tensorboard().
  TrainingArguments._log_to_tensorboard: TrainingArguments#_log_to_tensorboard().
  TrainingArguments.to_json_string: TrainingArguments#to_json_string().
  TrainingArguments.load_from_json: TrainingArguments#load_from_json().
  TrainingArguments.save_arguments: TrainingArguments#save_arguments().
  TrainingArguments._get_save_directory_milestone: TrainingArguments#_get_save_directory_milestone().
  TrainingArguments.aux_loss_enabled: TrainingArguments#aux_loss_enabled.
  TrainingArguments.do_eval: TrainingArguments#do_eval.
  TrainingArguments.esurge_max_num_seqs: TrainingArguments#esurge_max_num_seqs.
  TrainingArguments.performance_mode: TrainingArguments#performance_mode.
  TrainingArguments.pruning_module: TrainingArguments#pruning_module.
  TrainingArguments.save_directory: TrainingArguments#save_directory.
  TrainingArguments.merge_lora_before_save: TrainingArguments#merge_lora_before_save.
  TrainingArguments.save_total_limit: TrainingArguments#save_total_limit.
  TrainingArguments.weight_decay: TrainingArguments#weight_decay.
  TrainingArguments.is_process_zero: TrainingArguments#is_process_zero().
  TrainingArguments.sparse_module_type: TrainingArguments#sparse_module_type.
  TrainingArguments.from_dict: TrainingArguments#from_dict().
  TrainingArguments.__hash__: TrainingArguments#__hash__.
  AFFINE_SUPPORTED_BITS: AFFINE_SUPPORTED_BITS.
  _apply_training_args_legacy_aliases: _apply_training_args_legacy_aliases().
  TrainingArguments.backend: TrainingArguments#backend.
  TrainingArguments.evaluation_steps: TrainingArguments#evaluation_steps.
  TrainingArguments.extra_optimizer_kwargs: TrainingArguments#extra_optimizer_kwargs.
  TrainingArguments.report_metrics: TrainingArguments#report_metrics.
  TrainingArguments.log_steps: TrainingArguments#log_steps.
  TrainingArguments.max_training_steps: TrainingArguments#max_training_steps.
  TrainingArguments.generation_presence_penalty: TrainingArguments#generation_presence_penalty.
  TrainingArguments.generation_frequency_penalty: TrainingArguments#generation_frequency_penalty.
  TrainingArguments.generation_repetition_penalty: TrainingArguments#generation_repetition_penalty.
  TrainingArguments.log_training_generations_to_wandb: TrainingArguments#log_training_generations_to_wandb.
  TrainingArguments.esurge_hbm_utilization: TrainingArguments#esurge_hbm_utilization.
  TrainingArguments.esurge_min_input_pad: TrainingArguments#esurge_min_input_pad.
  TrainingArguments.esurge_page_size: TrainingArguments#esurge_page_size.
  TrainingArguments.esurge_max_num_batched_tokens: TrainingArguments#esurge_max_num_batched_tokens.
  TrainingArguments.esurge_enable_prefix_caching: TrainingArguments#esurge_enable_prefix_caching.
  TrainingArguments.save_interval_minutes: TrainingArguments#save_interval_minutes.
  TrainingArguments.save_steps: TrainingArguments#save_steps.
  TrainingArguments.shuffle_seed_train: TrainingArguments#shuffle_seed_train.
  TrainingArguments.trainer_prefix: TrainingArguments#trainer_prefix.
  TrainingArguments.track_memory: TrainingArguments#track_memory.
  TrainingArguments.use_grain: TrainingArguments#use_grain.
  TrainingArguments.weight_distribution_log_steps: TrainingArguments#weight_distribution_log_steps.
  TrainingArguments._time_to_seconds: TrainingArguments#_time_to_seconds().
  TrainingArguments._wandb_float_token: TrainingArguments#_wandb_float_token().
  TrainingArguments.to_dict: TrainingArguments#to_dict().
  get_safe_arr: get_safe_arr().
  TrainingArguments.custom_scheduler: TrainingArguments#custom_scheduler.
  TrainingArguments.dataloader_num_workers: TrainingArguments#dataloader_num_workers.
  TrainingArguments.do_last_save: TrainingArguments#do_last_save.
  TrainingArguments.grain_shard_index: TrainingArguments#grain_shard_index.
  TrainingArguments.grain_shard_count: TrainingArguments#grain_shard_count.
  TrainingArguments.ids_to_pop_from_dataset: TrainingArguments#ids_to_pop_from_dataset.
  TrainingArguments.init_tx: TrainingArguments#init_tx.
  TrainingArguments.jax_distributed_config: TrainingArguments#jax_distributed_config.
  TrainingArguments.max_evaluation_steps: TrainingArguments#max_evaluation_steps.
  TrainingArguments.max_sequence_length: TrainingArguments#max_sequence_length.
  TrainingArguments.quantization_block: TrainingArguments#quantization_block.
  TrainingArguments.per_epoch_training_steps: TrainingArguments#per_epoch_training_steps.
  TrainingArguments.per_epoch_evaluation_steps: TrainingArguments#per_epoch_evaluation_steps.
  TrainingArguments.metrics_to_show_in_rich_pbar: TrainingArguments#metrics_to_show_in_rich_pbar.
  TrainingArguments.generation_prompts: TrainingArguments#generation_prompts.
  TrainingArguments.generation_use_train_prompts: TrainingArguments#generation_use_train_prompts.
  TrainingArguments.generation_config_overrides: TrainingArguments#generation_config_overrides.
  TrainingArguments.generation_seed: TrainingArguments#generation_seed.
  TrainingArguments.generation_preview_print: TrainingArguments#generation_preview_print.
  TrainingArguments.generation_log_to_wandb: TrainingArguments#generation_log_to_wandb.
  TrainingArguments.esurge_use_tqdm: TrainingArguments#esurge_use_tqdm.
  TrainingArguments.esurge_silent_mode: TrainingArguments#esurge_silent_mode.
  TrainingArguments.esurge_runner_verbose: TrainingArguments#esurge_runner_verbose.
  TrainingArguments.offload_dataset: TrainingArguments#offload_dataset.
  TrainingArguments.offload_device_type: TrainingArguments#offload_device_type.
  TrainingArguments.offload_device_index: TrainingArguments#offload_device_index.
  TrainingArguments.process_zero_is_admin: TrainingArguments#process_zero_is_admin.
  TrainingArguments.progress_bar_type: TrainingArguments#progress_bar_type.
  TrainingArguments.report_steps: TrainingArguments#report_steps.
  TrainingArguments.save_optimizer_state: TrainingArguments#save_optimizer_state.
  TrainingArguments.save_tpu_preemption_checkpoints: TrainingArguments#save_tpu_preemption_checkpoints.
  TrainingArguments.step_start_point: TrainingArguments#step_start_point.
  TrainingArguments.force_step_start_point: TrainingArguments#force_step_start_point.
  TrainingArguments.resume_if_possible: TrainingArguments#resume_if_possible.
  TrainingArguments.tx_mu_dtype: TrainingArguments#tx_mu_dtype.
  TrainingArguments.wandb_entity: TrainingArguments#wandb_entity.
  TrainingArguments.wandb_name: TrainingArguments#wandb_name.
  TrainingArguments.weight_distribution_pattern: TrainingArguments#weight_distribution_pattern.
  TrainingArguments._wandb_token: TrainingArguments#_wandb_token().
  TrainingArguments._restructure_metric_name: TrainingArguments#_restructure_metric_name().
  TrainingArguments._create_wandb_histogram: TrainingArguments#_create_wandb_histogram().
  TrainingArguments._dict_from_json_file: TrainingArguments#_dict_from_json_file().
  TrainingArguments.auto_shard_states: TrainingArguments#auto_shard_states.
  TrainingArguments.dataloader_pin_memory: TrainingArguments#dataloader_pin_memory.
  TrainingArguments.do_train: TrainingArguments#do_train.
  TrainingArguments.frozen_parameters: TrainingArguments#frozen_parameters.
  TrainingArguments.is_fine_tuning: TrainingArguments#is_fine_tuning.
  TrainingArguments.log_grad_norms: TrainingArguments#log_grad_norms.
  TrainingArguments.low_mem_usage: TrainingArguments#low_mem_usage.
  TrainingArguments.model_parameters: TrainingArguments#model_parameters.
  TrainingArguments.generation_dataset_prompt_field: TrainingArguments#generation_dataset_prompt_field.
  TrainingArguments.remove_ckpt_after_load: TrainingArguments#remove_ckpt_after_load.
  TrainingArguments.remove_unused_columns: TrainingArguments#remove_unused_columns.
  TrainingArguments.merge_lora_before_tpu_preemption_save: TrainingArguments#merge_lora_before_tpu_preemption_save.
  TrainingArguments.sparsify_module: TrainingArguments#sparsify_module.
  TrainingArguments.state_apply_fn_kwarguments_to_model: TrainingArguments#state_apply_fn_kwarguments_to_model.
  TrainingArguments.train_on_inputs: TrainingArguments#train_on_inputs.
  TrainingArguments.use_data_collator: TrainingArguments#use_data_collator.
  TrainingArguments.verbose: TrainingArguments#verbose.
---
# Module: [`easydel/trainers/training_configurations.py`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py)

## Classes
### `TrainingArguments`
- def: [`easydel/trainers/training_configurations.py:194`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L194) — documented in [easydel-trainers-base_trainer](../../../concepts/easydel-trainers-base_trainer.md)
- doc: Comprehensive configuration class for training and evaluation.
- signature: `class TrainingArguments:`
- members:
  - `__post_init__(self, max_sequence_length: int | None, quantization_block: int | None)` — [`L929`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L929) — Post-initialization setup and validation. — documented in [easydel-trainers-training_configurations](../../../concepts/easydel-trainers-training_configurations.md)
  - `_coerce_float(value: tp.Any)` — [`L1071`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L1071) — Best-effort coercion of a value to a Python float.
  - `_coerce_int(value: tp.Any)` — [`L1101`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L1101) — Best-effort coercion of a value to a Python int.
  - `_create_wandb_histogram(self, value)` — [`L1869`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L1869) — Create a wandb.Histogram object from the given value.
  - `_dict_from_json_file(cls, json_file: str | os.PathLike)` — [`L1899`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L1899) — Load a JSON file and return its contents as a dictionary.
  - `_ensure_variables(self)` — [`L1058`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L1058) — Ensure all configuration variables are properly initialized. — documented in [easydel-trainers-training_configurations](../../../concepts/easydel-trainers-training_configurations.md)
  - `_get_save_directory(self, create: bool = True)` — [`L2048`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L2048) — Return the base checkpoint save directory.
  - `_get_save_directory_milestone(self, step, create: bool = True)` — [`L2061`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L2061) — Return the checkpoint directory for a specific training step.
  - `_handle_deprecated_max_sequence_length(self, max_sequence_length: int | None)` — [`L856`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L856) — Migrate the deprecated ``max_sequence_length`` value to ``max_length``.
  - `_handle_deprecated_quantization_block(self, quantization_block: int | None)` — [`L898`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L898) — Migrate the deprecated ``quantization_block`` value to ``quantization_group_size``.
  - `_inherit_generation_attr(attr, fallback_name)` — [`L1231`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L1231) — Copy a fallback attribute into a generation attribute if unset.
  - `_log_to_tensorboard(self, metrics: dict[str, tp.Any], step: int, log_as: tp.Literal["summary", "config"] | None = None)` — [`L1829`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L1829) — Log metrics to TensorBoard.
  - `_log_to_wandb(self, metrics: dict[str, tp.Any], step: int, log_as: tp.Literal["summary", "config"] | None = None)` — [`L1777`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L1777) — Log metrics to Weights & Biases (wandb).
  - `_restructure_metric_name(self, metric_name: str)` — [`L1691`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L1691) — Restructures the metric name for logging.
  - `_setup_distributed(self)` — [`L996`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L996) — Configure JAX for distributed training.
  - `_setup_logging(self)` — [`L1039`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L1039) — Configure logging infrastructure for training monitoring.
  - `_setup_optimizer(self)` — [`L1011`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L1011) — Configure optimizer and learning rate scheduler settings.
  - `_tensorboard(self)` — [`L1503`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L1503) — Lazy initialization of TensorBoard writer.
  - `_time_to_seconds(time_str: str)` — [`L1313`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L1313) — Convert a human-readable time string to seconds.
  - `_validate_config(self)` — [`L959`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L959) — Validate configuration settings for correctness and compatibility.
  - `_wandb_float_token(value: float | int | None, fallback: str = "none")` — [`L1603`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L1603) — Format a numeric value as a compact string token for W&B run names.
  - `_wandb_token(value: tp.Any, fallback: str = "na")` — [`L1590`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L1590) — Normalize arbitrary config values into safe W&B name tokens.
  - `build_wandb_run_name(self, model_name: str, size_in_billion: float | None = None)` — [`L1620`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L1620) — Build a structured default W&B run name.
  - `can_log_metrics(self)` — [`L799`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L799) — Whether this process should log metrics.
  - `ensure_checkpoint_path(self)` — [`L1365`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L1365) — Create the checkpoint directory if it doesn't exist.
  - `ensure_training_time_limit(self, time_passed)` — [`L1640`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L1640) — Check if training has exceeded the configured time limit.
  - `from_dict(cls, data: dict[str, tp.Any])` — [`L1938`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L1938) — Deserializes a dictionary into a TrainingArguments instance.
  - `get_checkpoint_policies(self)` — [`L1469`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L1469) — Convert save_steps configuration to CheckpointInterval policies.
  - `get_optimizer_and_scheduler(self, steps: int | None = None)` — [`L1396`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L1396) — Create and return the optimizer and learning rate scheduler.
  - `get_path(self)` — [`L1353`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L1353) — Get the path to the checkpoint directory.
  - `get_save_interval_timedelta(self)` — [`L1487`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L1487) — Get time-based checkpoint save interval as timedelta.
  - `get_streaming_checkpointer(self)` — [`L1447`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L1447) — Get the asynchronous checkpoint manager.
  - `get_tensorboard(self)` — [`L1525`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L1525) — Get the TensorBoard SummaryWriter for logging metrics.
  - `get_tx_template(self, possible_max: int | None = None)` — [`L1377`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L1377) — Get the optimizer transformation without a specific step count.
  - `get_wandb_init(self)` — [`L1541`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L1541) — Initialize Weights & Biases for experiment tracking.
  - `is_process_zero(self)` — [`L848`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L848) — Whether the current process is the rank-zero (main) process.
  - `load_arguments(cls, json_file: str | os.PathLike)` — [`L1984`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L1984) — Load training arguments from a JSON file.
  - `load_from_json(cls, config_dict)` — [`L2008`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L2008) — Reconstruct a TrainingArguments instance from a parsed JSON dictionary.
  - `log_metrics(self, metrics: MetricsType, step: int, log_as: tp.Literal["summary", "config"] | None = None)` — [`L1652`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L1652) — Log metrics to configured logging backends.
  - `log_watchers(self, state, step: int)` — [`L1756`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L1756) — Run all registered ``LogWatcher`` instances and log their metrics.
  - `log_weight_distribution(self, state, step: int)` — [`L1705`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L1705) — Log weight distribution histograms and statistics.
  - `offload_device(self)` — [`L825`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L825) — Return the JAX device used for parameter offloading.
  - `save_arguments(self, json_file_path: str | os.PathLike | ePathLike)` — [`L2030`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L2030) — Save training arguments to a JSON file.
  - `to_dict(self)` — [`L1910`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L1910) — Serializes this instance to a dictionary.
  - `to_json_string(self)` — [`L1972`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L1972) — Serializes this instance to a JSON string.
  - `training_time_seconds(self)` — [`L837`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L837) — Convert ``training_time_limit`` to total seconds.
  - `auto_shard_states` — [`L222`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L222)
  - `aux_loss_enabled` — [`L226`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L226)
  - `backend` — [`L230`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L230)
  - `benchmark_interval` — [`L511`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L511)
  - `benchmarks` — [`L515`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L515)
  - `clip_grad` — [`L234`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L234)
  - `custom_scheduler` — [`L238`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L238)
  - `dataloader_num_workers` — [`L242`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L242)
  - `dataloader_pin_memory` — [`L246`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L246)
  - `do_eval` — [`L250`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L250)
  - `do_last_save` — [`L254`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L254)
  - `do_train` — [`L258`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L258)
  - `esurge_data_parallelism_axis` — [`L565`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L565)
  - `esurge_enable_prefix_caching` — [`L561`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L561)
  - `esurge_hbm_utilization` — [`L527`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L527)
  - `esurge_max_num_batched_tokens` — [`L557`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L557)
  - `esurge_max_num_seq_buckets` — [`L537`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L537)
  - `esurge_max_num_seqs` — [`L531`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L531)
  - `esurge_min_input_pad` — [`L541`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L541)
  - `esurge_page_size` — [`L545`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L545)
  - `esurge_runner_verbose` — [`L553`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L553)
  - `esurge_silent_mode` — [`L549`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L549)
  - `esurge_use_tqdm` — [`L523`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L523)
  - `eval_batch_size` — [`L262`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L262)
  - `evaluation_steps` — [`L266`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L266)
  - `extra_optimizer_kwargs` — [`L270`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L270)
  - `force_step_start_point` — [`L698`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L698)
  - `frozen_parameters` — [`L274`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L274)
  - `generation_config_overrides` — [`L491`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L491)
  - `generation_dataset_prompt_field` — [`L483`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L483)
  - `generation_do_sample` — [`L451`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L451)
  - `generation_extra_kwargs` — [`L487`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L487)
  - `generation_frequency_penalty` — [`L439`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L439)
  - `generation_interval` — [`L467`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L467)
  - `generation_log_to_wandb` — [`L503`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L503)
  - `generation_max_new_tokens` — [`L459`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L459)
  - `generation_num_prompts` — [`L479`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L479)
  - `generation_num_return_sequences` — [`L455`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L455)
  - `generation_presence_penalty` — [`L435`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L435)
  - `generation_preview_print` — [`L499`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L499)
  - `generation_prompts` — [`L471`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L471)
  - `generation_repetition_penalty` — [`L443`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L443)
  - `generation_seed` — [`L495`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L495)
  - `generation_shard_inputs` — [`L463`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L463)
  - `generation_temperature` — [`L447`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L447)
  - `generation_top_k` — [`L431`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L431)
  - `generation_top_p` — [`L427`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L427)
  - `generation_use_train_prompts` — [`L475`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L475)
  - `gradient_accumulation_steps` — [`L286`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L286) — documented in [easydel-trainers-training_configurations](../../../concepts/easydel-trainers-training_configurations.md)
  - `grain_shard_count` — [`L282`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L282)
  - `grain_shard_index` — [`L278`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L278)
  - `ids_to_pop_from_dataset` — [`L290`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L290)
  - `init_tx` — [`L298`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L298)
  - `is_fine_tuning` — [`L294`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L294)
  - `jax_distributed_config` — [`L302`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L302)
  - `learning_rate` — [`L306`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L306)
  - `learning_rate_end` — [`L310`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L310)
  - `lmhead_chunksize` — [`L334`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L334)
  - `log_all_workers` — [`L314`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L314)
  - `log_grad_norms` — [`L318`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L318)
  - `log_steps` — [`L326`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L326)
  - `log_training_generations_to_wandb` — [`L507`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L507)
  - `loss_config` — [`L330`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L330) — documented in [easydel-trainers-training_configurations](../../../concepts/easydel-trainers-training_configurations.md)
  - `low_mem_usage` — [`L389`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L389)
  - `max_evaluation_steps` — [`L393`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L393)
  - `max_length` — [`L397`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L397) — documented in [easydel-trainers-base_trainer](../../../concepts/easydel-trainers-base_trainer.md)
  - `max_sequence_length` — [`L401`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L401)
  - `max_training_steps` — [`L403`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L403)
  - `merge_lora_before_save` — [`L629`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L629)
  - `merge_lora_before_tpu_preemption_save` — [`L639`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L639)
  - `metrics_to_show_in_rich_pbar` — [`L423`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L423)
  - `model_name` — [`L415`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L415)
  - `model_parameters` — [`L419`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L419)
  - `num_train_epochs` — [`L569`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L569)
  - `offload_dataset` — [`L573`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L573)
  - `offload_device_index` — [`L581`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L581)
  - `offload_device_type` — [`L577`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L577)
  - `optimizer` — [`L585`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L585)
  - `optimizer_kwargs` — [`L1025`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L1025)
  - `per_epoch_evaluation_steps` — [`L411`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L411)
  - `per_epoch_training_steps` — [`L407`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L407)
  - `performance_mode` — [`L589`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L589)
  - `process_zero_is_admin` — [`L597`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L597)
  - `progress_bar_type` — [`L601`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L601)
  - `pruning_module` — [`L593`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L593)
  - `quantization_bits` — [`L359`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L359) — documented in [easydel-trainers-training_configurations](../../../concepts/easydel-trainers-training_configurations.md)
  - `quantization_block` — [`L402`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L402)
  - `quantization_group_size` — [`L350`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L350) — documented in [easydel-trainers-training_configurations](../../../concepts/easydel-trainers-training_configurations.md)
  - `quantization_mode` — [`L338`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L338) — documented in [easydel-trainers-training_configurations](../../../concepts/easydel-trainers-training_configurations.md)
  - `remove_ckpt_after_load` — [`L605`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L605)
  - `remove_unused_columns` — [`L609`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L609)
  - `report_metrics` — [`L322`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L322)
  - `report_steps` — [`L613`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L613)
  - `resume_if_possible` — [`L709`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L709)
  - `save_directory` — [`L621`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L621)
  - `save_interval_minutes` — [`L617`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L617)
  - `save_optimizer_state` — [`L625`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L625)
  - `save_steps` — [`L648`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L648)
  - `save_total_limit` — [`L652`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L652)
  - `save_tpu_preemption_checkpoints` — [`L661`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L661)
  - `scheduler` — [`L670`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L670)
  - `shuffle_seed_train` — [`L674`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L674)
  - `shuffle_train_dataset` — [`L713`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L713)
  - `sparse_module_type` — [`L682`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L682)
  - `sparsify_module` — [`L678`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L678)
  - `state_apply_fn_kwarguments_to_model` — [`L686`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L686)
  - `step_partition_spec` — [`L690`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L690) — documented in [easydel-trainers-training_configurations](../../../concepts/easydel-trainers-training_configurations.md)
  - `step_start_point` — [`L694`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L694)
  - `straight_through_emulator` — [`L379`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L379)
  - `tensor_straight_through` — [`L369`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L369)
  - `total_batch_size` — [`L717`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L717)
  - `track_memory` — [`L741`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L741)
  - `train_on_inputs` — [`L725`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L725)
  - `trainer_prefix` — [`L729`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L729)
  - `training_time_limit` — [`L721`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L721)
  - `truncation_mode` — [`L733`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L733)
  - `tx_mu_dtype` — [`L737`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L737)
  - `use_data_collator` — [`L745`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L745)
  - `use_esurge_generation` — [`L519`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L519)
  - `use_grain` — [`L749`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L749)
  - `use_wandb` — [`L753`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L753)
  - `verbose` — [`L757`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L757)
  - `wandb_entity` — [`L761`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L761)
  - `wandb_name` — [`L765`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L765)
  - `warmup_steps` — [`L769`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L769)
  - `watchers` — [`L777`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L777)
  - `weight_decay` — [`L773`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L773)
  - `weight_distribution_log_steps` — [`L790`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L790)
  - `weight_distribution_pattern` — [`L786`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L786)
- protocol/private: `__hash__`[`L2084`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L2084), `_can_log_metrics`[`L795`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L795), `_im_a_hidden_checkpoint_manager`[`L796`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L796)
- uses (calls/refs, reference-scoped): [`LossConfig`](../infra/loss_utils.md#LossConfig), [`hash_fn`](../utils/compiling_utils.md#hash_fn), [`GRPOConfig`](group_relative_policy_optimization/grpo_config.md#GRPOConfig), [`DistillationConfig`](distillation_trainer/distillation_config.md#DistillationConfig), [`normalize_benchmark_configs`](../infra/elarge/benchmarking.md#normalize_benchmark_configs), [`MetricsType`](training_configurations.md#MetricsType), [`BenchmarkConfig`](../infra/elarge/types/eval.md#BenchmarkConfig), [`SFTConfig`](supervised_fine_tuning_trainer/sft_config.md#SFTConfig), [`BCOConfig`](binary_classifier_optimization_trainer/bco_config.md#BCOConfig), [`EmbeddingConfig`](embedding_trainer/embedding_config.md#EmbeddingConfig), [`KTOConfig`](kto_trainer/kto_config.md#KTOConfig), [`ORPOConfig`](odds_ratio_preference_optimization_trainer/orpo_config.md#ORPOConfig), [`PPOConfig`](proximal_policy_optimization_trainer/ppo_config.md#PPOConfig), [`SeqKDConfig`](seq_kd_trainer/seq_kd_config.md#SeqKDConfig), [`run_watchers`](metrics.md#run_watchers), [`RewardConfig`](reward_trainer/reward_config.md#RewardConfig), [`CPOConfig`](contrastive_preference_optimization_trainer/cpo_config.md#CPOConfig), [`DPOConfig`](direct_preference_optimization_trainer/dpo_config.md#DPOConfig), [`EasyDeLTimerError`](../infra/errors.md#EasyDeLTimerError), [`compute_weight_stats`](metrics.md#compute_weight_stats), [`initialize`](utils.md#JaxDistributedConfig.initialize), [`LogWatcher`](metrics.md#LogWatcher), [`AVAILABLE_BACKENDS`](training_configurations.md#AVAILABLE_BACKENDS.AVAILABLE_BACKENDS), [`mean`](metrics.md#MetricsHistogram.mean), [`logger`](training_configurations.md#logger), [`MetricsHistogram`](metrics.md#MetricsHistogram), [`QuantizationMode`](training_configurations.md#QuantizationMode), [`FIXED_QUANTIZATION_BITS_BY_MODE`](training_configurations.md#FIXED_QUANTIZATION_BITS_BY_MODE.FIXED_QUANTIZATION_BITS_BY_MODE), [`STE_QAT_QUANTIZATION_MODES`](training_configurations.md#STE_QAT_QUANTIZATION_MODES.STE_QAT_QUANTIZATION_MODES), [`STE_QAT_QUANTIZATION_MODES_DOC`](training_configurations.md#STE_QAT_QUANTIZATION_MODES_DOC), [`EasyDeLSchedulers`](../infra/etils.md#EasyDeLSchedulers), [`AVAILABLE_SPARSE_MODULE_TYPES`](../infra/etils.md#AVAILABLE_SPARSE_MODULE_TYPES), [`_parse_partition_spec`](training_configurations.md#_parse_partition_spec), [`std`](metrics.md#MetricsHistogram.std), [`EasyDeLOptimizers`](../infra/etils.md#EasyDeLOptimizers), [`JaxDistributedConfig`](utils.md#JaxDistributedConfig), [`AFFINE_SUPPORTED_BITS`](training_configurations.md#AFFINE_SUPPORTED_BITS), [`AVAILABLE_OPTIMIZERS`](../infra/etils.md#AVAILABLE_OPTIMIZERS), [`AVAILABLE_SCHEDULERS`](../infra/etils.md#AVAILABLE_SCHEDULERS), [`NONE`](../infra/etils.md#EasyDeLSchedulers.NONE)  (+7 more)
- used by: [`_import_structure`](../__init__.md#_import_structure), [`arguments`](base_trainer.md#BaseTrainer.arguments), [`generate_unified`](base_trainer.md#BaseTrainer.generate_unified), [`_preprocess_batch_input`](group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer._preprocess_batch_input), [`_preprocess_batch_input`](group_filtered_policy_optimization/gfpo_trainer.md#GFPOTrainer._preprocess_batch_input), [`_preprocess_batch_input`](proximal_policy_optimization_trainer/ppo_trainer.md#PPOTrainer._preprocess_batch_input), [`_preprocess_batch_input`](self_distillation_policy_optimization/sdpo_trainer.md#SDPOTrainer._preprocess_batch_input), [`_train_epoch`](trainer/trainer.md#Trainer._train_epoch), [`_preprocess_batch_input`](agentic_moshpit/agentic_moshpit_trainer.md#AgenticMoshPitTrainer._preprocess_batch_input), [`configure_functions`](distillation_trainer/distillation_trainer.md#DistillationTrainer.configure_functions), [`configure_functions`](direct_preference_optimization_trainer/dpo_trainer.md#DPOTrainer.configure_functions), [`configure_functions`](trainer/trainer.md#Trainer.configure_functions), [`configure_functions`](group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer.configure_functions), [`configure_functions`](generalized_knowledge_distillation_trainer/gkd_trainer.md#GKDTrainer.configure_functions), [`_get_information`](base_trainer.md#BaseTrainer._get_information), [`_eval_dataset_steps_auto_clamped`](base_trainer.md#BaseTrainer._eval_dataset_steps_auto_clamped), [`_tpu_preemption_sync_available`](base_trainer.md#BaseTrainer._tpu_preemption_sync_available), [`configure_functions`](on_policy_distillation_trainer/on_policy_distillation_trainer.md#OnPolicyDistillationTrainer.configure_functions), [`_eval_epoch`](trainer/trainer.md#Trainer._eval_epoch), [`configure_functions`](odds_ratio_preference_optimization_trainer/orpo_trainer.md#ORPOTrainer.configure_functions), [`configure_functions`](sparse_distillation_trainer/sparse_distillation_trainer.md#SparseDistillationTrainer.configure_functions), [`concatenated_forward`](kto_trainer/kto_trainer.md#KTOTrainer.concatenated_forward), [`configure_functions`](self_distillation_policy_optimization/sdpo_trainer.md#SDPOTrainer.configure_functions), [`configure_functions`](xpo_trainer/xpo_trainer.md#XPOTrainer.configure_functions), [`maybe_benchmark`](base_trainer.md#BaseTrainer.maybe_benchmark), [`configure_functions`](proximal_policy_optimization_trainer/ppo_trainer.md#PPOTrainer.configure_functions), [`concatenated_forward`](contrastive_preference_optimization_trainer/cpo_trainer.md#CPOTrainer.concatenated_forward), [`maybe_generate`](base_trainer.md#BaseTrainer.maybe_generate), [`configure_functions`](embedding_trainer/embedding_trainer.md#EmbeddingTrainer.configure_functions), [`__init__`](base_trainer.md#BaseTrainer.__init__), [`configure_functions`](reward_trainer/reward_trainer.md#RewardTrainer.configure_functions), [`_resolve_step_count`](base_trainer.md#BaseTrainer._resolve_step_count), [`concatenated_forward`](binary_classifier_optimization_trainer/bco_trainer.md#BCOTrainer.concatenated_forward), [`GRPOConfig`](group_relative_policy_optimization/grpo_config.md#GRPOConfig), [`_configure_state`](base_trainer.md#BaseTrainer._configure_state), [`_configure_tfds_dataloader`](base_trainer.md#BaseTrainer._configure_tfds_dataloader), [`_run_training_loop`](trainer/trainer.md#Trainer._run_training_loop), [`build_trainer`](../infra/elarge/model.md#eLargeModel.build_trainer), [`configure_functions`](contrastive_preference_optimization_trainer/cpo_trainer.md#CPOTrainer.configure_functions), [`configure_functions`](nash_md_trainer/nash_md_trainer.md#NashMDTrainer.configure_functions)  (+121 more)

## Functions
- `_apply_training_args_legacy_aliases(data: dict[str, tp.Any])` — [`L167`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L167) — Rewrite deprecated field names in a config dict to their current equivalents.
- `_get_max_sequence_length(self: TrainingArguments)` — [`L2087`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L2087) — Getter for the deprecated ``max_sequence_length`` property.
- `_get_quantization_block(self: TrainingArguments)` — [`L2120`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L2120) — Getter for the deprecated ``quantization_block`` property.
- `_normalize_partition_spec_entry(value: tp.Any)` — [`L109`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L109) — Recursively convert lists to tuples within a partition spec entry.
- `_parse_partition_spec(value: tp.Any)` — [`L125`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L125) — Parse a value into a JAX PartitionSpec.
- `_set_max_sequence_length(self: TrainingArguments, value: int | None)` — [`L2096`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L2096) — Setter for the deprecated ``max_sequence_length`` property.
- `_set_quantization_block(self: TrainingArguments, value: int | None)` — [`L2129`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L2129) — Setter for the deprecated ``quantization_block`` property.
- `get_safe_arr(xs)` — [`L93`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L93) — Convert single-element arrays to Python scalars for safe logging.

## Module values
- `AFFINE_SUPPORTED_BITS` — [`L83`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L83)
- `AVAILABLE_BACKENDS` — [`L189`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L189)
- `FIXED_QUANTIZATION_BITS_BY_MODE` — [`L84`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L84)
- `MetricsType` — [`L70`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L70)
- `QuantizationMode` — [`L73`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L73)
- `STE_QAT_QUANTIZATION_MODES` — [`L81`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L81)
- `STE_QAT_QUANTIZATION_MODES_DOC` — [`L82`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L82)
- `logger` — [`L71`](../../../../../../raw/code/EasyDeL/easydel/trainers/training_configurations.py#L71)


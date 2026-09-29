---
title: 'Module: easydel/trainers/base_trainer.py'
type: catalog
provenance: extracted
module: easydel/trainers/base_trainer.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.base_trainer`/
symbols:
  BaseTrainer.arguments: BaseTrainer#arguments.
  BaseTrainer.generate_unified: BaseTrainer#generate_unified().
  BaseTrainer._get_information: BaseTrainer#_get_information().
  BaseTrainer._eval_dataset_steps_auto_clamped: BaseTrainer#_eval_dataset_steps_auto_clamped.
  BaseTrainer._all_gather: BaseTrainer#_all_gather().
  BaseTrainer._tpu_preemption_sync_available: BaseTrainer#_tpu_preemption_sync_available.
  BaseTrainer.model_state: BaseTrainer#model_state.
  BaseTrainer._train_source: BaseTrainer#_train_source.
  BaseTrainer.maybe_benchmark: BaseTrainer#maybe_benchmark().
  BaseTrainer.maybe_generate: BaseTrainer#maybe_generate().
  logger: logger.
  BaseTrainer.__init__: BaseTrainer#__init__().
  BaseTrainer._resolve_step_count: BaseTrainer#_resolve_step_count().
  BaseTrainer._configure_state: BaseTrainer#_configure_state().
  BaseTrainer._configure_tfds_dataloader: BaseTrainer#_configure_tfds_dataloader().
  BaseTrainer._create_grain_dataloader: BaseTrainer#_create_grain_dataloader().
  BaseTrainer._configure_grain_dataloader: BaseTrainer#_configure_grain_dataloader().
  BaseTrainer.model: BaseTrainer#model().
  BaseTrainer._is_pretokenized: BaseTrainer#_is_pretokenized().
  BaseTrainer._esurge_init_kwargs: BaseTrainer#_esurge_init_kwargs().
  BaseTrainer._log_training_generations_to_wandb: BaseTrainer#_log_training_generations_to_wandb().
  BaseTrainer._get_preprocess_transform: BaseTrainer#_get_preprocess_transform().
  BaseTrainer: BaseTrainer#
  BaseTrainer._prepare_training_output: BaseTrainer#_prepare_training_output().
  BaseTrainer._configure_dataloaders: BaseTrainer#_configure_dataloaders().
  BaseTrainer._save_state: BaseTrainer#_save_state().
  BaseTrainer._setup_initial_metrics: BaseTrainer#_setup_initial_metrics().
  BaseTrainer.create_generate_function: BaseTrainer#create_generate_function().
  BaseTrainer.generate_aio: BaseTrainer#generate_aio().
  BaseTrainer._preprocess_batch_input: BaseTrainer#_preprocess_batch_input().
  BaseTrainer._configure_model: BaseTrainer#_configure_model().
  BaseTrainer._default_generation_kwargs: BaseTrainer#_default_generation_kwargs().
  BaseTrainer.processing_class: BaseTrainer#processing_class.
  BaseTrainer._configure_functions: BaseTrainer#_configure_functions().
  BaseTrainer.apply_training_hooks: BaseTrainer#apply_training_hooks().
  BaseTrainer.generate: BaseTrainer#generate().
  BaseTrainer.log_metrics: BaseTrainer#log_metrics().
  BaseTrainer._train_shared_fn_static_args: BaseTrainer#_train_shared_fn_static_args().
  log_debug_maybe: log_debug_maybe.
  BaseTrainer._eval_shared_fn_static_args: BaseTrainer#_eval_shared_fn_static_args().
  BaseTrainer.create_tf_dataset: BaseTrainer#create_tf_dataset().
  BaseTrainer._train_shared_fn_extra_args: BaseTrainer#_train_shared_fn_extra_args().
  BaseTrainer._prepare_generation_input: BaseTrainer#_prepare_generation_input().
  BaseTrainer._coerce_generation_texts: BaseTrainer#_coerce_generation_texts().
  BaseTrainer._eval_shared_fn_extra_args: BaseTrainer#_eval_shared_fn_extra_args().
  BaseTrainer._save_checkpoint_for_step: BaseTrainer#_save_checkpoint_for_step().
  BaseTrainer._add: BaseTrainer#_add().
  BaseTrainer._finalize_preview_results: BaseTrainer#_finalize_preview_results().
  BaseTrainer.create_progress_bar: BaseTrainer#create_progress_bar().
  GenerationResults.text: GenerationResults#text.
  BaseTrainer._set_dataset_size_metadata: BaseTrainer#_set_dataset_size_metadata().
  BaseTrainer._apply_step_start_point: BaseTrainer#_apply_step_start_point().
  BaseTrainer._configure_generation_function: BaseTrainer#_configure_generation_function().
  BaseTrainer.release_generation_runtime: BaseTrainer#release_generation_runtime().
  BaseTrainer._sample_prompts_from_sharded_source: BaseTrainer#_sample_prompts_from_sharded_source().
  BaseTrainer.configure_model: BaseTrainer#configure_model().
  BaseTrainer._cleanup_old_checkpoints: BaseTrainer#_cleanup_old_checkpoints().
  BaseTrainer._eos_token_id: BaseTrainer#_eos_token_id().
  BaseTrainer._eval_source: BaseTrainer#_eval_source.
  BaseTrainer.maybe_release_generation_runtime: BaseTrainer#maybe_release_generation_runtime().
  BaseTrainer.configure_dataloaders: BaseTrainer#configure_dataloaders().
  BaseTrainer._pad_token_id: BaseTrainer#_pad_token_id().
  BaseTrainer._apply_step_start_point_to_optimizer_state: BaseTrainer#_apply_step_start_point_to_optimizer_state().
  BaseTrainer._decode_prompt_batch: BaseTrainer#_decode_prompt_batch().
  BaseTrainer._initialize_attributes: BaseTrainer#_initialize_attributes().
  BaseTrainer.initialize_trainer_utils: BaseTrainer#initialize_trainer_utils().
  BaseTrainer.compile_aot: BaseTrainer#compile_aot().
  BaseTrainer.training_batch_size: BaseTrainer#training_batch_size().
  BaseTrainer._purify_batch: BaseTrainer#_purify_batch().
  GenerationResults.completion_ids: GenerationResults#completion_ids.
  BaseTrainer.eval_is_conversational: BaseTrainer#eval_is_conversational.
  BaseTrainer.save_pretrained: BaseTrainer#save_pretrained().
  BaseTrainer._apply_preprocess_transforms: BaseTrainer#_apply_preprocess_transforms().
  BaseTrainer._normalize_supported_generation_model_kwargs: BaseTrainer#_normalize_supported_generation_model_kwargs().
  BaseTrainer._cleanup_failed_esurge_generation: BaseTrainer#_cleanup_failed_esurge_generation().
  BaseTrainer._sample_random_example_from_shard: BaseTrainer#_sample_random_example_from_shard().
  GenerationResults.reasoning: GenerationResults#reasoning.
  GenerationResults.tool_calls: GenerationResults#tool_calls.
  BaseTrainer.mesh: BaseTrainer#mesh().
  BaseTrainer.on_step_end: BaseTrainer#on_step_end().
  BaseTrainer._collect_generation_prompts: BaseTrainer#_collect_generation_prompts().
  GenerationResults.prompt_ids: GenerationResults#prompt_ids.
  BaseTrainer.evaluation_batch_size: BaseTrainer#evaluation_batch_size().
  BaseTrainer._initialize_memory_tracking: BaseTrainer#_initialize_memory_tracking().
  BaseTrainer._create_dataloader_from_source: BaseTrainer#_create_dataloader_from_source().
  BaseTrainer._int_attr: BaseTrainer#_int_attr().
  BaseTrainer._coerce_generation_metadata_list: BaseTrainer#_coerce_generation_metadata_list().
  BaseTrainer.checkpointer: BaseTrainer#checkpointer.
  GenerationResults.prompt_mask: GenerationResults#prompt_mask.
  GenerationResults.sequences: GenerationResults#sequences.
  GenerationResults.raw_text: GenerationResults#raw_text.
  BaseTrainer._apply_runtime_model_config_overrides: BaseTrainer#_apply_runtime_model_config_overrides().
  BaseTrainer._coerce_optional_generation_texts: BaseTrainer#_coerce_optional_generation_texts().
  BaseTrainer._sum_known_shard_rows: BaseTrainer#_sum_known_shard_rows().
  BaseTrainer._sample_prompts_from_dataset: BaseTrainer#_sample_prompts_from_dataset().
  BaseTrainer._log_benchmark_results_to_wandb: BaseTrainer#_log_benchmark_results_to_wandb().
  BaseTrainer._initialize_wandb: BaseTrainer#_initialize_wandb().
  BaseTrainer._initialize_timer: BaseTrainer#_initialize_timer().
  BaseTrainer.create_tf_dataset_from_iterable: BaseTrainer#create_tf_dataset_from_iterable().
  BaseTrainer.save_callback: BaseTrainer#save_callback().
  BaseTrainer._save_to_torch: BaseTrainer#_save_to_torch().
  BaseTrainer._save_tpu_preemption_checkpoint: BaseTrainer#_save_tpu_preemption_checkpoint().
  BaseTrainer._make_attn_mask: BaseTrainer#_make_attn_mask().
  BaseTrainer.specs_to_name_sharding: BaseTrainer#specs_to_name_sharding().
  BaseTrainer._handle_training_interruption: BaseTrainer#_handle_training_interruption().
  BaseTrainer._build_reward_call_kwargs: BaseTrainer#_build_reward_call_kwargs().
  _ResolvedStepCount.steps: _ResolvedStepCount#steps.
  BaseTrainer._build_structured_assistant_messages: BaseTrainer#_build_structured_assistant_messages().
  BaseTrainer.benchmark_log_table: BaseTrainer#benchmark_log_table.
  BaseTrainer._maybe_insert: BaseTrainer#_maybe_insert().
  BaseTrainer._memory_optimization_hints: BaseTrainer#_memory_optimization_hints().
  BaseTrainer._discover_dataset_num_examples: BaseTrainer#_discover_dataset_num_examples().
  BaseTrainer._format_partition_rules: BaseTrainer#_format_partition_rules().
  BaseTrainer._should_save_tpu_preemption_checkpoint: BaseTrainer#_should_save_tpu_preemption_checkpoint().
  BaseTrainer.log_weight_distribution: BaseTrainer#log_weight_distribution().
  BaseTrainer.log_watchers: BaseTrainer#log_watchers().
  BaseTrainer._apply_runtime_model_config_overrides_to_state: BaseTrainer#_apply_runtime_model_config_overrides_to_state().
  BaseTrainer._peek_first_example: BaseTrainer#_peek_first_example().
  BaseTrainer._to_sharded_source: BaseTrainer#_to_sharded_source().
  BaseTrainer._default_generation_config_overrides: BaseTrainer#_default_generation_config_overrides().
  BaseTrainer._batch_decode_tokens: BaseTrainer#_batch_decode_tokens().
  BaseTrainer._normalize_single: BaseTrainer#_normalize_single().
  BaseTrainer.is_enable: BaseTrainer#is_enable().
  _ResolvedStepCount: _ResolvedStepCount#
  BaseTrainer.to_tf_dataloader: BaseTrainer#to_tf_dataloader().
  BaseTrainer._get_next_batch: BaseTrainer#_get_next_batch().
  _ResolvedStepCount.num_examples: _ResolvedStepCount#num_examples.
  _ResolvedStepCount.num_examples_exact: _ResolvedStepCount#num_examples_exact.
  _ResolvedStepCount.source_label: _ResolvedStepCount#source_label.
  _ResolvedStepCount.auto_discovered: _ResolvedStepCount#auto_discovered.
  _ResolvedStepCount.auto_clamped: _ResolvedStepCount#auto_clamped.
  GenerationResults.completion_mask: GenerationResults#completion_mask.
  GenerationResults.completion_prompts: GenerationResults#completion_prompts.
  BaseTrainer._seed_count: BaseTrainer#_seed_count().
  BaseTrainer._memory_optimization_trainer_name: BaseTrainer#_memory_optimization_trainer_name().
  BaseTrainer._format_memory_optimization_hints: BaseTrainer#_format_memory_optimization_hints().
  BaseTrainer.generate_function: BaseTrainer#generate_function.
  BaseTrainer._generation_rng: BaseTrainer#_generation_rng.
  BaseTrainer._prepare_generation_config: BaseTrainer#_prepare_generation_config().
  BaseTrainer._get_processing_class: BaseTrainer#_get_processing_class().
  BaseTrainer.configure_functions: BaseTrainer#configure_functions().
  BaseTrainer._maybe_remove_loaded_checkpoint: BaseTrainer#_maybe_remove_loaded_checkpoint().
  BaseTrainer._create_checkpointer: BaseTrainer#_create_checkpointer().
  BaseTrainer.start_training_hook: BaseTrainer#start_training_hook().
  BaseTrainer.start_evaluation_hook: BaseTrainer#start_evaluation_hook().
  BaseTrainer._should_run_evaluation: BaseTrainer#_should_run_evaluation().
  BaseTrainer._should_enable_tpu_preemption_checkpointing: BaseTrainer#_should_enable_tpu_preemption_checkpointing().
  BaseTrainer._hidden_rich_pbar: BaseTrainer#_hidden_rich_pbar.
  BaseTrainer._normalize_esurge_prompts: BaseTrainer#_normalize_esurge_prompts().
  BaseTrainer._length_is_exact: BaseTrainer#_length_is_exact().
  GenerationResults: GenerationResults#
  GenerationResults.generation_results: GenerationResults#generation_results.
  BaseTrainer.__setattr__: BaseTrainer#__setattr__().
  BaseTrainer._preemption_checkpoint_path: BaseTrainer#_preemption_checkpoint_path.
  BaseTrainer._reward_chat_template_tools: BaseTrainer#_reward_chat_template_tools().
  BaseTrainer.train_is_conversational: BaseTrainer#train_is_conversational.
  BaseTrainer.is_process_zero: BaseTrainer#is_process_zero().
  BaseTrainer.save_information: BaseTrainer#save_information().
  BaseTrainer._initialize_conversational_flags: BaseTrainer#_initialize_conversational_flags().
  BaseTrainer._preview_reasoning_entries: BaseTrainer#_preview_reasoning_entries().
  BaseTrainer._preview_tool_call_entries: BaseTrainer#_preview_tool_call_entries().
  BaseTrainer._save_readme: BaseTrainer#_save_readme().
  BaseTrainer.compile_function: BaseTrainer#compile_function().
  _ReiterableDataLoader: _ReiterableDataLoader#
  BaseTrainer._extract_reward_batch_sidechannels: BaseTrainer#_extract_reward_batch_sidechannels().
  BaseTrainer.generate_function_with_model_kwargs: BaseTrainer#generate_function_with_model_kwargs.
  BaseTrainer.preview_log_table: BaseTrainer#preview_log_table.
  BaseTrainer.training_generation_log_table: BaseTrainer#training_generation_log_table.
  BaseTrainer._apply_chat_template: BaseTrainer#_apply_chat_template().
  BaseTrainer._wandb_stringify_generation_value: BaseTrainer#_wandb_stringify_generation_value().
  BaseTrainer._auto_data_collator: BaseTrainer#_auto_data_collator().
  BaseTrainer._augment_memory_oom_exception: BaseTrainer#_augment_memory_oom_exception().
  BaseTrainer._normalize_tool_call_payloads: BaseTrainer#_normalize_tool_call_payloads().
  BaseTrainer.on_step_start: BaseTrainer#on_step_start().
  BaseTrainer.create_grain_collect_function: BaseTrainer#create_grain_collect_function().
  BaseTrainer.create_tfds_collect_function: BaseTrainer#create_tfds_collect_function().
  BaseTrainer.create_collect_function: BaseTrainer#create_collect_function().
  BaseTrainer._sync_checkpointer_after_callback_save: BaseTrainer#_sync_checkpointer_after_callback_save().
  BaseTrainer._restore_checkpoint_metadata: BaseTrainer#_restore_checkpoint_metadata().
  BaseTrainer._get_device_info: BaseTrainer#_get_device_info().
  BaseTrainer._fast_forward_batches: BaseTrainer#_fast_forward_batches().
  _ReiterableDataLoader.__iter__: _ReiterableDataLoader#__iter__().
  _ReiterableDataLoader.__len__: _ReiterableDataLoader#__len__().
  BaseTrainer.__str__: BaseTrainer#__str__.
  BaseTrainer._ensure_functions_compiled: BaseTrainer#_ensure_functions_compiled().
  BaseTrainer._one_to_all: BaseTrainer#_one_to_all().
  BaseTrainer._create_hf_model_config: BaseTrainer#_create_hf_model_config().
  _ReiterableDataLoader._length: _ReiterableDataLoader#_length.
  GenerationResults.decoded_prompts: GenerationResults#decoded_prompts.
  BaseTrainer._RUNTIME_MODEL_OVERRIDE_STATE_ATTRS: BaseTrainer#_RUNTIME_MODEL_OVERRIDE_STATE_ATTRS.
  BaseTrainer._is_memory_oom_exception: BaseTrainer#_is_memory_oom_exception().
  BaseTrainer._coerce_mapping_like: BaseTrainer#_coerce_mapping_like().
  BaseTrainer._train_dataset_num_examples: BaseTrainer#_train_dataset_num_examples.
  BaseTrainer._eval_dataset_num_examples: BaseTrainer#_eval_dataset_num_examples.
  BaseTrainer._strip_pad: BaseTrainer#_strip_pad().
  BaseTrainer._setup_static_metrics: BaseTrainer#_setup_static_metrics().
  DEFAULT_ARGS_JSON_NAME: DEFAULT_ARGS_JSON_NAME.
  _ReiterableDataLoader._factory: _ReiterableDataLoader#_factory.
  BaseTrainer._stack_per_example_outputs: BaseTrainer#_stack_per_example_outputs().
  BaseTrainer._steps_from_examples: BaseTrainer#_steps_from_examples().
  BaseTrainer._train_dataset_num_examples_exact: BaseTrainer#_train_dataset_num_examples_exact.
  BaseTrainer._train_dataset_size_source_label: BaseTrainer#_train_dataset_size_source_label.
  BaseTrainer._train_dataset_steps_auto_discovered: BaseTrainer#_train_dataset_steps_auto_discovered.
  BaseTrainer._train_dataset_steps_auto_clamped: BaseTrainer#_train_dataset_steps_auto_clamped.
  BaseTrainer._eval_dataset_num_examples_exact: BaseTrainer#_eval_dataset_num_examples_exact.
  BaseTrainer._eval_dataset_size_source_label: BaseTrainer#_eval_dataset_size_source_label.
  BaseTrainer._eval_dataset_steps_auto_discovered: BaseTrainer#_eval_dataset_steps_auto_discovered.
  BaseTrainer.latest_generation_samples: BaseTrainer#latest_generation_samples.
  BaseTrainer.latest_benchmark_results: BaseTrainer#latest_benchmark_results.
  BaseTrainer.__repr__: BaseTrainer#__repr__().
  BaseTrainer.finish: BaseTrainer#finish().
  BaseTrainer._pad_value_for_batch_key: BaseTrainer#_pad_value_for_batch_key().
  BaseTrainer.get_mtime: BaseTrainer#get_mtime().
  BaseTrainer._get_current_step: BaseTrainer#_get_current_step().
  BaseTrainer.count_model_parameters: BaseTrainer#count_model_parameters().
  _ReiterableDataLoader.__init__: _ReiterableDataLoader#__init__().
  BaseTrainer._sanitize_text_prompt: BaseTrainer#_sanitize_text_prompt().
  BaseTrainer.calculate_number_total_flops: BaseTrainer#calculate_number_total_flops().
---
# Module: [`easydel/trainers/base_trainer.py`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py)

## Classes
### `BaseTrainer`  ·  implements/extends BaseTrainerProtocol
- def: [`easydel/trainers/base_trainer.py:163`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L163)
- doc: Base trainer class implementing core training functionality for EasyDeL models.
- signature: `class BaseTrainer(BaseTrainerProtocol):`
- members:
  - `__init__(self, arguments: TrainingArguments | None = None, model_state: EasyDeLState | None = None, model: tp.type[EasyDeLBaseModule] | None = None, dataset_train: Dataset | IterableDataset | ShardedDataSource | None = None, dataset_eval: Dataset | IterableDataset | ShardedDataSource | None = None, data_collator: tp.Callable | None = None, finetune: bool = True, processing_class: PreTrainedTokenizerBase | None = None, **deprecated_kwargs)` — [`L226`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L226) — Initialize the BaseTrainer. — documented in [easydel-trainers-base_trainer](../../../concepts/easydel-trainers-base_trainer.md)
  - `__repr__(self)` — [`L1768`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1768) — Return string representation of the trainer.
  - `__setattr__(self, name: str, value: tp.Any)` — [`L205`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L205) — Set an attribute, applying runtime model config overrides for state attributes.
  - `_add(text: str)` — [`L602`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L602) — Append a hint string if it is not already present in the list.
  - `_all_gather(self, arr: jax.Array)` — [`L3969`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L3969) — Gather array from all devices to a single replicated array. — documented in [easydel-trainers-base_trainer](../../../concepts/easydel-trainers-base_trainer.md)
  - `_apply_preprocess_transforms(self)` — [`L1601`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1601) — Apply preprocessing transforms to data sources.
  - `_apply_runtime_model_config_overrides(self)` — [`L432`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L432) — Apply runtime model config overrides to all tracked model states.
  - `_apply_runtime_model_config_overrides_to_state(state: EasyDeLState | None, arguments: TrainingArguments)` — [`L398`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L398) — Propagate training argument overrides onto a model state's config.
  - `_apply_step_start_point(self)` — [`L447`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L447) — Initialize a fresh training state from ``step_start_point`` when requested.
  - `_apply_step_start_point_to_optimizer_state(self)` — [`L482`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L482) — Align optimizer and scheduler counters with ``step_start_point``.
  - `_augment_memory_oom_exception(self, exc: BaseException)` — [`L798`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L798) — Wrap an OOM exception with actionable memory optimization hints.
  - `_auto_data_collator(batch)` — [`L358`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L358) — Automatically collate a batch using the base collator.
  - `_batch_decode_tokens(self, token_ids: tp.Any)` — [`L3054`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L3054) — Decode a batch of token ID arrays into human-readable strings.
  - `_build_reward_call_kwargs(self, reward_func: tp.Callable[..., tp.Any], *, prompts: tp.Any, completions: tp.Any, max_length: int, raw_completions: tp.Any | None = None, prompt_texts: list[str] | None = None, completion_texts: list[str] | None = None, raw_text: list[str] | None = None, reasoning: list[str | None] | None = None, tool_calls: list[tp.Any | None] | None = None, batch: dict[str, tp.Any] | None = None, **extra_kwargs: tp.Any)` — [`L1077`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1077) — Build filtered kwargs for callable reward functions.
  - `_build_structured_assistant_messages(self, contents: list[str], *, tool_calls: list[tp.Any | None] | None = None)` — [`L1048`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1048) — Build assistant message payloads with normalized tool calls when present.
  - `_cleanup_old_checkpoints(self)` — [`L4799`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L4799) — Clean up old permanent checkpoints based on save_total_limit.
  - `_coerce_generation_metadata_list(values: collections.abc.Sequence[tp.Any] | tp.Any | None, *, target_len: int)` — [`L961`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L961) — Normalize non-text generation metadata to a fixed-length list.
  - `_coerce_generation_texts(values: str | collections.abc.Sequence[tp.Any] | None, *, fallback: str | collections.abc.Sequence[tp.Any] | None = None)` — [`L926`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L926) — Normalize generation text outputs into a list of strings.
  - `_coerce_mapping_like(value: tp.Any)` — [`L978`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L978) — Coerce JSON-string payloads into mapping-like objects when possible.
  - `_coerce_optional_generation_texts(values: str | collections.abc.Sequence[tp.Any] | None, *, target_len: int)` — [`L942`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L942) — Normalize optional generation text metadata to a fixed-length list.
  - `_collect_generation_prompts(self)` — [`L3077`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L3077) — Collect prompts for preview generation during training.
  - `_configure_dataloaders(self)` — [`L4026`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L4026) — Configures the dataloaders for training and evaluation.
  - `_configure_functions(self)` — [`L4070`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L4070) — Configures and JIT-compiles the training and evaluation step functions.
  - `_configure_generation_function(self)` — [`L1974`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1974) — Prepare a default generation function when the model supports generation.
  - `_configure_grain_dataloader(self)` — [`L4242`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L4242) — Configure Grain dataloaders for training and evaluation.
  - `_configure_model(self)` — [`L4053`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L4053) — Configures the model, optimizer, scheduler, and configuration.
  - `_configure_state(self)` — [`L4091`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L4091) — Configures and shards the model state across devices. — documented in [easydel-trainers-base_trainer](../../../concepts/easydel-trainers-base_trainer.md)
  - `_configure_tfds_dataloader(self)` — [`L4373`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L4373) — Configure TensorFlow Dataset dataloaders for training and evaluation. — documented in [easydel-trainers-base_trainer](../../../concepts/easydel-trainers-base_trainer.md)
  - `_create_checkpointer(self)` — [`L4786`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L4786) — Create and configure the Checkpointer instance.
  - `_create_dataloader_from_source(self, source: ShardedDataSource, batch_size: int, is_train: bool = True, shuffle: bool = False, num_epochs: int = 1, drop_remainder: bool = True)` — [`L4204`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L4204) — Create dataloader iterator from ShardedDataSource.
  - `_create_grain_dataloader(dataset: Dataset | IterableDataset, is_train: bool)` — [`L4257`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L4257) — Creates a Grain DataLoader from a Hugging Face Dataset. — documented in [easydel-trainers-base_trainer](../../../concepts/easydel-trainers-base_trainer.md)
  - `_create_hf_model_config(self, state: EasyDeLState, model_config, model_type)` — [`L5098`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L5098) — Create a HuggingFace ``AutoConfig`` populated from the EasyDeL model config.
  - `_decode_prompt_batch(processor: PreTrainedTokenizerBase | None, input_ids: jax.Array | np.ndarray, skip_special_tokens: bool = True, pad_token_id: int | None = None, pop_pad_tokens: bool = False, attention_mask: jax.Array | np.ndarray | None = None)` — [`L870`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L870) — Decode a batch of token IDs into prompt strings.
  - `_default_generation_config_overrides(self)` — [`L2066`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L2066) — Return a copy of user-specified generation config overrides, or None if empty.
  - `_default_generation_kwargs(self)` — [`L2012`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L2012) — Assemble default keyword arguments for `model.generate` based on training arguments.
  - `_discover_dataset_num_examples(self, dataset: Dataset | IterableDataset | ShardedDataSource | None, *, source: ShardedDataSource | None = None)` — [`L1294`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1294) — Discover dataset cardinality from the most informative available object.
  - `_ensure_functions_compiled(self)` — [`L1964`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1964) — Ensure training and evaluation functions are compiled.
  - `_eos_token_id(self)` — [`L1529`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1529) — Collect all unique EOS token IDs from the processor and model config.
  - `_esurge_init_kwargs(self, *, max_num_seqs: int | None = None)` — [`L3666`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L3666) — Build keyword arguments for initializing an eSurge inference runtime.
  - `_eval_shared_fn_extra_args(self)` — [`L1476`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1476) — Extra arguments passed to the shared evaluation function at each step.
  - `_eval_shared_fn_static_args(self)` — [`L1486`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1486) — Static (compile-time constant) arguments for the shared evaluation function.
  - `_extract_reward_batch_sidechannels(batch: tp.Any)` — [`L1112`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1112) — Preserve non-numeric batch metadata needed by callable reward functions.
  - `_fast_forward_batches(self, data_iter, dataloader, num_batches: int)` — [`L5442`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L5442) — Advance an iterator by discarding a fixed number of batches.
  - `_finalize_preview_results(records: list[dict[str, tp.Any]])` — [`L3433`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L3433) — Log and store completed preview generation results.
  - `_format_memory_optimization_hints(self)` — [`L780`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L780) — Format memory optimization hints into a single multiline message.
  - `_format_partition_rules(self)` — [`L4883`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L4883) — Format partition rules with proper indentation and formatting.
  - `_get_current_step(self, state)` — [`L4863`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L4863) — Get the current training step from state.
  - `_get_device_info(self)` — [`L4891`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L4891) — Get information about available devices.
  - `_get_information(self)` — [`L4903`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L4903) — Generate formatted information about the model and training setup.
  - `_get_next_batch(self, data_iter, dataloader)` — [`L5413`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L5413) — Get next batch from iterator, reinitializing if needed.
  - `_get_preprocess_transform(self)` — [`L1616`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1616) — Get trainer-specific preprocessing transform.
  - `_get_processing_class(self)` — [`L3028`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L3028) — Resolve the tokenizer or processor associated with the trainer.
  - `_handle_training_interruption(self, state: EasyDeLState, exception: Exception)` — [`L5358`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L5358) — Handle training interruption gracefully.
  - `_initialize_attributes(self)` — [`L1642`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1642) — Initialize all trainer attributes with default values.
  - `_initialize_conversational_flags(self, train_dataset: tp.Any, eval_dataset: tp.Any)` — [`L1187`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1187) — Detect whether train and eval datasets use conversational format.
  - `_initialize_memory_tracking(self)` — [`L1755`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1755) — Initialize memory monitoring for tracking GPU/TPU memory usage.
  - `_initialize_timer(self)` — [`L4016`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L4016) — Initialize the timer for performance monitoring.
  - `_initialize_wandb(self)` — [`L4005`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L4005) — Initialize Weights & Biases logging if enabled.
  - `_int_attr(name: str)` — [`L607`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L607) — Get an integer attribute from the training arguments by name.
  - `_is_memory_oom_exception(exc: BaseException)` — [`L530`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L530) — Determine whether an exception represents an out-of-memory error.
  - `_is_pretokenized(self)` — [`L1628`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1628) — Check if the training dataset is already tokenized.
  - `_length_is_exact(self, candidate: tp.Any)` — [`L1255`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1255) — Best-effort signal for whether ``len(candidate)`` is an exact count.
  - `_log_benchmark_results_to_wandb(self, *, step: int, benchmark_results: dict[str, dict[str, tp.Any]])` — [`L3782`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L3782) — Log benchmark summaries to W&B as an incremental table.
  - `_log_training_generations_to_wandb(self, *, state: EasyDeLState, prompts: tp.Any, completions: list[str] | tuple[str, ...] | None = None, prompt_mask: jax.Array | np.ndarray | None = None, completion_ids: jax.Array | np.ndarray | None = None, completion_mask: jax.Array | np.ndarray | None = None, completion_lengths: jax.Array | np.ndarray | None = None, generation_time: float | None = None, reasoning: list[tp.Any] | tuple[tp.Any, ...] | None = None, tool_calls: list[tp.Any] | tuple[tp.Any, ...] | None = None, source: str = "policy")` — [`L3836`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L3836) — Log rollout generations used for training to an incremental W&B table.
  - `_make_attn_mask(self, arr)` — [`L1553`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1553) — Build a causal attention mask that masks positions after the first EOS token.
  - `_maybe_insert(target: dict[str, tp.Any], key: str, value: tp.Any)` — [`L2018`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L2018) — Insert a key-value pair into target dict only if value is not None.
  - `_maybe_remove_loaded_checkpoint(self, checkpoint_path: str)` — [`L4764`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L4764) — Delete a checkpoint after load when explicitly requested.
  - `_memory_optimization_hints(self)` — [`L578`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L578) — Inspect training arguments and build a list of memory optimization hints.
  - `_memory_optimization_trainer_name(self)` — [`L562`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L562) — Return a human-readable trainer name for memory optimization messages.
  - `_normalize_esurge_prompts(prompts: tp.Any, apply_chat_template: bool)` — [`L821`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L821) — Normalize user-provided prompts into strings or chat conversations.
  - `_normalize_single(item: tp.Any)` — [`L827`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L827) — Normalize a single prompt item into a string or chat message list.
  - `_normalize_supported_generation_model_kwargs(self, state: EasyDeLState, model_kwargs: dict[str, tp.Any] | None)` — [`L2402`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L2402) — Normalize raw generation kwargs against the model's generation entrypoint.
  - `_normalize_tool_call_payloads(cls, tool_calls: tp.Any)` — [`L988`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L988) — Normalize structured tool-call payloads for chat-template rendering.
  - `_one_to_all(self, arr: jax.Array)` — [`L3956`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L3956) — Distribute array from one device to all devices.
  - `_pad_token_id(self)` — [`L1511`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1511) — Resolve the pad token ID from the processing class.
  - `_pad_value_for_batch_key(key: str, array: np.ndarray)` — [`L1899`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1899) — Choose a safe padding value for trainer-side list batch collation.
  - `_peek_first_example(dataset: tp.Any)` — [`L1151`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1151) — Retrieve the first example from a dataset without consuming it.
  - `_prepare_generation_config(self, generation_config: GenerationConfig | None)` — [`L1996`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1996) — Return a copy of the generation config to avoid mutating shared references.
  - `_prepare_generation_input(self, prompt: tp.Any, *, tools: tp.Any | None = None)` — [`L3246`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L3246) — Tokenize and pad a single prompt into model-ready input arrays.
  - `_prepare_training_output(self, state: EasyDeLState, run_exception: Exception | None = None)` — [`L5298`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L5298) — Finalize training by handling exceptions and saving the last checkpoint.
  - `_preprocess_batch_input(self, state: EasyDeLState, batch: dict[str, jax.Array], is_train: bool)` — [`L1848`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1848) — Preprocess a batch of input data before feeding to the model.
  - `_purify_batch(self, batch: dict)` — [`L1882`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1882) — Remove non-JAX-compatible fields from a batch.
  - `_resolve_step_count(self, dataset: Dataset | IterableDataset | ShardedDataSource | None, *, source: ShardedDataSource | None, is_train: bool, drop_remainder: bool)` — [`L1333`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1333) — Resolve configured steps and clamp to discovered dataset capacity when possible. — documented in [easydel-trainers-base_trainer](../../../concepts/easydel-trainers-base_trainer.md)
  - `_restore_checkpoint_metadata(checkpoint_dir: str | os.PathLike | ePathLike)` — [`L4683`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L4683) — Restore metadata fields that trainer-side state saves add on top of checkpointer metadata.
  - `_reward_chat_template_tools(self)` — [`L1069`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1069) — Resolve optional tool schemas for reward-side chat-template rendering.
  - `_sample_prompts_from_dataset(self, expected: int)` — [`L3098`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L3098) — Randomly sample prompt entries from the training dataset.
  - `_sample_prompts_from_sharded_source(self, source: ShardedDataSource[tp.Any], expected: int)` — [`L3167`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L3167) — Randomly sample raw examples from a :class:`ShardedDataSource`.
  - `_sample_random_example_from_shard(self, source: ShardedDataSource[tp.Any], shard_name: str, *, row_index: int | None = None, shard_rows: int | None = None)` — [`L3132`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L3132) — Sample a single example from one shard.
  - `_sanitize_text_prompt(prompt: str, processor: PreTrainedTokenizerBase | None)` — [`L1131`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1131) — Remove pad token occurrences from a decoded text prompt.
  - `_save_checkpoint_for_step(self, state: EasyDeLState, *, step: int, force: bool = False, merge_lora_before_save: bool = False)` — [`L4670`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L4670) — Run checkpointer policy evaluation and serialize the current trainer state.
  - `_save_readme(self, save_directory)` — [`L4874`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L4874) — Save training information as README.md in checkpoint directory.
  - `_save_state(self, state: EasyDeLState, save_directory: str | None = None, *args, merge_lora_before_save: bool = False, **kwargs)` — [`L4600`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L4600) — Save the current model state to a checkpoint.
  - `_save_to_torch(self, state: EasyDeLState, save_directory: str | os.PathLike, easystate_to_huggingface_model_kwargs: dict | None = None, torch_save_pretrained_kwargs: dict | None = None)` — [`L5071`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L5071) — Convert the JAX model to a PyTorch HuggingFace model and save it.
  - `_save_tpu_preemption_checkpoint(self, state: EasyDeLState, step: int)` — [`L5507`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L5507) — Save a coordinated TPU preemption checkpoint using standard trainer naming.
  - `_seed_count(path, leaf)` — [`L498`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L498) — Replace optimizer 'count' leaves with the requested step value.
  - `_setup_initial_metrics(self, state)` — [`L5375`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L5375) — Setup initial metrics logging.
  - `_setup_static_metrics(self)` — [`L5216`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L5216) — Initialize static training/evaluation metrics.
  - `_should_enable_tpu_preemption_checkpointing(self)` — [`L5477`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L5477) — Return whether TPU preemption-triggered checkpointing is enabled.
  - `_should_run_evaluation(self, current_step)` — [`L5279`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L5279) — Determine if evaluation should be run at current step.
  - `_should_save_tpu_preemption_checkpoint(self, step: int)` — [`L5486`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L5486) — Return True when JAX's preemption sync service reaches a safe save step.
  - `_stack_per_example_outputs(per_example: list[dict[str, tp.Any]])` — [`L320`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L320) — Stack per-example collator outputs into batched arrays.
  - `_strip_pad(tokens: list[int] | np.ndarray)` — [`L2719`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L2719) — Remove pad tokens for matching prompts across shuffled eSurge outputs.
  - `_sum_known_shard_rows(self, source: ShardedDataSource | None)` — [`L1274`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1274) — Sum ``ShardInfo.num_rows`` when every shard advertises it.
  - `_sync_checkpointer_after_callback_save(self, step: int)` — [`L4648`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L4648) — Mirror checkpointer bookkeeping for callback-driven saves.
  - `_to_sharded_source(self, dataset: Dataset | IterableDataset | ShardedDataSource | None)` — [`L1576`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1576) — Convert any dataset type to ShardedDataSource.
  - `_train_shared_fn_extra_args(self)` — [`L1471`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1471) — Extra arguments passed to the shared training function at each step.
  - `_train_shared_fn_static_args(self)` — [`L1481`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1481) — Static (compile-time constant) arguments for the shared training function.
  - `_wandb_stringify_generation_value(value: tp.Any)` — [`L3825`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L3825) — Convert prompt/completion metadata into a W&B-table-friendly string.
  - `apply_training_hooks(self, metrics: LossMetrics)` — [`L5168`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L5168) — Apply training hooks to check for stopping conditions.
  - `calculate_number_total_flops(self, params, is_training=True)` — [`L5144`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L5144) — Calculate total FLOPs for the model.
  - `compile_aot(self)` — [`L5224`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L5224) — Compile training and evaluation functions ahead-of-time.
  - `compile_function(function, dataloader, state, tag)` — [`L5242`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L5242) — Lower and AOT-compile a step function if not already compiled.
  - `configure_dataloaders(self)` — [`L4557`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L4557) — Configures the dataloaders for training and evaluation.
  - `configure_functions(self)` — [`L4190`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L4190) — Configures and JIT-compiles the training and evaluation step functions.
  - `configure_model(self)` — [`L4574`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L4574) — Configures the model, optimizer, scheduler, and configuration.
  - `count_model_parameters(prm)` — [`L5157`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L5157) — Count total number of model parameters.
  - `create_collect_function(self, max_sequence_length: int, truncation_mode: tp.Literal["keep_end", "keep_start"])` — [`L4169`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L4169) — Creates a function to collect and process batches of data for training or evaluation.
  - `create_generate_function(self, generation_config: GenerationConfig | None = None, *, shard_inputs: bool = True, config_overrides: dict[str, tp.Any] | None = None, accept_model_kwargs: bool = False, **generate_kwargs)` — [`L2078`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L2078) — Build and return a compiled generation function that mirrors the model's `generate`.
  - `create_grain_collect_function(self, max_sequence_length: int, truncation_mode: tp.Literal["keep_end", "keep_start"])` — [`L4127`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L4127) — Creates a function to collect and process batches of data for training or evaluation.
  - `create_progress_bar(self, total: int, desc: str = "", disabled: bool = False)` — [`L5524`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L5524) — Create a progress bar of the specified type.
  - `create_tf_dataset(dataset: Dataset, is_train: bool)` — [`L4406`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L4406) — Creates a TensorFlow dataset from a Hugging Face Dataset.
  - `create_tf_dataset_from_iterable(dataset: IterableDataset, is_train: bool)` — [`L4436`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L4436) — Creates a TensorFlow dataset from an iterable Hugging Face Dataset.
  - `create_tfds_collect_function(self, max_sequence_length: int, truncation_mode: tp.Literal["keep_end", "keep_start"])` — [`L4148`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L4148) — Creates a function to collect and process batches of data for training or evaluation.
  - `evaluation_batch_size(self)` — [`L1462`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1462) — Get the evaluation batch size.
  - `finish()` — [`L1779`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1779) — Clean up resources and finish any active logging sessions.
  - `generate(state: EasyDeLState, input_ids: jax.Array, attention_mask: jax.Array, model_kwargs: dict[str, tp.Any])` — [`L2130`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L2130) — Run model generation with additional model kwargs (e.g., multimodal inputs).
  - `generate_aio(self, input_ids: jax.Array | np.ndarray, attention_mask: jax.Array | np.ndarray | None = None, *, model_kwargs: dict[str, tp.Any] | None = None, state: EasyDeLState | None = None, generation_config: GenerationConfig | None = None, shard_inputs: bool | None = None, config_overrides: dict[str, tp.Any] | None = None, return_metadata: bool = False, all_gather: bool = False, **generate_kwargs)` — [`L2251`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L2251) — Convenience wrapper around the compiled generation function.
  - `generate_unified(self, input_ids: jax.Array | np.ndarray | None = None, attention_mask: jax.Array | np.ndarray | None = None, prompts: str | list[str] | None = None, *, model_kwargs: dict[str, tp.Any] | None = None, state: EasyDeLState | None = None, use_esurge: bool | None = None, apply_chat_template: bool = False, generation_config: GenerationConfig | None = None, shard_inputs: bool | None = None, config_overrides: dict[str, tp.Any] | None = None, release_runtime_after_generation: bool = True, clear_esurge_compiled_cache_after_generation: bool = False, all_gather: bool = False, **generate_kwargs)` — [`L2456`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L2456) — Unified generation interface supporting both compiled and eSurge generation. — documented in [easydel-trainers-base_trainer](../../../concepts/easydel-trainers-base_trainer.md)
  - `get_mtime(path)` — [`L4841`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L4841) — Return the modification time of a checkpoint directory.
  - `initialize_trainer_utils(self)` — [`L3980`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L3980) — Initializes various utilities used by the trainer.
  - `is_enable(self)` — [`L1445`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1445) — Check if operations are enabled for this process.
  - `is_process_zero(self)` — [`L1436`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1436) — Check if this is the main process (rank 0).
  - `log_metrics(self, metrics: MetricsType, pbar: BaseProgressBar, step: int, mode: str = "train", *, update_progress: bool = True, log_to_backends: bool = True, force_report: bool = False)` — [`L5598`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L5598) — Log metrics to configured backends and update the progress bar.
  - `log_watchers(self, state: EasyDeLState, step: int)` — [`L5586`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L5586) — Run registered LogWatcher instances and log their metrics.
  - `log_weight_distribution(self, state: EasyDeLState, step: int)` — [`L5574`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L5574) — Log weight distribution statistics.
  - `maybe_benchmark(self, state: EasyDeLState, step: int)` — [`L3714`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L3714) — Optionally run configured lm-eval benchmark suites during training. — documented in [easydel-trainers-base_trainer](../../../concepts/easydel-trainers-base_trainer.md)
  - `maybe_generate(self, state: EasyDeLState, step: int, metrics: MetricsType | None = None)` — [`L3380`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L3380) — Optionally run preview generation to monitor training progress. — documented in [easydel-trainers-base_trainer](../../../concepts/easydel-trainers-base_trainer.md)
  - `maybe_release_generation_runtime(self, results: GenerationResults, *, state: EasyDeLState | None = None, release_runtime_after_generation: bool = True, clear_esurge_compiled_cache_after_generation: bool = True)` — [`L2429`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L2429) — Optionally release generation runtime after materializing outputs.
  - `mesh(self)` — [`L1221`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1221) — Get the device mesh for distributed computation.
  - `model(self)` — [`L1212`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1212) — Get the model instance.
  - `on_step_end(self, state: EasyDeLState, metrics: MetricsType, step: int)` — [`L1818`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1818) — Hook method called at the end of each training step.
  - `on_step_start(self, state: EasyDeLState, step: int)` — [`L1792`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1792) — Hook method called at the start of each training step.
  - `release_generation_runtime(self, *, state: EasyDeLState | None = None, clear_esurge_compiled_cache: bool = False)` — [`L2379`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L2379) — Pause generation runtime while reclaiming rollout KV memory.
  - `save_callback(dest, mesh, meta, s=state)` — [`L4722`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L4722) — Checkpointer callback that serializes trainer state to disk.
  - `save_information(self, output_path: str | ePathLike)` — [`L5018`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L5018) — Save the generated information to a markdown file.
  - `save_pretrained(self, state: EasyDeLState, save_directory: str | None = None, to_torch: bool = False, easystate_to_huggingface_model_kwargs: dict | None = None, torch_save_pretrained_kwargs: dict | None = None)` — [`L5035`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L5035) — Save the model in either EasyDeL state format or PyTorch format.
  - `specs_to_name_sharding(self, tree, mesh=None)` — [`L5131`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L5131) — Convert partition specs to named sharding.
  - `start_evaluation_hook(self)` — [`L5207`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L5207) — Hook called at the start of evaluation.
  - `start_training_hook(self)` — [`L5198`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L5198) — Hook called at the start of training.
  - `to_tf_dataloader(dataset: Dataset | IterableDataset, is_train: bool)` — [`L4482`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L4482) — Converts a Hugging Face Dataset to a TensorFlow dataloader.
  - `training_batch_size(self)` — [`L1242`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1242) — Calculate the effective training batch size.
  - `arguments` — [`L192`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L192) — documented in [easydel-trainers-base_trainer](../../../concepts/easydel-trainers-base_trainer.md)
  - `benchmark_log_table` — [`L1745`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1745)
  - `checkpointer` — [`L4087`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L4087)
  - `eval_is_conversational` — [`L1201`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1201)
  - `generate_function` — [`L1739`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1739)
  - `generate_function_with_model_kwargs` — [`L1740`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1740)
  - `latest_benchmark_results` — [`L1742`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1742)
  - `latest_generation_samples` — [`L1741`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1741)
  - `model_state` — [`L193`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L193) — documented in [easydel-trainers-base_trainer](../../../concepts/easydel-trainers-base_trainer.md)
  - `preview_log_table` — [`L1743`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1743)
  - `processing_class` — [`L312`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L312)
  - `train_is_conversational` — [`L1200`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1200)
  - `training_generation_log_table` — [`L1744`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1744)
- protocol/private: `_RUNTIME_MODEL_OVERRIDE_STATE_ATTRS`[`L196`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L196), `__str__`[`L1776`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1776), `_apply_chat_template`[`L3286`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L3286), `_cleanup_failed_esurge_generation`[`L2667`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L2667), `_eval_dataset_num_examples`[`L1667`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1667), `_eval_dataset_num_examples_exact`[`L1668`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1668), `_eval_dataset_size_source_label`[`L1669`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1669), `_eval_dataset_steps_auto_clamped`[`L1671`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1671), `_eval_dataset_steps_auto_discovered`[`L1670`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1670), `_eval_source`[`L195`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L195), `_generation_rng`[`L1753`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1753), `_hidden_rich_pbar`[`L5566`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L5566), `_preemption_checkpoint_path`[`L267`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L267), `_preview_reasoning_entries`[`L3411`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L3411), `_preview_tool_call_entries`[`L3419`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L3419), `_set_dataset_size_metadata`[`L1320`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1320), `_steps_from_examples`[`L1347`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1347), `_tpu_preemption_sync_available`[`L268`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L268), `_train_dataset_num_examples`[`L1662`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1662), `_train_dataset_num_examples_exact`[`L1663`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1663), `_train_dataset_size_source_label`[`L1664`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1664), `_train_dataset_steps_auto_clamped`[`L1666`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1666), `_train_dataset_steps_auto_discovered`[`L1665`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L1665), `_train_source`[`L194`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L194)
- uses (calls/refs, reference-scoped): [`config`](../infra/base_module.md#EasyDeLBaseModule.config), [`EasyDeLBaseModule`](../infra/base_module.md#EasyDeLBaseModule), [`TaskType`](../infra/factory.md#TaskType), [`EasyDeLState`](../infra/base_state.md#EasyDeLState), [`ShardedDataSource`](../data/core/protocols.md#ShardedDataSource), [`_preprocess_batch_input`](group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer._preprocess_batch_input), [`CAUSAL_LM`](../infra/factory.md#TaskType.CAUSAL_LM), [`_preprocess_batch_input`](proximal_policy_optimization_trainer/ppo_trainer.md#PPOTrainer._preprocess_batch_input), [`LossMetrics`](../infra/loss_utils.md#LossMetrics), [`model`](../infra/base_state.md#EasyDeLState.model), [`generation_config`](../infra/mixins/generation.md#EasyGenerationMixin.generation_config), [`Transform`](../data/transforms/base.md#Transform), [`SamplingParams`](../inference/sampling_params.md#SamplingParams), [`configure_functions`](trainer/trainer.md#Trainer.configure_functions), [`BASE_MODULE`](../infra/factory.md#TaskType.BASE_MODULE), [`TrainingArguments`](training_configurations.md#TrainingArguments), [`flops_per_token`](../infra/base_module.md#EasyDeLBaseModule.flops_per_token), [`Trainer`](trainer/trainer.md#Trainer), [`max_length`](training_configurations.md#TrainingArguments.max_length), [`open_shard`](../data/core/protocols.md#ShardedDataSource.open_shard), [`shard_names`](../data/core/protocols.md#ShardedDataSource.shard_names), [`SEQUENCE_CLASSIFICATION`](../infra/factory.md#TaskType.SEQUENCE_CLASSIFICATION), [`mesh`](../infra/base_module.md#EasyDeLBaseModule.mesh), [`state_shardings`](trainer_protocol.md#BaseTrainerProtocol.state_shardings), [`TrainerConfigureFunctionOutput`](trainer_protocol.md#TrainerConfigureFunctionOutput), [`logger`](base_trainer.md#logger), [`prepare_inputs_for_generation`](../infra/mixins/generation.md#EasyGenerationMixin.prepare_inputs_for_generation), [`generate`](../infra/mixins/generation.md#EasyGenerationMixin.generate), [`get_esurge`](../infra/mixins/generation.md#EasyGenerationMixin.get_esurge), [`_preprocess_batch_input`](sparse_distillation_trainer/sparse_distillation_trainer.md#SparseDistillationTrainer._preprocess_batch_input), [`load_state`](../infra/base_state.md#EasyDeLState.load_state), [`_preprocess_batch_input`](on_policy_distillation_trainer/on_policy_distillation_trainer.md#OnPolicyDistillationTrainer._preprocess_batch_input), [`_preprocess_batch_input`](seq_kd_trainer/seq_kd_trainer.md#SeqKDTrainer._preprocess_batch_input), [`step`](../infra/base_state.md#EasyDeLState.step), [`loss_config`](training_configurations.md#TrainingArguments.loss_config), [`IMAGE_TEXT_TO_TEXT`](../infra/factory.md#TaskType.IMAGE_TEXT_TO_TEXT), [`RequestOutput`](../inference/esurge/esurge_engine.md#RequestOutput), [`gradient_accumulation_steps`](training_configurations.md#TrainingArguments.gradient_accumulation_steps), [`dtype`](../infra/base_module.md#EasyDeLBaseModule.dtype), [`create_grain_collect_function`](trainer/trainer.md#Trainer.create_grain_collect_function)  (+291 more)
- used by: [`_import_structure`](../__init__.md#_import_structure), [`_preprocess_batch_input`](group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer._preprocess_batch_input), [`_preprocess_batch_input`](group_filtered_policy_optimization/gfpo_trainer.md#GFPOTrainer._preprocess_batch_input), [`_preprocess_batch_input`](proximal_policy_optimization_trainer/ppo_trainer.md#PPOTrainer._preprocess_batch_input), [`_preprocess_batch_input`](self_distillation_policy_optimization/sdpo_trainer.md#SDPOTrainer._preprocess_batch_input), [`_train_epoch`](trainer/trainer.md#Trainer._train_epoch), [`_preprocess_batch_input`](agentic_moshpit/agentic_moshpit_trainer.md#AgenticMoshPitTrainer._preprocess_batch_input), [`configure_functions`](distillation_trainer/distillation_trainer.md#DistillationTrainer.configure_functions), [`configure_functions`](direct_preference_optimization_trainer/dpo_trainer.md#DPOTrainer.configure_functions), [`configure_functions`](trainer/trainer.md#Trainer.configure_functions), [`configure_functions`](group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer.configure_functions), [`Trainer`](trainer/trainer.md#Trainer), [`configure_functions`](generalized_knowledge_distillation_trainer/gkd_trainer.md#GKDTrainer.configure_functions), [`_preprocess_batch_input`](xpo_trainer/xpo_trainer.md#XPOTrainer._preprocess_batch_input), [`configure_functions`](on_policy_distillation_trainer/on_policy_distillation_trainer.md#OnPolicyDistillationTrainer.configure_functions), [`_eval_epoch`](trainer/trainer.md#Trainer._eval_epoch), [`_preprocess_batch_input`](nash_md_trainer/nash_md_trainer.md#NashMDTrainer._preprocess_batch_input), [`configure_functions`](odds_ratio_preference_optimization_trainer/orpo_trainer.md#ORPOTrainer.configure_functions), [`configure_functions`](sparse_distillation_trainer/sparse_distillation_trainer.md#SparseDistillationTrainer.configure_functions), [`concatenated_forward`](kto_trainer/kto_trainer.md#KTOTrainer.concatenated_forward), [`configure_functions`](self_distillation_policy_optimization/sdpo_trainer.md#SDPOTrainer.configure_functions), [`configure_functions`](xpo_trainer/xpo_trainer.md#XPOTrainer.configure_functions), [`configure_functions`](proximal_policy_optimization_trainer/ppo_trainer.md#PPOTrainer.configure_functions), [`concatenated_forward`](contrastive_preference_optimization_trainer/cpo_trainer.md#CPOTrainer.concatenated_forward), [`configure_functions`](embedding_trainer/embedding_trainer.md#EmbeddingTrainer.configure_functions), [`configure_functions`](reward_trainer/reward_trainer.md#RewardTrainer.configure_functions), [`concatenated_forward`](binary_classifier_optimization_trainer/bco_trainer.md#BCOTrainer.concatenated_forward), [`_run_training_loop`](trainer/trainer.md#Trainer._run_training_loop), [`configure_functions`](contrastive_preference_optimization_trainer/cpo_trainer.md#CPOTrainer.configure_functions), [`configure_functions`](nash_md_trainer/nash_md_trainer.md#NashMDTrainer.configure_functions), [`_score_auxiliary_rewards`](agentic_moshpit/agentic_moshpit_trainer.md#AgenticMoshPitTrainer._score_auxiliary_rewards), [`_preprocess_batch_input`](sparse_distillation_trainer/sparse_distillation_trainer.md#SparseDistillationTrainer._preprocess_batch_input), [`configure_functions`](group_filtered_policy_optimization/gfpo_trainer.md#GFPOTrainer.configure_functions), [`_preprocess_batch_input`](on_policy_distillation_trainer/on_policy_distillation_trainer.md#OnPolicyDistillationTrainer._preprocess_batch_input), [`_preprocess_batch_input`](seq_kd_trainer/seq_kd_trainer.md#SeqKDTrainer._preprocess_batch_input), [`compute_reference_log_probs`](direct_preference_optimization_trainer/dpo_trainer.md#DPOTrainer.compute_reference_log_probs), [`configure_dataloaders`](direct_preference_optimization_trainer/dpo_trainer.md#DPOTrainer.configure_dataloaders), [`train`](../infra/elarge/model.md#eLargeModel.train), [`input_data_collator_grain`](contrastive_preference_optimization_trainer/cpo_trainer.md#CPOTrainer.input_data_collator_grain), [`_execute_train_step`](trainer/trainer.md#Trainer._execute_train_step)  (+134 more)

### `GenerationResults`  ·  implements/extends NamedTuple
- def: [`easydel/trainers/base_trainer.py:131`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L131)
- doc: Results from unified generation containing both text and token representations.
- signature: `class GenerationResults(NamedTuple):`
- members:
  - `completion_ids` — [`L153`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L153)
  - `completion_mask` — [`L154`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L154)
  - `completion_prompts` — [`L156`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L156)
  - `decoded_prompts` — [`L155`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L155)
  - `generation_results` — [`L149`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L149)
  - `prompt_ids` — [`L150`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L150)
  - `prompt_mask` — [`L151`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L151)
  - `raw_text` — [`L160`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L160)
  - `reasoning` — [`L158`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L158)
  - `sequences` — [`L152`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L152)
  - `text` — [`L157`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L157)
  - `tool_calls` — [`L159`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L159)
- used by: [`generate_unified`](base_trainer.md#BaseTrainer.generate_unified), [`_preprocess_batch_input`](group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer._preprocess_batch_input), [`_preprocess_batch_input`](group_filtered_policy_optimization/gfpo_trainer.md#GFPOTrainer._preprocess_batch_input), [`_preprocess_batch_input`](proximal_policy_optimization_trainer/ppo_trainer.md#PPOTrainer._preprocess_batch_input), [`_preprocess_batch_input`](self_distillation_policy_optimization/sdpo_trainer.md#SDPOTrainer._preprocess_batch_input), [`_preprocess_batch_input`](xpo_trainer/xpo_trainer.md#XPOTrainer._preprocess_batch_input), [`_preprocess_batch_input`](nash_md_trainer/nash_md_trainer.md#NashMDTrainer._preprocess_batch_input), [`maybe_generate`](base_trainer.md#BaseTrainer.maybe_generate), [`_preprocess_batch_input`](sparse_distillation_trainer/sparse_distillation_trainer.md#SparseDistillationTrainer._preprocess_batch_input), [`_preprocess_batch_input`](on_policy_distillation_trainer/on_policy_distillation_trainer.md#OnPolicyDistillationTrainer._preprocess_batch_input), [`_preprocess_batch_input`](seq_kd_trainer/seq_kd_trainer.md#SeqKDTrainer._preprocess_batch_input), [`generate_fn`](agentic_moshpit/agentic_moshpit_trainer.md#AgenticMoshPitTrainer.generate_fn), [`maybe_release_generation_runtime`](base_trainer.md#BaseTrainer.maybe_release_generation_runtime)

### `_ReiterableDataLoader`
- def: [`easydel/trainers/base_trainer.py:106`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L106)
- doc: Small wrapper that recreates a fresh iterator on every `iter(...)` call.
- signature: `class _ReiterableDataLoader:`
- protocol/private: `__init__`[`L109`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L109), `__iter__`[`L113`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L113), `__len__`[`L116`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L116), `_factory`[`L110`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L110), `_length`[`L111`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L111)
- used by: [`_configure_tfds_dataloader`](base_trainer.md#BaseTrainer._configure_tfds_dataloader), [`_configure_grain_dataloader`](base_trainer.md#BaseTrainer._configure_grain_dataloader)

### `_ResolvedStepCount`  ·  implements/extends NamedTuple
- def: [`easydel/trainers/base_trainer.py:122`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L122)
- signature: `class _ResolvedStepCount(NamedTuple):`
- members:
  - `auto_clamped` — [`L128`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L128)
  - `auto_discovered` — [`L127`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L127)
  - `num_examples` — [`L124`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L124)
  - `num_examples_exact` — [`L125`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L125)
  - `source_label` — [`L126`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L126)
  - `steps` — [`L123`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L123)
- used by: [`_resolve_step_count`](base_trainer.md#BaseTrainer._resolve_step_count), [`_configure_tfds_dataloader`](base_trainer.md#BaseTrainer._configure_tfds_dataloader), [`_configure_grain_dataloader`](base_trainer.md#BaseTrainer._configure_grain_dataloader), [`_set_dataset_size_metadata`](base_trainer.md#BaseTrainer._set_dataset_size_metadata)

## Module values
- `DEFAULT_ARGS_JSON_NAME` — [`L103`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L103)
- `log_debug_maybe` — [`L101`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L101)
- `logger` — [`L99`](../../../../../../raw/code/EasyDeL/easydel/trainers/base_trainer.py#L99)


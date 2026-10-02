---
title: 'Module: easydel/trainers/group_relative_policy_optimization/grpo_trainer.py'
type: catalog
provenance: extracted
module: easydel/trainers/group_relative_policy_optimization/grpo_trainer.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.group_relative_policy_optimization.grpo_trainer`/
symbols:
  GRPOTrainer._preprocess_batch_input: GRPOTrainer#_preprocess_batch_input().
  GRPOTrainer.configure_functions: GRPOTrainer#configure_functions().
  GRPOTrainer.arguments: GRPOTrainer#arguments.
  GRPOTrainer: GRPOTrainer#
  GRPOTrainer.__init__: GRPOTrainer#__init__().
  GRPOTrainer.reward_funcs: GRPOTrainer#reward_funcs.
  GRPOTrainer.ref_state: GRPOTrainer#ref_state.
  GRPOTrainer.on_step_end: GRPOTrainer#on_step_end().
  GRPOTrainer.processing_class: GRPOTrainer#processing_class.
  GRPOTrainer.compute_refmodel_logps: GRPOTrainer#compute_refmodel_logps.
  GRPOTrainer.ref_logps_chunk_size: GRPOTrainer#ref_logps_chunk_size.
  GRPOTrainer.scale_rewards: GRPOTrainer#scale_rewards.
  GRPOTrainer.apply_fn: GRPOTrainer#apply_fn().
  GRPOTrainer._compute_refmodel_logps: GRPOTrainer#_compute_refmodel_logps().
  GRPOTrainer.num_generations: GRPOTrainer#num_generations.
  GRPOTrainer._get_preprocess_transform: GRPOTrainer#_get_preprocess_transform().
  GRPOTrainer.create_grain_collect_function: GRPOTrainer#create_grain_collect_function().
  GRPOTrainer.create_tfds_collect_function: GRPOTrainer#create_tfds_collect_function().
  GRPOTrainer.importance_sampling_level: GRPOTrainer#importance_sampling_level.
  GRPOTrainer.step_sharding: GRPOTrainer#step_sharding().
  GRPOTrainer._is_pretokenized: GRPOTrainer#_is_pretokenized().
  GRPOTrainer.reward_weights: GRPOTrainer#reward_weights.
  RewardFunc: RewardFunc.
  GRPOTrainer.reward_processing_classes: GRPOTrainer#reward_processing_classes.
  GRPOTrainer.loss_type: GRPOTrainer#loss_type.
  GRPOTrainer.epsilon: GRPOTrainer#epsilon.
  GRPOTrainer.epsilon_high: GRPOTrainer#epsilon_high.
  GRPOTrainer.delta: GRPOTrainer#delta.
  GRPOTrainer.top_entropy_quantile: GRPOTrainer#top_entropy_quantile.
  GRPOTrainer.reward_func_names: GRPOTrainer#reward_func_names.
  GRPOTrainer.padding_value: GRPOTrainer#padding_value.
  delete_tree: delete_tree().
  GRPOTrainer.truncation_mode: GRPOTrainer#truncation_mode.
  GRPOTrainer.data_tokenize_fn: GRPOTrainer#data_tokenize_fn.
  _fileaf: _fileaf().
  logger: logger.
---
# Module: [`easydel/trainers/group_relative_policy_optimization/grpo_trainer.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_trainer.py)

## Classes
### `GRPOTrainer`  ·  implements/extends Trainer
- def: [`easydel/trainers/group_relative_policy_optimization/grpo_trainer.py:83`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_trainer.py#L83)
- doc: Group Relative Policy Optimization trainer for RLHF.
- signature: `class GRPOTrainer(Trainer):`
- members:
  - `_get_preprocess_transform(self)` — [`L257`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_trainer.py#L257) — Get GRPO preprocessing transform for ShardedDataSource.
  - `_is_pretokenized(self)` — [`L269`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_trainer.py#L269) — Check if dataset already has tokenized fields.
  - `apply_fn(gd, gs, gt, batch)` — [`L200`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_trainer.py#L200)
  - `configure_functions(self)` — [`L312`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_trainer.py#L312) — Configures and JIT-compiles the training and evaluation step functions.
  - `create_grain_collect_function(self, max_sequence_length: int, truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end")` — [`L279`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_trainer.py#L279) — Create data collator for Grain data loading.
  - `create_tfds_collect_function(self, max_sequence_length: int, truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end")` — [`L292`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_trainer.py#L292) — Create data collator for TFDS data loading.
  - `on_step_end(self, state: EasyDeLState, metrics: MetricsType, step: int)` — [`L733`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_trainer.py#L733) — Post-step hook that optionally synchronizes the reference model.
  - `step_sharding(self)` — [`L306`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_trainer.py#L306)
  - `arguments` — [`L122`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_trainer.py#L122)
  - `compute_refmodel_logps` — [`L414`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_trainer.py#L414)
  - `data_tokenize_fn` — [`L246`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_trainer.py#L246)
  - `delta` — [`L147`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_trainer.py#L147)
  - `epsilon` — [`L145`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_trainer.py#L145)
  - `epsilon_high` — [`L146`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_trainer.py#L146)
  - `importance_sampling_level` — [`L148`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_trainer.py#L148)
  - `loss_type` — [`L144`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_trainer.py#L144)
  - `num_generations` — [`L240`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_trainer.py#L240)
  - `padding_value` — [`L171`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_trainer.py#L171)
  - `processing_class` — [`L167`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_trainer.py#L167)
  - `ref_logps_chunk_size` — [`L155`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_trainer.py#L155)
  - `ref_state` — [`L160`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_trainer.py#L160)
  - `reward_func_names` — [`L238`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_trainer.py#L238)
  - `reward_funcs` — [`L174`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_trainer.py#L174)
  - `reward_processing_classes` — [`L123`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_trainer.py#L123)
  - `reward_weights` — [`L234`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_trainer.py#L234)
  - `scale_rewards` — [`L151`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_trainer.py#L151)
  - `top_entropy_quantile` — [`L154`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_trainer.py#L154)
  - `truncation_mode` — [`L143`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_trainer.py#L143)
- protocol/private: `__init__`[`L125`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_trainer.py#L125), `_compute_refmodel_logps`[`L392`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_trainer.py#L392), `_preprocess_batch_input`[`L440`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_trainer.py#L440)
- uses (calls/refs, reference-scoped): [`config`](../../infra/base_module.md#EasyDeLBaseModule.config), [`EasyDeLBaseModule`](../../infra/base_module.md#EasyDeLBaseModule), [`EasyDeLState`](../../infra/base_state.md#EasyDeLState), [`ShardedDataSource`](../../data/core/protocols.md#ShardedDataSource), [`generate_unified`](../base_trainer.md#BaseTrainer.generate_unified), [`_preprocess_batch_input`](../group_filtered_policy_optimization/gfpo_trainer.md#GFPOTrainer._preprocess_batch_input), [`_preprocess_batch_input`](../self_distillation_policy_optimization/sdpo_trainer.md#SDPOTrainer._preprocess_batch_input), [`model`](../../infra/base_state.md#EasyDeLState.model), [`Registry`](../../utils/registery.md#Registry), [`_preprocess_batch_input`](../agentic_moshpit/agentic_moshpit_trainer.md#AgenticMoshPitTrainer._preprocess_batch_input), [`Trainer`](../trainer/trainer.md#Trainer), [`ProcessingClassType`](../../infra/utils.md#ProcessingClassType), [`_all_gather`](../base_trainer.md#BaseTrainer._all_gather), [`max_length`](../training_configurations.md#TrainingArguments.max_length), [`open_shard`](../../data/core/protocols.md#ShardedDataSource.open_shard), `pad_token_id`, [`_preprocess_batch_input`](../xpo_trainer/xpo_trainer.md#XPOTrainer._preprocess_batch_input), [`shard_names`](../../data/core/protocols.md#ShardedDataSource.shard_names), [`_preprocess_batch_input`](../nash_md_trainer/nash_md_trainer.md#NashMDTrainer._preprocess_batch_input), [`_train_source`](../base_trainer.md#BaseTrainer._train_source), [`configure_functions`](../self_distillation_policy_optimization/sdpo_trainer.md#SDPOTrainer.configure_functions), [`configure_functions`](../xpo_trainer/xpo_trainer.md#XPOTrainer.configure_functions), [`mesh`](../../infra/base_module.md#EasyDeLBaseModule.mesh), [`capture_time`](../../utils/helpers.md#capture_time), [`register`](../../utils/registery.md#Registry.register), [`state_shardings`](../trainer_protocol.md#BaseTrainerProtocol.state_shardings), [`TrainerConfigureFunctionOutput`](../trainer_protocol.md#TrainerConfigureFunctionOutput), [`__init__`](../base_trainer.md#BaseTrainer.__init__), [`graphstate`](../../infra/base_state.md#EasyDeLState.graphstate), [`step_partition_spec`](../training_configurations.md#TrainingArguments.step_partition_spec), [`GRPOConfig`](grpo_config.md#GRPOConfig), [`configure_functions`](../nash_md_trainer/nash_md_trainer.md#NashMDTrainer.configure_functions), [`to_state`](../../infra/base_module.md#EasyDeLBaseModule.to_state), [`configure_functions`](../group_filtered_policy_optimization/gfpo_trainer.md#GFPOTrainer.configure_functions), [`model`](../base_trainer.md#BaseTrainer.model), [`_log_training_generations_to_wandb`](../base_trainer.md#BaseTrainer._log_training_generations_to_wandb), [`loss_config`](../training_configurations.md#TrainingArguments.loss_config), [`resolve_straight_through_emulator`](../training_utils.md#resolve_straight_through_emulator), [`gradient_accumulation_steps`](../training_configurations.md#TrainingArguments.gradient_accumulation_steps), [`get_per_token_logps`](_fn.md#get_per_token_logps)  (+87 more)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`_preprocess_batch_input`](../group_filtered_policy_optimization/gfpo_trainer.md#GFPOTrainer._preprocess_batch_input), [`_preprocess_batch_input`](../self_distillation_policy_optimization/sdpo_trainer.md#SDPOTrainer._preprocess_batch_input), [`_preprocess_batch_input`](../agentic_moshpit/agentic_moshpit_trainer.md#AgenticMoshPitTrainer._preprocess_batch_input), [`configure_functions`](../trainer/trainer.md#Trainer.configure_functions), [`Trainer`](../trainer/trainer.md#Trainer), [`_preprocess_batch_input`](../xpo_trainer/xpo_trainer.md#XPOTrainer._preprocess_batch_input), [`_preprocess_batch_input`](../nash_md_trainer/nash_md_trainer.md#NashMDTrainer._preprocess_batch_input), [`configure_functions`](../self_distillation_policy_optimization/sdpo_trainer.md#SDPOTrainer.configure_functions), [`_score_auxiliary_rewards`](../agentic_moshpit/agentic_moshpit_trainer.md#AgenticMoshPitTrainer._score_auxiliary_rewards), [`configure_functions`](../group_filtered_policy_optimization/gfpo_trainer.md#GFPOTrainer.configure_functions), [`_is_pretokenized`](../base_trainer.md#BaseTrainer._is_pretokenized), [`_get_preprocess_transform`](../base_trainer.md#BaseTrainer._get_preprocess_transform), [`__init__`](../rlvr_trainer/rlvr_trainer.md#RLVRTrainer.__init__), [`create_grain_collect_function`](../trainer/trainer.md#Trainer.create_grain_collect_function), [`create_tfds_collect_function`](../trainer/trainer.md#Trainer.create_tfds_collect_function), [`_score_rewards`](../xpo_trainer/xpo_trainer.md#XPOTrainer._score_rewards), [`_preprocess_batch_input`](../base_trainer.md#BaseTrainer._preprocess_batch_input), [`_score_rewards`](../nash_md_trainer/nash_md_trainer.md#NashMDTrainer._score_rewards), [`wrapped_env_factory`](../agentic_moshpit/agentic_moshpit_trainer.md#AgenticMoshPitTrainer.wrapped_env_factory), [`_tool_call_parser`](../agentic_moshpit/agentic_moshpit_trainer.md#AgenticMoshPitTrainer._tool_call_parser), [`__init__`](../group_sequence_policy_optimization/gspo_trainer.md#GSPOTrainer.__init__), [`_rollout_manager`](../agentic_moshpit/agentic_moshpit_trainer.md#AgenticMoshPitTrainer._rollout_manager), [`__init__`](../nash_md_trainer/nash_md_trainer.md#NashMDTrainer.__init__), [`on_step_end`](../base_trainer.md#BaseTrainer.on_step_end), [`teacher_prompt_length`](../self_distillation_policy_optimization/sdpo_trainer.md#SDPOTrainer.teacher_prompt_length), [`AgenticMoshPitTrainer`](../agentic_moshpit/agentic_moshpit_trainer.md#AgenticMoshPitTrainer), [`NashMDTrainer`](../nash_md_trainer/nash_md_trainer.md#NashMDTrainer), [`SDPOTrainer`](../self_distillation_policy_optimization/sdpo_trainer.md#SDPOTrainer), [`XPOTrainer`](../xpo_trainer/xpo_trainer.md#XPOTrainer), [`create_grain_collect_function`](../agentic_moshpit/agentic_moshpit_trainer.md#AgenticMoshPitTrainer.create_grain_collect_function), [`create_tfds_collect_function`](../agentic_moshpit/agentic_moshpit_trainer.md#AgenticMoshPitTrainer.create_tfds_collect_function), [`_get_reward_processing_classes`](../xpo_trainer/xpo_trainer.md#XPOTrainer._get_reward_processing_classes), [`_tokenize_feedback_separators`](../self_distillation_policy_optimization/sdpo_trainer.md#SDPOTrainer._tokenize_feedback_separators), [`loss_type_id`](../xpo_trainer/xpo_trainer.md#XPOTrainer.loss_type_id), [`GFPOTrainer`](../group_filtered_policy_optimization/gfpo_trainer.md#GFPOTrainer), [`GSPOTrainer`](../group_sequence_policy_optimization/gspo_trainer.md#GSPOTrainer), [`RLVRTrainer`](../rlvr_trainer/rlvr_trainer.md#RLVRTrainer), [`group_filter_func`](../group_filtered_policy_optimization/gfpo_trainer.md#GFPOTrainer.group_filter_func)

## Functions
- `_fileaf(x)` — [`L70`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_trainer.py#L70)
- `delete_tree(pytree)` — [`L74`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_trainer.py#L74)

## Module values
- `RewardFunc` — [`L67`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_trainer.py#L67)
- `logger` — [`L66`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_trainer.py#L66)


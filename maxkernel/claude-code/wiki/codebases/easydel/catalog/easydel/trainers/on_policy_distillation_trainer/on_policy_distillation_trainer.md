---
title: 'Module: easydel/trainers/on_policy_distillation_trainer/on_policy_distillation_trainer.py'
type: catalog
provenance: extracted
module: easydel/trainers/on_policy_distillation_trainer/on_policy_distillation_trainer.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.on_policy_distillation_trainer.on_policy_distillation_trainer`/
symbols:
  OnPolicyDistillationTrainer.configure_functions: OnPolicyDistillationTrainer#configure_functions().
  OnPolicyDistillationTrainer._preprocess_batch_input: OnPolicyDistillationTrainer#_preprocess_batch_input().
  OnPolicyDistillationTrainer.arguments: OnPolicyDistillationTrainer#arguments.
  OnPolicyDistillationTrainer.__init__: OnPolicyDistillationTrainer#__init__().
  OnPolicyDistillationTrainer._get_preprocess_transform: OnPolicyDistillationTrainer#_get_preprocess_transform().
  OnPolicyDistillationTrainer.create_grain_collect_function: OnPolicyDistillationTrainer#create_grain_collect_function().
  OnPolicyDistillationTrainer.create_tfds_collect_function: OnPolicyDistillationTrainer#create_tfds_collect_function().
  OnPolicyDistillationTrainer.teacher_state: OnPolicyDistillationTrainer#teacher_state.
  OnPolicyDistillationTrainer._is_pretokenized: OnPolicyDistillationTrainer#_is_pretokenized().
  OnPolicyDistillationTrainer: OnPolicyDistillationTrainer#
  logger: logger.
  OnPolicyDistillationTrainer.padding_value: OnPolicyDistillationTrainer#padding_value.
  OnPolicyDistillationTrainer._train_shared_fn_extra_args: OnPolicyDistillationTrainer#_train_shared_fn_extra_args().
  OnPolicyDistillationTrainer._eval_shared_fn_extra_args: OnPolicyDistillationTrainer#_eval_shared_fn_extra_args().
  OnPolicyDistillationTrainer.processing_class: OnPolicyDistillationTrainer#processing_class.
---
# Module: [`easydel/trainers/on_policy_distillation_trainer/on_policy_distillation_trainer.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/on_policy_distillation_trainer/on_policy_distillation_trainer.py)

## Classes
### `OnPolicyDistillationTrainer`  ·  implements/extends Trainer
- def: [`easydel/trainers/on_policy_distillation_trainer/on_policy_distillation_trainer.py:45`](../../../../../../../raw/code/EasyDeL/easydel/trainers/on_policy_distillation_trainer/on_policy_distillation_trainer.py#L45)
- doc: On-policy knowledge distillation trainer.
- signature: `class OnPolicyDistillationTrainer(Trainer):`
- members:
  - `_get_preprocess_transform(self)` — [`L126`](../../../../../../../raw/code/EasyDeL/easydel/trainers/on_policy_distillation_trainer/on_policy_distillation_trainer.py#L126) — Get preprocessing transform for prompt-only datasets.
  - `_is_pretokenized(self)` — [`L136`](../../../../../../../raw/code/EasyDeL/easydel/trainers/on_policy_distillation_trainer/on_policy_distillation_trainer.py#L136) — Check whether the source already yields token IDs.
  - `_preprocess_batch_input(self, state: EasyDeLState, batch: dict[str, tp.Any], is_train: bool)` — [`L242`](../../../../../../../raw/code/EasyDeL/easydel/trainers/on_policy_distillation_trainer/on_policy_distillation_trainer.py#L242) — Generate completions from prompts and prepare the distillation batch.
  - `configure_functions(self)` — [`L172`](../../../../../../../raw/code/EasyDeL/easydel/trainers/on_policy_distillation_trainer/on_policy_distillation_trainer.py#L172) — Configure and JIT-compile the training and evaluation step functions.
  - `create_grain_collect_function(self, max_sequence_length: int, truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end")` — [`L146`](../../../../../../../raw/code/EasyDeL/easydel/trainers/on_policy_distillation_trainer/on_policy_distillation_trainer.py#L146) — Create Grain data collator for prompt-only batches.
  - `create_tfds_collect_function(self, max_sequence_length: int, truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end")` — [`L159`](../../../../../../../raw/code/EasyDeL/easydel/trainers/on_policy_distillation_trainer/on_policy_distillation_trainer.py#L159) — Create TFDS data collator for prompt-only batches.
  - `arguments` — [`L83`](../../../../../../../raw/code/EasyDeL/easydel/trainers/on_policy_distillation_trainer/on_policy_distillation_trainer.py#L83)
  - `padding_value` — [`L115`](../../../../../../../raw/code/EasyDeL/easydel/trainers/on_policy_distillation_trainer/on_policy_distillation_trainer.py#L115)
  - `processing_class` — [`L110`](../../../../../../../raw/code/EasyDeL/easydel/trainers/on_policy_distillation_trainer/on_policy_distillation_trainer.py#L110)
  - `teacher_state` — [`L82`](../../../../../../../raw/code/EasyDeL/easydel/trainers/on_policy_distillation_trainer/on_policy_distillation_trainer.py#L82)
- protocol/private: `__init__`[`L85`](../../../../../../../raw/code/EasyDeL/easydel/trainers/on_policy_distillation_trainer/on_policy_distillation_trainer.py#L85), `_eval_shared_fn_extra_args`[`L344`](../../../../../../../raw/code/EasyDeL/easydel/trainers/on_policy_distillation_trainer/on_policy_distillation_trainer.py#L344), `_train_shared_fn_extra_args`[`L340`](../../../../../../../raw/code/EasyDeL/easydel/trainers/on_policy_distillation_trainer/on_policy_distillation_trainer.py#L340)
- uses (calls/refs, reference-scoped): [`EasyDeLBaseModule`](../../infra/base_module.md#EasyDeLBaseModule), [`EasyDeLState`](../../infra/base_state.md#EasyDeLState), [`ShardedDataSource`](../../data/core/protocols.md#ShardedDataSource), [`generate_unified`](../base_trainer.md#BaseTrainer.generate_unified), [`model`](../../infra/base_state.md#EasyDeLState.model), [`flops_per_token`](../../infra/base_module.md#EasyDeLBaseModule.flops_per_token), [`Trainer`](../trainer/trainer.md#Trainer), [`ProcessingClassType`](../../infra/utils.md#ProcessingClassType), [`_all_gather`](../base_trainer.md#BaseTrainer._all_gather), [`open_shard`](../../data/core/protocols.md#ShardedDataSource.open_shard), [`shard_names`](../../data/core/protocols.md#ShardedDataSource.shard_names), [`_train_source`](../base_trainer.md#BaseTrainer._train_source), [`mesh`](../../infra/base_module.md#EasyDeLBaseModule.mesh), [`capture_time`](../../utils/helpers.md#capture_time), [`state_shardings`](../trainer_protocol.md#BaseTrainerProtocol.state_shardings), [`TrainerConfigureFunctionOutput`](../trainer_protocol.md#TrainerConfigureFunctionOutput), [`__init__`](../base_trainer.md#BaseTrainer.__init__), [`step_partition_spec`](../training_configurations.md#TrainingArguments.step_partition_spec), [`to_state`](../../infra/base_module.md#EasyDeLBaseModule.to_state), [`model`](../base_trainer.md#BaseTrainer.model), [`_log_training_generations_to_wandb`](../base_trainer.md#BaseTrainer._log_training_generations_to_wandb), [`loss_config`](../training_configurations.md#TrainingArguments.loss_config), [`resolve_straight_through_emulator`](../training_utils.md#resolve_straight_through_emulator), [`gradient_accumulation_steps`](../training_configurations.md#TrainingArguments.gradient_accumulation_steps), [`scheduler`](../trainer_protocol.md#BaseTrainerProtocol.scheduler), [`quantization_mode`](../training_configurations.md#TrainingArguments.quantization_mode), [`quantization_bits`](../training_configurations.md#TrainingArguments.quantization_bits), [`quantization_group_size`](../training_configurations.md#TrainingArguments.quantization_group_size), [`get_streaming_checkpointer`](../training_configurations.md#TrainingArguments.get_streaming_checkpointer), [`on_policy_distillation_step`](_fn.md#on_policy_distillation_step), [`_train_shared_fn_static_args`](../base_trainer.md#BaseTrainer._train_shared_fn_static_args), [`_eval_shared_fn_static_args`](../base_trainer.md#BaseTrainer._eval_shared_fn_static_args), [`GRPOPreprocessTransform`](../prompt_transforms.md#GRPOPreprocessTransform), [`shardings`](../../infra/base_state.md#EasyDeLState.shardings), [`ensure_checkpoint_path`](../training_configurations.md#TrainingArguments.ensure_checkpoint_path), [`OnPolicyDistillationConfig`](on_policy_distillation_config.md#OnPolicyDistillationConfig), [`straight_through_emulator`](../training_configurations.md#TrainingArguments.straight_through_emulator), [`tensor_straight_through`](../training_configurations.md#TrainingArguments.tensor_straight_through), [`_purify_batch`](../base_trainer.md#BaseTrainer._purify_batch), [`completion_ids`](../base_trainer.md#GenerationResults.completion_ids)  (+17 more)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`configure_functions`](../trainer/trainer.md#Trainer.configure_functions), [`Trainer`](../trainer/trainer.md#Trainer), [`_is_pretokenized`](../base_trainer.md#BaseTrainer._is_pretokenized), [`_get_preprocess_transform`](../base_trainer.md#BaseTrainer._get_preprocess_transform), [`create_grain_collect_function`](../trainer/trainer.md#Trainer.create_grain_collect_function), [`create_tfds_collect_function`](../trainer/trainer.md#Trainer.create_tfds_collect_function), [`_preprocess_batch_input`](../base_trainer.md#BaseTrainer._preprocess_batch_input), [`_train_shared_fn_extra_args`](../base_trainer.md#BaseTrainer._train_shared_fn_extra_args), [`_eval_shared_fn_extra_args`](../base_trainer.md#BaseTrainer._eval_shared_fn_extra_args)

## Module values
- `logger` — [`L41`](../../../../../../../raw/code/EasyDeL/easydel/trainers/on_policy_distillation_trainer/on_policy_distillation_trainer.py#L41)


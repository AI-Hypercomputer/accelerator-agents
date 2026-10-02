---
title: 'Module: easydel/trainers/sparse_distillation_trainer/sparse_distillation_trainer.py'
type: catalog
provenance: extracted
module: easydel/trainers/sparse_distillation_trainer/sparse_distillation_trainer.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.sparse_distillation_trainer.sparse_distillation_trainer`/
symbols:
  SparseDistillationTrainer.configure_functions: SparseDistillationTrainer#configure_functions().
  SparseDistillationTrainer._preprocess_batch_input: SparseDistillationTrainer#_preprocess_batch_input().
  SparseDistillationTrainer.arguments: SparseDistillationTrainer#arguments.
  SparseDistillationTrainer.__init__: SparseDistillationTrainer#__init__().
  SparseDistillationTrainer._get_preprocess_transform: SparseDistillationTrainer#_get_preprocess_transform().
  SparseDistillationTrainer.create_grain_collect_function: SparseDistillationTrainer#create_grain_collect_function().
  SparseDistillationTrainer.create_tfds_collect_function: SparseDistillationTrainer#create_tfds_collect_function().
  SparseDistillationTrainer._is_pretokenized: SparseDistillationTrainer#_is_pretokenized().
  SparseDistillationTrainer.teacher_state: SparseDistillationTrainer#teacher_state.
  SparseDistillationTrainer: SparseDistillationTrainer#
  SparseDistillationTrainer.teacher_fn: SparseDistillationTrainer#teacher_fn.
  logger: logger.
  SparseDistillationTrainer.padding_value: SparseDistillationTrainer#padding_value.
  SparseTeacherFn: SparseTeacherFn.
  SparseDistillationTrainer.processing_class: SparseDistillationTrainer#processing_class.
---
# Module: [`easydel/trainers/sparse_distillation_trainer/sparse_distillation_trainer.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/sparse_distillation_trainer/sparse_distillation_trainer.py)

## Classes
### `SparseDistillationTrainer`  ·  implements/extends Trainer
- def: [`easydel/trainers/sparse_distillation_trainer/sparse_distillation_trainer.py:63`](../../../../../../../raw/code/EasyDeL/easydel/trainers/sparse_distillation_trainer/sparse_distillation_trainer.py#L63)
- doc: Sparse (gray-box) knowledge distillation trainer.
- signature: `class SparseDistillationTrainer(Trainer):`
- members:
  - `_get_preprocess_transform(self)` — [`L155`](../../../../../../../raw/code/EasyDeL/easydel/trainers/sparse_distillation_trainer/sparse_distillation_trainer.py#L155) — Get preprocessing transform for prompt-only datasets.
  - `_is_pretokenized(self)` — [`L165`](../../../../../../../raw/code/EasyDeL/easydel/trainers/sparse_distillation_trainer/sparse_distillation_trainer.py#L165) — Check whether the source already yields token IDs.
  - `_preprocess_batch_input(self, state: EasyDeLState, batch: dict[str, tp.Any], is_train: bool)` — [`L265`](../../../../../../../raw/code/EasyDeL/easydel/trainers/sparse_distillation_trainer/sparse_distillation_trainer.py#L265) — Generate completions and score with teacher to get top-k logprobs.
  - `configure_functions(self)` — [`L201`](../../../../../../../raw/code/EasyDeL/easydel/trainers/sparse_distillation_trainer/sparse_distillation_trainer.py#L201) — Configure and JIT-compile training and evaluation step functions.
  - `create_grain_collect_function(self, max_sequence_length: int, truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end")` — [`L175`](../../../../../../../raw/code/EasyDeL/easydel/trainers/sparse_distillation_trainer/sparse_distillation_trainer.py#L175) — Create Grain data collator for prompt-only batches.
  - `create_tfds_collect_function(self, max_sequence_length: int, truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end")` — [`L188`](../../../../../../../raw/code/EasyDeL/easydel/trainers/sparse_distillation_trainer/sparse_distillation_trainer.py#L188) — Create TFDS data collator for prompt-only batches.
  - `arguments` — [`L106`](../../../../../../../raw/code/EasyDeL/easydel/trainers/sparse_distillation_trainer/sparse_distillation_trainer.py#L106)
  - `padding_value` — [`L144`](../../../../../../../raw/code/EasyDeL/easydel/trainers/sparse_distillation_trainer/sparse_distillation_trainer.py#L144)
  - `processing_class` — [`L139`](../../../../../../../raw/code/EasyDeL/easydel/trainers/sparse_distillation_trainer/sparse_distillation_trainer.py#L139)
  - `teacher_fn` — [`L105`](../../../../../../../raw/code/EasyDeL/easydel/trainers/sparse_distillation_trainer/sparse_distillation_trainer.py#L105)
  - `teacher_state` — [`L104`](../../../../../../../raw/code/EasyDeL/easydel/trainers/sparse_distillation_trainer/sparse_distillation_trainer.py#L104)
- protocol/private: `__init__`[`L108`](../../../../../../../raw/code/EasyDeL/easydel/trainers/sparse_distillation_trainer/sparse_distillation_trainer.py#L108)
- uses (calls/refs, reference-scoped): [`EasyDeLBaseModule`](../../infra/base_module.md#EasyDeLBaseModule), [`EasyDeLState`](../../infra/base_state.md#EasyDeLState), [`ShardedDataSource`](../../data/core/protocols.md#ShardedDataSource), [`generate_unified`](../base_trainer.md#BaseTrainer.generate_unified), [`model`](../../infra/base_state.md#EasyDeLState.model), [`flops_per_token`](../../infra/base_module.md#EasyDeLBaseModule.flops_per_token), [`Trainer`](../trainer/trainer.md#Trainer), [`ProcessingClassType`](../../infra/utils.md#ProcessingClassType), [`_all_gather`](../base_trainer.md#BaseTrainer._all_gather), [`open_shard`](../../data/core/protocols.md#ShardedDataSource.open_shard), [`shard_names`](../../data/core/protocols.md#ShardedDataSource.shard_names), [`_train_source`](../base_trainer.md#BaseTrainer._train_source), [`mesh`](../../infra/base_module.md#EasyDeLBaseModule.mesh), [`capture_time`](../../utils/helpers.md#capture_time), [`state_shardings`](../trainer_protocol.md#BaseTrainerProtocol.state_shardings), [`TrainerConfigureFunctionOutput`](../trainer_protocol.md#TrainerConfigureFunctionOutput), [`__init__`](../base_trainer.md#BaseTrainer.__init__), [`step_partition_spec`](../training_configurations.md#TrainingArguments.step_partition_spec), [`to_state`](../../infra/base_module.md#EasyDeLBaseModule.to_state), [`model`](../base_trainer.md#BaseTrainer.model), [`_log_training_generations_to_wandb`](../base_trainer.md#BaseTrainer._log_training_generations_to_wandb), [`loss_config`](../training_configurations.md#TrainingArguments.loss_config), [`resolve_straight_through_emulator`](../training_utils.md#resolve_straight_through_emulator), [`gradient_accumulation_steps`](../training_configurations.md#TrainingArguments.gradient_accumulation_steps), [`scheduler`](../trainer_protocol.md#BaseTrainerProtocol.scheduler), [`filter_kwargs_for_callable`](../training_utils.md#filter_kwargs_for_callable), [`quantization_mode`](../training_configurations.md#TrainingArguments.quantization_mode), [`quantization_bits`](../training_configurations.md#TrainingArguments.quantization_bits), [`quantization_group_size`](../training_configurations.md#TrainingArguments.quantization_group_size), [`get_streaming_checkpointer`](../training_configurations.md#TrainingArguments.get_streaming_checkpointer), [`sanitize_model_call_kwargs`](../training_utils.md#sanitize_model_call_kwargs), [`sparse_distillation_step`](_fn.md#sparse_distillation_step), [`_train_shared_fn_static_args`](../base_trainer.md#BaseTrainer._train_shared_fn_static_args), [`_eval_shared_fn_static_args`](../base_trainer.md#BaseTrainer._eval_shared_fn_static_args), [`GRPOPreprocessTransform`](../prompt_transforms.md#GRPOPreprocessTransform), [`ensure_checkpoint_path`](../training_configurations.md#TrainingArguments.ensure_checkpoint_path), [`SparseDistillationConfig`](sparse_distillation_config.md#SparseDistillationConfig), [`straight_through_emulator`](../training_configurations.md#TrainingArguments.straight_through_emulator), [`tensor_straight_through`](../training_configurations.md#TrainingArguments.tensor_straight_through), [`_purify_batch`](../base_trainer.md#BaseTrainer._purify_batch)  (+13 more)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`configure_functions`](../trainer/trainer.md#Trainer.configure_functions), [`Trainer`](../trainer/trainer.md#Trainer), [`_is_pretokenized`](../base_trainer.md#BaseTrainer._is_pretokenized), [`_get_preprocess_transform`](../base_trainer.md#BaseTrainer._get_preprocess_transform), [`create_grain_collect_function`](../trainer/trainer.md#Trainer.create_grain_collect_function), [`create_tfds_collect_function`](../trainer/trainer.md#Trainer.create_tfds_collect_function), [`_preprocess_batch_input`](../base_trainer.md#BaseTrainer._preprocess_batch_input)

## Module values
- `SparseTeacherFn` — [`L54`](../../../../../../../raw/code/EasyDeL/easydel/trainers/sparse_distillation_trainer/sparse_distillation_trainer.py#L54)
- `logger` — [`L59`](../../../../../../../raw/code/EasyDeL/easydel/trainers/sparse_distillation_trainer/sparse_distillation_trainer.py#L59)


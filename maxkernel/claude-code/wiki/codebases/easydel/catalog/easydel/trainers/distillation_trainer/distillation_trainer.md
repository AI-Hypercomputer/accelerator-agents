---
title: 'Module: easydel/trainers/distillation_trainer/distillation_trainer.py'
type: catalog
provenance: extracted
module: easydel/trainers/distillation_trainer/distillation_trainer.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.distillation_trainer.distillation_trainer`/
symbols:
  DistillationTrainer.configure_functions: DistillationTrainer#configure_functions().
  DistillationTrainer.arguments: DistillationTrainer#arguments.
  DistillationTrainer.__init__: DistillationTrainer#__init__().
  DistillationTrainer._get_preprocess_transform: DistillationTrainer#_get_preprocess_transform().
  DistillationTrainer.teacher_state: DistillationTrainer#teacher_state.
  DistillationTrainer._is_pretokenized: DistillationTrainer#_is_pretokenized().
  DistillationTrainer: DistillationTrainer#
  logger: logger.
  DistillationTrainer._preprocess_batch_input: DistillationTrainer#_preprocess_batch_input().
  DistillationTrainer._train_shared_fn_extra_args: DistillationTrainer#_train_shared_fn_extra_args().
  DistillationTrainer._eval_shared_fn_extra_args: DistillationTrainer#_eval_shared_fn_extra_args().
---
# Module: [`easydel/trainers/distillation_trainer/distillation_trainer.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/distillation_trainer/distillation_trainer.py)

## Classes
### `DistillationTrainer`  ·  implements/extends Trainer
- def: [`easydel/trainers/distillation_trainer/distillation_trainer.py:43`](../../../../../../../raw/code/EasyDeL/easydel/trainers/distillation_trainer/distillation_trainer.py#L43)
- doc: Knowledge distillation trainer for model compression.
- signature: `class DistillationTrainer(Trainer):`
- members:
  - `_get_preprocess_transform(self)` — [`L211`](../../../../../../../raw/code/EasyDeL/easydel/trainers/distillation_trainer/distillation_trainer.py#L211) — Tokenize raw text examples for distillation when needed.
  - `_is_pretokenized(self)` — [`L227`](../../../../../../../raw/code/EasyDeL/easydel/trainers/distillation_trainer/distillation_trainer.py#L227) — Check whether the source already yields token IDs.
  - `_preprocess_batch_input(self, state: EasyDeLState, batch: dict[str, tp.Any], is_train: bool)` — [`L237`](../../../../../../../raw/code/EasyDeL/easydel/trainers/distillation_trainer/distillation_trainer.py#L237) — Normalize completion masks/labels for mixed SFT + pretrain distillation batches.
  - `configure_functions(self)` — [`L120`](../../../../../../../raw/code/EasyDeL/easydel/trainers/distillation_trainer/distillation_trainer.py#L120) — Configures and JIT-compiles the training and evaluation step functions.
  - `arguments` — [`L82`](../../../../../../../raw/code/EasyDeL/easydel/trainers/distillation_trainer/distillation_trainer.py#L82)
  - `teacher_state` — [`L81`](../../../../../../../raw/code/EasyDeL/easydel/trainers/distillation_trainer/distillation_trainer.py#L81)
- protocol/private: `__init__`[`L84`](../../../../../../../raw/code/EasyDeL/easydel/trainers/distillation_trainer/distillation_trainer.py#L84), `_eval_shared_fn_extra_args`[`L287`](../../../../../../../raw/code/EasyDeL/easydel/trainers/distillation_trainer/distillation_trainer.py#L287), `_train_shared_fn_extra_args`[`L283`](../../../../../../../raw/code/EasyDeL/easydel/trainers/distillation_trainer/distillation_trainer.py#L283)
- uses (calls/refs, reference-scoped): [`EasyDeLBaseModule`](../../infra/base_module.md#EasyDeLBaseModule), [`EasyDeLState`](../../infra/base_state.md#EasyDeLState), [`model`](../../infra/base_state.md#EasyDeLState.model), [`flops_per_token`](../../infra/base_module.md#EasyDeLBaseModule.flops_per_token), [`Trainer`](../trainer/trainer.md#Trainer), [`ProcessingClassType`](../../infra/utils.md#ProcessingClassType), [`max_length`](../training_configurations.md#TrainingArguments.max_length), [`open_shard`](../../data/core/protocols.md#ShardedDataSource.open_shard), [`shard_names`](../../data/core/protocols.md#ShardedDataSource.shard_names), [`_train_source`](../base_trainer.md#BaseTrainer._train_source), [`mesh`](../../infra/base_module.md#EasyDeLBaseModule.mesh), [`state_shardings`](../trainer_protocol.md#BaseTrainerProtocol.state_shardings), [`TrainerConfigureFunctionOutput`](../trainer_protocol.md#TrainerConfigureFunctionOutput), [`__init__`](../base_trainer.md#BaseTrainer.__init__), [`step_partition_spec`](../training_configurations.md#TrainingArguments.step_partition_spec), [`to_state`](../../infra/base_module.md#EasyDeLBaseModule.to_state), [`model`](../base_trainer.md#BaseTrainer.model), [`loss_config`](../training_configurations.md#TrainingArguments.loss_config), [`resolve_straight_through_emulator`](../training_utils.md#resolve_straight_through_emulator), [`gradient_accumulation_steps`](../training_configurations.md#TrainingArguments.gradient_accumulation_steps), [`scheduler`](../trainer_protocol.md#BaseTrainerProtocol.scheduler), [`quantization_mode`](../training_configurations.md#TrainingArguments.quantization_mode), [`quantization_bits`](../training_configurations.md#TrainingArguments.quantization_bits), [`quantization_group_size`](../training_configurations.md#TrainingArguments.quantization_group_size), [`_preprocess_batch_input`](../base_trainer.md#BaseTrainer._preprocess_batch_input), [`distillation_step`](_fn.md#distillation_step), [`get_streaming_checkpointer`](../training_configurations.md#TrainingArguments.get_streaming_checkpointer), [`processing_class`](../base_trainer.md#BaseTrainer.processing_class), [`DistillationConfig`](distillation_config.md#DistillationConfig), [`_train_shared_fn_static_args`](../base_trainer.md#BaseTrainer._train_shared_fn_static_args), [`_eval_shared_fn_static_args`](../base_trainer.md#BaseTrainer._eval_shared_fn_static_args), [`shardings`](../../infra/base_state.md#EasyDeLState.shardings), [`ensure_checkpoint_path`](../training_configurations.md#TrainingArguments.ensure_checkpoint_path), [`straight_through_emulator`](../training_configurations.md#TrainingArguments.straight_through_emulator), [`tensor_straight_through`](../training_configurations.md#TrainingArguments.tensor_straight_through), [`SFTPreprocessTransform`](../prompt_transforms.md#SFTPreprocessTransform), [`_extra_forward_flops_per_token`](../trainer_protocol.md#BaseTrainerProtocol._extra_forward_flops_per_token), [`_extra_backward_flops_per_token`](../trainer_protocol.md#BaseTrainerProtocol._extra_backward_flops_per_token), [`alpha`](distillation_config.md#DistillationConfig.alpha), [`attention_loss_weight`](distillation_config.md#DistillationConfig.attention_loss_weight)  (+8 more)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`configure_functions`](../trainer/trainer.md#Trainer.configure_functions), [`Trainer`](../trainer/trainer.md#Trainer), [`_is_pretokenized`](../base_trainer.md#BaseTrainer._is_pretokenized), [`_get_preprocess_transform`](../base_trainer.md#BaseTrainer._get_preprocess_transform), [`_preprocess_batch_input`](../base_trainer.md#BaseTrainer._preprocess_batch_input), [`_train_shared_fn_extra_args`](../base_trainer.md#BaseTrainer._train_shared_fn_extra_args), [`_eval_shared_fn_extra_args`](../base_trainer.md#BaseTrainer._eval_shared_fn_extra_args)

## Module values
- `logger` — [`L39`](../../../../../../../raw/code/EasyDeL/easydel/trainers/distillation_trainer/distillation_trainer.py#L39)


---
title: 'Module: easydel/trainers/reward_trainer/reward_trainer.py'
type: catalog
provenance: extracted
module: easydel/trainers/reward_trainer/reward_trainer.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.reward_trainer.reward_trainer`/
symbols:
  RewardTrainer.configure_functions: RewardTrainer#configure_functions().
  RewardTrainer.input_data_collator_grain: RewardTrainer#input_data_collator_grain.
  RewardTrainer.__init__: RewardTrainer#__init__().
  RewardTrainer._get_preprocess_transform: RewardTrainer#_get_preprocess_transform().
  RewardTrainer.input_data_collator_tfds: RewardTrainer#input_data_collator_tfds.
  RewardTrainer._is_pretokenized: RewardTrainer#_is_pretokenized().
  RewardTrainer: RewardTrainer#
  logger: logger.
  RewardTrainer.create_grain_collect_function: RewardTrainer#create_grain_collect_function().
  RewardTrainer.create_tfds_collect_function: RewardTrainer#create_tfds_collect_function().
---
# Module: [`easydel/trainers/reward_trainer/reward_trainer.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/reward_trainer/reward_trainer.py)

## Classes
### `RewardTrainer`  ·  implements/extends Trainer
- def: [`easydel/trainers/reward_trainer/reward_trainer.py:45`](../../../../../../../raw/code/EasyDeL/easydel/trainers/reward_trainer/reward_trainer.py#L45)
- doc: Reward model trainer for RLHF pipelines.
- signature: `class RewardTrainer(Trainer):`
- members:
  - `_get_preprocess_transform(self)` — [`L125`](../../../../../../../raw/code/EasyDeL/easydel/trainers/reward_trainer/reward_trainer.py#L125) — Get Reward Model preprocessing transform for ShardedDataSource.
  - `_is_pretokenized(self)` — [`L136`](../../../../../../../raw/code/EasyDeL/easydel/trainers/reward_trainer/reward_trainer.py#L136) — Check if dataset already has tokenized fields.
  - `configure_functions(self)` — [`L146`](../../../../../../../raw/code/EasyDeL/easydel/trainers/reward_trainer/reward_trainer.py#L146) — Configure and JIT-compile training and evaluation step functions.
  - `create_grain_collect_function(self, max_sequence_length: int, truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end")` — [`L206`](../../../../../../../raw/code/EasyDeL/easydel/trainers/reward_trainer/reward_trainer.py#L206) — Create data collection function for Grain batching.
  - `create_tfds_collect_function(self, max_sequence_length: int, truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end")` — [`L214`](../../../../../../../raw/code/EasyDeL/easydel/trainers/reward_trainer/reward_trainer.py#L214) — Create data collection function for TFDS batching.
  - `input_data_collator_grain` — [`L104`](../../../../../../../raw/code/EasyDeL/easydel/trainers/reward_trainer/reward_trainer.py#L104)
  - `input_data_collator_tfds` — [`L99`](../../../../../../../raw/code/EasyDeL/easydel/trainers/reward_trainer/reward_trainer.py#L99)
- protocol/private: `__init__`[`L78`](../../../../../../../raw/code/EasyDeL/easydel/trainers/reward_trainer/reward_trainer.py#L78)
- uses (calls/refs, reference-scoped): [`EasyDeLBaseModule`](../../infra/base_module.md#EasyDeLBaseModule), [`EasyDeLState`](../../infra/base_state.md#EasyDeLState), [`ShardedDataSource`](../../data/core/protocols.md#ShardedDataSource), [`arguments`](../base_trainer.md#BaseTrainer.arguments), [`Trainer`](../trainer/trainer.md#Trainer), [`ProcessingClassType`](../../infra/utils.md#ProcessingClassType), [`max_length`](../training_configurations.md#TrainingArguments.max_length), [`open_shard`](../../data/core/protocols.md#ShardedDataSource.open_shard), [`shard_names`](../../data/core/protocols.md#ShardedDataSource.shard_names), [`_train_source`](../base_trainer.md#BaseTrainer._train_source), [`mesh`](../../infra/base_module.md#EasyDeLBaseModule.mesh), [`state_shardings`](../trainer_protocol.md#BaseTrainerProtocol.state_shardings), [`TrainerConfigureFunctionOutput`](../trainer_protocol.md#TrainerConfigureFunctionOutput), [`__init__`](../base_trainer.md#BaseTrainer.__init__), [`step_partition_spec`](../training_configurations.md#TrainingArguments.step_partition_spec), [`to_state`](../../infra/base_module.md#EasyDeLBaseModule.to_state), [`model`](../base_trainer.md#BaseTrainer.model), [`loss_config`](../training_configurations.md#TrainingArguments.loss_config), [`resolve_straight_through_emulator`](../training_utils.md#resolve_straight_through_emulator), [`gradient_accumulation_steps`](../training_configurations.md#TrainingArguments.gradient_accumulation_steps), [`scheduler`](../trainer_protocol.md#BaseTrainerProtocol.scheduler), [`quantization_mode`](../training_configurations.md#TrainingArguments.quantization_mode), [`quantization_bits`](../training_configurations.md#TrainingArguments.quantization_bits), [`quantization_group_size`](../training_configurations.md#TrainingArguments.quantization_group_size), [`get_streaming_checkpointer`](../training_configurations.md#TrainingArguments.get_streaming_checkpointer), [`processing_class`](../base_trainer.md#BaseTrainer.processing_class), [`_train_shared_fn_static_args`](../base_trainer.md#BaseTrainer._train_shared_fn_static_args), [`training_step`](_fn.md#training_step), [`_eval_shared_fn_static_args`](../base_trainer.md#BaseTrainer._eval_shared_fn_static_args), [`ensure_checkpoint_path`](../training_configurations.md#TrainingArguments.ensure_checkpoint_path), [`evaluation_step`](_fn.md#evaluation_step), [`RewardConfig`](reward_config.md#RewardConfig), [`straight_through_emulator`](../training_configurations.md#TrainingArguments.straight_through_emulator), [`tensor_straight_through`](../training_configurations.md#TrainingArguments.tensor_straight_through), [`truncation_mode`](../training_configurations.md#TrainingArguments.truncation_mode), [`RewardPreprocessTransform`](../prompt_transforms.md#RewardPreprocessTransform), [`RewardDataCollatorWithPaddingGrain`](../utils.md#RewardDataCollatorWithPaddingGrain), [`RewardDataCollatorWithPaddingTFDS`](../utils.md#RewardDataCollatorWithPaddingTFDS), [`max_length`](reward_config.md#RewardConfig.max_length)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`configure_functions`](../trainer/trainer.md#Trainer.configure_functions), [`Trainer`](../trainer/trainer.md#Trainer), [`_is_pretokenized`](../base_trainer.md#BaseTrainer._is_pretokenized), [`_get_preprocess_transform`](../base_trainer.md#BaseTrainer._get_preprocess_transform), [`create_grain_collect_function`](../trainer/trainer.md#Trainer.create_grain_collect_function), [`create_tfds_collect_function`](../trainer/trainer.md#Trainer.create_tfds_collect_function)

## Module values
- `logger` — [`L41`](../../../../../../../raw/code/EasyDeL/easydel/trainers/reward_trainer/reward_trainer.py#L41)


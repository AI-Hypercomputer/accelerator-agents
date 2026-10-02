---
title: 'Module: easydel/trainers/odds_ratio_preference_optimization_trainer/orpo_trainer.py'
type: catalog
provenance: extracted
module: easydel/trainers/odds_ratio_preference_optimization_trainer/orpo_trainer.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.odds_ratio_preference_optimization_trainer.orpo_trainer`/
symbols:
  ORPOTrainer.configure_functions: ORPOTrainer#configure_functions().
  ORPOTrainer.arguments: ORPOTrainer#arguments.
  ORPOTrainer.__init__: ORPOTrainer#__init__().
  ORPOTrainer._get_preprocess_transform: ORPOTrainer#_get_preprocess_transform().
  ORPOTrainer.input_data_collator_tfds: ORPOTrainer#input_data_collator_tfds.
  ORPOTrainer.input_data_collator_grain: ORPOTrainer#input_data_collator_grain.
  ORPOTrainer.padding_value: ORPOTrainer#padding_value.
  ORPOTrainer._is_pretokenized: ORPOTrainer#_is_pretokenized().
  ORPOTrainer._stored_metrics: ORPOTrainer#_stored_metrics.
  ORPOTrainer.concatenated_forward: ORPOTrainer#concatenated_forward.
  ORPOTrainer: ORPOTrainer#
  logger: logger.
  ORPOTrainer.is_encoder_decoder: ORPOTrainer#is_encoder_decoder.
  ORPOTrainer.truncation_mode: ORPOTrainer#truncation_mode.
  ORPOTrainer.create_grain_collect_function: ORPOTrainer#create_grain_collect_function().
  ORPOTrainer.create_tfds_collect_function: ORPOTrainer#create_tfds_collect_function().
  ORPOTrainer.processing_class: ORPOTrainer#processing_class.
---
# Module: [`easydel/trainers/odds_ratio_preference_optimization_trainer/orpo_trainer.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/odds_ratio_preference_optimization_trainer/orpo_trainer.py)

## Classes
### `ORPOTrainer`  ·  implements/extends Trainer
- def: [`easydel/trainers/odds_ratio_preference_optimization_trainer/orpo_trainer.py:48`](../../../../../../../raw/code/EasyDeL/easydel/trainers/odds_ratio_preference_optimization_trainer/orpo_trainer.py#L48)
- doc: Odds Ratio Preference Optimization trainer.
- signature: `class ORPOTrainer(Trainer):`
- members:
  - `_get_preprocess_transform(self)` — [`L159`](../../../../../../../raw/code/EasyDeL/easydel/trainers/odds_ratio_preference_optimization_trainer/orpo_trainer.py#L159) — Get ORPO preprocessing transform for ShardedDataSource.
  - `_is_pretokenized(self)` — [`L173`](../../../../../../../raw/code/EasyDeL/easydel/trainers/odds_ratio_preference_optimization_trainer/orpo_trainer.py#L173) — Check if dataset already has tokenized fields.
  - `configure_functions(self)` — [`L183`](../../../../../../../raw/code/EasyDeL/easydel/trainers/odds_ratio_preference_optimization_trainer/orpo_trainer.py#L183) — Configure and JIT-compile training and evaluation step functions.
  - `create_grain_collect_function(self, max_sequence_length: int, truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end")` — [`L261`](../../../../../../../raw/code/EasyDeL/easydel/trainers/odds_ratio_preference_optimization_trainer/orpo_trainer.py#L261) — Create data collection function for Grain batching.
  - `create_tfds_collect_function(self, max_sequence_length: int, truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end")` — [`L269`](../../../../../../../raw/code/EasyDeL/easydel/trainers/odds_ratio_preference_optimization_trainer/orpo_trainer.py#L269) — Create data collection function for TFDS batching.
  - `arguments` — [`L81`](../../../../../../../raw/code/EasyDeL/easydel/trainers/odds_ratio_preference_optimization_trainer/orpo_trainer.py#L81)
  - `concatenated_forward` — [`L251`](../../../../../../../raw/code/EasyDeL/easydel/trainers/odds_ratio_preference_optimization_trainer/orpo_trainer.py#L251)
  - `input_data_collator_grain` — [`L132`](../../../../../../../raw/code/EasyDeL/easydel/trainers/odds_ratio_preference_optimization_trainer/orpo_trainer.py#L132)
  - `input_data_collator_tfds` — [`L120`](../../../../../../../raw/code/EasyDeL/easydel/trainers/odds_ratio_preference_optimization_trainer/orpo_trainer.py#L120)
  - `is_encoder_decoder` — [`L102`](../../../../../../../raw/code/EasyDeL/easydel/trainers/odds_ratio_preference_optimization_trainer/orpo_trainer.py#L102)
  - `padding_value` — [`L106`](../../../../../../../raw/code/EasyDeL/easydel/trainers/odds_ratio_preference_optimization_trainer/orpo_trainer.py#L106)
  - `processing_class` — [`L101`](../../../../../../../raw/code/EasyDeL/easydel/trainers/odds_ratio_preference_optimization_trainer/orpo_trainer.py#L101)
  - `truncation_mode` — [`L100`](../../../../../../../raw/code/EasyDeL/easydel/trainers/odds_ratio_preference_optimization_trainer/orpo_trainer.py#L100)
- protocol/private: `__init__`[`L83`](../../../../../../../raw/code/EasyDeL/easydel/trainers/odds_ratio_preference_optimization_trainer/orpo_trainer.py#L83), `_stored_metrics`[`L145`](../../../../../../../raw/code/EasyDeL/easydel/trainers/odds_ratio_preference_optimization_trainer/orpo_trainer.py#L145)
- uses (calls/refs, reference-scoped): [`EasyDeLBaseModule`](../../infra/base_module.md#EasyDeLBaseModule), [`EasyDeLState`](../../infra/base_state.md#EasyDeLState), [`ShardedDataSource`](../../data/core/protocols.md#ShardedDataSource), [`Trainer`](../trainer/trainer.md#Trainer), [`ProcessingClassType`](../../infra/utils.md#ProcessingClassType), [`open_shard`](../../data/core/protocols.md#ShardedDataSource.open_shard), [`shard_names`](../../data/core/protocols.md#ShardedDataSource.shard_names), [`_train_source`](../base_trainer.md#BaseTrainer._train_source), [`mesh`](../../infra/base_module.md#EasyDeLBaseModule.mesh), [`state_shardings`](../trainer_protocol.md#BaseTrainerProtocol.state_shardings), [`TrainerConfigureFunctionOutput`](../trainer_protocol.md#TrainerConfigureFunctionOutput), [`__init__`](../base_trainer.md#BaseTrainer.__init__), [`step_partition_spec`](../training_configurations.md#TrainingArguments.step_partition_spec), [`to_state`](../../infra/base_module.md#EasyDeLBaseModule.to_state), [`model`](../base_trainer.md#BaseTrainer.model), [`loss_config`](../training_configurations.md#TrainingArguments.loss_config), [`resolve_straight_through_emulator`](../training_utils.md#resolve_straight_through_emulator), [`gradient_accumulation_steps`](../training_configurations.md#TrainingArguments.gradient_accumulation_steps), [`scheduler`](../trainer_protocol.md#BaseTrainerProtocol.scheduler), [`quantization_mode`](../training_configurations.md#TrainingArguments.quantization_mode), [`quantization_bits`](../training_configurations.md#TrainingArguments.quantization_bits), [`quantization_group_size`](../training_configurations.md#TrainingArguments.quantization_group_size), [`get_streaming_checkpointer`](../training_configurations.md#TrainingArguments.get_streaming_checkpointer), [`orpo_step`](_fn.md#orpo_step), [`concatenated_forward`](_fn.md#concatenated_forward), [`_train_shared_fn_static_args`](../base_trainer.md#BaseTrainer._train_shared_fn_static_args), [`_eval_shared_fn_static_args`](../base_trainer.md#BaseTrainer._eval_shared_fn_static_args), [`ensure_checkpoint_path`](../training_configurations.md#TrainingArguments.ensure_checkpoint_path), [`ORPOConfig`](orpo_config.md#ORPOConfig), [`straight_through_emulator`](../training_configurations.md#TrainingArguments.straight_through_emulator), [`tensor_straight_through`](../training_configurations.md#TrainingArguments.tensor_straight_through), [`truncation_mode`](../training_configurations.md#TrainingArguments.truncation_mode), [`_extra_forward_flops_per_token`](../trainer_protocol.md#BaseTrainerProtocol._extra_forward_flops_per_token), [`ORPOPreprocessTransform`](../prompt_transforms.md#ORPOPreprocessTransform), [`_extra_backward_flops_per_token`](../trainer_protocol.md#BaseTrainerProtocol._extra_backward_flops_per_token), [`max_completion_length`](orpo_config.md#ORPOConfig.max_completion_length), [`max_prompt_length`](orpo_config.md#ORPOConfig.max_prompt_length), [`is_encoder_decoder`](orpo_config.md#ORPOConfig.is_encoder_decoder), [`label_pad_token_id`](orpo_config.md#ORPOConfig.label_pad_token_id), [`logprob_vocab_chunk_size`](orpo_config.md#ORPOConfig.logprob_vocab_chunk_size)  (+5 more)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`configure_functions`](../trainer/trainer.md#Trainer.configure_functions), [`Trainer`](../trainer/trainer.md#Trainer), [`_is_pretokenized`](../base_trainer.md#BaseTrainer._is_pretokenized), [`_get_preprocess_transform`](../base_trainer.md#BaseTrainer._get_preprocess_transform), [`create_grain_collect_function`](../trainer/trainer.md#Trainer.create_grain_collect_function), [`create_tfds_collect_function`](../trainer/trainer.md#Trainer.create_tfds_collect_function)

## Module values
- `logger` — [`L44`](../../../../../../../raw/code/EasyDeL/easydel/trainers/odds_ratio_preference_optimization_trainer/orpo_trainer.py#L44)


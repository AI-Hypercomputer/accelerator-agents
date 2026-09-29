---
title: 'Module: easydel/trainers/kto_trainer/kto_trainer.py'
type: catalog
provenance: extracted
module: easydel/trainers/kto_trainer/kto_trainer.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.kto_trainer.kto_trainer`/
symbols:
  KTOTrainer.concatenated_forward: KTOTrainer#concatenated_forward.
  KTOTrainer.arguments: KTOTrainer#arguments.
  KTOTrainer.padding_value: KTOTrainer#padding_value.
  KTOTrainer.__init__: KTOTrainer#__init__().
  KTOTrainer.forward_fn: KTOTrainer#forward_fn().
  KTOTrainer.configure_functions: KTOTrainer#configure_functions().
  KTOTrainer.is_encoder_decoder: KTOTrainer#is_encoder_decoder.
  KTOTrainer._get_preprocess_transform: KTOTrainer#_get_preprocess_transform().
  KTOTrainer.aux_loss_coef: KTOTrainer#aux_loss_coef.
  KTOTrainer._preprocess_kto_dataset: KTOTrainer#_preprocess_kto_dataset().
  KTOTrainer._is_pretokenized: KTOTrainer#_is_pretokenized().
  KTOTrainer: KTOTrainer#
  logger: logger.
  KTOTrainer.reference_state: KTOTrainer#reference_state.
  KTOTrainer.calculate_kl: KTOTrainer#calculate_kl.
  KTOTrainer.aux_loss_enabled: KTOTrainer#aux_loss_enabled.
  KTOTrainer.create_grain_collect_function: KTOTrainer#create_grain_collect_function().
  KTOTrainer.create_tfds_collect_function: KTOTrainer#create_tfds_collect_function().
  KTOTrainer.input_data_collator_tfds: KTOTrainer#input_data_collator_tfds.
  KTOTrainer.input_data_collator_grain: KTOTrainer#input_data_collator_grain.
  KTOTrainer.processing_class: KTOTrainer#processing_class.
---
# Module: [`easydel/trainers/kto_trainer/kto_trainer.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/kto_trainer.py)

## Classes
### `KTOTrainer`  ·  implements/extends Trainer
- def: [`easydel/trainers/kto_trainer/kto_trainer.py:53`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/kto_trainer.py#L53)
- doc: Kahneman-Tversky Optimization trainer.
- signature: `class KTOTrainer(Trainer):`
- members:
  - `_get_preprocess_transform(self)` — [`L225`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/kto_trainer.py#L225) — Get KTO preprocessing transform for ShardedDataSource.
  - `_is_pretokenized(self)` — [`L238`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/kto_trainer.py#L238) — Check if dataset already has tokenized fields.
  - `_preprocess_kto_dataset(dataset, processing_class, arguments)` — [`L176`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/kto_trainer.py#L176) — Preprocess dataset for KTO training.
  - `configure_functions(self)` — [`L248`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/kto_trainer.py#L248) — Configure JIT-compiled training and evaluation functions.
  - `create_grain_collect_function(self, max_sequence_length: int, truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end")` — [`L338`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/kto_trainer.py#L338) — Create data collator for Grain data loading.
  - `create_tfds_collect_function(self, max_sequence_length: int, truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end")` — [`L354`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/kto_trainer.py#L354) — Create data collator for TFDS data loading.
  - `forward_fn(model, batch)` — [`L264`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/kto_trainer.py#L264)
  - `arguments` — [`L70`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/kto_trainer.py#L70)
  - `aux_loss_coef` — [`L155`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/kto_trainer.py#L155)
  - `aux_loss_enabled` — [`L154`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/kto_trainer.py#L154)
  - `calculate_kl` — [`L153`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/kto_trainer.py#L153)
  - `concatenated_forward` — [`L277`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/kto_trainer.py#L277)
  - `input_data_collator_grain` — [`L149`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/kto_trainer.py#L149)
  - `input_data_collator_tfds` — [`L148`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/kto_trainer.py#L148)
  - `is_encoder_decoder` — [`L110`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/kto_trainer.py#L110)
  - `padding_value` — [`L116`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/kto_trainer.py#L116)
  - `processing_class` — [`L151`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/kto_trainer.py#L151)
  - `reference_state` — [`L152`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/kto_trainer.py#L152)
- protocol/private: `__init__`[`L72`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/kto_trainer.py#L72)
- uses (calls/refs, reference-scoped): [`config`](../../infra/base_module.md#EasyDeLBaseModule.config), [`EasyDeLBaseModule`](../../infra/base_module.md#EasyDeLBaseModule), [`EasyDeLState`](../../infra/base_state.md#EasyDeLState), [`ShardedDataSource`](../../data/core/protocols.md#ShardedDataSource), [`model`](../../infra/base_state.md#EasyDeLState.model), [`Trainer`](../trainer/trainer.md#Trainer), [`ProcessingClassType`](../../infra/utils.md#ProcessingClassType), [`open_shard`](../../data/core/protocols.md#ShardedDataSource.open_shard), [`model_state`](../base_trainer.md#BaseTrainer.model_state), [`shard_names`](../../data/core/protocols.md#ShardedDataSource.shard_names), [`_train_source`](../base_trainer.md#BaseTrainer._train_source), [`mesh`](../../infra/base_module.md#EasyDeLBaseModule.mesh), [`state_shardings`](../trainer_protocol.md#BaseTrainerProtocol.state_shardings), [`TrainerConfigureFunctionOutput`](../trainer_protocol.md#TrainerConfigureFunctionOutput), [`__init__`](../base_trainer.md#BaseTrainer.__init__), [`step_partition_spec`](../training_configurations.md#TrainingArguments.step_partition_spec), [`to_state`](../../infra/base_module.md#EasyDeLBaseModule.to_state), [`model`](../base_trainer.md#BaseTrainer.model), [`loss_config`](../training_configurations.md#TrainingArguments.loss_config), [`resolve_straight_through_emulator`](../training_utils.md#resolve_straight_through_emulator), [`gradient_accumulation_steps`](../training_configurations.md#TrainingArguments.gradient_accumulation_steps), [`scheduler`](../trainer_protocol.md#BaseTrainerProtocol.scheduler), [`quantization_mode`](../training_configurations.md#TrainingArguments.quantization_mode), [`quantization_bits`](../training_configurations.md#TrainingArguments.quantization_bits), [`quantization_group_size`](../training_configurations.md#TrainingArguments.quantization_group_size), [`training_step`](_fn.md#training_step), [`get_streaming_checkpointer`](../training_configurations.md#TrainingArguments.get_streaming_checkpointer), [`_train_shared_fn_static_args`](../base_trainer.md#BaseTrainer._train_shared_fn_static_args), [`_eval_shared_fn_static_args`](../base_trainer.md#BaseTrainer._eval_shared_fn_static_args), [`_train_shared_fn_extra_args`](../base_trainer.md#BaseTrainer._train_shared_fn_extra_args), [`_eval_shared_fn_extra_args`](../base_trainer.md#BaseTrainer._eval_shared_fn_extra_args), [`deepcopy_model`](../../utils/traversals.md#deepcopy_model), [`shardings`](../../infra/base_state.md#EasyDeLState.shardings), [`concatenated_forward`](../binary_classifier_optimization_trainer/_fn.md#concatenated_forward), [`KTOConfig`](kto_config.md#KTOConfig), [`evaluation_step`](_fn.md#evaluation_step), [`maybe_apply_chat_template`](../prompt_utils.md#maybe_apply_chat_template), [`straight_through_emulator`](../training_configurations.md#TrainingArguments.straight_through_emulator), [`tensor_straight_through`](../training_configurations.md#TrainingArguments.tensor_straight_through), [`maybe_extract_prompt`](../prompt_utils.md#maybe_extract_prompt)  (+19 more)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`configure_functions`](../trainer/trainer.md#Trainer.configure_functions), [`Trainer`](../trainer/trainer.md#Trainer), [`_is_pretokenized`](../base_trainer.md#BaseTrainer._is_pretokenized), [`_get_preprocess_transform`](../base_trainer.md#BaseTrainer._get_preprocess_transform), [`create_grain_collect_function`](../trainer/trainer.md#Trainer.create_grain_collect_function), [`create_tfds_collect_function`](../trainer/trainer.md#Trainer.create_tfds_collect_function)

## Module values
- `logger` — [`L49`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/kto_trainer.py#L49)


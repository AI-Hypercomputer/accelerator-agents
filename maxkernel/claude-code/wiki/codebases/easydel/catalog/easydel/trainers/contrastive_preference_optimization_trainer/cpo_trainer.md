---
title: 'Module: easydel/trainers/contrastive_preference_optimization_trainer/cpo_trainer.py'
type: catalog
provenance: extracted
module: easydel/trainers/contrastive_preference_optimization_trainer/cpo_trainer.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.contrastive_preference_optimization_trainer.cpo_trainer`/
symbols:
  CPOTrainer.concatenated_forward: CPOTrainer#concatenated_forward.
  CPOTrainer.configure_functions: CPOTrainer#configure_functions().
  CPOTrainer.arguments: CPOTrainer#arguments.
  CPOTrainer.input_data_collator_grain: CPOTrainer#input_data_collator_grain.
  CPOTrainer.__init__: CPOTrainer#__init__().
  CPOTrainer.is_encoder_decoder: CPOTrainer#is_encoder_decoder.
  CPOTrainer._get_preprocess_transform: CPOTrainer#_get_preprocess_transform().
  CPOTrainer.input_data_collator_tfds: CPOTrainer#input_data_collator_tfds.
  CPOTrainer.padding_value: CPOTrainer#padding_value.
  CPOTrainer._is_pretokenized: CPOTrainer#_is_pretokenized().
  CPOTrainer._stored_metrics: CPOTrainer#_stored_metrics.
  CPOTrainer: CPOTrainer#
  logger: logger.
  CPOTrainer.on_step_end: CPOTrainer#on_step_end().
  CPOTrainer.truncation_mode: CPOTrainer#truncation_mode.
  CPOTrainer.create_grain_collect_function: CPOTrainer#create_grain_collect_function().
  CPOTrainer.create_tfds_collect_function: CPOTrainer#create_tfds_collect_function().
  CPOTrainer.processing_class: CPOTrainer#processing_class.
  CPOTrainer._train_shared_fn_extra_args: CPOTrainer#_train_shared_fn_extra_args().
  CPOTrainer._eval_shared_fn_extra_args: CPOTrainer#_eval_shared_fn_extra_args().
---
# Module: [`easydel/trainers/contrastive_preference_optimization_trainer/cpo_trainer.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/cpo_trainer.py)

## Classes
### `CPOTrainer`  ·  implements/extends Trainer
- def: [`easydel/trainers/contrastive_preference_optimization_trainer/cpo_trainer.py:52`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/cpo_trainer.py#L52)
- doc: Contrastive Preference Optimization (CPO) trainer.
- signature: `class CPOTrainer(Trainer):`
- members:
  - `_get_preprocess_transform(self)` — [`L151`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/cpo_trainer.py#L151) — Get CPO preprocessing transform for ShardedDataSource.
  - `_is_pretokenized(self)` — [`L164`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/cpo_trainer.py#L164) — Check if dataset already has tokenized fields.
  - `configure_functions(self)` — [`L174`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/cpo_trainer.py#L174) — Configure JIT-compiled training and evaluation functions.
  - `create_grain_collect_function(self, max_sequence_length: int, truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end")` — [`L274`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/cpo_trainer.py#L274) — Create data collator for Grain data loading.
  - `create_tfds_collect_function(self, max_sequence_length: int, truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end")` — [`L290`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/cpo_trainer.py#L290) — Create data collator for TFDS data loading.
  - `on_step_end(self, state: EasyDeLState, metrics: MetricsType, step: int)` — [`L314`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/cpo_trainer.py#L314) — Called at the end of each training step.
  - `arguments` — [`L67`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/cpo_trainer.py#L67)
  - `concatenated_forward` — [`L201`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/cpo_trainer.py#L201)
  - `input_data_collator_grain` — [`L126`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/cpo_trainer.py#L126)
  - `input_data_collator_tfds` — [`L119`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/cpo_trainer.py#L119)
  - `is_encoder_decoder` — [`L96`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/cpo_trainer.py#L96)
  - `padding_value` — [`L105`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/cpo_trainer.py#L105)
  - `processing_class` — [`L86`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/cpo_trainer.py#L86)
  - `truncation_mode` — [`L87`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/cpo_trainer.py#L87)
- protocol/private: `__init__`[`L69`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/cpo_trainer.py#L69), `_eval_shared_fn_extra_args`[`L311`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/cpo_trainer.py#L311), `_stored_metrics`[`L88`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/cpo_trainer.py#L88), `_train_shared_fn_extra_args`[`L307`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/cpo_trainer.py#L307)
- uses (calls/refs, reference-scoped): [`config`](../../infra/base_module.md#EasyDeLBaseModule.config), [`EasyDeLBaseModule`](../../infra/base_module.md#EasyDeLBaseModule), [`EasyDeLState`](../../infra/base_state.md#EasyDeLState), [`ShardedDataSource`](../../data/core/protocols.md#ShardedDataSource), [`model`](../../infra/base_state.md#EasyDeLState.model), [`Trainer`](../trainer/trainer.md#Trainer), [`ProcessingClassType`](../../infra/utils.md#ProcessingClassType), [`open_shard`](../../data/core/protocols.md#ShardedDataSource.open_shard), [`model_state`](../base_trainer.md#BaseTrainer.model_state), [`shard_names`](../../data/core/protocols.md#ShardedDataSource.shard_names), [`_train_source`](../base_trainer.md#BaseTrainer._train_source), [`mesh`](../../infra/base_module.md#EasyDeLBaseModule.mesh), [`state_shardings`](../trainer_protocol.md#BaseTrainerProtocol.state_shardings), [`TrainerConfigureFunctionOutput`](../trainer_protocol.md#TrainerConfigureFunctionOutput), [`__init__`](../base_trainer.md#BaseTrainer.__init__), [`step_partition_spec`](../training_configurations.md#TrainingArguments.step_partition_spec), [`to_state`](../../infra/base_module.md#EasyDeLBaseModule.to_state), [`model`](../base_trainer.md#BaseTrainer.model), [`loss_config`](../training_configurations.md#TrainingArguments.loss_config), [`resolve_straight_through_emulator`](../training_utils.md#resolve_straight_through_emulator), [`gradient_accumulation_steps`](../training_configurations.md#TrainingArguments.gradient_accumulation_steps), [`scheduler`](../trainer_protocol.md#BaseTrainerProtocol.scheduler), [`quantization_mode`](../training_configurations.md#TrainingArguments.quantization_mode), [`quantization_bits`](../training_configurations.md#TrainingArguments.quantization_bits), [`quantization_group_size`](../training_configurations.md#TrainingArguments.quantization_group_size), [`get_streaming_checkpointer`](../training_configurations.md#TrainingArguments.get_streaming_checkpointer), [`training_step`](_fn.md#training_step), [`_train_shared_fn_static_args`](../base_trainer.md#BaseTrainer._train_shared_fn_static_args), [`MetricsType`](../training_configurations.md#MetricsType), [`_eval_shared_fn_static_args`](../base_trainer.md#BaseTrainer._eval_shared_fn_static_args), [`concatenated_forward`](_fn.md#concatenated_forward), [`ensure_checkpoint_path`](../training_configurations.md#TrainingArguments.ensure_checkpoint_path), [`evaluation_step`](_fn.md#evaluation_step), [`straight_through_emulator`](../training_configurations.md#TrainingArguments.straight_through_emulator), [`tensor_straight_through`](../training_configurations.md#TrainingArguments.tensor_straight_through), [`CPOConfig`](cpo_config.md#CPOConfig), [`_extra_forward_flops_per_token`](../trainer_protocol.md#BaseTrainerProtocol._extra_forward_flops_per_token), [`sharded_evaluation_step_function`](../trainer_protocol.md#BaseTrainerProtocol.sharded_evaluation_step_function), [`sharded_training_step_function`](../trainer_protocol.md#BaseTrainerProtocol.sharded_training_step_function), [`CPOPreprocessTransform`](../prompt_transforms.md#CPOPreprocessTransform)  (+19 more)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`configure_functions`](../trainer/trainer.md#Trainer.configure_functions), [`Trainer`](../trainer/trainer.md#Trainer), [`_is_pretokenized`](../base_trainer.md#BaseTrainer._is_pretokenized), [`_get_preprocess_transform`](../base_trainer.md#BaseTrainer._get_preprocess_transform), [`create_grain_collect_function`](../trainer/trainer.md#Trainer.create_grain_collect_function), [`create_tfds_collect_function`](../trainer/trainer.md#Trainer.create_tfds_collect_function), [`_train_shared_fn_extra_args`](../base_trainer.md#BaseTrainer._train_shared_fn_extra_args), [`_eval_shared_fn_extra_args`](../base_trainer.md#BaseTrainer._eval_shared_fn_extra_args), [`on_step_end`](../base_trainer.md#BaseTrainer.on_step_end)

## Module values
- `logger` — [`L48`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/cpo_trainer.py#L48)


---
title: 'Module: easydel/trainers/direct_preference_optimization_trainer/dpo_trainer.py'
type: catalog
provenance: extracted
module: easydel/trainers/direct_preference_optimization_trainer/dpo_trainer.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.direct_preference_optimization_trainer.dpo_trainer`/
symbols:
  DPOTrainer.configure_functions: DPOTrainer#configure_functions().
  DPOTrainer.arguments: DPOTrainer#arguments.
  DPOTrainer.compute_reference_log_probs: DPOTrainer#compute_reference_log_probs().
  DPOTrainer.configure_dataloaders: DPOTrainer#configure_dataloaders().
  DPOTrainer.concatenated_forward: DPOTrainer#concatenated_forward.
  DPOTrainer.on_step_end: DPOTrainer#on_step_end().
  DPOTrainer._build_preprocess_transform: DPOTrainer#_build_preprocess_transform().
  DPOTrainer._precompute_reference_log_probs_for_split: DPOTrainer#_precompute_reference_log_probs_for_split().
  DPOTrainer.input_data_collator_tfds: DPOTrainer#input_data_collator_tfds.
  DPOTrainer.input_data_collator_grain: DPOTrainer#input_data_collator_grain.
  DPOTrainer.__init__: DPOTrainer#__init__().
  DPOTrainer.reference_state: DPOTrainer#reference_state.
  DPOTrainer._build_source_from_dataset: DPOTrainer#_build_source_from_dataset().
  DPOTrainer.padding_value: DPOTrainer#padding_value.
  DPOTrainer._source_is_pretokenized: DPOTrainer#_source_is_pretokenized().
  DPOTrainer._source_has_reference_logps: DPOTrainer#_source_has_reference_logps().
  DPOTrainer._stored_metrics: DPOTrainer#_stored_metrics.
  DPOTrainer._get_preprocess_transform: DPOTrainer#_get_preprocess_transform().
  logger: logger.
  DPOTrainer._is_pretokenized: DPOTrainer#_is_pretokenized().
  DPOTrainer: DPOTrainer#
  DPOTrainer._precomputed_eval_ref_log_probs: DPOTrainer#_precomputed_eval_ref_log_probs.
  DPOTrainer._precomputed_train_ref_log_probs: DPOTrainer#_precomputed_train_ref_log_probs.
  DPOTrainer.truncation_mode: DPOTrainer#truncation_mode.
  DPOTrainer.is_encoder_decoder: DPOTrainer#is_encoder_decoder.
  DPOTrainer.create_grain_collect_function: DPOTrainer#create_grain_collect_function().
  DPOTrainer.create_tfds_collect_function: DPOTrainer#create_tfds_collect_function().
  DPOTrainer._train_shared_fn_extra_args: DPOTrainer#_train_shared_fn_extra_args().
  DPOTrainer._eval_shared_fn_extra_args: DPOTrainer#_eval_shared_fn_extra_args().
  DPOTrainer.processing_class: DPOTrainer#processing_class.
---
# Module: [`easydel/trainers/direct_preference_optimization_trainer/dpo_trainer.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_trainer.py)

## Classes
### `DPOTrainer`  ·  implements/extends Trainer
- def: [`easydel/trainers/direct_preference_optimization_trainer/dpo_trainer.py:51`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_trainer.py#L51)
- doc: Trainer for Direct Preference Optimization (DPO).
- signature: `class DPOTrainer(Trainer):`
- members:
  - `_get_preprocess_transform(self)` — [`L177`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_trainer.py#L177) — Get DPO preprocessing transform for ShardedDataSource.
  - `_is_pretokenized(self)` — [`L233`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_trainer.py#L233) — Check if dataset already has DPO tokenized fields.
  - `compute_reference_log_probs(self, padded_batch: dict)` — [`L438`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_trainer.py#L438) — Compute log probabilities of the reference model for a batch.
  - `configure_dataloaders(self)` — [`L356`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_trainer.py#L356) — Configure dataloaders with optional precomputed reference log probs.
  - `configure_functions(self)` — [`L237`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_trainer.py#L237) — Configure and JIT-compile training and evaluation step functions.
  - `create_grain_collect_function(self, max_sequence_length: int, truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end")` — [`L340`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_trainer.py#L340) — Create data collection function for Grain batching.
  - `create_tfds_collect_function(self, max_sequence_length: int, truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end")` — [`L348`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_trainer.py#L348) — Create data collection function for TFDS batching.
  - `on_step_end(self, state: EasyDeLState, metrics: MetricsType, step: int)` — [`L471`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_trainer.py#L471) — Hook called at the end of each step for reference model sync.
  - `arguments` — [`L89`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_trainer.py#L89)
  - `concatenated_forward` — [`L326`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_trainer.py#L326)
  - `input_data_collator_grain` — [`L144`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_trainer.py#L144)
  - `input_data_collator_tfds` — [`L133`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_trainer.py#L133)
  - `is_encoder_decoder` — [`L111`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_trainer.py#L111)
  - `padding_value` — [`L117`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_trainer.py#L117)
  - `processing_class` — [`L110`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_trainer.py#L110)
  - `reference_state` — [`L166`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_trainer.py#L166)
  - `truncation_mode` — [`L109`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_trainer.py#L109)
- protocol/private: `__init__`[`L91`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_trainer.py#L91), `_build_preprocess_transform`[`L193`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_trainer.py#L193), `_build_source_from_dataset`[`L224`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_trainer.py#L224), `_eval_shared_fn_extra_args`[`L468`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_trainer.py#L468), `_precompute_reference_log_probs_for_split`[`L384`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_trainer.py#L384), `_precomputed_eval_ref_log_probs`[`L113`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_trainer.py#L113), `_precomputed_train_ref_log_probs`[`L112`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_trainer.py#L112), `_source_has_reference_logps`[`L213`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_trainer.py#L213), `_source_is_pretokenized`[`L203`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_trainer.py#L203), `_stored_metrics`[`L156`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_trainer.py#L156), `_train_shared_fn_extra_args`[`L464`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_trainer.py#L464)
- uses (calls/refs, reference-scoped): [`EasyDeLBaseModule`](../../infra/base_module.md#EasyDeLBaseModule), [`EasyDeLState`](../../infra/base_state.md#EasyDeLState), [`ShardedDataSource`](../../data/core/protocols.md#ShardedDataSource), [`model`](../../infra/base_state.md#EasyDeLState.model), [`flops_per_token`](../../infra/base_module.md#EasyDeLBaseModule.flops_per_token), [`Trainer`](../trainer/trainer.md#Trainer), [`ProcessingClassType`](../../infra/utils.md#ProcessingClassType), [`open_shard`](../../data/core/protocols.md#ShardedDataSource.open_shard), [`model_state`](../base_trainer.md#BaseTrainer.model_state), [`shard_names`](../../data/core/protocols.md#ShardedDataSource.shard_names), [`_train_source`](../base_trainer.md#BaseTrainer._train_source), [`mesh`](../../infra/base_module.md#EasyDeLBaseModule.mesh), [`state_shardings`](../trainer_protocol.md#BaseTrainerProtocol.state_shardings), [`TrainerConfigureFunctionOutput`](../trainer_protocol.md#TrainerConfigureFunctionOutput), [`__init__`](../base_trainer.md#BaseTrainer.__init__), [`graphstate`](../../infra/base_state.md#EasyDeLState.graphstate), [`step_partition_spec`](../training_configurations.md#TrainingArguments.step_partition_spec), [`to_state`](../../infra/base_module.md#EasyDeLBaseModule.to_state), [`model`](../base_trainer.md#BaseTrainer.model), [`loss_config`](../training_configurations.md#TrainingArguments.loss_config), [`resolve_straight_through_emulator`](../training_utils.md#resolve_straight_through_emulator), [`gradient_accumulation_steps`](../training_configurations.md#TrainingArguments.gradient_accumulation_steps), [`training_step`](_fn.md#training_step), [`scheduler`](../trainer_protocol.md#BaseTrainerProtocol.scheduler), [`quantization_mode`](../training_configurations.md#TrainingArguments.quantization_mode), [`quantization_bits`](../training_configurations.md#TrainingArguments.quantization_bits), [`quantization_group_size`](../training_configurations.md#TrainingArguments.quantization_group_size), [`get_streaming_checkpointer`](../training_configurations.md#TrainingArguments.get_streaming_checkpointer), [`_train_shared_fn_static_args`](../base_trainer.md#BaseTrainer._train_shared_fn_static_args), [`MetricsType`](../training_configurations.md#MetricsType), [`_eval_shared_fn_static_args`](../base_trainer.md#BaseTrainer._eval_shared_fn_static_args), [`deepcopy_model`](../../utils/traversals.md#deepcopy_model), [`shardings`](../../infra/base_state.md#EasyDeLState.shardings), [`concatenated_forward`](_fn.md#concatenated_forward), [`evaluation_step`](_fn.md#evaluation_step), [`ensure_checkpoint_path`](../training_configurations.md#TrainingArguments.ensure_checkpoint_path), [`_eval_source`](../base_trainer.md#BaseTrainer._eval_source), [`configure_dataloaders`](../base_trainer.md#BaseTrainer.configure_dataloaders), [`straight_through_emulator`](../training_configurations.md#TrainingArguments.straight_through_emulator), [`tensor_straight_through`](../training_configurations.md#TrainingArguments.tensor_straight_through)  (+32 more)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`configure_functions`](../trainer/trainer.md#Trainer.configure_functions), [`Trainer`](../trainer/trainer.md#Trainer), [`_is_pretokenized`](../base_trainer.md#BaseTrainer._is_pretokenized), [`_get_preprocess_transform`](../base_trainer.md#BaseTrainer._get_preprocess_transform), [`create_grain_collect_function`](../trainer/trainer.md#Trainer.create_grain_collect_function), [`create_tfds_collect_function`](../trainer/trainer.md#Trainer.create_tfds_collect_function), [`_train_shared_fn_extra_args`](../base_trainer.md#BaseTrainer._train_shared_fn_extra_args), [`_eval_shared_fn_extra_args`](../base_trainer.md#BaseTrainer._eval_shared_fn_extra_args), [`configure_dataloaders`](../base_trainer.md#BaseTrainer.configure_dataloaders), [`on_step_end`](../base_trainer.md#BaseTrainer.on_step_end)

## Module values
- `logger` — [`L47`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_trainer.py#L47)


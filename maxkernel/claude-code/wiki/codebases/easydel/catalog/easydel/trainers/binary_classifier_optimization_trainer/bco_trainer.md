---
title: 'Module: easydel/trainers/binary_classifier_optimization_trainer/bco_trainer.py'
type: catalog
provenance: extracted
module: easydel/trainers/binary_classifier_optimization_trainer/bco_trainer.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.binary_classifier_optimization_trainer.bco_trainer`/
symbols:
  BCOTrainer.concatenated_forward: BCOTrainer#concatenated_forward.
  BCOTrainer.arguments: BCOTrainer#arguments.
  BCOTrainer.forward_fn: BCOTrainer#forward_fn().
  BCOTrainer._train_density_ratio_classifier: BCOTrainer#_train_density_ratio_classifier().
  BCOTrainer.configure_functions: BCOTrainer#configure_functions().
  BCOTrainer._preprocess_batch_input: BCOTrainer#_preprocess_batch_input().
  BCOTrainer.input_data_collator_grain: BCOTrainer#input_data_collator_grain.
  BCOTrainer.__init__: BCOTrainer#__init__().
  BCOTrainer._get_preprocess_transform: BCOTrainer#_get_preprocess_transform().
  BCOTrainer.is_encoder_decoder: BCOTrainer#is_encoder_decoder.
  BCOTrainer.input_data_collator_tfds: BCOTrainer#input_data_collator_tfds.
  BCOTrainer.clf_weights: BCOTrainer#clf_weights.
  BCOTrainer._rng: BCOTrainer#_rng.
  BCOTrainer.padding_value: BCOTrainer#padding_value.
  BCOTrainer.compute_reference_log_probs: BCOTrainer#compute_reference_log_probs().
  BCOTrainer._vectorize_prompt: BCOTrainer#_vectorize_prompt().
  BCOTrainer._is_pretokenized: BCOTrainer#_is_pretokenized().
  logger: logger.
  BCOTrainer.reference_state: BCOTrainer#reference_state.
  BCOTrainer: BCOTrainer#
  BCOTrainer.embedding_tokenizer: BCOTrainer#embedding_tokenizer.
  BCOTrainer.running: BCOTrainer#running.
  BCOTrainer.processing_class: BCOTrainer#processing_class.
  BCOTrainer.embedding_func: BCOTrainer#embedding_func.
  BCOTrainer.beta: BCOTrainer#beta.
  BCOTrainer.max_length: BCOTrainer#max_length.
  BCOTrainer.max_prompt_length: BCOTrainer#max_prompt_length.
  BCOTrainer.max_completion_length: BCOTrainer#max_completion_length.
  BCOTrainer.truncation_mode: BCOTrainer#truncation_mode.
  BCOTrainer.label_pad_token_id: BCOTrainer#label_pad_token_id.
  BCOTrainer.create_grain_collect_function: BCOTrainer#create_grain_collect_function().
  BCOTrainer.create_tfds_collect_function: BCOTrainer#create_tfds_collect_function().
  BCOTrainer._fit_logistic_regression: BCOTrainer#_fit_logistic_regression().
  BCOTrainer._precomputed_train_ref_log_probs: BCOTrainer#_precomputed_train_ref_log_probs.
  BCOTrainer._precomputed_eval_ref_log_probs: BCOTrainer#_precomputed_eval_ref_log_probs.
---
# Module: [`easydel/trainers/binary_classifier_optimization_trainer/bco_trainer.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_trainer.py)

## Classes
### `BCOTrainer`  ·  implements/extends Trainer
- def: [`easydel/trainers/binary_classifier_optimization_trainer/bco_trainer.py:49`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_trainer.py#L49)
- doc: Binary Classifier Optimization (BCO) trainer.
- signature: `class BCOTrainer(Trainer):`
- members:
  - `_fit_logistic_regression(self, embeddings: np.ndarray, labels: np.ndarray, lr: float = 0.1, max_iter: int = 500, tol: float = 0.00001)` — [`L304`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_trainer.py#L304) — Fit logistic regression classifier using gradient descent.
  - `_get_preprocess_transform(self)` — [`L183`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_trainer.py#L183) — Get BCO preprocessing transform for ShardedDataSource.
  - `_is_pretokenized(self)` — [`L201`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_trainer.py#L201) — Check if dataset already has tokenized fields.
  - `_preprocess_batch_input(self, state: EasyDeLState, batch: dict[str, np.ndarray], is_train: bool)` — [`L469`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_trainer.py#L469) — Preprocess batch by adding running moments and UDM weights.
  - `_train_density_ratio_classifier(self)` — [`L236`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_trainer.py#L236) — Train UDM density ratio classifier by iterating over transformed source.
  - `_vectorize_prompt(self, input_ids: np.ndarray, attention_mask: np.ndarray)` — [`L211`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_trainer.py#L211) — Convert prompt tokens to embeddings using the embedding function.
  - `compute_reference_log_probs(self, batch: dict[str, np.ndarray])` — [`L453`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_trainer.py#L453) — Compute reference model log probabilities for a batch.
  - `configure_functions(self)` — [`L346`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_trainer.py#L346) — Configure JIT-compiled training and evaluation functions.
  - `create_grain_collect_function(self, max_sequence_length: int, truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end")` — [`L421`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_trainer.py#L421) — Create data collator for Grain data loading.
  - `create_tfds_collect_function(self, max_sequence_length: int, truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end")` — [`L437`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_trainer.py#L437) — Create data collator for TFDS data loading.
  - `forward_fn(model, batch)` — [`L362`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_trainer.py#L362)
  - `arguments` — [`L68`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_trainer.py#L68)
  - `beta` — [`L93`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_trainer.py#L93)
  - `clf_weights` — [`L161`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_trainer.py#L161)
  - `concatenated_forward` — [`L375`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_trainer.py#L375)
  - `embedding_func` — [`L91`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_trainer.py#L91)
  - `embedding_tokenizer` — [`L92`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_trainer.py#L92)
  - `input_data_collator_grain` — [`L141`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_trainer.py#L141)
  - `input_data_collator_tfds` — [`L134`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_trainer.py#L134)
  - `is_encoder_decoder` — [`L111`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_trainer.py#L111)
  - `label_pad_token_id` — [`L156`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_trainer.py#L156)
  - `max_completion_length` — [`L154`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_trainer.py#L154)
  - `max_length` — [`L152`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_trainer.py#L152)
  - `max_prompt_length` — [`L153`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_trainer.py#L153)
  - `padding_value` — [`L120`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_trainer.py#L120)
  - `processing_class` — [`L90`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_trainer.py#L90)
  - `reference_state` — [`L158`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_trainer.py#L158)
  - `running` — [`L94`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_trainer.py#L94)
  - `truncation_mode` — [`L155`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_trainer.py#L155)
- protocol/private: `__init__`[`L70`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_trainer.py#L70), `_precomputed_eval_ref_log_probs`[`L160`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_trainer.py#L160), `_precomputed_train_ref_log_probs`[`L159`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_trainer.py#L159), `_rng`[`L96`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_trainer.py#L96)
- uses (calls/refs, reference-scoped): [`config`](../../infra/base_module.md#EasyDeLBaseModule.config), [`EasyDeLBaseModule`](../../infra/base_module.md#EasyDeLBaseModule), [`EasyDeLState`](../../infra/base_state.md#EasyDeLState), [`ShardedDataSource`](../../data/core/protocols.md#ShardedDataSource), [`model`](../../infra/base_state.md#EasyDeLState.model), [`Trainer`](../trainer/trainer.md#Trainer), [`ProcessingClassType`](../../infra/utils.md#ProcessingClassType), [`open_shard`](../../data/core/protocols.md#ShardedDataSource.open_shard), [`model_state`](../base_trainer.md#BaseTrainer.model_state), [`shard_names`](../../data/core/protocols.md#ShardedDataSource.shard_names), [`_train_source`](../base_trainer.md#BaseTrainer._train_source), [`mesh`](../../infra/base_module.md#EasyDeLBaseModule.mesh), [`state_shardings`](../trainer_protocol.md#BaseTrainerProtocol.state_shardings), [`TrainerConfigureFunctionOutput`](../trainer_protocol.md#TrainerConfigureFunctionOutput), [`__init__`](../base_trainer.md#BaseTrainer.__init__), [`step_partition_spec`](../training_configurations.md#TrainingArguments.step_partition_spec), [`to_state`](../../infra/base_module.md#EasyDeLBaseModule.to_state), [`model`](../base_trainer.md#BaseTrainer.model), [`loss_config`](../training_configurations.md#TrainingArguments.loss_config), [`resolve_straight_through_emulator`](../training_utils.md#resolve_straight_through_emulator), [`gradient_accumulation_steps`](../training_configurations.md#TrainingArguments.gradient_accumulation_steps), [`scheduler`](../trainer_protocol.md#BaseTrainerProtocol.scheduler), [`quantization_mode`](../training_configurations.md#TrainingArguments.quantization_mode), [`quantization_bits`](../training_configurations.md#TrainingArguments.quantization_bits), [`quantization_group_size`](../training_configurations.md#TrainingArguments.quantization_group_size), [`get_streaming_checkpointer`](../training_configurations.md#TrainingArguments.get_streaming_checkpointer), [`_train_shared_fn_static_args`](../base_trainer.md#BaseTrainer._train_shared_fn_static_args), [`training_step`](_fn.md#training_step), [`_eval_shared_fn_static_args`](../base_trainer.md#BaseTrainer._eval_shared_fn_static_args), [`_train_shared_fn_extra_args`](../base_trainer.md#BaseTrainer._train_shared_fn_extra_args), [`_eval_shared_fn_extra_args`](../base_trainer.md#BaseTrainer._eval_shared_fn_extra_args), [`deepcopy_model`](../../utils/traversals.md#deepcopy_model), [`shardings`](../../infra/base_state.md#EasyDeLState.shardings), [`concatenated_forward`](_fn.md#concatenated_forward), [`BCOConfig`](bco_config.md#BCOConfig), [`straight_through_emulator`](../training_configurations.md#TrainingArguments.straight_through_emulator), [`tensor_straight_through`](../training_configurations.md#TrainingArguments.tensor_straight_through), [`_purify_batch`](../base_trainer.md#BaseTrainer._purify_batch), [`evaluation_step`](_fn.md#evaluation_step), [`sharded_evaluation_step_function`](../trainer_protocol.md#BaseTrainerProtocol.sharded_evaluation_step_function)  (+19 more)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`configure_functions`](../trainer/trainer.md#Trainer.configure_functions), [`Trainer`](../trainer/trainer.md#Trainer), [`_is_pretokenized`](../base_trainer.md#BaseTrainer._is_pretokenized), [`_get_preprocess_transform`](../base_trainer.md#BaseTrainer._get_preprocess_transform), [`create_grain_collect_function`](../trainer/trainer.md#Trainer.create_grain_collect_function), [`create_tfds_collect_function`](../trainer/trainer.md#Trainer.create_tfds_collect_function), [`_preprocess_batch_input`](../base_trainer.md#BaseTrainer._preprocess_batch_input)

## Module values
- `logger` — [`L45`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_trainer.py#L45)


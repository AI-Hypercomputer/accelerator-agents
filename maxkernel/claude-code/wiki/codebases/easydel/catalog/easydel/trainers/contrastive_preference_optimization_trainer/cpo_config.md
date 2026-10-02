---
title: 'Module: easydel/trainers/contrastive_preference_optimization_trainer/cpo_config.py'
type: catalog
provenance: extracted
module: easydel/trainers/contrastive_preference_optimization_trainer/cpo_config.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.contrastive_preference_optimization_trainer.cpo_config`/
symbols:
  CPOConfig.__post_init__: CPOConfig#__post_init__().
  CPOConfig: CPOConfig#
  CPOConfig.loss_type: CPOConfig#loss_type.
  LOSS_TYPES: LOSS_TYPES.
  CPOConfig.max_prompt_length: CPOConfig#max_prompt_length.
  CPOConfig.max_completion_length: CPOConfig#max_completion_length.
  CPOConfig.label_pad_token_id: CPOConfig#label_pad_token_id.
  CPOConfig.logprob_vocab_chunk_size: CPOConfig#logprob_vocab_chunk_size.
  CPOConfig.is_encoder_decoder: CPOConfig#is_encoder_decoder.
  CPOConfig.cpo_alpha: CPOConfig#cpo_alpha.
  CPOConfig.padding_value: CPOConfig#padding_value.
  CPOConfig.max_length: CPOConfig#max_length.
  CPOConfig.beta: CPOConfig#beta.
  CPOConfig.label_smoothing: CPOConfig#label_smoothing.
  CPOConfig.simpo_gamma: CPOConfig#simpo_gamma.
  CPOConfig.alpha: CPOConfig#alpha.
  CPOConfig.truncation_mode: CPOConfig#truncation_mode.
  CPOConfig.disable_dropout: CPOConfig#disable_dropout.
  CPOConfig.trainer_prefix: CPOConfig#trainer_prefix.
  CPOConfig.dataset_num_proc: CPOConfig#dataset_num_proc.
  CPOConfig.__hash__: CPOConfig#__hash__.
---
# Module: [`easydel/trainers/contrastive_preference_optimization_trainer/cpo_config.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/cpo_config.py)

## Classes
### `CPOConfig`  ·  implements/extends TrainingArguments
- def: [`easydel/trainers/contrastive_preference_optimization_trainer/cpo_config.py:29`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/cpo_config.py#L29)
- doc: Configuration class for Contrastive Preference Optimization (CPO) training.
- signature: `class CPOConfig(TrainingArguments):`
- members:
  - `alpha` — [`L87`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/cpo_config.py#L87)
  - `beta` — [`L50`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/cpo_config.py#L50)
  - `cpo_alpha` — [`L74`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/cpo_config.py#L74)
  - `dataset_num_proc` — [`L140`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/cpo_config.py#L140)
  - `disable_dropout` — [`L70`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/cpo_config.py#L70)
  - `is_encoder_decoder` — [`L136`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/cpo_config.py#L136)
  - `label_pad_token_id` — [`L96`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/cpo_config.py#L96)
  - `label_smoothing` — [`L59`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/cpo_config.py#L59)
  - `logprob_vocab_chunk_size` — [`L127`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/cpo_config.py#L127)
  - `loss_type` — [`L63`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/cpo_config.py#L63)
  - `max_completion_length` — [`L119`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/cpo_config.py#L119)
  - `max_length` — [`L111`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/cpo_config.py#L111)
  - `max_prompt_length` — [`L115`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/cpo_config.py#L115)
  - `padding_value` — [`L100`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/cpo_config.py#L100)
  - `simpo_gamma` — [`L83`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/cpo_config.py#L83)
  - `trainer_prefix` — [`L46`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/cpo_config.py#L46)
  - `truncation_mode` — [`L104`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/cpo_config.py#L104)
- protocol/private: `__hash__`[`L169`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/cpo_config.py#L169), `__post_init__`[`L145`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/cpo_config.py#L145)
- uses (calls/refs, reference-scoped): [`TrainingArguments`](../training_configurations.md#TrainingArguments), [`hash_fn`](../../utils/compiling_utils.md#hash_fn), [`__post_init__`](../training_configurations.md#TrainingArguments.__post_init__), [`_handle_deprecated_max_sequence_length`](../training_configurations.md#TrainingArguments._handle_deprecated_max_sequence_length), [`LOSS_TYPES`](cpo_config.md#LOSS_TYPES)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`TrainingArguments`](../training_configurations.md#TrainingArguments), [`concatenated_forward`](cpo_trainer.md#CPOTrainer.concatenated_forward), [`configure_functions`](cpo_trainer.md#CPOTrainer.configure_functions), [`arguments`](cpo_trainer.md#CPOTrainer.arguments), [`input_data_collator_grain`](cpo_trainer.md#CPOTrainer.input_data_collator_grain), [`__init__`](cpo_trainer.md#CPOTrainer.__init__), [`is_encoder_decoder`](cpo_trainer.md#CPOTrainer.is_encoder_decoder), [`_get_preprocess_transform`](cpo_trainer.md#CPOTrainer._get_preprocess_transform), [`input_data_collator_tfds`](cpo_trainer.md#CPOTrainer.input_data_collator_tfds), [`padding_value`](cpo_trainer.md#CPOTrainer.padding_value), [`_stored_metrics`](cpo_trainer.md#CPOTrainer._stored_metrics), [`truncation_mode`](cpo_trainer.md#CPOTrainer.truncation_mode)

## Module values
- `LOSS_TYPES` — [`L24`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/cpo_config.py#L24)


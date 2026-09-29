---
title: 'Module: easydel/trainers/binary_classifier_optimization_trainer/bco_config.py'
type: catalog
provenance: extracted
module: easydel/trainers/binary_classifier_optimization_trainer/bco_config.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.binary_classifier_optimization_trainer.bco_config`/BCOConfig#
symbols:
  BCOConfig: ''
  BCOConfig.__post_init__: __post_init__().
  BCOConfig.max_prompt_length: max_prompt_length.
  BCOConfig.max_completion_length: max_completion_length.
  BCOConfig.label_pad_token_id: label_pad_token_id.
  BCOConfig.max_length: max_length.
  BCOConfig.logprob_vocab_chunk_size: logprob_vocab_chunk_size.
  BCOConfig.is_encoder_decoder: is_encoder_decoder.
  BCOConfig.beta: beta.
  BCOConfig.padding_value: padding_value.
  BCOConfig.truncation_mode: truncation_mode.
  BCOConfig.prompt_sample_size: prompt_sample_size.
  BCOConfig.min_density_ratio: min_density_ratio.
  BCOConfig.max_density_ratio: max_density_ratio.
  BCOConfig.trainer_prefix: trainer_prefix.
  BCOConfig.disable_dropout: disable_dropout.
  BCOConfig.generate_during_eval: generate_during_eval.
  BCOConfig.precompute_ref_log_probs: precompute_ref_log_probs.
  BCOConfig.model_init_kwargs: model_init_kwargs.
  BCOConfig.ref_model_init_kwargs: ref_model_init_kwargs.
  BCOConfig.dataset_num_proc: dataset_num_proc.
  BCOConfig.__hash__: __hash__.
---
# Module: [`easydel/trainers/binary_classifier_optimization_trainer/bco_config.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_config.py)

## Classes
### `BCOConfig`  ·  implements/extends TrainingArguments
- def: [`easydel/trainers/binary_classifier_optimization_trainer/bco_config.py:28`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_config.py#L28)
- doc: Configuration container for Binary Classifier Optimisation (BCO) training.
- signature: `class BCOConfig(TrainingArguments):`
- members:
  - `beta` — [`L35`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_config.py#L35)
  - `dataset_num_proc` — [`L104`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_config.py#L104)
  - `disable_dropout` — [`L76`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_config.py#L76)
  - `generate_during_eval` — [`L80`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_config.py#L80)
  - `is_encoder_decoder` — [`L84`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_config.py#L84)
  - `label_pad_token_id` — [`L41`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_config.py#L41)
  - `logprob_vocab_chunk_size` — [`L63`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_config.py#L63)
  - `max_completion_length` — [`L57`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_config.py#L57)
  - `max_density_ratio` — [`L116`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_config.py#L116)
  - `max_length` — [`L49`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_config.py#L49)
  - `max_prompt_length` — [`L53`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_config.py#L53)
  - `min_density_ratio` — [`L112`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_config.py#L112)
  - `model_init_kwargs` — [`L94`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_config.py#L94)
  - `padding_value` — [`L45`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_config.py#L45)
  - `precompute_ref_log_probs` — [`L88`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_config.py#L88)
  - `prompt_sample_size` — [`L108`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_config.py#L108)
  - `ref_model_init_kwargs` — [`L98`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_config.py#L98)
  - `trainer_prefix` — [`L31`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_config.py#L31)
  - `truncation_mode` — [`L72`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_config.py#L72)
- protocol/private: `__hash__`[`L140`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_config.py#L140), `__post_init__`[`L121`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/bco_config.py#L121)
- uses (calls/refs, reference-scoped): [`Registry`](../../utils/registery.md#Registry), [`TrainingArguments`](../training_configurations.md#TrainingArguments), [`hash_fn`](../../utils/compiling_utils.md#hash_fn), [`register`](../../utils/registery.md#Registry.register), [`__post_init__`](../training_configurations.md#TrainingArguments.__post_init__), [`_handle_deprecated_max_sequence_length`](../training_configurations.md#TrainingArguments._handle_deprecated_max_sequence_length)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`TrainingArguments`](../training_configurations.md#TrainingArguments), [`concatenated_forward`](bco_trainer.md#BCOTrainer.concatenated_forward), [`arguments`](bco_trainer.md#BCOTrainer.arguments), [`forward_fn`](bco_trainer.md#BCOTrainer.forward_fn), [`_train_density_ratio_classifier`](bco_trainer.md#BCOTrainer._train_density_ratio_classifier), [`_preprocess_batch_input`](bco_trainer.md#BCOTrainer._preprocess_batch_input), [`input_data_collator_grain`](bco_trainer.md#BCOTrainer.input_data_collator_grain), [`__init__`](bco_trainer.md#BCOTrainer.__init__), [`_get_preprocess_transform`](bco_trainer.md#BCOTrainer._get_preprocess_transform), [`is_encoder_decoder`](bco_trainer.md#BCOTrainer.is_encoder_decoder), [`input_data_collator_tfds`](bco_trainer.md#BCOTrainer.input_data_collator_tfds), [`_rng`](bco_trainer.md#BCOTrainer._rng), [`padding_value`](bco_trainer.md#BCOTrainer.padding_value), [`beta`](bco_trainer.md#BCOTrainer.beta), [`label_pad_token_id`](bco_trainer.md#BCOTrainer.label_pad_token_id), [`max_completion_length`](bco_trainer.md#BCOTrainer.max_completion_length), [`max_length`](bco_trainer.md#BCOTrainer.max_length), [`max_prompt_length`](bco_trainer.md#BCOTrainer.max_prompt_length), [`truncation_mode`](bco_trainer.md#BCOTrainer.truncation_mode)


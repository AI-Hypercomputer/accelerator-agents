---
title: 'Module: easydel/trainers/odds_ratio_preference_optimization_trainer/orpo_config.py'
type: catalog
provenance: extracted
module: easydel/trainers/odds_ratio_preference_optimization_trainer/orpo_config.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.odds_ratio_preference_optimization_trainer.orpo_config`/ORPOConfig#
symbols:
  ORPOConfig: ''
  ORPOConfig.__post_init__: __post_init__().
  ORPOConfig.max_prompt_length: max_prompt_length.
  ORPOConfig.max_completion_length: max_completion_length.
  ORPOConfig.logprob_vocab_chunk_size: logprob_vocab_chunk_size.
  ORPOConfig.label_pad_token_id: label_pad_token_id.
  ORPOConfig.padding_value: padding_value.
  ORPOConfig.is_encoder_decoder: is_encoder_decoder.
  ORPOConfig.max_length: max_length.
  ORPOConfig.beta: beta.
  ORPOConfig.trainer_prefix: trainer_prefix.
  ORPOConfig.learning_rate: learning_rate.
  ORPOConfig.disable_dropout: disable_dropout.
  ORPOConfig.generate_during_eval: generate_during_eval.
  ORPOConfig.dataset_num_proc: dataset_num_proc.
  ORPOConfig.__hash__: __hash__.
---
# Module: [`easydel/trainers/odds_ratio_preference_optimization_trainer/orpo_config.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/odds_ratio_preference_optimization_trainer/orpo_config.py)

## Classes
### `ORPOConfig`  ·  implements/extends TrainingArguments
- def: [`easydel/trainers/odds_ratio_preference_optimization_trainer/orpo_config.py:25`](../../../../../../../raw/code/EasyDeL/easydel/trainers/odds_ratio_preference_optimization_trainer/orpo_config.py#L25)
- doc: Configuration class for Odds Ratio Preference Optimization training.
- signature: `class ORPOConfig(TrainingArguments):`
- members:
  - `__post_init__(self, max_sequence_length: int | None, quantization_block: int | None)` — [`L140`](../../../../../../../raw/code/EasyDeL/easydel/trainers/odds_ratio_preference_optimization_trainer/orpo_config.py#L140) — Post-initialization processing.
  - `beta` — [`L111`](../../../../../../../raw/code/EasyDeL/easydel/trainers/odds_ratio_preference_optimization_trainer/orpo_config.py#L111)
  - `dataset_num_proc` — [`L135`](../../../../../../../raw/code/EasyDeL/easydel/trainers/odds_ratio_preference_optimization_trainer/orpo_config.py#L135)
  - `disable_dropout` — [`L115`](../../../../../../../raw/code/EasyDeL/easydel/trainers/odds_ratio_preference_optimization_trainer/orpo_config.py#L115)
  - `generate_during_eval` — [`L127`](../../../../../../../raw/code/EasyDeL/easydel/trainers/odds_ratio_preference_optimization_trainer/orpo_config.py#L127)
  - `is_encoder_decoder` — [`L131`](../../../../../../../raw/code/EasyDeL/easydel/trainers/odds_ratio_preference_optimization_trainer/orpo_config.py#L131)
  - `label_pad_token_id` — [`L119`](../../../../../../../raw/code/EasyDeL/easydel/trainers/odds_ratio_preference_optimization_trainer/orpo_config.py#L119)
  - `learning_rate` — [`L83`](../../../../../../../raw/code/EasyDeL/easydel/trainers/odds_ratio_preference_optimization_trainer/orpo_config.py#L83)
  - `logprob_vocab_chunk_size` — [`L102`](../../../../../../../raw/code/EasyDeL/easydel/trainers/odds_ratio_preference_optimization_trainer/orpo_config.py#L102)
  - `max_completion_length` — [`L95`](../../../../../../../raw/code/EasyDeL/easydel/trainers/odds_ratio_preference_optimization_trainer/orpo_config.py#L95)
  - `max_length` — [`L87`](../../../../../../../raw/code/EasyDeL/easydel/trainers/odds_ratio_preference_optimization_trainer/orpo_config.py#L87)
  - `max_prompt_length` — [`L91`](../../../../../../../raw/code/EasyDeL/easydel/trainers/odds_ratio_preference_optimization_trainer/orpo_config.py#L91)
  - `padding_value` — [`L123`](../../../../../../../raw/code/EasyDeL/easydel/trainers/odds_ratio_preference_optimization_trainer/orpo_config.py#L123)
  - `trainer_prefix` — [`L79`](../../../../../../../raw/code/EasyDeL/easydel/trainers/odds_ratio_preference_optimization_trainer/orpo_config.py#L79)
- protocol/private: `__hash__`[`L171`](../../../../../../../raw/code/EasyDeL/easydel/trainers/odds_ratio_preference_optimization_trainer/orpo_config.py#L171)
- uses (calls/refs, reference-scoped): [`Registry`](../../utils/registery.md#Registry), [`TrainingArguments`](../training_configurations.md#TrainingArguments), [`hash_fn`](../../utils/compiling_utils.md#hash_fn), [`register`](../../utils/registery.md#Registry.register), [`__post_init__`](../training_configurations.md#TrainingArguments.__post_init__), [`_handle_deprecated_max_sequence_length`](../training_configurations.md#TrainingArguments._handle_deprecated_max_sequence_length)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`TrainingArguments`](../training_configurations.md#TrainingArguments), [`configure_functions`](orpo_trainer.md#ORPOTrainer.configure_functions), [`arguments`](orpo_trainer.md#ORPOTrainer.arguments), [`__init__`](orpo_trainer.md#ORPOTrainer.__init__), [`_get_preprocess_transform`](orpo_trainer.md#ORPOTrainer._get_preprocess_transform), [`input_data_collator_grain`](orpo_trainer.md#ORPOTrainer.input_data_collator_grain), [`input_data_collator_tfds`](orpo_trainer.md#ORPOTrainer.input_data_collator_tfds), [`padding_value`](orpo_trainer.md#ORPOTrainer.padding_value), [`is_encoder_decoder`](orpo_trainer.md#ORPOTrainer.is_encoder_decoder)


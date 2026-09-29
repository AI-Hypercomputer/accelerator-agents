---
title: 'Module: easydel/trainers/kto_trainer/kto_config.py'
type: catalog
provenance: extracted
module: easydel/trainers/kto_trainer/kto_config.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.kto_trainer.kto_config`/KTOConfig#
symbols:
  KTOConfig: ''
  KTOConfig.__post_init__: __post_init__().
  KTOConfig.max_completion_length: max_completion_length.
  KTOConfig.max_prompt_length: max_prompt_length.
  KTOConfig.max_length: max_length.
  KTOConfig.logprob_vocab_chunk_size: logprob_vocab_chunk_size.
  KTOConfig.label_pad_token_id: label_pad_token_id.
  KTOConfig.is_encoder_decoder: is_encoder_decoder.
  KTOConfig.loss_type: loss_type.
  KTOConfig.padding_value: padding_value.
  KTOConfig.beta: beta.
  KTOConfig.desirable_weight: desirable_weight.
  KTOConfig.undesirable_weight: undesirable_weight.
  KTOConfig.truncation_mode: truncation_mode.
  KTOConfig.disable_dropout: disable_dropout.
  KTOConfig.trainer_prefix: trainer_prefix.
  KTOConfig.dataset_num_proc: dataset_num_proc.
  KTOConfig.precompute_ref_log_probs: precompute_ref_log_probs.
  KTOConfig.__hash__: __hash__.
---
# Module: [`easydel/trainers/kto_trainer/kto_config.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/kto_config.py)

## Classes
### `KTOConfig`  ·  implements/extends TrainingArguments
- def: [`easydel/trainers/kto_trainer/kto_config.py:27`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/kto_config.py#L27)
- doc: Configuration for the `~easydel.trainers.KTOTrainer`.
- signature: `class KTOConfig(TrainingArguments):`
- members:
  - `beta` — [`L55`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/kto_config.py#L55)
  - `dataset_num_proc` — [`L91`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/kto_config.py#L91)
  - `desirable_weight` — [`L59`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/kto_config.py#L59)
  - `disable_dropout` — [`L87`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/kto_config.py#L87)
  - `is_encoder_decoder` — [`L83`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/kto_config.py#L83)
  - `label_pad_token_id` — [`L71`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/kto_config.py#L71)
  - `logprob_vocab_chunk_size` — [`L46`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/kto_config.py#L46)
  - `loss_type` — [`L67`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/kto_config.py#L67)
  - `max_completion_length` — [`L42`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/kto_config.py#L42)
  - `max_length` — [`L34`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/kto_config.py#L34)
  - `max_prompt_length` — [`L38`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/kto_config.py#L38)
  - `padding_value` — [`L75`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/kto_config.py#L75)
  - `precompute_ref_log_probs` — [`L95`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/kto_config.py#L95)
  - `trainer_prefix` — [`L30`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/kto_config.py#L30)
  - `truncation_mode` — [`L79`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/kto_config.py#L79)
  - `undesirable_weight` — [`L63`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/kto_config.py#L63)
- protocol/private: `__hash__`[`L117`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/kto_config.py#L117), `__post_init__`[`L100`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/kto_config.py#L100)
- uses (calls/refs, reference-scoped): [`Registry`](../../utils/registery.md#Registry), [`TrainingArguments`](../training_configurations.md#TrainingArguments), [`hash_fn`](../../utils/compiling_utils.md#hash_fn), [`register`](../../utils/registery.md#Registry.register), [`__post_init__`](../training_configurations.md#TrainingArguments.__post_init__), [`_handle_deprecated_max_sequence_length`](../training_configurations.md#TrainingArguments._handle_deprecated_max_sequence_length)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`TrainingArguments`](../training_configurations.md#TrainingArguments), [`concatenated_forward`](kto_trainer.md#KTOTrainer.concatenated_forward), [`arguments`](kto_trainer.md#KTOTrainer.arguments), [`padding_value`](kto_trainer.md#KTOTrainer.padding_value), [`__init__`](kto_trainer.md#KTOTrainer.__init__), [`forward_fn`](kto_trainer.md#KTOTrainer.forward_fn), [`is_encoder_decoder`](kto_trainer.md#KTOTrainer.is_encoder_decoder), [`_get_preprocess_transform`](kto_trainer.md#KTOTrainer._get_preprocess_transform), [`aux_loss_coef`](kto_trainer.md#KTOTrainer.aux_loss_coef), [`calculate_kl`](kto_trainer.md#KTOTrainer.calculate_kl)


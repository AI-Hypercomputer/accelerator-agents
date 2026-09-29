---
title: 'Module: easydel/trainers/direct_preference_optimization_trainer/dpo_config.py'
type: catalog
provenance: extracted
module: easydel/trainers/direct_preference_optimization_trainer/dpo_config.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.direct_preference_optimization_trainer.dpo_config`/
symbols:
  DPOConfig.__post_init__: DPOConfig#__post_init__().
  DPOConfig: DPOConfig#
  LOSS_FN_VARIANTS: LOSS_FN_VARIANTS.
  DPOConfig.loss_type: DPOConfig#loss_type.
  DPOConfig.label_pad_token_id: DPOConfig#label_pad_token_id.
  DPOConfig.max_completion_length: DPOConfig#max_completion_length.
  DPOConfig.is_encoder_decoder: DPOConfig#is_encoder_decoder.
  DPOConfig.logprob_vocab_chunk_size: DPOConfig#logprob_vocab_chunk_size.
  DPOConfig.max_prompt_length: DPOConfig#max_prompt_length.
  DPOConfig.padding_value: DPOConfig#padding_value.
  DPOConfig.max_length: DPOConfig#max_length.
  DPOConfig.beta: DPOConfig#beta.
  DPOConfig.label_smoothing: DPOConfig#label_smoothing.
  DPOConfig.precompute_ref_log_probs: DPOConfig#precompute_ref_log_probs.
  DPOConfig.reference_free: DPOConfig#reference_free.
  DPOConfig.sync_ref_model: DPOConfig#sync_ref_model.
  DPOConfig.ref_model_sync_steps: DPOConfig#ref_model_sync_steps.
  DPOConfig.trainer_prefix: DPOConfig#trainer_prefix.
  DPOConfig.use_weighting: DPOConfig#use_weighting.
  DPOConfig.disable_dropout: DPOConfig#disable_dropout.
  DPOConfig.dataset_num_proc: DPOConfig#dataset_num_proc.
  DPOConfig.force_use_ref_model: DPOConfig#force_use_ref_model.
  DPOConfig.learning_rate: DPOConfig#learning_rate.
  DPOConfig.ref_model_mixup_alpha: DPOConfig#ref_model_mixup_alpha.
  DPOConfig.rpo_alpha: DPOConfig#rpo_alpha.
  DPOConfig.tools: DPOConfig#tools.
  DPOConfig.__hash__: DPOConfig#__hash__.
---
# Module: [`easydel/trainers/direct_preference_optimization_trainer/dpo_config.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_config.py)

## Classes
### `DPOConfig`  ·  implements/extends TrainingArguments
- def: [`easydel/trainers/direct_preference_optimization_trainer/dpo_config.py:41`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_config.py#L41)
- doc: Configuration class for Direct Preference Optimization (DPO) training.
- signature: `class DPOConfig(TrainingArguments):`
- members:
  - `__post_init__(self, max_sequence_length: int | None, quantization_block: int | None)` — [`L215`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_config.py#L215) — Post-initialization processing to derive dependent parameters.
  - `beta` — [`L105`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_config.py#L105)
  - `dataset_num_proc` — [`L169`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_config.py#L169)
  - `disable_dropout` — [`L161`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_config.py#L161)
  - `force_use_ref_model` — [`L177`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_config.py#L177)
  - `is_encoder_decoder` — [`L157`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_config.py#L157)
  - `label_pad_token_id` — [`L135`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_config.py#L135)
  - `label_smoothing` — [`L114`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_config.py#L114)
  - `learning_rate` — [`L185`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_config.py#L185)
  - `logprob_vocab_chunk_size` — [`L201`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_config.py#L201)
  - `loss_type` — [`L122`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_config.py#L122)
  - `max_completion_length` — [`L151`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_config.py#L151)
  - `max_length` — [`L143`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_config.py#L143)
  - `max_prompt_length` — [`L147`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_config.py#L147)
  - `padding_value` — [`L139`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_config.py#L139)
  - `precompute_ref_log_probs` — [`L165`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_config.py#L165)
  - `ref_model_mixup_alpha` — [`L189`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_config.py#L189)
  - `ref_model_sync_steps` — [`L193`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_config.py#L193)
  - `reference_free` — [`L173`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_config.py#L173)
  - `rpo_alpha` — [`L197`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_config.py#L197)
  - `sync_ref_model` — [`L181`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_config.py#L181)
  - `tools` — [`L210`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_config.py#L210)
  - `trainer_prefix` — [`L101`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_config.py#L101)
  - `use_weighting` — [`L131`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_config.py#L131)
- protocol/private: `__hash__`[`L234`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_config.py#L234)
- uses (calls/refs, reference-scoped): [`TrainingArguments`](../training_configurations.md#TrainingArguments), [`hash_fn`](../../utils/compiling_utils.md#hash_fn), [`__post_init__`](../training_configurations.md#TrainingArguments.__post_init__), [`_handle_deprecated_max_sequence_length`](../training_configurations.md#TrainingArguments._handle_deprecated_max_sequence_length), [`LOSS_FN_VARIANTS`](dpo_config.md#LOSS_FN_VARIANTS)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`configure_functions`](dpo_trainer.md#DPOTrainer.configure_functions), [`TrainingArguments`](../training_configurations.md#TrainingArguments), [`arguments`](dpo_trainer.md#DPOTrainer.arguments), [`compute_reference_log_probs`](dpo_trainer.md#DPOTrainer.compute_reference_log_probs), [`configure_dataloaders`](dpo_trainer.md#DPOTrainer.configure_dataloaders), [`on_step_end`](dpo_trainer.md#DPOTrainer.on_step_end), [`_build_preprocess_transform`](dpo_trainer.md#DPOTrainer._build_preprocess_transform), [`input_data_collator_grain`](dpo_trainer.md#DPOTrainer.input_data_collator_grain), [`input_data_collator_tfds`](dpo_trainer.md#DPOTrainer.input_data_collator_tfds), [`__init__`](dpo_trainer.md#DPOTrainer.__init__), [`padding_value`](dpo_trainer.md#DPOTrainer.padding_value), [`_precomputed_eval_ref_log_probs`](dpo_trainer.md#DPOTrainer._precomputed_eval_ref_log_probs), [`is_encoder_decoder`](dpo_trainer.md#DPOTrainer.is_encoder_decoder)

## Module values
- `LOSS_FN_VARIANTS` — [`L23`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/dpo_config.py#L23)


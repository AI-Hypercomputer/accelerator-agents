---
title: 'Module: easydel/trainers/seq_kd_trainer/seq_kd_config.py'
type: catalog
provenance: extracted
module: easydel/trainers/seq_kd_trainer/seq_kd_config.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.seq_kd_trainer.seq_kd_config`/SeqKDConfig#
symbols:
  SeqKDConfig: ''
  SeqKDConfig.__post_init__: __post_init__().
  SeqKDConfig.max_prompt_length: max_prompt_length.
  SeqKDConfig.max_completion_length: max_completion_length.
  SeqKDConfig.skip_apply_chat_template: skip_apply_chat_template.
  SeqKDConfig.trainer_prefix: trainer_prefix.
  SeqKDConfig.remove_unused_columns: remove_unused_columns.
  SeqKDConfig.num_generations_per_prompt: num_generations_per_prompt.
  SeqKDConfig.temperature_sampling: temperature_sampling.
  SeqKDConfig.top_k: top_k.
  SeqKDConfig.top_p: top_p.
  SeqKDConfig.presence_penalty: presence_penalty.
  SeqKDConfig.frequency_penalty: frequency_penalty.
  SeqKDConfig.repetition_penalty: repetition_penalty.
  SeqKDConfig.__hash__: __hash__.
---
# Module: [`easydel/trainers/seq_kd_trainer/seq_kd_config.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/seq_kd_trainer/seq_kd_config.py)

## Classes
### `SeqKDConfig`  ·  implements/extends TrainingArguments
- def: [`easydel/trainers/seq_kd_trainer/seq_kd_config.py:34`](../../../../../../../raw/code/EasyDeL/easydel/trainers/seq_kd_trainer/seq_kd_config.py#L34)
- doc: Configuration for Sequence-level Knowledge Distillation training.
- signature: `class SeqKDConfig(TrainingArguments):`
- members:
  - `frequency_penalty` — [`L91`](../../../../../../../raw/code/EasyDeL/easydel/trainers/seq_kd_trainer/seq_kd_config.py#L91)
  - `max_completion_length` — [`L67`](../../../../../../../raw/code/EasyDeL/easydel/trainers/seq_kd_trainer/seq_kd_config.py#L67)
  - `max_prompt_length` — [`L63`](../../../../../../../raw/code/EasyDeL/easydel/trainers/seq_kd_trainer/seq_kd_config.py#L63)
  - `num_generations_per_prompt` — [`L71`](../../../../../../../raw/code/EasyDeL/easydel/trainers/seq_kd_trainer/seq_kd_config.py#L71)
  - `presence_penalty` — [`L87`](../../../../../../../raw/code/EasyDeL/easydel/trainers/seq_kd_trainer/seq_kd_config.py#L87)
  - `remove_unused_columns` — [`L59`](../../../../../../../raw/code/EasyDeL/easydel/trainers/seq_kd_trainer/seq_kd_config.py#L59)
  - `repetition_penalty` — [`L95`](../../../../../../../raw/code/EasyDeL/easydel/trainers/seq_kd_trainer/seq_kd_config.py#L95)
  - `skip_apply_chat_template` — [`L99`](../../../../../../../raw/code/EasyDeL/easydel/trainers/seq_kd_trainer/seq_kd_config.py#L99)
  - `temperature_sampling` — [`L75`](../../../../../../../raw/code/EasyDeL/easydel/trainers/seq_kd_trainer/seq_kd_config.py#L75)
  - `top_k` — [`L79`](../../../../../../../raw/code/EasyDeL/easydel/trainers/seq_kd_trainer/seq_kd_config.py#L79)
  - `top_p` — [`L83`](../../../../../../../raw/code/EasyDeL/easydel/trainers/seq_kd_trainer/seq_kd_config.py#L83)
  - `trainer_prefix` — [`L55`](../../../../../../../raw/code/EasyDeL/easydel/trainers/seq_kd_trainer/seq_kd_config.py#L55)
- protocol/private: `__hash__`[`L133`](../../../../../../../raw/code/EasyDeL/easydel/trainers/seq_kd_trainer/seq_kd_config.py#L133), `__post_init__`[`L104`](../../../../../../../raw/code/EasyDeL/easydel/trainers/seq_kd_trainer/seq_kd_config.py#L104)
- uses (calls/refs, reference-scoped): [`Registry`](../../utils/registery.md#Registry), [`TrainingArguments`](../training_configurations.md#TrainingArguments), [`max_length`](../training_configurations.md#TrainingArguments.max_length), [`hash_fn`](../../utils/compiling_utils.md#hash_fn), [`register`](../../utils/registery.md#Registry.register), [`__post_init__`](../training_configurations.md#TrainingArguments.__post_init__)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`TrainingArguments`](../training_configurations.md#TrainingArguments), [`_preprocess_batch_input`](seq_kd_trainer.md#SeqKDTrainer._preprocess_batch_input), [`__init__`](seq_kd_trainer.md#SeqKDTrainer.__init__), [`_get_preprocess_transform`](seq_kd_trainer.md#SeqKDTrainer._get_preprocess_transform), [`arguments`](seq_kd_trainer.md#SeqKDTrainer.arguments), [`create_grain_collect_function`](seq_kd_trainer.md#SeqKDTrainer.create_grain_collect_function), [`create_tfds_collect_function`](seq_kd_trainer.md#SeqKDTrainer.create_tfds_collect_function)


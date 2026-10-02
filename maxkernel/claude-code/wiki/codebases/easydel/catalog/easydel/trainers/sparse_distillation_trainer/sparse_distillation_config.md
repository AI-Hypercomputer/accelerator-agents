---
title: 'Module: easydel/trainers/sparse_distillation_trainer/sparse_distillation_config.py'
type: catalog
provenance: extracted
module: easydel/trainers/sparse_distillation_trainer/sparse_distillation_config.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.sparse_distillation_trainer.sparse_distillation_config`/SparseDistillationConfig#
symbols:
  SparseDistillationConfig: ''
  SparseDistillationConfig.__post_init__: __post_init__().
  SparseDistillationConfig.max_prompt_length: max_prompt_length.
  SparseDistillationConfig.max_completion_length: max_completion_length.
  SparseDistillationConfig.top_k_teacher: top_k_teacher.
  SparseDistillationConfig.skip_apply_chat_template: skip_apply_chat_template.
  SparseDistillationConfig.trainer_prefix: trainer_prefix.
  SparseDistillationConfig.remove_unused_columns: remove_unused_columns.
  SparseDistillationConfig.num_generations_per_prompt: num_generations_per_prompt.
  SparseDistillationConfig.temperature_sampling: temperature_sampling.
  SparseDistillationConfig.top_k: top_k.
  SparseDistillationConfig.top_p: top_p.
  SparseDistillationConfig.presence_penalty: presence_penalty.
  SparseDistillationConfig.frequency_penalty: frequency_penalty.
  SparseDistillationConfig.repetition_penalty: repetition_penalty.
  SparseDistillationConfig.__hash__: __hash__.
---
# Module: [`easydel/trainers/sparse_distillation_trainer/sparse_distillation_config.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/sparse_distillation_trainer/sparse_distillation_config.py)

## Classes
### `SparseDistillationConfig`  ·  implements/extends DistillationConfig
- def: [`easydel/trainers/sparse_distillation_trainer/sparse_distillation_config.py:34`](../../../../../../../raw/code/EasyDeL/easydel/trainers/sparse_distillation_trainer/sparse_distillation_config.py#L34)
- doc: Configuration for sparse (gray-box) knowledge distillation.
- signature: `class SparseDistillationConfig(DistillationConfig):`
- members:
  - `frequency_penalty` — [`L105`](../../../../../../../raw/code/EasyDeL/easydel/trainers/sparse_distillation_trainer/sparse_distillation_config.py#L105)
  - `max_completion_length` — [`L78`](../../../../../../../raw/code/EasyDeL/easydel/trainers/sparse_distillation_trainer/sparse_distillation_config.py#L78)
  - `max_prompt_length` — [`L74`](../../../../../../../raw/code/EasyDeL/easydel/trainers/sparse_distillation_trainer/sparse_distillation_config.py#L74)
  - `num_generations_per_prompt` — [`L82`](../../../../../../../raw/code/EasyDeL/easydel/trainers/sparse_distillation_trainer/sparse_distillation_config.py#L82)
  - `presence_penalty` — [`L101`](../../../../../../../raw/code/EasyDeL/easydel/trainers/sparse_distillation_trainer/sparse_distillation_config.py#L101)
  - `remove_unused_columns` — [`L62`](../../../../../../../raw/code/EasyDeL/easydel/trainers/sparse_distillation_trainer/sparse_distillation_config.py#L62)
  - `repetition_penalty` — [`L109`](../../../../../../../raw/code/EasyDeL/easydel/trainers/sparse_distillation_trainer/sparse_distillation_config.py#L109)
  - `skip_apply_chat_template` — [`L113`](../../../../../../../raw/code/EasyDeL/easydel/trainers/sparse_distillation_trainer/sparse_distillation_config.py#L113)
  - `temperature_sampling` — [`L86`](../../../../../../../raw/code/EasyDeL/easydel/trainers/sparse_distillation_trainer/sparse_distillation_config.py#L86)
  - `top_k` — [`L93`](../../../../../../../raw/code/EasyDeL/easydel/trainers/sparse_distillation_trainer/sparse_distillation_config.py#L93)
  - `top_k_teacher` — [`L66`](../../../../../../../raw/code/EasyDeL/easydel/trainers/sparse_distillation_trainer/sparse_distillation_config.py#L66)
  - `top_p` — [`L97`](../../../../../../../raw/code/EasyDeL/easydel/trainers/sparse_distillation_trainer/sparse_distillation_config.py#L97)
  - `trainer_prefix` — [`L58`](../../../../../../../raw/code/EasyDeL/easydel/trainers/sparse_distillation_trainer/sparse_distillation_config.py#L58)
- protocol/private: `__hash__`[`L150`](../../../../../../../raw/code/EasyDeL/easydel/trainers/sparse_distillation_trainer/sparse_distillation_config.py#L150), `__post_init__`[`L118`](../../../../../../../raw/code/EasyDeL/easydel/trainers/sparse_distillation_trainer/sparse_distillation_config.py#L118)
- uses (calls/refs, reference-scoped): [`Registry`](../../utils/registery.md#Registry), [`max_length`](../training_configurations.md#TrainingArguments.max_length), [`hash_fn`](../../utils/compiling_utils.md#hash_fn), [`register`](../../utils/registery.md#Registry.register), [`__post_init__`](../distillation_trainer/distillation_config.md#DistillationConfig.__post_init__), [`DistillationConfig`](../distillation_trainer/distillation_config.md#DistillationConfig)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`_preprocess_batch_input`](sparse_distillation_trainer.md#SparseDistillationTrainer._preprocess_batch_input), [`DistillationConfig`](../distillation_trainer/distillation_config.md#DistillationConfig), [`arguments`](sparse_distillation_trainer.md#SparseDistillationTrainer.arguments), [`__init__`](sparse_distillation_trainer.md#SparseDistillationTrainer.__init__), [`_get_preprocess_transform`](sparse_distillation_trainer.md#SparseDistillationTrainer._get_preprocess_transform), [`create_grain_collect_function`](sparse_distillation_trainer.md#SparseDistillationTrainer.create_grain_collect_function), [`create_tfds_collect_function`](sparse_distillation_trainer.md#SparseDistillationTrainer.create_tfds_collect_function)


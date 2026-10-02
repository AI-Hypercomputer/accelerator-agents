---
title: 'Module: easydel/trainers/on_policy_distillation_trainer/on_policy_distillation_config.py'
type: catalog
provenance: extracted
module: easydel/trainers/on_policy_distillation_trainer/on_policy_distillation_config.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.on_policy_distillation_trainer.on_policy_distillation_config`/OnPolicyDistillationConfig#
symbols:
  OnPolicyDistillationConfig: ''
  OnPolicyDistillationConfig.__post_init__: __post_init__().
  OnPolicyDistillationConfig.max_prompt_length: max_prompt_length.
  OnPolicyDistillationConfig.max_completion_length: max_completion_length.
  OnPolicyDistillationConfig.generate_with_teacher: generate_with_teacher.
  OnPolicyDistillationConfig.skip_apply_chat_template: skip_apply_chat_template.
  OnPolicyDistillationConfig.trainer_prefix: trainer_prefix.
  OnPolicyDistillationConfig.remove_unused_columns: remove_unused_columns.
  OnPolicyDistillationConfig.num_generations_per_prompt: num_generations_per_prompt.
  OnPolicyDistillationConfig.temperature_sampling: temperature_sampling.
  OnPolicyDistillationConfig.top_k: top_k.
  OnPolicyDistillationConfig.top_p: top_p.
  OnPolicyDistillationConfig.presence_penalty: presence_penalty.
  OnPolicyDistillationConfig.frequency_penalty: frequency_penalty.
  OnPolicyDistillationConfig.repetition_penalty: repetition_penalty.
  OnPolicyDistillationConfig.__hash__: __hash__.
---
# Module: [`easydel/trainers/on_policy_distillation_trainer/on_policy_distillation_config.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/on_policy_distillation_trainer/on_policy_distillation_config.py)

## Classes
### `OnPolicyDistillationConfig`  ·  implements/extends DistillationConfig
- def: [`easydel/trainers/on_policy_distillation_trainer/on_policy_distillation_config.py:26`](../../../../../../../raw/code/EasyDeL/easydel/trainers/on_policy_distillation_trainer/on_policy_distillation_config.py#L26)
- doc: Configuration for on-policy knowledge distillation training.
- signature: `class OnPolicyDistillationConfig(DistillationConfig):`
- members:
  - `frequency_penalty` — [`L92`](../../../../../../../raw/code/EasyDeL/easydel/trainers/on_policy_distillation_trainer/on_policy_distillation_config.py#L92)
  - `generate_with_teacher` — [`L100`](../../../../../../../raw/code/EasyDeL/easydel/trainers/on_policy_distillation_trainer/on_policy_distillation_config.py#L100)
  - `max_completion_length` — [`L65`](../../../../../../../raw/code/EasyDeL/easydel/trainers/on_policy_distillation_trainer/on_policy_distillation_config.py#L65)
  - `max_prompt_length` — [`L61`](../../../../../../../raw/code/EasyDeL/easydel/trainers/on_policy_distillation_trainer/on_policy_distillation_config.py#L61)
  - `num_generations_per_prompt` — [`L69`](../../../../../../../raw/code/EasyDeL/easydel/trainers/on_policy_distillation_trainer/on_policy_distillation_config.py#L69)
  - `presence_penalty` — [`L88`](../../../../../../../raw/code/EasyDeL/easydel/trainers/on_policy_distillation_trainer/on_policy_distillation_config.py#L88)
  - `remove_unused_columns` — [`L57`](../../../../../../../raw/code/EasyDeL/easydel/trainers/on_policy_distillation_trainer/on_policy_distillation_config.py#L57)
  - `repetition_penalty` — [`L96`](../../../../../../../raw/code/EasyDeL/easydel/trainers/on_policy_distillation_trainer/on_policy_distillation_config.py#L96)
  - `skip_apply_chat_template` — [`L107`](../../../../../../../raw/code/EasyDeL/easydel/trainers/on_policy_distillation_trainer/on_policy_distillation_config.py#L107)
  - `temperature_sampling` — [`L73`](../../../../../../../raw/code/EasyDeL/easydel/trainers/on_policy_distillation_trainer/on_policy_distillation_config.py#L73)
  - `top_k` — [`L80`](../../../../../../../raw/code/EasyDeL/easydel/trainers/on_policy_distillation_trainer/on_policy_distillation_config.py#L80)
  - `top_p` — [`L84`](../../../../../../../raw/code/EasyDeL/easydel/trainers/on_policy_distillation_trainer/on_policy_distillation_config.py#L84)
  - `trainer_prefix` — [`L53`](../../../../../../../raw/code/EasyDeL/easydel/trainers/on_policy_distillation_trainer/on_policy_distillation_config.py#L53)
- protocol/private: `__hash__`[`L141`](../../../../../../../raw/code/EasyDeL/easydel/trainers/on_policy_distillation_trainer/on_policy_distillation_config.py#L141), `__post_init__`[`L112`](../../../../../../../raw/code/EasyDeL/easydel/trainers/on_policy_distillation_trainer/on_policy_distillation_config.py#L112)
- uses (calls/refs, reference-scoped): [`Registry`](../../utils/registery.md#Registry), [`max_length`](../training_configurations.md#TrainingArguments.max_length), [`hash_fn`](../../utils/compiling_utils.md#hash_fn), [`register`](../../utils/registery.md#Registry.register), [`__post_init__`](../distillation_trainer/distillation_config.md#DistillationConfig.__post_init__), [`DistillationConfig`](../distillation_trainer/distillation_config.md#DistillationConfig)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`_preprocess_batch_input`](on_policy_distillation_trainer.md#OnPolicyDistillationTrainer._preprocess_batch_input), [`arguments`](on_policy_distillation_trainer.md#OnPolicyDistillationTrainer.arguments), [`DistillationConfig`](../distillation_trainer/distillation_config.md#DistillationConfig), [`__init__`](on_policy_distillation_trainer.md#OnPolicyDistillationTrainer.__init__), [`_get_preprocess_transform`](on_policy_distillation_trainer.md#OnPolicyDistillationTrainer._get_preprocess_transform), [`create_grain_collect_function`](on_policy_distillation_trainer.md#OnPolicyDistillationTrainer.create_grain_collect_function), [`create_tfds_collect_function`](on_policy_distillation_trainer.md#OnPolicyDistillationTrainer.create_tfds_collect_function)


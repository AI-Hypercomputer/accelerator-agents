---
title: 'Module: easydel/trainers/distillation_trainer/distillation_config.py'
type: catalog
provenance: extracted
module: easydel/trainers/distillation_trainer/distillation_config.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.distillation_trainer.distillation_config`/DistillationConfig#
symbols:
  DistillationConfig.__post_init__: __post_init__().
  DistillationConfig: ''
  DistillationConfig.temperature: temperature.
  DistillationConfig.alpha: alpha.
  DistillationConfig.hidden_state_loss_weight: hidden_state_loss_weight.
  DistillationConfig.attention_loss_weight: attention_loss_weight.
  DistillationConfig.logits_chunk_size: logits_chunk_size.
  DistillationConfig.hidden_state_layers: hidden_state_layers.
  DistillationConfig.attention_layers: attention_layers.
  DistillationConfig.completion_only_loss: completion_only_loss.
  DistillationConfig.assistant_only_loss: assistant_only_loss.
  DistillationConfig.hidden_state_loss: hidden_state_loss.
  DistillationConfig.attention_normalize: attention_normalize.
  DistillationConfig.trainer_prefix: trainer_prefix.
  DistillationConfig.dataset_text_field: dataset_text_field.
  DistillationConfig.__hash__: __hash__.
---
# Module: [`easydel/trainers/distillation_trainer/distillation_config.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/distillation_trainer/distillation_config.py)

## Classes
### `DistillationConfig`  ·  implements/extends TrainingArguments
- def: [`easydel/trainers/distillation_trainer/distillation_config.py:27`](../../../../../../../raw/code/EasyDeL/easydel/trainers/distillation_trainer/distillation_config.py#L27)
- doc: Configuration class for knowledge distillation training.
- signature: `class DistillationConfig(TrainingArguments):`
- members:
  - `alpha` — [`L75`](../../../../../../../raw/code/EasyDeL/easydel/trainers/distillation_trainer/distillation_config.py#L75)
  - `assistant_only_loss` — [`L86`](../../../../../../../raw/code/EasyDeL/easydel/trainers/distillation_trainer/distillation_config.py#L86)
  - `attention_layers` — [`L130`](../../../../../../../raw/code/EasyDeL/easydel/trainers/distillation_trainer/distillation_config.py#L130)
  - `attention_loss_weight` — [`L121`](../../../../../../../raw/code/EasyDeL/easydel/trainers/distillation_trainer/distillation_config.py#L121)
  - `attention_normalize` — [`L139`](../../../../../../../raw/code/EasyDeL/easydel/trainers/distillation_trainer/distillation_config.py#L139)
  - `completion_only_loss` — [`L95`](../../../../../../../raw/code/EasyDeL/easydel/trainers/distillation_trainer/distillation_config.py#L95)
  - `dataset_text_field` — [`L82`](../../../../../../../raw/code/EasyDeL/easydel/trainers/distillation_trainer/distillation_config.py#L82)
  - `hidden_state_layers` — [`L108`](../../../../../../../raw/code/EasyDeL/easydel/trainers/distillation_trainer/distillation_config.py#L108)
  - `hidden_state_loss` — [`L117`](../../../../../../../raw/code/EasyDeL/easydel/trainers/distillation_trainer/distillation_config.py#L117)
  - `hidden_state_loss_weight` — [`L99`](../../../../../../../raw/code/EasyDeL/easydel/trainers/distillation_trainer/distillation_config.py#L99)
  - `logits_chunk_size` — [`L148`](../../../../../../../raw/code/EasyDeL/easydel/trainers/distillation_trainer/distillation_config.py#L148)
  - `temperature` — [`L68`](../../../../../../../raw/code/EasyDeL/easydel/trainers/distillation_trainer/distillation_config.py#L68)
  - `trainer_prefix` — [`L65`](../../../../../../../raw/code/EasyDeL/easydel/trainers/distillation_trainer/distillation_config.py#L65)
- protocol/private: `__hash__`[`L197`](../../../../../../../raw/code/EasyDeL/easydel/trainers/distillation_trainer/distillation_config.py#L197), `__post_init__`[`L162`](../../../../../../../raw/code/EasyDeL/easydel/trainers/distillation_trainer/distillation_config.py#L162)
- uses (calls/refs, reference-scoped): [`Registry`](../../utils/registery.md#Registry), [`TrainingArguments`](../training_configurations.md#TrainingArguments), [`hash_fn`](../../utils/compiling_utils.md#hash_fn), [`register`](../../utils/registery.md#Registry.register), [`__post_init__`](../training_configurations.md#TrainingArguments.__post_init__), [`OnPolicyDistillationConfig`](../on_policy_distillation_trainer/on_policy_distillation_config.md#OnPolicyDistillationConfig), [`SparseDistillationConfig`](../sparse_distillation_trainer/sparse_distillation_config.md#SparseDistillationConfig)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`configure_functions`](distillation_trainer.md#DistillationTrainer.configure_functions), [`TrainingArguments`](../training_configurations.md#TrainingArguments), [`configure_functions`](../on_policy_distillation_trainer/on_policy_distillation_trainer.md#OnPolicyDistillationTrainer.configure_functions), [`configure_functions`](../sparse_distillation_trainer/sparse_distillation_trainer.md#SparseDistillationTrainer.configure_functions), [`arguments`](distillation_trainer.md#DistillationTrainer.arguments), [`__init__`](distillation_trainer.md#DistillationTrainer.__init__), [`OnPolicyDistillationConfig`](../on_policy_distillation_trainer/on_policy_distillation_config.md#OnPolicyDistillationConfig), [`SparseDistillationConfig`](../sparse_distillation_trainer/sparse_distillation_config.md#SparseDistillationConfig), [`__post_init__`](../sparse_distillation_trainer/sparse_distillation_config.md#SparseDistillationConfig.__post_init__), [`__post_init__`](../on_policy_distillation_trainer/on_policy_distillation_config.md#OnPolicyDistillationConfig.__post_init__)


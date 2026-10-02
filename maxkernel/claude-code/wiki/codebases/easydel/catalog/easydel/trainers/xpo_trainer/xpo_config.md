---
title: 'Module: easydel/trainers/xpo_trainer/xpo_config.py'
type: catalog
provenance: extracted
module: easydel/trainers/xpo_trainer/xpo_config.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.xpo_trainer.xpo_config`/XPOConfig#
symbols:
  XPOConfig: ''
  XPOConfig.__post_init__: __post_init__().
  XPOConfig.beta: beta.
  XPOConfig.alpha: alpha.
  XPOConfig.missing_eos_penalty: missing_eos_penalty.
  XPOConfig.loss_type: loss_type.
  XPOConfig.trainer_prefix: trainer_prefix.
  XPOConfig.__hash__: __hash__.
---
# Module: [`easydel/trainers/xpo_trainer/xpo_config.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/xpo_trainer/xpo_config.py)

## Classes
### `XPOConfig`  ·  implements/extends GRPOConfig
- def: [`easydel/trainers/xpo_trainer/xpo_config.py:29`](../../../../../../../raw/code/EasyDeL/easydel/trainers/xpo_trainer/xpo_config.py#L29)
- doc: Configuration for the XPO (Exploratory Preference Optimization) trainer.
- signature: `class XPOConfig(GRPOConfig):`
- members:
  - `alpha` — [`L64`](../../../../../../../raw/code/EasyDeL/easydel/trainers/xpo_trainer/xpo_config.py#L64)
  - `beta` — [`L56`](../../../../../../../raw/code/EasyDeL/easydel/trainers/xpo_trainer/xpo_config.py#L56)
  - `loss_type` — [`L51`](../../../../../../../raw/code/EasyDeL/easydel/trainers/xpo_trainer/xpo_config.py#L51)
  - `missing_eos_penalty` — [`L72`](../../../../../../../raw/code/EasyDeL/easydel/trainers/xpo_trainer/xpo_config.py#L72)
  - `trainer_prefix` — [`L46`](../../../../../../../raw/code/EasyDeL/easydel/trainers/xpo_trainer/xpo_config.py#L46)
- protocol/private: `__hash__`[`L96`](../../../../../../../raw/code/EasyDeL/easydel/trainers/xpo_trainer/xpo_config.py#L96), `__post_init__`[`L80`](../../../../../../../raw/code/EasyDeL/easydel/trainers/xpo_trainer/xpo_config.py#L80)
- uses (calls/refs, reference-scoped): [`Registry`](../../utils/registery.md#Registry), [`hash_fn`](../../utils/compiling_utils.md#hash_fn), [`register`](../../utils/registery.md#Registry.register), [`GRPOConfig`](../group_relative_policy_optimization/grpo_config.md#GRPOConfig), [`__post_init__`](../group_relative_policy_optimization/grpo_config.md#GRPOConfig.__post_init__), [`_handle_deprecated_max_sequence_length`](../training_configurations.md#TrainingArguments._handle_deprecated_max_sequence_length)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`_preprocess_batch_input`](xpo_trainer.md#XPOTrainer._preprocess_batch_input), [`GRPOConfig`](../group_relative_policy_optimization/grpo_config.md#GRPOConfig), [`arguments`](xpo_trainer.md#XPOTrainer.arguments), [`__init__`](xpo_trainer.md#XPOTrainer.__init__), [`loss_type_id`](xpo_trainer.md#XPOTrainer.loss_type_id), [`_alpha_schedule`](xpo_trainer.md#XPOTrainer._alpha_schedule), [`_beta_schedule`](xpo_trainer.md#XPOTrainer._beta_schedule)


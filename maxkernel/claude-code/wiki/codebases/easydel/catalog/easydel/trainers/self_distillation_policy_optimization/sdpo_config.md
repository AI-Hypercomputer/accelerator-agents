---
title: 'Module: easydel/trainers/self_distillation_policy_optimization/sdpo_config.py'
type: catalog
provenance: extracted
module: easydel/trainers/self_distillation_policy_optimization/sdpo_config.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.self_distillation_policy_optimization.sdpo_config`/SDPOConfig#
symbols:
  SDPOConfig: ''
  SDPOConfig.__post_init__: __post_init__().
  SDPOConfig.max_feedback_length: max_feedback_length.
  SDPOConfig.distillation_type: distillation_type.
  SDPOConfig.beta: beta.
  SDPOConfig.trainer_prefix: trainer_prefix.
  SDPOConfig.__hash__: __hash__.
---
# Module: [`easydel/trainers/self_distillation_policy_optimization/sdpo_config.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/self_distillation_policy_optimization/sdpo_config.py)

## Classes
### `SDPOConfig`  ·  implements/extends GRPOConfig
- def: [`easydel/trainers/self_distillation_policy_optimization/sdpo_config.py:25`](../../../../../../../raw/code/EasyDeL/easydel/trainers/self_distillation_policy_optimization/sdpo_config.py#L25)
- doc: Configuration for Self-Distillation Policy Optimization (SDPO).
- signature: `class SDPOConfig(GRPOConfig):`
- members:
  - `beta` — [`L89`](../../../../../../../raw/code/EasyDeL/easydel/trainers/self_distillation_policy_optimization/sdpo_config.py#L89)
  - `distillation_type` — [`L74`](../../../../../../../raw/code/EasyDeL/easydel/trainers/self_distillation_policy_optimization/sdpo_config.py#L74)
  - `max_feedback_length` — [`L61`](../../../../../../../raw/code/EasyDeL/easydel/trainers/self_distillation_policy_optimization/sdpo_config.py#L61)
  - `trainer_prefix` — [`L56`](../../../../../../../raw/code/EasyDeL/easydel/trainers/self_distillation_policy_optimization/sdpo_config.py#L56)
- protocol/private: `__hash__`[`L120`](../../../../../../../raw/code/EasyDeL/easydel/trainers/self_distillation_policy_optimization/sdpo_config.py#L120), `__post_init__`[`L101`](../../../../../../../raw/code/EasyDeL/easydel/trainers/self_distillation_policy_optimization/sdpo_config.py#L101)
- uses (calls/refs, reference-scoped): [`Registry`](../../utils/registery.md#Registry), [`max_length`](../training_configurations.md#TrainingArguments.max_length), [`hash_fn`](../../utils/compiling_utils.md#hash_fn), [`register`](../../utils/registery.md#Registry.register), [`GRPOConfig`](../group_relative_policy_optimization/grpo_config.md#GRPOConfig), [`__post_init__`](../group_relative_policy_optimization/grpo_config.md#GRPOConfig.__post_init__), [`max_prompt_length`](../group_relative_policy_optimization/grpo_config.md#GRPOConfig.max_prompt_length), [`max_completion_length`](../group_relative_policy_optimization/grpo_config.md#GRPOConfig.max_completion_length)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`_preprocess_batch_input`](sdpo_trainer.md#SDPOTrainer._preprocess_batch_input), [`configure_functions`](sdpo_trainer.md#SDPOTrainer.configure_functions), [`GRPOConfig`](../group_relative_policy_optimization/grpo_config.md#GRPOConfig), [`arguments`](sdpo_trainer.md#SDPOTrainer.arguments), [`_configure_teacher_context`](sdpo_trainer.md#SDPOTrainer._configure_teacher_context), [`__init__`](sdpo_trainer.md#SDPOTrainer.__init__), [`_effective_feedback_length`](sdpo_trainer.md#SDPOTrainer._effective_feedback_length)


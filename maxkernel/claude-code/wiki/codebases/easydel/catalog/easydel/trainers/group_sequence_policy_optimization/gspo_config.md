---
title: 'Module: easydel/trainers/group_sequence_policy_optimization/gspo_config.py'
type: catalog
provenance: extracted
module: easydel/trainers/group_sequence_policy_optimization/gspo_config.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.group_sequence_policy_optimization.gspo_config`/GSPOConfig#
symbols:
  GSPOConfig: ''
  GSPOConfig.trainer_prefix: trainer_prefix.
  GSPOConfig.importance_sampling_level: importance_sampling_level.
  GSPOConfig.epsilon: epsilon.
  GSPOConfig.epsilon_high: epsilon_high.
  GSPOConfig.beta: beta.
  GSPOConfig.loss_type: loss_type.
  GSPOConfig.__hash__: __hash__.
---
# Module: [`easydel/trainers/group_sequence_policy_optimization/gspo_config.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_sequence_policy_optimization/gspo_config.py)

## Classes
### `GSPOConfig`  ·  implements/extends GRPOConfig
- def: [`easydel/trainers/group_sequence_policy_optimization/gspo_config.py:25`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_sequence_policy_optimization/gspo_config.py#L25)
- doc: Configuration class for Group Sequence Policy Optimization training.
- signature: `class GSPOConfig(GRPOConfig):`
- members:
  - `beta` — [`L85`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_sequence_policy_optimization/gspo_config.py#L85)
  - `epsilon` — [`L71`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_sequence_policy_optimization/gspo_config.py#L71)
  - `epsilon_high` — [`L78`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_sequence_policy_optimization/gspo_config.py#L78)
  - `importance_sampling_level` — [`L64`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_sequence_policy_optimization/gspo_config.py#L64)
  - `loss_type` — [`L91`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_sequence_policy_optimization/gspo_config.py#L91)
  - `trainer_prefix` — [`L60`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_sequence_policy_optimization/gspo_config.py#L60)
- protocol/private: `__hash__`[`L99`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_sequence_policy_optimization/gspo_config.py#L99)
- uses (calls/refs, reference-scoped): [`Registry`](../../utils/registery.md#Registry), [`hash_fn`](../../utils/compiling_utils.md#hash_fn), [`register`](../../utils/registery.md#Registry.register), [`GRPOConfig`](../group_relative_policy_optimization/grpo_config.md#GRPOConfig)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`GRPOConfig`](../group_relative_policy_optimization/grpo_config.md#GRPOConfig), [`__init__`](gspo_trainer.md#GSPOTrainer.__init__), [`arguments`](gspo_trainer.md#GSPOTrainer.arguments)


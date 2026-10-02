---
title: 'Module: easydel/trainers/group_filtered_policy_optimization/gfpo_config.py'
type: catalog
provenance: extracted
module: easydel/trainers/group_filtered_policy_optimization/gfpo_config.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.group_filtered_policy_optimization.gfpo_config`/GFPOConfig#
symbols:
  GFPOConfig: ''
  GFPOConfig.num_remains_in_group: num_remains_in_group.
  GFPOConfig.__post_init__: __post_init__().
  GFPOConfig.filter_by_length: filter_by_length.
  GFPOConfig.filter_by_efficiency: filter_by_efficiency.
  GFPOConfig.length_weight: length_weight.
  GFPOConfig.efficiency_weight: efficiency_weight.
  GFPOConfig.trainer_prefix: trainer_prefix.
  GFPOConfig.__hash__: __hash__.
---
# Module: [`easydel/trainers/group_filtered_policy_optimization/gfpo_config.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_filtered_policy_optimization/gfpo_config.py)

## Classes
### `GFPOConfig`  ·  implements/extends GRPOConfig
- def: [`easydel/trainers/group_filtered_policy_optimization/gfpo_config.py:25`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_filtered_policy_optimization/gfpo_config.py#L25)
- doc: Configuration class for Group Filtered Policy Optimization training.
- signature: `class GFPOConfig(GRPOConfig):`
- members:
  - `__post_init__(self, max_sequence_length: int | None, quantization_block: int | None)` — [`L104`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_filtered_policy_optimization/gfpo_config.py#L104) — Post initialization to validate GFPO-specific parameters.
  - `efficiency_weight` — [`L96`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_filtered_policy_optimization/gfpo_config.py#L96)
  - `filter_by_efficiency` — [`L82`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_filtered_policy_optimization/gfpo_config.py#L82)
  - `filter_by_length` — [`L76`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_filtered_policy_optimization/gfpo_config.py#L76)
  - `length_weight` — [`L89`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_filtered_policy_optimization/gfpo_config.py#L89)
  - `num_remains_in_group` — [`L68`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_filtered_policy_optimization/gfpo_config.py#L68)
  - `trainer_prefix` — [`L64`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_filtered_policy_optimization/gfpo_config.py#L64)
- protocol/private: `__hash__`[`L124`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_filtered_policy_optimization/gfpo_config.py#L124)
- uses (calls/refs, reference-scoped): [`Registry`](../../utils/registery.md#Registry), [`hash_fn`](../../utils/compiling_utils.md#hash_fn), [`register`](../../utils/registery.md#Registry.register), [`GRPOConfig`](../group_relative_policy_optimization/grpo_config.md#GRPOConfig), [`__post_init__`](../group_relative_policy_optimization/grpo_config.md#GRPOConfig.__post_init__), [`num_generations`](../group_relative_policy_optimization/grpo_config.md#GRPOConfig.num_generations)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`_preprocess_batch_input`](gfpo_trainer.md#GFPOTrainer._preprocess_batch_input), [`GRPOConfig`](../group_relative_policy_optimization/grpo_config.md#GRPOConfig), [`configure_functions`](gfpo_trainer.md#GFPOTrainer.configure_functions), [`arguments`](gfpo_trainer.md#GFPOTrainer.arguments), [`__init__`](gfpo_trainer.md#GFPOTrainer.__init__), [`_default_filter_func`](gfpo_trainer.md#GFPOTrainer._default_filter_func)


---
title: 'Module: easydel/trainers/rlvr_trainer/rlvr_config.py'
type: catalog
provenance: extracted
module: easydel/trainers/rlvr_trainer/rlvr_config.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.rlvr_trainer.rlvr_config`/RLVRConfig#
symbols:
  RLVRConfig: ''
  RLVRConfig.__post_init__: __post_init__().
  RLVRConfig.answer_key: answer_key.
  RLVRConfig.length_penalty_target: length_penalty_target.
  RLVRConfig.reward_clip_range: reward_clip_range.
  RLVRConfig.format_pattern: format_pattern.
  RLVRConfig.format_reward_weight: format_reward_weight.
  RLVRConfig.length_penalty_weight: length_penalty_weight.
  RLVRConfig.max_len_mask: max_len_mask.
  RLVRConfig.trainer_prefix: trainer_prefix.
  RLVRConfig.test_key: test_key.
  RLVRConfig.difficulty_key: difficulty_key.
  RLVRConfig.difficulty_loss_weight: difficulty_loss_weight.
  RLVRConfig.__hash__: __hash__.
---
# Module: [`easydel/trainers/rlvr_trainer/rlvr_config.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/rlvr_config.py)

## Classes
### `RLVRConfig`  ·  implements/extends GRPOConfig
- def: [`easydel/trainers/rlvr_trainer/rlvr_config.py:38`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/rlvr_config.py#L38)
- doc: Configuration for Reinforcement Learning with Verifiable Rewards.
- signature: `class RLVRConfig(GRPOConfig):`
- members:
  - `answer_key` — [`L94`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/rlvr_config.py#L94)
  - `difficulty_key` — [`L132`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/rlvr_config.py#L132)
  - `difficulty_loss_weight` — [`L136`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/rlvr_config.py#L136)
  - `format_pattern` — [`L104`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/rlvr_config.py#L104)
  - `format_reward_weight` — [`L108`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/rlvr_config.py#L108)
  - `length_penalty_target` — [`L112`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/rlvr_config.py#L112)
  - `length_penalty_weight` — [`L118`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/rlvr_config.py#L118)
  - `max_len_mask` — [`L122`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/rlvr_config.py#L122)
  - `reward_clip_range` — [`L126`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/rlvr_config.py#L126)
  - `test_key` — [`L100`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/rlvr_config.py#L100)
  - `trainer_prefix` — [`L89`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/rlvr_config.py#L89)
- protocol/private: `__hash__`[`L161`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/rlvr_config.py#L161), `__post_init__`[`L141`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/rlvr_config.py#L141)
- uses (calls/refs, reference-scoped): [`Registry`](../../utils/registery.md#Registry), [`hash_fn`](../../utils/compiling_utils.md#hash_fn), [`register`](../../utils/registery.md#Registry.register), [`GRPOConfig`](../group_relative_policy_optimization/grpo_config.md#GRPOConfig), [`__post_init__`](../group_relative_policy_optimization/grpo_config.md#GRPOConfig.__post_init__), [`mask_truncated_completions`](../group_relative_policy_optimization/grpo_config.md#GRPOConfig.mask_truncated_completions)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`GRPOConfig`](../group_relative_policy_optimization/grpo_config.md#GRPOConfig), [`__init__`](rlvr_trainer.md#RLVRTrainer.__init__), [`_build_verifiers`](rlvr_trainer.md#RLVRTrainer._build_verifiers), [`arguments`](rlvr_trainer.md#RLVRTrainer.arguments)


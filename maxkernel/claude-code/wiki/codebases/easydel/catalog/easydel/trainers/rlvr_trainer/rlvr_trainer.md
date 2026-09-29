---
title: 'Module: easydel/trainers/rlvr_trainer/rlvr_trainer.py'
type: catalog
provenance: extracted
module: easydel/trainers/rlvr_trainer/rlvr_trainer.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.rlvr_trainer.rlvr_trainer`/
symbols:
  RLVRTrainer.__init__: RLVRTrainer#__init__().
  RLVRTrainer._build_verifiers: RLVRTrainer#_build_verifiers().
  RewardFunc: RewardFunc.
  RLVRTrainer._merge_reward_processing_classes: RLVRTrainer#_merge_reward_processing_classes().
  RLVRTrainer: RLVRTrainer#
  RLVRTrainer._coerce_reward_func_list: RLVRTrainer#_coerce_reward_func_list().
  RLVRTrainer._coerce_processing_class_list: RLVRTrainer#_coerce_processing_class_list().
  RLVRTrainer.arguments: RLVRTrainer#arguments.
  logger: logger.
---
# Module: [`easydel/trainers/rlvr_trainer/rlvr_trainer.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/rlvr_trainer.py)

## Classes
### `RLVRTrainer`  ·  implements/extends GRPOTrainer
- def: [`easydel/trainers/rlvr_trainer/rlvr_trainer.py:66`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/rlvr_trainer.py#L66)
- doc: Reinforcement Learning with Verifiable Rewards trainer.
- signature: `class RLVRTrainer(GRPOTrainer):`
- members:
  - `_build_verifiers(config: RLVRConfig)` — [`L194`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/rlvr_trainer.py#L194) — Construct reward verifiers from config.
  - `arguments` — [`L123`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/rlvr_trainer.py#L123)
- protocol/private: `__init__`[`L125`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/rlvr_trainer.py#L125), `_coerce_processing_class_list`[`L226`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/rlvr_trainer.py#L226), `_coerce_reward_func_list`[`L218`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/rlvr_trainer.py#L218), `_merge_reward_processing_classes`[`L247`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/rlvr_trainer.py#L247)
- uses (calls/refs, reference-scoped): [`EasyDeLBaseModule`](../../infra/base_module.md#EasyDeLBaseModule), [`EasyDeLState`](../../infra/base_state.md#EasyDeLState), [`ShardedDataSource`](../../data/core/protocols.md#ShardedDataSource), [`ProcessingClassType`](../../infra/utils.md#ProcessingClassType), [`GRPOTrainer`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer), [`__init__`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer.__init__), [`RLVRConfig`](rlvr_config.md#RLVRConfig), [`RewardFunc`](rlvr_trainer.md#RewardFunc), [`reward_weights`](../group_relative_policy_optimization/grpo_config.md#GRPOConfig.reward_weights), [`answer_key`](rlvr_config.md#RLVRConfig.answer_key), [`length_penalty_target`](rlvr_config.md#RLVRConfig.length_penalty_target), [`MathVerifier`](reward_verifiers.md#MathVerifier), [`FormatVerifier`](reward_verifiers.md#FormatVerifier), [`LengthPenaltyVerifier`](reward_verifiers.md#LengthPenaltyVerifier), [`format_pattern`](rlvr_config.md#RLVRConfig.format_pattern), [`format_reward_weight`](rlvr_config.md#RLVRConfig.format_reward_weight), [`length_penalty_weight`](rlvr_config.md#RLVRConfig.length_penalty_weight), [`logger`](rlvr_trainer.md#logger)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`GRPOTrainer`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer)

## Module values
- `RewardFunc` — [`L62`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/rlvr_trainer.py#L62)
- `logger` — [`L60`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/rlvr_trainer.py#L60)


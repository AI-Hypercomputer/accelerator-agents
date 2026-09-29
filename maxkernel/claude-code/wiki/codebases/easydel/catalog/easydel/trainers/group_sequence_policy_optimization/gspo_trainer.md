---
title: 'Module: easydel/trainers/group_sequence_policy_optimization/gspo_trainer.py'
type: catalog
provenance: extracted
module: easydel/trainers/group_sequence_policy_optimization/gspo_trainer.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.group_sequence_policy_optimization.gspo_trainer`/
symbols:
  GSPOTrainer.__init__: GSPOTrainer#__init__().
  RewardFunc: RewardFunc.
  GSPOTrainer: GSPOTrainer#
  GSPOTrainer.arguments: GSPOTrainer#arguments.
---
# Module: [`easydel/trainers/group_sequence_policy_optimization/gspo_trainer.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_sequence_policy_optimization/gspo_trainer.py)

## Classes
### `GSPOTrainer`  ·  implements/extends GRPOTrainer
- def: [`easydel/trainers/group_sequence_policy_optimization/gspo_trainer.py:37`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_sequence_policy_optimization/gspo_trainer.py#L37)
- doc: Group Sequence Policy Optimization trainer for RLHF.
- signature: `class GSPOTrainer(GRPOTrainer):`
- members:
  - `__init__(self, arguments: GSPOConfig, model: EasyDeLBaseModule | EasyDeLState | None, reward_funcs: RewardFunc | list[RewardFunc], train_dataset: Dataset | IterableDataset | ShardedDataSource | None = None, eval_dataset: Dataset | IterableDataset | ShardedDataSource | dict[str, Dataset] | None = None, processing_class: ProcessingClassType = None, reward_processing_classes: ProcessingClassType = None, data_tokenize_fn: tp.Callable | None = None)` — [`L85`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_sequence_policy_optimization/gspo_trainer.py#L85) — Initialize the GSPO trainer.
  - `arguments` — [`L83`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_sequence_policy_optimization/gspo_trainer.py#L83)
- uses (calls/refs, reference-scoped): [`EasyDeLBaseModule`](../../infra/base_module.md#EasyDeLBaseModule), [`EasyDeLState`](../../infra/base_state.md#EasyDeLState), [`ShardedDataSource`](../../data/core/protocols.md#ShardedDataSource), [`ProcessingClassType`](../../infra/utils.md#ProcessingClassType), [`GRPOTrainer`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer), [`__init__`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer.__init__), [`GSPOConfig`](gspo_config.md#GSPOConfig), [`RewardFunc`](gspo_trainer.md#RewardFunc)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`GRPOTrainer`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer)

## Module values
- `RewardFunc` — [`L33`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_sequence_policy_optimization/gspo_trainer.py#L33)


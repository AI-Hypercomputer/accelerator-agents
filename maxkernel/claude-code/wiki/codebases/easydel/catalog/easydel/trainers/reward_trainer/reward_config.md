---
title: 'Module: easydel/trainers/reward_trainer/reward_config.py'
type: catalog
provenance: extracted
module: easydel/trainers/reward_trainer/reward_config.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.reward_trainer.reward_config`/RewardConfig#
symbols:
  RewardConfig: ''
  RewardConfig.max_length: max_length.
  RewardConfig.trainer_prefix: trainer_prefix.
  RewardConfig.disable_dropout: disable_dropout.
  RewardConfig.dataset_num_proc: dataset_num_proc.
  RewardConfig.center_rewards_coefficient: center_rewards_coefficient.
  RewardConfig.remove_unused_columns: remove_unused_columns.
  RewardConfig.__hash__: __hash__.
---
# Module: [`easydel/trainers/reward_trainer/reward_config.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/reward_trainer/reward_config.py)

## Classes
### `RewardConfig`  ·  implements/extends TrainingArguments
- def: [`easydel/trainers/reward_trainer/reward_config.py:24`](../../../../../../../raw/code/EasyDeL/easydel/trainers/reward_trainer/reward_config.py#L24)
- doc: Configuration class for Reward Model training.
- signature: `class RewardConfig(TrainingArguments):`
- members:
  - `center_rewards_coefficient` — [`L91`](../../../../../../../raw/code/EasyDeL/easydel/trainers/reward_trainer/reward_config.py#L91)
  - `dataset_num_proc` — [`L87`](../../../../../../../raw/code/EasyDeL/easydel/trainers/reward_trainer/reward_config.py#L87)
  - `disable_dropout` — [`L83`](../../../../../../../raw/code/EasyDeL/easydel/trainers/reward_trainer/reward_config.py#L83)
  - `max_length` — [`L76`](../../../../../../../raw/code/EasyDeL/easydel/trainers/reward_trainer/reward_config.py#L76)
  - `remove_unused_columns` — [`L95`](../../../../../../../raw/code/EasyDeL/easydel/trainers/reward_trainer/reward_config.py#L95)
  - `trainer_prefix` — [`L72`](../../../../../../../raw/code/EasyDeL/easydel/trainers/reward_trainer/reward_config.py#L72)
- protocol/private: `__hash__`[`L103`](../../../../../../../raw/code/EasyDeL/easydel/trainers/reward_trainer/reward_config.py#L103)
- uses (calls/refs, reference-scoped): [`Registry`](../../utils/registery.md#Registry), [`TrainingArguments`](../training_configurations.md#TrainingArguments), [`hash_fn`](../../utils/compiling_utils.md#hash_fn), [`register`](../../utils/registery.md#Registry.register)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`TrainingArguments`](../training_configurations.md#TrainingArguments), [`input_data_collator_grain`](reward_trainer.md#RewardTrainer.input_data_collator_grain), [`__init__`](reward_trainer.md#RewardTrainer.__init__), [`input_data_collator_tfds`](reward_trainer.md#RewardTrainer.input_data_collator_tfds)


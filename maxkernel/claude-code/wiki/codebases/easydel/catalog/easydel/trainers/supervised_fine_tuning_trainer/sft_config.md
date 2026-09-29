---
title: 'Module: easydel/trainers/supervised_fine_tuning_trainer/sft_config.py'
type: catalog
provenance: extracted
module: easydel/trainers/supervised_fine_tuning_trainer/sft_config.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.supervised_fine_tuning_trainer.sft_config`/SFTConfig#
symbols:
  SFTConfig: ''
  SFTConfig.dataset_text_field: dataset_text_field.
  SFTConfig.packing: packing.
  SFTConfig.trainer_prefix: trainer_prefix.
  SFTConfig.add_special_tokens: add_special_tokens.
  SFTConfig.packing_strategy: packing_strategy.
  SFTConfig.assistant_only_loss: assistant_only_loss.
  SFTConfig.learning_rate: learning_rate.
  SFTConfig.dataset_num_proc: dataset_num_proc.
  SFTConfig.dataset_batch_size: dataset_batch_size.
  SFTConfig.dataset_kwargs: dataset_kwargs.
  SFTConfig.eval_packing: eval_packing.
  SFTConfig.num_of_sequences: num_of_sequences.
  SFTConfig.__hash__: __hash__.
---
# Module: [`easydel/trainers/supervised_fine_tuning_trainer/sft_config.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/supervised_fine_tuning_trainer/sft_config.py)

## Classes
### `SFTConfig`  ·  implements/extends TrainingArguments
- def: [`easydel/trainers/supervised_fine_tuning_trainer/sft_config.py:25`](../../../../../../../raw/code/EasyDeL/easydel/trainers/supervised_fine_tuning_trainer/sft_config.py#L25)
- doc: Configuration class for the \[`SFTTrainer`\].
- signature: `class SFTConfig(TrainingArguments):`
- members:
  - `add_special_tokens` — [`L58`](../../../../../../../raw/code/EasyDeL/easydel/trainers/supervised_fine_tuning_trainer/sft_config.py#L58)
  - `assistant_only_loss` — [`L76`](../../../../../../../raw/code/EasyDeL/easydel/trainers/supervised_fine_tuning_trainer/sft_config.py#L76)
  - `dataset_batch_size` — [`L94`](../../../../../../../raw/code/EasyDeL/easydel/trainers/supervised_fine_tuning_trainer/sft_config.py#L94)
  - `dataset_kwargs` — [`L98`](../../../../../../../raw/code/EasyDeL/easydel/trainers/supervised_fine_tuning_trainer/sft_config.py#L98)
  - `dataset_num_proc` — [`L90`](../../../../../../../raw/code/EasyDeL/easydel/trainers/supervised_fine_tuning_trainer/sft_config.py#L90)
  - `dataset_text_field` — [`L54`](../../../../../../../raw/code/EasyDeL/easydel/trainers/supervised_fine_tuning_trainer/sft_config.py#L54)
  - `eval_packing` — [`L102`](../../../../../../../raw/code/EasyDeL/easydel/trainers/supervised_fine_tuning_trainer/sft_config.py#L102)
  - `learning_rate` — [`L86`](../../../../../../../raw/code/EasyDeL/easydel/trainers/supervised_fine_tuning_trainer/sft_config.py#L86)
  - `num_of_sequences` — [`L106`](../../../../../../../raw/code/EasyDeL/easydel/trainers/supervised_fine_tuning_trainer/sft_config.py#L106)
  - `packing` — [`L62`](../../../../../../../raw/code/EasyDeL/easydel/trainers/supervised_fine_tuning_trainer/sft_config.py#L62)
  - `packing_strategy` — [`L69`](../../../../../../../raw/code/EasyDeL/easydel/trainers/supervised_fine_tuning_trainer/sft_config.py#L69)
  - `trainer_prefix` — [`L50`](../../../../../../../raw/code/EasyDeL/easydel/trainers/supervised_fine_tuning_trainer/sft_config.py#L50)
- protocol/private: `__hash__`[`L111`](../../../../../../../raw/code/EasyDeL/easydel/trainers/supervised_fine_tuning_trainer/sft_config.py#L111)
- uses (calls/refs, reference-scoped): [`Registry`](../../utils/registery.md#Registry), [`TrainingArguments`](../training_configurations.md#TrainingArguments), [`hash_fn`](../../utils/compiling_utils.md#hash_fn), [`register`](../../utils/registery.md#Registry.register), [`GKDConfig`](../generalized_knowledge_distillation_trainer/gkd_config.md#GKDConfig)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`TrainingArguments`](../training_configurations.md#TrainingArguments), [`__init__`](sft_trainer.md#SFTTrainer.__init__), [`GKDConfig`](../generalized_knowledge_distillation_trainer/gkd_config.md#GKDConfig), [`_dataset_text_field`](sft_trainer.md#SFTTrainer._dataset_text_field)


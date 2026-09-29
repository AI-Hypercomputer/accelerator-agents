---
title: 'Module: easydel/trainers/supervised_fine_tuning_trainer/sft_trainer.py'
type: catalog
provenance: extracted
module: easydel/trainers/supervised_fine_tuning_trainer/sft_trainer.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.supervised_fine_tuning_trainer.sft_trainer`/
symbols:
  SFTTrainer.__init__: SFTTrainer#__init__().
  SFTTrainer._apply_preprocess_transforms: SFTTrainer#_apply_preprocess_transforms().
  SFTTrainer._get_preprocess_transform: SFTTrainer#_get_preprocess_transform().
  SFTTrainer._dataset_text_field: SFTTrainer#_dataset_text_field.
  SFTTrainer: SFTTrainer#
  SFTTrainer._is_pretokenized: SFTTrainer#_is_pretokenized().
  SFTTrainer._preprocess_batch_input: SFTTrainer#_preprocess_batch_input().
  logger: logger.
  SFTTrainer.tokenizer: SFTTrainer#tokenizer.
  SFTTrainer._formatting_func: SFTTrainer#_formatting_func.
---
# Module: [`easydel/trainers/supervised_fine_tuning_trainer/sft_trainer.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/supervised_fine_tuning_trainer/sft_trainer.py)

## Classes
### `SFTTrainer`  ·  implements/extends Trainer
- def: [`easydel/trainers/supervised_fine_tuning_trainer/sft_trainer.py:40`](../../../../../../../raw/code/EasyDeL/easydel/trainers/supervised_fine_tuning_trainer/sft_trainer.py#L40)
- doc: Supervised Fine-Tuning trainer for language models.
- signature: `class SFTTrainer(Trainer):`
- members:
  - `_apply_preprocess_transforms(self)` — [`L172`](../../../../../../../raw/code/EasyDeL/easydel/trainers/supervised_fine_tuning_trainer/sft_trainer.py#L172) — Apply preprocessing transforms including optional packing.
  - `_get_preprocess_transform(self)` — [`L132`](../../../../../../../raw/code/EasyDeL/easydel/trainers/supervised_fine_tuning_trainer/sft_trainer.py#L132) — Get SFT preprocessing transform for ShardedDataSource.
  - `_is_pretokenized(self)` — [`L162`](../../../../../../../raw/code/EasyDeL/easydel/trainers/supervised_fine_tuning_trainer/sft_trainer.py#L162) — Check if dataset already has tokenized fields.
  - `tokenizer` — [`L116`](../../../../../../../raw/code/EasyDeL/easydel/trainers/supervised_fine_tuning_trainer/sft_trainer.py#L116)
- protocol/private: `__init__`[`L85`](../../../../../../../raw/code/EasyDeL/easydel/trainers/supervised_fine_tuning_trainer/sft_trainer.py#L85), `_dataset_text_field`[`L118`](../../../../../../../raw/code/EasyDeL/easydel/trainers/supervised_fine_tuning_trainer/sft_trainer.py#L118), `_formatting_func`[`L117`](../../../../../../../raw/code/EasyDeL/easydel/trainers/supervised_fine_tuning_trainer/sft_trainer.py#L117), `_preprocess_batch_input`[`L227`](../../../../../../../raw/code/EasyDeL/easydel/trainers/supervised_fine_tuning_trainer/sft_trainer.py#L227)
- uses (calls/refs, reference-scoped): [`EasyDeLBaseModule`](../../infra/base_module.md#EasyDeLBaseModule), [`EasyDeLState`](../../infra/base_state.md#EasyDeLState), [`ShardedDataSource`](../../data/core/protocols.md#ShardedDataSource), [`arguments`](../base_trainer.md#BaseTrainer.arguments), [`Trainer`](../trainer/trainer.md#Trainer), [`ProcessingClassType`](../../infra/utils.md#ProcessingClassType), [`max_length`](../training_configurations.md#TrainingArguments.max_length), [`open_shard`](../../data/core/protocols.md#ShardedDataSource.open_shard), [`shard_names`](../../data/core/protocols.md#ShardedDataSource.shard_names), [`_train_source`](../base_trainer.md#BaseTrainer._train_source), [`__init__`](../base_trainer.md#BaseTrainer.__init__), [`to_state`](../../infra/base_module.md#EasyDeLBaseModule.to_state), [`_preprocess_batch_input`](../base_trainer.md#BaseTrainer._preprocess_batch_input), [`processing_class`](../base_trainer.md#BaseTrainer.processing_class), [`SFTConfig`](sft_config.md#SFTConfig), [`_eval_source`](../base_trainer.md#BaseTrainer._eval_source), [`_preprocess_batch_input`](../generalized_knowledge_distillation_trainer/gkd_trainer.md#GKDTrainer._preprocess_batch_input), [`_apply_preprocess_transforms`](../base_trainer.md#BaseTrainer._apply_preprocess_transforms), [`PackedShardedSource`](../../data/transforms/pack.md#PackedShardedSource), [`SFTPreprocessTransform`](../prompt_transforms.md#SFTPreprocessTransform), [`get_formatting_func_from_dataset`](../utils.md#get_formatting_func_from_dataset), [`DataCollatorForCompletionOnlyLM`](../utils.md#DataCollatorForCompletionOnlyLM), [`logger`](sft_trainer.md#logger), [`GKDTrainer`](../generalized_knowledge_distillation_trainer/gkd_trainer.md#GKDTrainer), [`dataset_text_field`](sft_config.md#SFTConfig.dataset_text_field), [`packing`](sft_config.md#SFTConfig.packing)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`Trainer`](../trainer/trainer.md#Trainer), [`_is_pretokenized`](../base_trainer.md#BaseTrainer._is_pretokenized), [`_get_preprocess_transform`](../base_trainer.md#BaseTrainer._get_preprocess_transform), [`_preprocess_batch_input`](../base_trainer.md#BaseTrainer._preprocess_batch_input), [`teacher_state`](../generalized_knowledge_distillation_trainer/gkd_trainer.md#GKDTrainer.teacher_state), [`_apply_preprocess_transforms`](../base_trainer.md#BaseTrainer._apply_preprocess_transforms), [`pad_token_id`](../generalized_knowledge_distillation_trainer/gkd_trainer.md#GKDTrainer.pad_token_id), [`GKDTrainer`](../generalized_knowledge_distillation_trainer/gkd_trainer.md#GKDTrainer)

## Module values
- `logger` — [`L36`](../../../../../../../raw/code/EasyDeL/easydel/trainers/supervised_fine_tuning_trainer/sft_trainer.py#L36)


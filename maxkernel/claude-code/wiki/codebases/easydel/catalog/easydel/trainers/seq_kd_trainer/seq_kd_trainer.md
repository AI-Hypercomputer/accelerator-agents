---
title: 'Module: easydel/trainers/seq_kd_trainer/seq_kd_trainer.py'
type: catalog
provenance: extracted
module: easydel/trainers/seq_kd_trainer/seq_kd_trainer.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.seq_kd_trainer.seq_kd_trainer`/
symbols:
  SeqKDTrainer._preprocess_batch_input: SeqKDTrainer#_preprocess_batch_input().
  SeqKDTrainer.__init__: SeqKDTrainer#__init__().
  SeqKDTrainer._get_preprocess_transform: SeqKDTrainer#_get_preprocess_transform().
  SeqKDTrainer.arguments: SeqKDTrainer#arguments.
  SeqKDTrainer.create_grain_collect_function: SeqKDTrainer#create_grain_collect_function().
  SeqKDTrainer.create_tfds_collect_function: SeqKDTrainer#create_tfds_collect_function().
  SeqKDTrainer._is_pretokenized: SeqKDTrainer#_is_pretokenized().
  SeqKDTrainer: SeqKDTrainer#
  logger: logger.
  SeqKDTrainer.teacher_state: SeqKDTrainer#teacher_state.
  SeqKDTrainer.padding_value: SeqKDTrainer#padding_value.
  SeqKDTrainer.teacher_fn: SeqKDTrainer#teacher_fn.
  SeqKDTrainer.processing_class: SeqKDTrainer#processing_class.
---
# Module: [`easydel/trainers/seq_kd_trainer/seq_kd_trainer.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/seq_kd_trainer/seq_kd_trainer.py)

## Classes
### `SeqKDTrainer`  ·  implements/extends Trainer
- def: [`easydel/trainers/seq_kd_trainer/seq_kd_trainer.py:49`](../../../../../../../raw/code/EasyDeL/easydel/trainers/seq_kd_trainer/seq_kd_trainer.py#L49)
- doc: Sequence-level Knowledge Distillation trainer.
- signature: `class SeqKDTrainer(Trainer):`
- members:
  - `_get_preprocess_transform(self)` — [`L139`](../../../../../../../raw/code/EasyDeL/easydel/trainers/seq_kd_trainer/seq_kd_trainer.py#L139) — Get preprocessing transform for prompt-only datasets.
  - `_is_pretokenized(self)` — [`L150`](../../../../../../../raw/code/EasyDeL/easydel/trainers/seq_kd_trainer/seq_kd_trainer.py#L150) — Check whether the source already yields token IDs.
  - `_preprocess_batch_input(self, state: EasyDeLState, batch: dict[str, tp.Any], is_train: bool)` — [`L186`](../../../../../../../raw/code/EasyDeL/easydel/trainers/seq_kd_trainer/seq_kd_trainer.py#L186) — Generate completions from teacher and prepare a CE training batch.
  - `create_grain_collect_function(self, max_sequence_length: int, truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end")` — [`L160`](../../../../../../../raw/code/EasyDeL/easydel/trainers/seq_kd_trainer/seq_kd_trainer.py#L160) — Create Grain data collator for prompt-only batches.
  - `create_tfds_collect_function(self, max_sequence_length: int, truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end")` — [`L173`](../../../../../../../raw/code/EasyDeL/easydel/trainers/seq_kd_trainer/seq_kd_trainer.py#L173) — Create TFDS data collator for prompt-only batches.
  - `arguments` — [`L90`](../../../../../../../raw/code/EasyDeL/easydel/trainers/seq_kd_trainer/seq_kd_trainer.py#L90)
  - `padding_value` — [`L128`](../../../../../../../raw/code/EasyDeL/easydel/trainers/seq_kd_trainer/seq_kd_trainer.py#L128)
  - `processing_class` — [`L123`](../../../../../../../raw/code/EasyDeL/easydel/trainers/seq_kd_trainer/seq_kd_trainer.py#L123)
  - `teacher_fn` — [`L89`](../../../../../../../raw/code/EasyDeL/easydel/trainers/seq_kd_trainer/seq_kd_trainer.py#L89)
  - `teacher_state` — [`L88`](../../../../../../../raw/code/EasyDeL/easydel/trainers/seq_kd_trainer/seq_kd_trainer.py#L88)
- protocol/private: `__init__`[`L92`](../../../../../../../raw/code/EasyDeL/easydel/trainers/seq_kd_trainer/seq_kd_trainer.py#L92)
- uses (calls/refs, reference-scoped): [`EasyDeLBaseModule`](../../infra/base_module.md#EasyDeLBaseModule), [`EasyDeLState`](../../infra/base_state.md#EasyDeLState), [`ShardedDataSource`](../../data/core/protocols.md#ShardedDataSource), [`generate_unified`](../base_trainer.md#BaseTrainer.generate_unified), [`Trainer`](../trainer/trainer.md#Trainer), [`ProcessingClassType`](../../infra/utils.md#ProcessingClassType), [`_all_gather`](../base_trainer.md#BaseTrainer._all_gather), [`open_shard`](../../data/core/protocols.md#ShardedDataSource.open_shard), [`shard_names`](../../data/core/protocols.md#ShardedDataSource.shard_names), [`_train_source`](../base_trainer.md#BaseTrainer._train_source), [`capture_time`](../../utils/helpers.md#capture_time), [`__init__`](../base_trainer.md#BaseTrainer.__init__), [`to_state`](../../infra/base_module.md#EasyDeLBaseModule.to_state), [`_log_training_generations_to_wandb`](../base_trainer.md#BaseTrainer._log_training_generations_to_wandb), [`GRPOPreprocessTransform`](../prompt_transforms.md#GRPOPreprocessTransform), [`_coerce_generation_texts`](../base_trainer.md#BaseTrainer._coerce_generation_texts), [`text`](../base_trainer.md#GenerationResults.text), [`SeqKDConfig`](seq_kd_config.md#SeqKDConfig), [`_purify_batch`](../base_trainer.md#BaseTrainer._purify_batch), [`completion_ids`](../base_trainer.md#GenerationResults.completion_ids), [`prompt_ids`](../base_trainer.md#GenerationResults.prompt_ids), [`GRPODataCollatorGrain`](../utils.md#GRPODataCollatorGrain), [`GRPODataCollatorTFDS`](../utils.md#GRPODataCollatorTFDS), [`prompt_mask`](../base_trainer.md#GenerationResults.prompt_mask), [`_make_attn_mask`](../base_trainer.md#BaseTrainer._make_attn_mask), [`max_prompt_length`](seq_kd_config.md#SeqKDConfig.max_prompt_length), [`max_completion_length`](seq_kd_config.md#SeqKDConfig.max_completion_length), [`skip_apply_chat_template`](seq_kd_config.md#SeqKDConfig.skip_apply_chat_template)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`Trainer`](../trainer/trainer.md#Trainer), [`_is_pretokenized`](../base_trainer.md#BaseTrainer._is_pretokenized), [`_get_preprocess_transform`](../base_trainer.md#BaseTrainer._get_preprocess_transform), [`create_grain_collect_function`](../trainer/trainer.md#Trainer.create_grain_collect_function), [`create_tfds_collect_function`](../trainer/trainer.md#Trainer.create_tfds_collect_function), [`_preprocess_batch_input`](../base_trainer.md#BaseTrainer._preprocess_batch_input)

## Module values
- `logger` — [`L45`](../../../../../../../raw/code/EasyDeL/easydel/trainers/seq_kd_trainer/seq_kd_trainer.py#L45)


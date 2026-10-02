---
title: 'Module: easydel/trainers/embedding_trainer/embedding_trainer.py'
type: catalog
provenance: extracted
module: easydel/trainers/embedding_trainer/embedding_trainer.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.embedding_trainer.embedding_trainer`/
symbols:
  EmbeddingTrainer.configure_functions: EmbeddingTrainer#configure_functions().
  EmbeddingTrainer._get_preprocess_transform: EmbeddingTrainer#_get_preprocess_transform().
  EmbeddingTrainer.arguments: EmbeddingTrainer#arguments.
  EmbeddingTrainer.__init__: EmbeddingTrainer#__init__().
  EmbeddingTrainer.create_tfds_collect_function: EmbeddingTrainer#create_tfds_collect_function().
  EmbeddingTrainer.create_grain_collect_function: EmbeddingTrainer#create_grain_collect_function().
  EmbeddingTrainer._is_pretokenized: EmbeddingTrainer#_is_pretokenized().
  EmbeddingTrainer.padding_value: EmbeddingTrainer#padding_value.
  EmbeddingTrainer: EmbeddingTrainer#
  logger: logger.
  EmbeddingTrainer._embedding_config: EmbeddingTrainer#_embedding_config.
---
# Module: [`easydel/trainers/embedding_trainer/embedding_trainer.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/embedding_trainer/embedding_trainer.py)

## Classes
### `EmbeddingTrainer`  ·  implements/extends Trainer
- def: [`easydel/trainers/embedding_trainer/embedding_trainer.py:63`](../../../../../../../raw/code/EasyDeL/easydel/trainers/embedding_trainer/embedding_trainer.py#L63)
- doc: Contrastive embedding trainer for dense text representations.
- signature: `class EmbeddingTrainer(Trainer):`
- members:
  - `_get_preprocess_transform(self)` — [`L140`](../../../../../../../raw/code/EasyDeL/easydel/trainers/embedding_trainer/embedding_trainer.py#L140) — Tokenize query/positive/negative text columns.
  - `_is_pretokenized(self)` — [`L158`](../../../../../../../raw/code/EasyDeL/easydel/trainers/embedding_trainer/embedding_trainer.py#L158) — Check if the dataset is already tokenized.
  - `configure_functions(self)` — [`L166`](../../../../../../../raw/code/EasyDeL/easydel/trainers/embedding_trainer/embedding_trainer.py#L166) — Configure JIT-compiled training/eval step functions.
  - `create_grain_collect_function(self, max_sequence_length: int, truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end")` — [`L230`](../../../../../../../raw/code/EasyDeL/easydel/trainers/embedding_trainer/embedding_trainer.py#L230) — Create a Grain data collator for embedding training.
  - `create_tfds_collect_function(self, max_sequence_length: int, truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end")` — [`L218`](../../../../../../../raw/code/EasyDeL/easydel/trainers/embedding_trainer/embedding_trainer.py#L218) — Create a data collator for embedding training.
  - `arguments` — [`L102`](../../../../../../../raw/code/EasyDeL/easydel/trainers/embedding_trainer/embedding_trainer.py#L102)
  - `padding_value` — [`L119`](../../../../../../../raw/code/EasyDeL/easydel/trainers/embedding_trainer/embedding_trainer.py#L119)
- protocol/private: `__init__`[`L104`](../../../../../../../raw/code/EasyDeL/easydel/trainers/embedding_trainer/embedding_trainer.py#L104), `_embedding_config`[`L116`](../../../../../../../raw/code/EasyDeL/easydel/trainers/embedding_trainer/embedding_trainer.py#L116)
- uses (calls/refs, reference-scoped): [`EasyDeLBaseModule`](../../infra/base_module.md#EasyDeLBaseModule), [`EasyDeLState`](../../infra/base_state.md#EasyDeLState), [`ShardedDataSource`](../../data/core/protocols.md#ShardedDataSource), [`Trainer`](../trainer/trainer.md#Trainer), [`ProcessingClassType`](../../infra/utils.md#ProcessingClassType), [`open_shard`](../../data/core/protocols.md#ShardedDataSource.open_shard), [`shard_names`](../../data/core/protocols.md#ShardedDataSource.shard_names), [`_train_source`](../base_trainer.md#BaseTrainer._train_source), [`mesh`](../../infra/base_module.md#EasyDeLBaseModule.mesh), [`state_shardings`](../trainer_protocol.md#BaseTrainerProtocol.state_shardings), [`TrainerConfigureFunctionOutput`](../trainer_protocol.md#TrainerConfigureFunctionOutput), [`__init__`](../base_trainer.md#BaseTrainer.__init__), [`step_partition_spec`](../training_configurations.md#TrainingArguments.step_partition_spec), [`model`](../base_trainer.md#BaseTrainer.model), [`loss_config`](../training_configurations.md#TrainingArguments.loss_config), [`gradient_accumulation_steps`](../training_configurations.md#TrainingArguments.gradient_accumulation_steps), [`scheduler`](../trainer_protocol.md#BaseTrainerProtocol.scheduler), [`embedding_training_step`](_fn.md#embedding_training_step), [`get_streaming_checkpointer`](../training_configurations.md#TrainingArguments.get_streaming_checkpointer), [`processing_class`](../base_trainer.md#BaseTrainer.processing_class), [`_train_shared_fn_static_args`](../base_trainer.md#BaseTrainer._train_shared_fn_static_args), [`_eval_shared_fn_static_args`](../base_trainer.md#BaseTrainer._eval_shared_fn_static_args), [`_train_shared_fn_extra_args`](../base_trainer.md#BaseTrainer._train_shared_fn_extra_args), [`evaluation_step`](../trainer/_fn.md#evaluation_step), [`ensure_checkpoint_path`](../training_configurations.md#TrainingArguments.ensure_checkpoint_path), [`EmbeddingConfig`](embedding_config.md#EmbeddingConfig), [`EmbeddingPreprocessTransform`](../prompt_transforms.md#EmbeddingPreprocessTransform), [`max_length`](embedding_config.md#EmbeddingConfig.max_length), [`negative_field`](embedding_config.md#EmbeddingConfig.negative_field), [`EmbeddingDataCollatorGrain`](../utils.md#EmbeddingDataCollatorGrain), [`EmbeddingDataCollatorTFDS`](../utils.md#EmbeddingDataCollatorTFDS), [`loss_type`](embedding_config.md#EmbeddingConfig.loss_type), [`margin`](embedding_config.md#EmbeddingConfig.margin), [`matryoshka_dims`](embedding_config.md#EmbeddingConfig.matryoshka_dims), [`normalize_embeddings`](embedding_config.md#EmbeddingConfig.normalize_embeddings), [`positive_field`](embedding_config.md#EmbeddingConfig.positive_field), [`query_field`](embedding_config.md#EmbeddingConfig.query_field), [`temperature`](embedding_config.md#EmbeddingConfig.temperature)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`configure_functions`](../trainer/trainer.md#Trainer.configure_functions), [`Trainer`](../trainer/trainer.md#Trainer), [`_is_pretokenized`](../base_trainer.md#BaseTrainer._is_pretokenized), [`_get_preprocess_transform`](../base_trainer.md#BaseTrainer._get_preprocess_transform), [`create_grain_collect_function`](../trainer/trainer.md#Trainer.create_grain_collect_function), [`create_tfds_collect_function`](../trainer/trainer.md#Trainer.create_tfds_collect_function)

## Module values
- `logger` — [`L59`](../../../../../../../raw/code/EasyDeL/easydel/trainers/embedding_trainer/embedding_trainer.py#L59)


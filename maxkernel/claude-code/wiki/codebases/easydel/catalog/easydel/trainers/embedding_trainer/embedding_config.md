---
title: 'Module: easydel/trainers/embedding_trainer/embedding_config.py'
type: catalog
provenance: extracted
module: easydel/trainers/embedding_trainer/embedding_config.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.embedding_trainer.embedding_config`/EmbeddingConfig#
symbols:
  EmbeddingConfig: ''
  EmbeddingConfig.negative_field: negative_field.
  EmbeddingConfig.max_length: max_length.
  EmbeddingConfig.loss_type: loss_type.
  EmbeddingConfig.temperature: temperature.
  EmbeddingConfig.margin: margin.
  EmbeddingConfig.query_field: query_field.
  EmbeddingConfig.positive_field: positive_field.
  EmbeddingConfig.matryoshka_dims: matryoshka_dims.
  EmbeddingConfig.normalize_embeddings: normalize_embeddings.
  EmbeddingConfig.trainer_prefix: trainer_prefix.
  EmbeddingConfig.pooling_strategy: pooling_strategy.
  EmbeddingConfig.dataset_num_proc: dataset_num_proc.
  EmbeddingConfig.__hash__: __hash__.
---
# Module: [`easydel/trainers/embedding_trainer/embedding_config.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/embedding_trainer/embedding_config.py)

## Classes
### `EmbeddingConfig`  ·  implements/extends TrainingArguments
- def: [`easydel/trainers/embedding_trainer/embedding_config.py:28`](../../../../../../../raw/code/EasyDeL/easydel/trainers/embedding_trainer/embedding_config.py#L28)
- doc: Configuration for contrastive embedding model training.
- signature: `class EmbeddingConfig(TrainingArguments):`
- members:
  - `dataset_num_proc` — [`L151`](../../../../../../../raw/code/EasyDeL/easydel/trainers/embedding_trainer/embedding_config.py#L151)
  - `loss_type` — [`L78`](../../../../../../../raw/code/EasyDeL/easydel/trainers/embedding_trainer/embedding_config.py#L78)
  - `margin` — [`L100`](../../../../../../../raw/code/EasyDeL/easydel/trainers/embedding_trainer/embedding_config.py#L100)
  - `matryoshka_dims` — [`L125`](../../../../../../../raw/code/EasyDeL/easydel/trainers/embedding_trainer/embedding_config.py#L125)
  - `max_length` — [`L120`](../../../../../../../raw/code/EasyDeL/easydel/trainers/embedding_trainer/embedding_config.py#L120)
  - `negative_field` — [`L115`](../../../../../../../raw/code/EasyDeL/easydel/trainers/embedding_trainer/embedding_config.py#L115)
  - `normalize_embeddings` — [`L136`](../../../../../../../raw/code/EasyDeL/easydel/trainers/embedding_trainer/embedding_config.py#L136)
  - `pooling_strategy` — [`L141`](../../../../../../../raw/code/EasyDeL/easydel/trainers/embedding_trainer/embedding_config.py#L141)
  - `positive_field` — [`L110`](../../../../../../../raw/code/EasyDeL/easydel/trainers/embedding_trainer/embedding_config.py#L110)
  - `query_field` — [`L105`](../../../../../../../raw/code/EasyDeL/easydel/trainers/embedding_trainer/embedding_config.py#L105)
  - `temperature` — [`L90`](../../../../../../../raw/code/EasyDeL/easydel/trainers/embedding_trainer/embedding_config.py#L90)
  - `trainer_prefix` — [`L73`](../../../../../../../raw/code/EasyDeL/easydel/trainers/embedding_trainer/embedding_config.py#L73)
- protocol/private: `__hash__`[`L156`](../../../../../../../raw/code/EasyDeL/easydel/trainers/embedding_trainer/embedding_config.py#L156)
- uses (calls/refs, reference-scoped): [`Registry`](../../utils/registery.md#Registry), [`TrainingArguments`](../training_configurations.md#TrainingArguments), [`hash_fn`](../../utils/compiling_utils.md#hash_fn), [`register`](../../utils/registery.md#Registry.register)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`TrainingArguments`](../training_configurations.md#TrainingArguments), [`configure_functions`](embedding_trainer.md#EmbeddingTrainer.configure_functions), [`_get_preprocess_transform`](embedding_trainer.md#EmbeddingTrainer._get_preprocess_transform), [`arguments`](embedding_trainer.md#EmbeddingTrainer.arguments), [`__init__`](embedding_trainer.md#EmbeddingTrainer.__init__), [`create_grain_collect_function`](embedding_trainer.md#EmbeddingTrainer.create_grain_collect_function), [`create_tfds_collect_function`](embedding_trainer.md#EmbeddingTrainer.create_tfds_collect_function)


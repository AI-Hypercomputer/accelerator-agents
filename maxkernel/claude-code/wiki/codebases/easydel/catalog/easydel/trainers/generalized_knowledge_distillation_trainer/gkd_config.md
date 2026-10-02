---
title: 'Module: easydel/trainers/generalized_knowledge_distillation_trainer/gkd_config.py'
type: catalog
provenance: extracted
module: easydel/trainers/generalized_knowledge_distillation_trainer/gkd_config.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.generalized_knowledge_distillation_trainer.gkd_config`/GKDConfig#
symbols:
  GKDConfig: ''
  GKDConfig.__post_init__: __post_init__().
  GKDConfig.lmbda: lmbda.
  GKDConfig.temperature: temperature.
  GKDConfig.beta: beta.
  GKDConfig.__hash__: __hash__.
  GKDConfig.max_new_tokens: max_new_tokens.
  GKDConfig.disable_dropout: disable_dropout.
  GKDConfig.seq_kd: seq_kd.
  GKDConfig.trainer_prefix: trainer_prefix.
---
# Module: [`easydel/trainers/generalized_knowledge_distillation_trainer/gkd_config.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/generalized_knowledge_distillation_trainer/gkd_config.py)

## Classes
### `GKDConfig`  ·  implements/extends SFTConfig
- def: [`easydel/trainers/generalized_knowledge_distillation_trainer/gkd_config.py:27`](../../../../../../../raw/code/EasyDeL/easydel/trainers/generalized_knowledge_distillation_trainer/gkd_config.py#L27)
- doc: Configuration for the `~easydel.trainers.GKDTrainer`.
- signature: `class GKDConfig(SFTConfig):`
- members:
  - `beta` — [`L63`](../../../../../../../raw/code/EasyDeL/easydel/trainers/generalized_knowledge_distillation_trainer/gkd_config.py#L63)
  - `disable_dropout` — [`L74`](../../../../../../../raw/code/EasyDeL/easydel/trainers/generalized_knowledge_distillation_trainer/gkd_config.py#L74)
  - `lmbda` — [`L56`](../../../../../../../raw/code/EasyDeL/easydel/trainers/generalized_knowledge_distillation_trainer/gkd_config.py#L56)
  - `max_new_tokens` — [`L70`](../../../../../../../raw/code/EasyDeL/easydel/trainers/generalized_knowledge_distillation_trainer/gkd_config.py#L70)
  - `seq_kd` — [`L78`](../../../../../../../raw/code/EasyDeL/easydel/trainers/generalized_knowledge_distillation_trainer/gkd_config.py#L78)
  - `temperature` — [`L52`](../../../../../../../raw/code/EasyDeL/easydel/trainers/generalized_knowledge_distillation_trainer/gkd_config.py#L52)
  - `trainer_prefix` — [`L48`](../../../../../../../raw/code/EasyDeL/easydel/trainers/generalized_knowledge_distillation_trainer/gkd_config.py#L48)
- protocol/private: `__hash__`[`L83`](../../../../../../../raw/code/EasyDeL/easydel/trainers/generalized_knowledge_distillation_trainer/gkd_config.py#L83), `__post_init__`[`L85`](../../../../../../../raw/code/EasyDeL/easydel/trainers/generalized_knowledge_distillation_trainer/gkd_config.py#L85)
- uses (calls/refs, reference-scoped): [`Registry`](../../utils/registery.md#Registry), [`hash_fn`](../../utils/compiling_utils.md#hash_fn), [`register`](../../utils/registery.md#Registry.register), [`__post_init__`](../training_configurations.md#TrainingArguments.__post_init__), [`SFTConfig`](../supervised_fine_tuning_trainer/sft_config.md#SFTConfig)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`configure_functions`](gkd_trainer.md#GKDTrainer.configure_functions), [`arguments`](gkd_trainer.md#GKDTrainer.arguments), [`teacher_state`](gkd_trainer.md#GKDTrainer.teacher_state), [`generation_config`](gkd_trainer.md#GKDTrainer.generation_config), [`SFTConfig`](../supervised_fine_tuning_trainer/sft_config.md#SFTConfig), [`__init__`](gkd_trainer.md#GKDTrainer.__init__), [`lmbda`](gkd_trainer.md#GKDTrainer.lmbda), [`seq_kd`](gkd_trainer.md#GKDTrainer.seq_kd)


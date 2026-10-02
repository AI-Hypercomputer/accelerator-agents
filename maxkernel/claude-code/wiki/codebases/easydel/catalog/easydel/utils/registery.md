---
title: 'Module: easydel/utils/registery.py'
type: catalog
provenance: extracted
module: easydel/utils/registery.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.utils.registery`/
symbols:
  Registry: Registry#
  Registry.register: Registry#register().
  Registry._registry: Registry#_registry.
  Registry.do_register: Registry#do_register().
  _CategoryType: _CategoryType.
  Registry.unregister: Registry#unregister().
  Registry.get_or_raise: Registry#get_or_raise().
  Registry.clear: Registry#clear().
  _T: _T.
  Registry.get: Registry#get().
  Registry._lock: Registry#_lock.
  Registry._metadata: Registry#_metadata.
  Registry.list_implementations: Registry#list_implementations().
  Registry.register_as: Registry#register_as().
  Registry.exists: Registry#exists().
  Registry.get_metadata: Registry#get_metadata().
  Registry.list_categories: Registry#list_categories().
  Registry.get_category_registry: Registry#get_category_registry().
  Registry.decorator: Registry#decorator().
  logger: logger.
  Registry.create: Registry#create().
  Registry.info: Registry#info().
  RegistryError: RegistryError#
---
# Module: [`easydel/utils/registery.py`](../../../../../../raw/code/EasyDeL/easydel/utils/registery.py)

## Classes
### `Registry`
- def: [`easydel/utils/registery.py:33`](../../../../../../raw/code/EasyDeL/easydel/utils/registery.py#L33)
- doc: Thread-safe registry for managing implementations across different categories.
- signature: `class Registry:`
- members:
  - `clear(cls, category: _CategoryType | None = None)` — [`L318`](../../../../../../raw/code/EasyDeL/easydel/utils/registery.py#L318) — Clear registry.
  - `create(cls, category: _CategoryType, impl_name: str, *args, **kwargs)` — [`L336`](../../../../../../raw/code/EasyDeL/easydel/utils/registery.py#L336) — Create an instance of a registered implementation.
  - `decorator(_cls: type[_T])` — [`L87`](../../../../../../raw/code/EasyDeL/easydel/utils/registery.py#L87)
  - `do_register(cls, category: _CategoryType, impl_names: str | Sequence[str], impl_cls: type[_T], metadata: dict[str, tp.Any] | None = None, overwrite: bool = False)` — [`L94`](../../../../../../raw/code/EasyDeL/easydel/utils/registery.py#L94) — Register an implementation under one or more names in a category.
  - `exists(cls, category: _CategoryType, impl_name: str)` — [`L279`](../../../../../../raw/code/EasyDeL/easydel/utils/registery.py#L279) — Check if an implementation exists.
  - `get(cls, category: _CategoryType, impl_name: str, default: type[_T] | None = None)` — [`L230`](../../../../../../raw/code/EasyDeL/easydel/utils/registery.py#L230) — Get a registered implementation.
  - `get_category_registry(cls, category: _CategoryType, wakeup: bool = True)` — [`L307`](../../../../../../raw/code/EasyDeL/easydel/utils/registery.py#L307) — Get all implementations in a category.
  - `get_metadata(cls, category: _CategoryType, impl_name: str)` — [`L285`](../../../../../../raw/code/EasyDeL/easydel/utils/registery.py#L285) — Get metadata for a registered implementation.
  - `get_or_raise(cls, category: _CategoryType, impl_name: str, wakeup: bool = True)` — [`L251`](../../../../../../raw/code/EasyDeL/easydel/utils/registery.py#L251) — Get a registered implementation or raise error.
  - `info(cls)` — [`L362`](../../../../../../raw/code/EasyDeL/easydel/utils/registery.py#L362) — Get information about the registry state.
  - `list_categories(cls)` — [`L295`](../../../../../../raw/code/EasyDeL/easydel/utils/registery.py#L295) — List all registered categories.
  - `list_implementations(cls, category: _CategoryType)` — [`L301`](../../../../../../raw/code/EasyDeL/easydel/utils/registery.py#L301) — List all implementations in a category.
  - `register(cls, category: _CategoryType, impl_names: str | Sequence[str] | None = None, metadata: dict[str, tp.Any] | None = None, overwrite: bool = False)` — [`L56`](../../../../../../raw/code/EasyDeL/easydel/utils/registery.py#L56) — Decorator for registering implementations with the Registry.
  - `register_as(cls, category: _CategoryType, impl_names: str | Sequence[str], metadata: dict[str, tp.Any] | None = None, overwrite: bool = False)` — [`L157`](../../../../../../raw/code/EasyDeL/easydel/utils/registery.py#L157) — Decorator for registering implementations.
  - `unregister(cls, category: _CategoryType, impl_name: str, raise_if_missing: bool = True)` — [`L188`](../../../../../../raw/code/EasyDeL/easydel/utils/registery.py#L188) — Unregister an implementation.
- protocol/private: `_lock`[`L52`](../../../../../../raw/code/EasyDeL/easydel/utils/registery.py#L52), `_metadata`[`L53`](../../../../../../raw/code/EasyDeL/easydel/utils/registery.py#L53), `_registry`[`L51`](../../../../../../raw/code/EasyDeL/easydel/utils/registery.py#L51)
- uses (calls/refs, reference-scoped): [`_CategoryType`](registery.md#_CategoryType), [`_T`](registery.md#_T), [`logger`](registery.md#logger), [`RegistryError`](registery.md#RegistryError)
- used by: [`_import_structure`](../__init__.md#_import_structure), [`eSurge`](../inference/esurge/esurge_engine.md#eSurge), [`GRPOConfig`](../trainers/group_relative_policy_optimization/grpo_config.md#GRPOConfig), [`GRPOTrainer`](../trainers/group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer), [`DistillationConfig`](../trainers/distillation_trainer/distillation_config.md#DistillationConfig), [`SFTConfig`](../trainers/supervised_fine_tuning_trainer/sft_config.md#SFTConfig), [`RLVRConfig`](../trainers/rlvr_trainer/rlvr_config.md#RLVRConfig), [`AgenticMoshPitConfig`](../trainers/agentic_moshpit/agentic_moshpit_config.md#AgenticMoshPitConfig), [`BCOConfig`](../trainers/binary_classifier_optimization_trainer/bco_config.md#BCOConfig), [`EmbeddingConfig`](../trainers/embedding_trainer/embedding_config.md#EmbeddingConfig), [`GFPOConfig`](../trainers/group_filtered_policy_optimization/gfpo_config.md#GFPOConfig), [`GSPOConfig`](../trainers/group_sequence_policy_optimization/gspo_config.md#GSPOConfig), [`KTOConfig`](../trainers/kto_trainer/kto_config.md#KTOConfig), [`ORPOConfig`](../trainers/odds_ratio_preference_optimization_trainer/orpo_config.md#ORPOConfig), [`OnPolicyDistillationConfig`](../trainers/on_policy_distillation_trainer/on_policy_distillation_config.md#OnPolicyDistillationConfig), [`PPOConfig`](../trainers/proximal_policy_optimization_trainer/ppo_config.md#PPOConfig), [`SDPOConfig`](../trainers/self_distillation_policy_optimization/sdpo_config.md#SDPOConfig), [`SeqKDConfig`](../trainers/seq_kd_trainer/seq_kd_config.md#SeqKDConfig), [`SparseDistillationConfig`](../trainers/sparse_distillation_trainer/sparse_distillation_config.md#SparseDistillationConfig), [`RewardFunc`](../trainers/rlvr_trainer/rlvr_trainer.md#RewardFunc), [`NashMDConfig`](../trainers/nash_md_trainer/nash_md_config.md#NashMDConfig), [`RewardConfig`](../trainers/reward_trainer/reward_config.md#RewardConfig), [`XPOConfig`](../trainers/xpo_trainer/xpo_config.md#XPOConfig), [`GKDConfig`](../trainers/generalized_knowledge_distillation_trainer/gkd_config.md#GKDConfig), [`vWhisperInference`](../inference/vwhisper/core.md#vWhisperInference), [`RewardFunc`](../trainers/group_sequence_policy_optimization/gspo_trainer.md#RewardFunc), [`RewardFunc`](../trainers/proximal_policy_optimization_trainer/ppo_trainer.md#RewardFunc), [`logger`](../trainers/trainer/trainer.md#logger), [`LOSS_FN_VARIANTS`](../trainers/direct_preference_optimization_trainer/dpo_config.md#LOSS_FN_VARIANTS), [`AgenticMoshPitTrainer`](../trainers/agentic_moshpit/agentic_moshpit_trainer.md#AgenticMoshPitTrainer), [`NashMDTrainer`](../trainers/nash_md_trainer/nash_md_trainer.md#NashMDTrainer), [`SDPOTrainer`](../trainers/self_distillation_policy_optimization/sdpo_trainer.md#SDPOTrainer), [`XPOTrainer`](../trainers/xpo_trainer/xpo_trainer.md#XPOTrainer), [`get_trainer_class`](../infra/elarge/types/training.md#get_trainer_class), [`get_training_arguments_class`](../infra/elarge/types/training.md#get_training_arguments_class), [`logger`](../trainers/binary_classifier_optimization_trainer/bco_trainer.md#logger), [`logger`](../trainers/generalized_knowledge_distillation_trainer/gkd_trainer.md#logger), [`logger`](../trainers/ray_scaler/distributed_trainer.md#logger), [`AVAILABLE_BACKENDS`](../trainers/training_configurations.md#AVAILABLE_BACKENDS.AVAILABLE_BACKENDS), [`GroupFilterFunc`](../trainers/group_filtered_policy_optimization/gfpo_trainer.md#GroupFilterFunc)  (+12 more)

### `RegistryError`  ·  implements/extends Exception
- def: [`easydel/utils/registery.py:27`](../../../../../../raw/code/EasyDeL/easydel/utils/registery.py#L27)
- doc: Base exception for Registry errors.
- signature: `class RegistryError(Exception):`
- used by: [`do_register`](registery.md#Registry.do_register), [`unregister`](registery.md#Registry.unregister), [`get_or_raise`](registery.md#Registry.get_or_raise)

## Module values
- `_CategoryType` — [`L22`](../../../../../../raw/code/EasyDeL/easydel/utils/registery.py#L22)
- `_T` — [`L21`](../../../../../../raw/code/EasyDeL/easydel/utils/registery.py#L21)
- `logger` — [`L24`](../../../../../../raw/code/EasyDeL/easydel/utils/registery.py#L24)


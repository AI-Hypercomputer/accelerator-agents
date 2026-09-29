---
title: 'Module: easydel/trainers/nash_md_trainer/nash_md_config.py'
type: catalog
provenance: extracted
module: easydel/trainers/nash_md_trainer/nash_md_config.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.nash_md_trainer.nash_md_config`/NashMDConfig#
symbols:
  NashMDConfig: ''
  NashMDConfig.__post_init__: __post_init__().
  NashMDConfig.beta: beta.
  NashMDConfig.mixture_coef: mixture_coef.
  NashMDConfig.missing_eos_penalty: missing_eos_penalty.
  NashMDConfig.trainer_prefix: trainer_prefix.
  NashMDConfig.__hash__: __hash__.
---
# Module: [`easydel/trainers/nash_md_trainer/nash_md_config.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/nash_md_trainer/nash_md_config.py)

## Classes
### `NashMDConfig`  ·  implements/extends GRPOConfig
- def: [`easydel/trainers/nash_md_trainer/nash_md_config.py:28`](../../../../../../../raw/code/EasyDeL/easydel/trainers/nash_md_trainer/nash_md_config.py#L28)
- doc: Configuration for the `~easydel.trainers.NashMDTrainer`.
- signature: `class NashMDConfig(GRPOConfig):`
- members:
  - `beta` — [`L56`](../../../../../../../raw/code/EasyDeL/easydel/trainers/nash_md_trainer/nash_md_config.py#L56)
  - `missing_eos_penalty` — [`L72`](../../../../../../../raw/code/EasyDeL/easydel/trainers/nash_md_trainer/nash_md_config.py#L72)
  - `mixture_coef` — [`L64`](../../../../../../../raw/code/EasyDeL/easydel/trainers/nash_md_trainer/nash_md_config.py#L64)
  - `trainer_prefix` — [`L51`](../../../../../../../raw/code/EasyDeL/easydel/trainers/nash_md_trainer/nash_md_config.py#L51)
- protocol/private: `__hash__`[`L94`](../../../../../../../raw/code/EasyDeL/easydel/trainers/nash_md_trainer/nash_md_config.py#L94), `__post_init__`[`L80`](../../../../../../../raw/code/EasyDeL/easydel/trainers/nash_md_trainer/nash_md_config.py#L80)
- uses (calls/refs, reference-scoped): [`Registry`](../../utils/registery.md#Registry), [`hash_fn`](../../utils/compiling_utils.md#hash_fn), [`register`](../../utils/registery.md#Registry.register), [`GRPOConfig`](../group_relative_policy_optimization/grpo_config.md#GRPOConfig), [`__post_init__`](../group_relative_policy_optimization/grpo_config.md#GRPOConfig.__post_init__)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`GRPOConfig`](../group_relative_policy_optimization/grpo_config.md#GRPOConfig), [`arguments`](nash_md_trainer.md#NashMDTrainer.arguments), [`__init__`](nash_md_trainer.md#NashMDTrainer.__init__), [`_beta_schedule`](nash_md_trainer.md#NashMDTrainer._beta_schedule), [`_mixture_schedule`](nash_md_trainer.md#NashMDTrainer._mixture_schedule), [`missing_eos_penalty`](nash_md_trainer.md#NashMDTrainer.missing_eos_penalty)


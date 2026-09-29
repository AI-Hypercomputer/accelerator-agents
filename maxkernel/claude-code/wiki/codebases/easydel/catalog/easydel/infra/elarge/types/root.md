---
title: 'Module: easydel/infra/elarge/types/root.py'
type: catalog
provenance: extracted
module: easydel/infra/elarge/types/root.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.infra.elarge.types.root`/eLMConfig#
symbols:
  eLMConfig: ''
  eLMConfig.model: model.
  eLMConfig.teacher_model: teacher_model.
  eLMConfig.reference_model: reference_model.
  eLMConfig.loader: loader.
  eLMConfig.sharding: sharding.
  eLMConfig.platform: platform.
  eLMConfig.quantization: quantization.
  eLMConfig.base_config: base_config.
  eLMConfig.mixture: mixture.
  eLMConfig.esurge: esurge.
  eLMConfig.trainer: trainer.
  eLMConfig.eval: eval.
---
# Module: [`easydel/infra/elarge/types/root.py`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/root.py)

## Classes
### `eLMConfig`  ·  implements/extends _TypedDict
- def: [`easydel/infra/elarge/types/root.py:34`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/root.py#L34)
- doc: Complete ELM (EasyDeL Large Model) configuration structure.
- signature: `class eLMConfig(TypedDict, total=False):`
- members:
  - `base_config` — [`L66`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/root.py#L66)
  - `esurge` — [`L68`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/root.py#L68)
  - `eval` — [`L70`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/root.py#L70)
  - `loader` — [`L62`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/root.py#L62)
  - `mixture` — [`L67`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/root.py#L67)
  - `model` — [`L59`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/root.py#L59)
  - `platform` — [`L64`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/root.py#L64)
  - `quantization` — [`L65`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/root.py#L65)
  - `reference_model` — [`L61`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/root.py#L61)
  - `sharding` — [`L63`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/root.py#L63)
  - `teacher_model` — [`L60`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/root.py#L60)
  - `trainer` — [`L69`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/root.py#L69)
- uses (calls/refs, reference-scoped): [`TrainerConfig`](training.md#TrainerConfig), [`EvalKwargs`](eval.md#EvalKwargs), [`DataMixtureCfg`](data.md#DataMixtureCfg), [`PlatformCfg`](infra.md#PlatformCfg), [`ModelCfg`](model.md#ModelCfg), [`BaseCfg`](engine.md#BaseCfg), [`LoaderCfg`](model.md#LoaderCfg), [`QuantizationCfg`](quantization.md#QuantizationCfg), [`ShardingCfg`](infra.md#ShardingCfg), [`eSurgeCfg`](engine.md#eSurgeCfg)
- used by: [`build_model`](../builders.md#build_model), [`normalize`](../processing.md#normalize), [`build_esurge`](../builders.md#build_esurge), [`build_sharded_source`](../builders.md#build_sharded_source), [`resolve_task`](../processing.md#resolve_task), [`materialize_base_config`](../processing.md#materialize_base_config), [`to_load_state_kwargs`](../builders.md#to_load_state_kwargs), [`to_from_pretrained_kwargs`](../builders.md#to_from_pretrained_kwargs), [`build_dataset`](../builders.md#build_dataset), [`to_data_mixture_kwargs`](../builders.md#to_data_mixture_kwargs), [`build_tokenized_dataset`](../builders.md#build_tokenized_dataset), [`save_elm_config`](../processing.md#save_elm_config), [`to_esurge_kwargs`](../builders.md#to_esurge_kwargs), [`load_elm_config`](../processing.md#load_elm_config), [`validate`](../processing.md#validate), [`config`](../model.md#eLargeModel.config), [`DEFAULTS`](../defaults.md#DEFAULTS.DEFAULTS), [`__init__`](../model.md#eLargeModel.__init__)


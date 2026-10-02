---
title: 'Module: easydel/trainers/proximal_policy_optimization_trainer/modeling_value_head.py'
type: catalog
provenance: extracted
module: easydel/trainers/proximal_policy_optimization_trainer/modeling_value_head.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.proximal_policy_optimization_trainer.modeling_value_head`/
symbols:
  CausalLMWithValueHead.__init__: CausalLMWithValueHead#__init__().
  CausalLMWithValueHead.model: CausalLMWithValueHead#model.
  CausalLMWithValueHead.value_head: CausalLMWithValueHead#value_head.
  CausalLMWithValueHead: CausalLMWithValueHead#
  CausalLMWithValueHead.generate: CausalLMWithValueHead#generate().
  CausalLMWithValueHead.flops_per_token: CausalLMWithValueHead#flops_per_token().
  CausalLMWithValueHead.esurge_graphdef: CausalLMWithValueHead#esurge_graphdef().
  CausalLMWithValueHead.esurge_compatible_model: CausalLMWithValueHead#esurge_compatible_model().
  CausalLMWithValueHead.get_esurge: CausalLMWithValueHead#get_esurge().
  CausalLMWithValueHead.esurge_generate: CausalLMWithValueHead#esurge_generate().
  CausalLMWithValueHead._call_esurge_engine: CausalLMWithValueHead#_call_esurge_engine().
  CausalLMWithValueHead.pause_esurge: CausalLMWithValueHead#pause_esurge().
  CausalLMWithValueHead.resume_esurge: CausalLMWithValueHead#resume_esurge().
  CausalLMWithValueHead.list_esurge_engines: CausalLMWithValueHead#list_esurge_engines().
  CausalLMWithValueHead.__call__: CausalLMWithValueHead#__call__().
  _infer_hidden_size: _infer_hidden_size().
  CausalLMWithValueHead.config_class: CausalLMWithValueHead#config_class.
  CausalLMWithValueHead.base_model_prefix: CausalLMWithValueHead#base_model_prefix.
---
# Module: [`easydel/trainers/proximal_policy_optimization_trainer/modeling_value_head.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/modeling_value_head.py)

## Classes
### `CausalLMWithValueHead`  ·  implements/extends EasyDeLBaseModule
- def: [`easydel/trainers/proximal_policy_optimization_trainer/modeling_value_head.py:57`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/modeling_value_head.py#L57)
- doc: A lightweight wrapper that adds a scalar value head to a causal LM.
- signature: `class CausalLMWithValueHead(EasyDeLBaseModule):`
- members:
  - `__call__(self, *args, **kwargs)` — [`L118`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/modeling_value_head.py#L118) — Forward pass delegated to the underlying model.
  - `__init__(self, base_model: EasyDeLBaseModule, *, rngs: nn.Rngs | None = None)` — [`L84`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/modeling_value_head.py#L84) — Initialize the CausalLMWithValueHead wrapper.
  - `_call_esurge_engine(self, *args, **kwargs)` — [`L153`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/modeling_value_head.py#L153) — Call a pre-resolved eSurge engine via the underlying model helper.
  - `esurge_compatible_model(self)` — [`L141`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/modeling_value_head.py#L141) — Get the eSurge compatible model from the underlying model.
  - `esurge_generate(self, *args, **kwargs)` — [`L149`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/modeling_value_head.py#L149) — Generate using eSurge from the underlying model.
  - `esurge_graphdef(self)` — [`L136`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/modeling_value_head.py#L136) — Get the eSurge graph definition from the underlying model.
  - `flops_per_token(self, *args, **kwargs)` — [`L126`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/modeling_value_head.py#L126) — Calculate FLOPs per token for the underlying model.
  - `generate(self, *args, **kwargs)` — [`L122`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/modeling_value_head.py#L122) — Generate sequences using the underlying model.
  - `get_esurge(self, *args, **kwargs)` — [`L145`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/modeling_value_head.py#L145) — Get eSurge engine from the underlying model.
  - `list_esurge_engines(self, *args, **kwargs)` — [`L165`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/modeling_value_head.py#L165) — List eSurge engines from the underlying model.
  - `pause_esurge(self, *args, **kwargs)` — [`L157`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/modeling_value_head.py#L157) — Pause eSurge engine in the underlying model.
  - `resume_esurge(self, *args, **kwargs)` — [`L161`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/modeling_value_head.py#L161) — Resume eSurge engine in the underlying model.
  - `base_model_prefix` — [`L82`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/modeling_value_head.py#L82)
  - `config_class` — [`L81`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/modeling_value_head.py#L81)
  - `model` — [`L105`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/modeling_value_head.py#L105)
  - `value_head` — [`L107`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/modeling_value_head.py#L107)
- uses (calls/refs, reference-scoped): [`config`](../../infra/base_module.md#EasyDeLBaseModule.config), [`EasyDeLBaseModule`](../../infra/base_module.md#EasyDeLBaseModule), [`__init__`](../../infra/base_module.md#EasyDeLBaseModule.__init__), [`flops_per_token`](../../infra/base_module.md#EasyDeLBaseModule.flops_per_token), [`generate`](../../infra/mixins/generation.md#EasyGenerationMixin.generate), [`get_esurge`](../../infra/mixins/generation.md#EasyGenerationMixin.get_esurge), [`dtype`](../../infra/base_module.md#EasyDeLBaseModule.dtype), [`param_dtype`](../../infra/base_module.md#EasyDeLBaseModule.param_dtype), [`esurge_compatible_model`](../../infra/mixins/generation.md#EasyGenerationMixin.esurge_compatible_model), [`rngs`](../../infra/base_module.md#EasyDeLBaseModule.rngs), [`resume_esurge`](../../infra/mixins/generation.md#EasyGenerationMixin.resume_esurge), [`pause_esurge`](../../infra/mixins/generation.md#EasyGenerationMixin.pause_esurge), [`esurge_generate`](../../infra/mixins/generation.md#EasyGenerationMixin.esurge_generate), [`precision`](../../infra/base_module.md#EasyDeLBaseModule.precision), [`list_esurge_engines`](../../infra/mixins/generation.md#EasyGenerationMixin.list_esurge_engines), [`_call_esurge_engine`](../../infra/mixins/generation.md#EasyGenerationMixin._call_esurge_engine), [`esurge_graphdef`](../../infra/mixins/generation.md#EasyGenerationMixin.esurge_graphdef), [`_infer_hidden_size`](modeling_value_head.md#_infer_hidden_size)
- used by: [`EasyDeLBaseModule`](../../infra/base_module.md#EasyDeLBaseModule), [`flops_per_token`](../../infra/base_module.md#EasyDeLBaseModule.flops_per_token), [`generate`](../../infra/mixins/generation.md#EasyGenerationMixin.generate), [`get_esurge`](../../infra/mixins/generation.md#EasyGenerationMixin.get_esurge), [`__init__`](ppo_trainer.md#PPOTrainer.__init__), [`esurge_compatible_model`](../../infra/mixins/generation.md#EasyGenerationMixin.esurge_compatible_model), [`resume_esurge`](../../infra/mixins/generation.md#EasyGenerationMixin.resume_esurge), [`pause_esurge`](../../infra/mixins/generation.md#EasyGenerationMixin.pause_esurge), [`esurge_generate`](../../infra/mixins/generation.md#EasyGenerationMixin.esurge_generate), [`list_esurge_engines`](../../infra/mixins/generation.md#EasyGenerationMixin.list_esurge_engines), [`_call_esurge_engine`](../../infra/mixins/generation.md#EasyGenerationMixin._call_esurge_engine), [`esurge_graphdef`](../../infra/mixins/generation.md#EasyGenerationMixin.esurge_graphdef)

## Functions
- `_infer_hidden_size(config)` — [`L32`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/modeling_value_head.py#L32) — Infer the hidden size from a model configuration.


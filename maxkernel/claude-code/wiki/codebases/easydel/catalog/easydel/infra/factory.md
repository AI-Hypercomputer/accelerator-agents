---
title: 'Module: easydel/infra/factory.py'
type: catalog
provenance: extracted
module: easydel/infra/factory.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.infra.factory`/
symbols:
  TaskType: TaskType#
  register_module: register_module.
  register_config: register_config.
  TaskType.CAUSAL_LM: TaskType#CAUSAL_LM.
  TaskType.BASE_MODULE: TaskType#BASE_MODULE.
  TaskType.SEQUENCE_CLASSIFICATION: TaskType#SEQUENCE_CLASSIFICATION.
  TaskType.IMAGE_TEXT_TO_TEXT: TaskType#IMAGE_TEXT_TO_TEXT.
  registry: registry.
  Registry.get_config: Registry#get_config().
  TaskType.BASE_VISION: TaskType#BASE_VISION.
  Registry.wrapper: Registry#wrapper().
  TaskType.SPEECH_SEQUENCE_TO_SEQUENCE: TaskType#SPEECH_SEQUENCE_TO_SEQUENCE.
  TaskType.EMBEDDING: TaskType#EMBEDDING.
  Registry.register_config: Registry#register_config().
  Registry.register_module: Registry#register_module().
  Registry.get_module_registration: Registry#get_module_registration().
  TaskType.SEQUENCE_TO_SEQUENCE: TaskType#SEQUENCE_TO_SEQUENCE.
  TaskType.ANY_TO_ANY: TaskType#ANY_TO_ANY.
  TaskType.VISION_LM: TaskType#VISION_LM.
  TaskType.ZERO_SHOT_IMAGE_CLASSIFICATION: TaskType#ZERO_SHOT_IMAGE_CLASSIFICATION.
  TaskType.AUTO_BIND: TaskType#AUTO_BIND.
  T: T.
  ConfigType: ConfigType#
  TaskType.DIFFUSION_LM: TaskType#DIFFUSION_LM.
  Registry._config_registry: Registry#_config_registry.
  Registry._task_registry: Registry#_task_registry.
  Registry.task_registry: Registry#task_registry().
  TaskType.IMAGE_CLASSIFICATION: TaskType#IMAGE_CLASSIFICATION.
  TaskType.AUDIO_CLASSIFICATION: TaskType#AUDIO_CLASSIFICATION.
  Registry.config_registry: Registry#config_registry().
  ModuleRegistration.module: ModuleRegistration#module.
  Registry._str: Registry#_str().
  ModuleRegistration: ModuleRegistration#
  ModuleRegistration.config: ModuleRegistration#config.
  ConfigType.MODULE_CONFIG: ConfigType#MODULE_CONFIG.
  _SENTINEL: _SENTINEL.
  Registry: Registry#
  ModuleRegistration.embedding_layer_names: ModuleRegistration#embedding_layer_names.
  ModuleRegistration.layernorm_names: ModuleRegistration#layernorm_names.
  Registry.__init__: Registry#__init__().
---
# Module: [`easydel/infra/factory.py`](../../../../../../raw/code/EasyDeL/easydel/infra/factory.py)

## Classes
### `ConfigType`  ·  implements/extends StrEnum
- def: [`easydel/infra/factory.py:92`](../../../../../../raw/code/EasyDeL/easydel/infra/factory.py#L92)
- doc: Enumeration of configuration types that can be registered in the registry.
- signature: `class ConfigType(StrEnum):`
- members:
  - `MODULE_CONFIG` — [`L114`](../../../../../../raw/code/EasyDeL/easydel/infra/factory.py#L114)
- used by: [`_import_structure`](../__init__.md#_import_structure), [`get_config`](factory.md#Registry.get_config), [`register_config`](factory.md#Registry.register_config), [`_config_registry`](factory.md#Registry._config_registry), [`config_registry`](factory.md#Registry.config_registry)

### `ModuleRegistration`
- def: [`easydel/infra/factory.py:201`](../../../../../../raw/code/EasyDeL/easydel/infra/factory.py#L201)
- doc: Container for registered EasyDeL module metadata.
- signature: `class ModuleRegistration:`
- members:
  - `config` — [`L250`](../../../../../../raw/code/EasyDeL/easydel/infra/factory.py#L250) — ---
  - `embedding_layer_names` — [`L251`](../../../../../../raw/code/EasyDeL/easydel/infra/factory.py#L251)
  - `layernorm_names` — [`L252`](../../../../../../raw/code/EasyDeL/easydel/infra/factory.py#L252)
  - `module` — [`L249`](../../../../../../raw/code/EasyDeL/easydel/infra/factory.py#L249) — ---
- uses (calls/refs, reference-scoped): [`EasyDeLBaseModule`](base_module.md#EasyDeLBaseModule), [`EasyDeLBaseConfig`](base_config.md#EasyDeLBaseConfig)
- used by: [`get_modules_by_type`](../modules/auto/auto_configuration.md#get_modules_by_type), [`from_config`](../modules/auto/auto_modeling.md#BaseAutoEasyModel.from_config), [`wrapper`](factory.md#Registry.wrapper), [`get_module_registration`](factory.md#Registry.get_module_registration), [`_task_registry`](factory.md#Registry._task_registry), [`task_registry`](factory.md#Registry.task_registry)

### `Registry`
- def: [`easydel/infra/factory.py:255`](../../../../../../raw/code/EasyDeL/easydel/infra/factory.py#L255)
- doc: Central registry for managing EasyDeL configurations and modules.
- signature: `class Registry:`
- members:
  - `__init__(self)` — [`L310`](../../../../../../raw/code/EasyDeL/easydel/infra/factory.py#L310) — Initialize the registry with empty configuration and task registries.
  - `_str(self)` — [`L380`](../../../../../../raw/code/EasyDeL/easydel/infra/factory.py#L380) — Generate a pretty-printed string representation.
  - `config_registry(self)` — [`L658`](../../../../../../raw/code/EasyDeL/easydel/infra/factory.py#L658) — Provide read-only access to the configuration registry.
  - `get_config(self, config_type: str, config_field: ConfigType = ConfigType.MODULE_CONFIG)` — [`L502`](../../../../../../raw/code/EasyDeL/easydel/infra/factory.py#L502) — Retrieve a registered configuration class by its type identifier.
  - `get_module_registration(self, task_type: TaskType | tp.Literal["causal-language-model", "sequence-classification", "vision-language-model", "audio-classification", "base-module", "sequence-to-sequence",], model_type: str)` — [`L553`](../../../../../../raw/code/EasyDeL/easydel/infra/factory.py#L553) — Retrieve the registration information for a specific module.
  - `register_config(self, config_type: str, config_field: ConfigType = ConfigType.MODULE_CONFIG)` — [`L321`](../../../../../../raw/code/EasyDeL/easydel/infra/factory.py#L321) — Create a decorator for registering a configuration class.
  - `register_module(self, task_type: TaskType, config: type[EasyDeLBaseConfig], model_type: str, embedding_layer_names: list[str] | None = None, layernorm_names: list[str] | None = None)` — [`L408`](../../../../../../raw/code/EasyDeL/easydel/infra/factory.py#L408) — Create a decorator for registering an EasyDeL module class.
  - `task_registry(self)` — [`L634`](../../../../../../raw/code/EasyDeL/easydel/infra/factory.py#L634) — Provide read-only access to the task-based module registry.
  - `wrapper(obj: T)` — [`L370`](../../../../../../raw/code/EasyDeL/easydel/infra/factory.py#L370) — Inner decorator that performs the actual registration.
- protocol/private: `_config_registry`[`L317`](../../../../../../raw/code/EasyDeL/easydel/infra/factory.py#L317), `_task_registry`[`L319`](../../../../../../raw/code/EasyDeL/easydel/infra/factory.py#L319)
- uses (calls/refs, reference-scoped): [`TaskType`](factory.md#TaskType), [`EasyDeLBaseConfig`](base_config.md#EasyDeLBaseConfig), [`ConfigType`](factory.md#ConfigType), [`T`](factory.md#T), [`printify_nnx`](mixins/protocol.md#printify_nnx), [`ModuleRegistration`](factory.md#ModuleRegistration), [`MODULE_CONFIG`](factory.md#ConfigType.MODULE_CONFIG), [`_SENTINEL`](factory.md#_SENTINEL)
- used by: [`register_module`](factory.md#register_module), [`register_config`](factory.md#register_config), [`get_modules_by_type`](../modules/auto/auto_configuration.md#get_modules_by_type), [`from_config`](../modules/auto/auto_modeling.md#BaseAutoEasyModel.from_config), [`registry`](factory.md#registry), [`_has_registered_gemma4_vision_backend`](../modules/gemma4/modeling_gemma4.md#_has_registered_gemma4_vision_backend), [`vision_config`](../modules/aya_vision/aya_vision_configuration.md#AyaVisionConfig.vision_config), [`vision_config`](../modules/mistral3/mistral3_configuration.md#Mistral3Config.vision_config), [`vision_feature_layer`](../modules/llava/llava_configuration.md#LlavaConfig.vision_feature_layer), [`vision_feature_layer`](../modules/mistral3/mistral3_configuration.md#Mistral3Config.vision_feature_layer), [`vision_feature_layer`](../modules/aya_vision/aya_vision_configuration.md#AyaVisionConfig.vision_feature_layer), [`vision_config`](../modules/llava/llava_configuration.md#LlavaConfig.vision_config)

### `TaskType`  ·  implements/extends StrEnum
- def: [`easydel/infra/factory.py:117`](../../../../../../raw/code/EasyDeL/easydel/infra/factory.py#L117) — documented in [easydel-infra-factory](../../../concepts/easydel-infra-factory.md)
- doc: Enumeration of supported model task types in the EasyDeL registry.
- signature: `class TaskType(StrEnum):`
- members:
  - `ANY_TO_ANY` — [`L196`](../../../../../../raw/code/EasyDeL/easydel/infra/factory.py#L196)
  - `AUDIO_CLASSIFICATION` — [`L193`](../../../../../../raw/code/EasyDeL/easydel/infra/factory.py#L193)
  - `AUTO_BIND` — [`L197`](../../../../../../raw/code/EasyDeL/easydel/infra/factory.py#L197)
  - `BASE_MODULE` — [`L187`](../../../../../../raw/code/EasyDeL/easydel/infra/factory.py#L187)
  - `BASE_VISION` — [`L188`](../../../../../../raw/code/EasyDeL/easydel/infra/factory.py#L188)
  - `CAUSAL_LM` — [`L183`](../../../../../../raw/code/EasyDeL/easydel/infra/factory.py#L183) — documented in [easydel-infra-factory](../../../concepts/easydel-infra-factory.md)
  - `DIFFUSION_LM` — [`L185`](../../../../../../raw/code/EasyDeL/easydel/infra/factory.py#L185)
  - `EMBEDDING` — [`L195`](../../../../../../raw/code/EasyDeL/easydel/infra/factory.py#L195)
  - `IMAGE_CLASSIFICATION` — [`L194`](../../../../../../raw/code/EasyDeL/easydel/infra/factory.py#L194)
  - `IMAGE_TEXT_TO_TEXT` — [`L186`](../../../../../../raw/code/EasyDeL/easydel/infra/factory.py#L186)
  - `SEQUENCE_CLASSIFICATION` — [`L192`](../../../../../../raw/code/EasyDeL/easydel/infra/factory.py#L192)
  - `SEQUENCE_TO_SEQUENCE` — [`L189`](../../../../../../raw/code/EasyDeL/easydel/infra/factory.py#L189)
  - `SPEECH_SEQUENCE_TO_SEQUENCE` — [`L190`](../../../../../../raw/code/EasyDeL/easydel/infra/factory.py#L190)
  - `VISION_LM` — [`L184`](../../../../../../raw/code/EasyDeL/easydel/infra/factory.py#L184)
  - `ZERO_SHOT_IMAGE_CLASSIFICATION` — [`L191`](../../../../../../raw/code/EasyDeL/easydel/infra/factory.py#L191)
- used by: [`_import_structure`](../__init__.md#_import_structure), [`__all__`](../modules/whisper/modeling_whisper.md#__all__), [`_get_information`](../trainers/base_trainer.md#BaseTrainer._get_information), [`flops_per_token`](utils.md#flops_per_token), [`build_model`](elarge/builders.md#build_model), [`load_state`](base_state.md#EasyDeLState.load_state), [`infer_task_from_hf_config`](../modules/auto/auto_configuration.md#infer_task_from_hf_config), [`get_torch_loader`](mixins/bridge.md#EasyBridgeMixin.get_torch_loader), [`from_pretrained`](../modules/auto/auto_configuration.md#AutoEasyDeLConfig.from_pretrained), [`AUTO_MODEL_FACTORY_REGISTRY`](../modules/_base/_auto_mapper.md#AUTO_MODEL_FACTORY_REGISTRY.AUTO_MODEL_FACTORY_REGISTRY), [`get_modules_by_type`](../modules/auto/auto_configuration.md#get_modules_by_type), [`build_esurge`](elarge/builders.md#build_esurge), [`from_config`](../modules/auto/auto_configuration.md#AutoShardAndGatherFunctions.from_config), [`Qwen3NextModel`](../modules/qwen3_next/modeling_qwen3_next.md#Qwen3NextModel), [`Qwen3NextForCausalLM`](../modules/qwen3_next/modeling_qwen3_next.md#Qwen3NextForCausalLM), [`Glm4vModel`](../modules/glm4v/modeling_glm4v.md#Glm4vModel), [`TASK_ALIASES`](elarge/processing.md#TASK_ALIASES.TASK_ALIASES), [`TASK_ALIASES`](../modules/auto/auto_configuration.md#TASK_ALIASES.TASK_ALIASES), [`RobertaModel`](../modules/roberta/modeling_roberta.md#RobertaModel), [`Qwen3Model`](../modules/qwen3/modeling_qwen3.md#Qwen3Model), [`Qwen3VLModel`](../modules/qwen3_vl/modeling_qwen3_vl.md#Qwen3VLModel), [`Qwen3VLMoeModel`](../modules/qwen3_vl_moe/modeling_qwen3_vl_moe.md#Qwen3VLMoeModel), [`resolve_task`](elarge/processing.md#resolve_task), [`Qwen2Model`](../modules/qwen2/modeling_qwen.md#Qwen2Model), [`DeepseekV3ForCausalLM`](../modules/deepseek_v3/modeling_deepseek.md#DeepseekV3ForCausalLM), [`Glm4vVisionModel`](../modules/glm4v/modeling_glm4v.md#Glm4vVisionModel), [`Gemma3TextModel`](../modules/gemma3/modeling_gemma3.md#Gemma3TextModel), [`Llama4ForCausalLM`](../modules/llama4/modeling_llama4.md#Llama4ForCausalLM), [`Qwen3OmniMoeThinkerForConditionalGeneration`](../modules/qwen3_omni_moe/modeling_qwen3_omni_moe.md#Qwen3OmniMoeThinkerForConditionalGeneration), [`Qwen3_5Model`](../modules/qwen3_5/modeling_qwen3_5.md#Qwen3_5Model), [`Qwen3_5MoeModel`](../modules/qwen3_5_moe/modeling_qwen3_5_moe.md#Qwen3_5MoeModel), [`WhisperForConditionalGeneration`](../modules/whisper/modeling_whisper.md#WhisperForConditionalGeneration), [`ArcticModel`](../modules/arctic/modeling_arctic.md#ArcticModel), [`Cohere2Model`](../modules/cohere2/modeling_cohere2.md#Cohere2Model), [`CohereModel`](../modules/cohere/modeling_cohere.md#CohereModel), [`DbrxModel`](../modules/dbrx/modeling_dbrx.md#DbrxModel), [`Exaone4Model`](../modules/exaone4/modeling_exaone4.md#Exaone4Model), [`ExaoneModel`](../modules/exaone/modeling_exaone.md#ExaoneModel), [`Gemma2Model`](../modules/gemma2/modeling_gemma2.md#Gemma2Model), [`GemmaModel`](../modules/gemma/modeling_gemma.md#GemmaModel)  (+309 more)

## Module values
- `T` — [`L88`](../../../../../../raw/code/EasyDeL/easydel/infra/factory.py#L88)
- `_SENTINEL` — [`L89`](../../../../../../raw/code/EasyDeL/easydel/infra/factory.py#L89)
- `register_config` — [`L688`](../../../../../../raw/code/EasyDeL/easydel/infra/factory.py#L688) — documented in [easydel-infra-factory](../../../concepts/easydel-infra-factory.md)
- `register_module` — [`L689`](../../../../../../raw/code/EasyDeL/easydel/infra/factory.py#L689) — documented in [easydel-infra-factory](../../../concepts/easydel-infra-factory.md)
- `registry` — [`L684`](../../../../../../raw/code/EasyDeL/easydel/infra/factory.py#L684) — documented in [easydel-infra-factory](../../../concepts/easydel-infra-factory.md)


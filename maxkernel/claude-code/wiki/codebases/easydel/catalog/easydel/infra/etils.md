---
title: 'Module: easydel/infra/etils.py'
type: catalog
provenance: extracted
module: easydel/infra/etils.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.infra.etils`/
symbols:
  EasyDeLGradientCheckPointers: EasyDeLGradientCheckPointers#
  EasyDeLGradientCheckPointers.NONE: EasyDeLGradientCheckPointers#NONE.
  EasyDeLBackends: EasyDeLBackends#
  EasyDeLPlatforms: EasyDeLPlatforms#
  AVAILABLE_ATTENTION_MECHANISMS: AVAILABLE_ATTENTION_MECHANISMS.
  AVAILABLE_GRADIENT_CHECKPOINT_TARGETS: AVAILABLE_GRADIENT_CHECKPOINT_TARGETS.
  GRADIENT_CHECKPOINT_TARGETS.GRADIENT_CHECKPOINT_TARGETS: GRADIENT_CHECKPOINT_TARGETS.GRADIENT_CHECKPOINT_TARGETS.
  EasyDeLSchedulers: EasyDeLSchedulers#
  AVAILABLE_MOE_METHODS: AVAILABLE_MOE_METHODS.
  AVAILABLE_SPARSE_MODULE_TYPES: AVAILABLE_SPARSE_MODULE_TYPES.
  EasyDeLOptimizers: EasyDeLOptimizers#
  define_flags_with_default: define_flags_with_default().
  EasyDeLSchedulers.NONE: EasyDeLSchedulers#NONE.
  AVAILABLE_GRADIENT_CHECKPOINTS: AVAILABLE_GRADIENT_CHECKPOINTS.
  AVAILABLE_SCHEDULERS: AVAILABLE_SCHEDULERS.
  AVAILABLE_OPTIMIZERS: AVAILABLE_OPTIMIZERS.
  DEFAULT_ATTENTION_MECHANISM: DEFAULT_ATTENTION_MECHANISM.
  DEFAULT_MLA_ATTENTION_MECHANISM: DEFAULT_MLA_ATTENTION_MECHANISM.
  EasyDeLOptimizers.ADAMW: EasyDeLOptimizers#ADAMW.
  EasyDeLGradientCheckPointers.CHECKPOINT_DOTS: EasyDeLGradientCheckPointers#CHECKPOINT_DOTS.
  EasyDeLBackends.CPU: EasyDeLBackends#CPU.
  EasyDeLBackends.GPU: EasyDeLBackends#GPU.
  EasyDeLBackends.TPU: EasyDeLBackends#TPU.
  EasyDeLBackends.TT: EasyDeLBackends#TT.
  StoreTupleAction: StoreTupleAction#
  EasyDeLOptimizers.ADAFACTOR: EasyDeLOptimizers#ADAFACTOR.
  EasyDeLOptimizers.MARS: EasyDeLOptimizers#MARS.
  EasyDeLOptimizers.MUON: EasyDeLOptimizers#MUON.
  EasyDeLOptimizers.RMSPROP: EasyDeLOptimizers#RMSPROP.
  EasyDeLOptimizers.LION: EasyDeLOptimizers#LION.
  EasyDeLOptimizers.SKEW: EasyDeLOptimizers#SKEW.
  EasyDeLOptimizers.QUAD: EasyDeLOptimizers#QUAD.
  EasyDeLSchedulers.LINEAR: EasyDeLSchedulers#LINEAR.
  EasyDeLSchedulers.COSINE: EasyDeLSchedulers#COSINE.
  EasyDeLGradientCheckPointers.EVERYTHING_SAVEABLE: EasyDeLGradientCheckPointers#EVERYTHING_SAVEABLE.
  EasyDeLGradientCheckPointers.NOTHING_SAVEABLE: EasyDeLGradientCheckPointers#NOTHING_SAVEABLE.
  EasyDeLGradientCheckPointers.CHECKPOINT_DOTS_WITH_NO_BATCH_DMIS: EasyDeLGradientCheckPointers#CHECKPOINT_DOTS_WITH_NO_BATCH_DMIS.
  EasyDeLGradientCheckPointers.DOTS_SAVEABLE: EasyDeLGradientCheckPointers#DOTS_SAVEABLE.
  EasyDeLGradientCheckPointers.DOTS_WITH_NO_BATCH_DIMS_AVAILABLE: EasyDeLGradientCheckPointers#DOTS_WITH_NO_BATCH_DIMS_AVAILABLE.
  EasyDeLGradientCheckPointers.SAVE_ANYTHING_EXCEPT_THESE_NAMES: EasyDeLGradientCheckPointers#SAVE_ANYTHING_EXCEPT_THESE_NAMES.
  EasyDeLGradientCheckPointers.SAVE_ANY_NAMES_BUT_THESE: EasyDeLGradientCheckPointers#SAVE_ANY_NAMES_BUT_THESE.
  EasyDeLGradientCheckPointers.SAVE_ONLY_THESE_NAMES: EasyDeLGradientCheckPointers#SAVE_ONLY_THESE_NAMES.
  EasyDeLGradientCheckPointers.MLP_NOTSAVEABLE: EasyDeLGradientCheckPointers#MLP_NOTSAVEABLE.
  EasyDeLGradientCheckPointers.ATTN_NOTSAVEABLE: EasyDeLGradientCheckPointers#ATTN_NOTSAVEABLE.
  EasyDeLGradientCheckPointers.MLP_ATTN_NOTSAVEABLE: EasyDeLGradientCheckPointers#MLP_ATTN_NOTSAVEABLE.
  EasyDeLGradientCheckPointers.SAVE_FROM_BOTH_POLICIES: EasyDeLGradientCheckPointers#SAVE_FROM_BOTH_POLICIES.
  EasyDeLPlatforms.JAX: EasyDeLPlatforms#JAX.
  EasyDeLPlatforms.TRITON: EasyDeLPlatforms#TRITON.
  EasyDeLPlatforms.PALLAS: EasyDeLPlatforms#PALLAS.
  StoreTupleAction.__call__: StoreTupleAction#__call__().
---
# Module: [`easydel/infra/etils.py`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py)

## Classes
### `EasyDeLBackends`  ·  implements/extends StrEnum
- def: [`easydel/infra/etils.py:260`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py#L260)
- doc: Enumeration of JAX backend types supported by EasyDeL.
- signature: `class EasyDeLBackends(StrEnum):`
- members:
  - `CPU` — [`L288`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py#L288)
  - `GPU` — [`L289`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py#L289)
  - `TPU` — [`L290`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py#L290)
  - `TT` — [`L291`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py#L291)
- used by: [`_import_structure`](../__init__.md#_import_structure), [`__init__`](base_config.md#EasyDeLBaseConfig.__init__), [`huggingface_to_easydel_sequential`](mixins/bridge.md#EasyBridgeMixin.huggingface_to_easydel_sequential), [`_from_torch_pretrained`](mixins/bridge.md#EasyBridgeMixin._from_torch_pretrained), [`from_pretrained`](mixins/bridge.md#EasyBridgeMixin.from_pretrained), [`add_basic_configurations`](base_config.md#EasyDeLBaseConfig.add_basic_configurations), [`__call__`](../operations/_base_operation.md#BaseOperation.__call__), [`load_state`](base_state.md#EasyDeLState.load_state), [`from_pretrained`](../modules/auto/auto_configuration.md#AutoEasyDeLConfig.from_pretrained), [`from_pretrained`](../modules/auto/auto_modeling.md#BaseAutoEasyModel.from_pretrained), [`__post_init__`](../operations/_operation_meta.md#OperationMetadata.__post_init__), [`from_pretrained`](../modules/auto/auto_modeling.md#BaseAutoEasyState.from_pretrained), [`from_pretrained`](../modules/auto/auto_configuration.md#AutoShardAndGatherFunctions.from_pretrained), [`_from_easydel_params`](../modules/auto/auto_modeling.md#BaseAutoEasyModel._from_easydel_params), [`_from_torch_pretrained`](../modules/auto/auto_modeling.md#BaseAutoEasyModel._from_torch_pretrained), [`backend`](../operations/_operation_meta.md#OperationMetadata.backend), [`__all__`](../operations/_operation_meta.md#__all__), [`EasyDeLBackends`](../operations/_base_operation.md#BaseOperation.EasyDeLBackends), [`backend`](base_config.md#EasyDeLBaseConfigDict.backend), [`backend`](elarge/types/infra.md#PlatformCfg.backend)

### `EasyDeLGradientCheckPointers`  ·  implements/extends StrEnum
- def: [`easydel/infra/etils.py:159`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py#L159)
- doc: Enumeration of gradient checkpointing strategies available in EasyDeL.
- signature: `class EasyDeLGradientCheckPointers(StrEnum):`
- members:
  - `ATTN_NOTSAVEABLE` — [`L222`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py#L222)
  - `CHECKPOINT_DOTS` — [`L213`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py#L213)
  - `CHECKPOINT_DOTS_WITH_NO_BATCH_DMIS` — [`L214`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py#L214)
  - `DOTS_SAVEABLE` — [`L216`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py#L216)
  - `DOTS_WITH_NO_BATCH_DIMS_AVAILABLE` — [`L217`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py#L217)
  - `EVERYTHING_SAVEABLE` — [`L211`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py#L211)
  - `MLP_ATTN_NOTSAVEABLE` — [`L223`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py#L223)
  - `MLP_NOTSAVEABLE` — [`L221`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py#L221)
  - `NONE` — [`L215`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py#L215) — documented in [easydel-infra-utils](../../../concepts/easydel-infra-utils.md)
  - `NOTHING_SAVEABLE` — [`L212`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py#L212)
  - `SAVE_ANYTHING_EXCEPT_THESE_NAMES` — [`L218`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py#L218)
  - `SAVE_ANY_NAMES_BUT_THESE` — [`L219`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py#L219)
  - `SAVE_FROM_BOTH_POLICIES` — [`L224`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py#L224)
  - `SAVE_ONLY_THESE_NAMES` — [`L220`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py#L220)
- used by: [`_import_structure`](../__init__.md#_import_structure), [`auto_remat`](utils.md#auto_remat), [`__init__`](base_config.md#EasyDeLBaseConfig.__init__), [`huggingface_to_easydel_sequential`](mixins/bridge.md#EasyBridgeMixin.huggingface_to_easydel_sequential), [`add_basic_configurations`](base_config.md#EasyDeLBaseConfig.add_basic_configurations), [`_generate_compatible_graphother`](base_module.md#EasyDeLBaseModule._generate_compatible_graphother), [`_generate_compatible_graphdef`](base_module.md#EasyDeLBaseModule._generate_compatible_graphdef), [`attach_custom_arguments`](../modules/gidd/gidd_configuration.md#GiddConfig.attach_custom_arguments), [`get_gradient_checkpoint_policy`](utils.md#get_gradient_checkpoint_policy), [`__init__`](../modules/dbrx/dbrx_configuration.md#DbrxConfig.__init__), [`__init__`](../modules/mosaic_mpt/mosaic_configuration.md#MptConfig.__init__), [`__init__`](../modules/opt/opt_configuration.md#OPTConfig.__init__), [`CONFIG_VARIABLES`](../trainers/ray_scaler/distributed_trainer.md#RayDistributedTrainer.CONFIG_VARIABLES), [`__init__`](../modules/llama/llama_configuration.md#LlamaConfig.__init__), [`gradient_checkpointing`](base_config.md#EasyDeLBaseConfigDict.gradient_checkpointing), [`__init__`](../modules/arctic/arctic_configuration.md#ArcticConfig.__init__), [`__init__`](../modules/cohere/cohere_configuration.md#CohereConfig.__init__), [`__init__`](../modules/cohere2/cohere2_configuration.md#Cohere2Config.__init__), [`__init__`](../modules/deepseek_v2/deepseek_configuration.md#DeepseekV2Config.__init__), [`__init__`](../modules/exaone/exaone_configuration.md#ExaoneConfig.__init__), [`__init__`](../modules/falcon/falcon_configuration.md#FalconConfig.__init__), [`__init__`](../modules/falcon_mamba/falcon_mamba_configuration.md#FalconMambaConfig.__init__), [`__init__`](../modules/gemma/gemma_configuration.md#GemmaConfig.__init__), [`__init__`](../modules/gemma2/gemma2_configuration.md#Gemma2Config.__init__), [`__init__`](../modules/gemma3/gemma3_configuration.md#Gemma3TextConfig.__init__), [`__init__`](../modules/gemma4/gemma4_configuration.md#Gemma4TextConfig.__init__), [`__init__`](../modules/gemma4/gemma4_configuration.md#Gemma4VisionConfig.__init__), [`__init__`](../modules/gidd/gidd_configuration.md#GiddConfig.__init__), [`__init__`](../modules/gpt2/gpt2_configuration.md#GPT2Config.__init__), [`__init__`](../modules/gpt_j/gpt_j_configuration.md#GPTJConfig.__init__), [`__init__`](../modules/gpt_neox/gpt_neox_configuration.md#GPTNeoXConfig.__init__), [`__init__`](../modules/grok_1/grok_1_configuration.md#Grok1Config.__init__), [`__init__`](../modules/internlm2/internlm2_configuration.md#InternLM2Config.__init__), [`__init__`](../modules/mamba/mamba_configuration.md#MambaConfig.__init__), [`__init__`](../modules/mamba2/mamba2_configuration.md#Mamba2Config.__init__), [`__init__`](../modules/mistral/mistral_configuration.md#MistralConfig.__init__), [`__init__`](../modules/mixtral/mixtral_configuration.md#MixtralConfig.__init__), [`__init__`](../modules/olmo/olmo_configuration.md#OlmoConfig.__init__), [`__init__`](../modules/olmo2/olmo2_configuration.md#Olmo2Config.__init__), [`__init__`](../modules/olmo3/olmo3_configuration.md#Olmo3Config.__init__)  (+10 more)

### `EasyDeLOptimizers`  ·  implements/extends StrEnum
- def: [`easydel/infra/etils.py:79`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py#L79)
- doc: Enumeration of available optimizers in the EasyDeL library.
- signature: `class EasyDeLOptimizers(StrEnum):`
- members:
  - `ADAFACTOR` — [`L116`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py#L116)
  - `ADAMW` — [`L117`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py#L117)
  - `LION` — [`L121`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py#L121)
  - `MARS` — [`L118`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py#L118)
  - `MUON` — [`L119`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py#L119)
  - `QUAD` — [`L123`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py#L123)
  - `RMSPROP` — [`L120`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py#L120)
  - `SKEW` — [`L122`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py#L122)
- used by: [`_import_structure`](../__init__.md#_import_structure), [`optimizer`](../trainers/training_configurations.md#TrainingArguments.optimizer)

### `EasyDeLPlatforms`  ·  implements/extends StrEnum
- def: [`easydel/infra/etils.py:227`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py#L227)
- doc: Enumeration of platforms or kernel execution backends supported by EasyDeL.
- signature: `class EasyDeLPlatforms(StrEnum):`
- members:
  - `JAX` — [`L255`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py#L255)
  - `PALLAS` — [`L257`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py#L257)
  - `TRITON` — [`L256`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py#L256)
- used by: [`_import_structure`](../__init__.md#_import_structure), [`__init__`](base_config.md#EasyDeLBaseConfig.__init__), [`huggingface_to_easydel_sequential`](mixins/bridge.md#EasyBridgeMixin.huggingface_to_easydel_sequential), [`_from_torch_pretrained`](mixins/bridge.md#EasyBridgeMixin._from_torch_pretrained), [`from_pretrained`](mixins/bridge.md#EasyBridgeMixin.from_pretrained), [`add_basic_configurations`](base_config.md#EasyDeLBaseConfig.add_basic_configurations), [`load_state`](base_state.md#EasyDeLState.load_state), [`from_pretrained`](../modules/auto/auto_configuration.md#AutoEasyDeLConfig.from_pretrained), [`from_pretrained`](../modules/auto/auto_modeling.md#BaseAutoEasyModel.from_pretrained), [`from_pretrained`](../modules/auto/auto_modeling.md#BaseAutoEasyState.from_pretrained), [`from_pretrained`](../modules/auto/auto_configuration.md#AutoShardAndGatherFunctions.from_pretrained), [`_from_easydel_params`](../modules/auto/auto_modeling.md#BaseAutoEasyModel._from_easydel_params), [`_from_torch_pretrained`](../modules/auto/auto_modeling.md#BaseAutoEasyModel._from_torch_pretrained), [`__all__`](../operations/_operation_meta.md#__all__), [`PlatformCfg`](elarge/types/infra.md#PlatformCfg), [`platform`](../operations/_operation_meta.md#OperationMetadata.platform), [`platform`](base_config.md#EasyDeLBaseConfigDict.platform), [`platform`](elarge/types/quantization.md#QuantizationCfg.platform)

### `EasyDeLSchedulers`  ·  implements/extends StrEnum
- def: [`easydel/infra/etils.py:126`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py#L126)
- doc: Enumeration of available learning rate schedulers in EasyDeL.
- signature: `class EasyDeLSchedulers(StrEnum):`
- members:
  - `COSINE` — [`L156`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py#L156)
  - `LINEAR` — [`L155`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py#L155)
  - `NONE` — [`L154`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py#L154)
- used by: [`_import_structure`](../__init__.md#_import_structure), [`get_optimizer_and_scheduler`](../trainers/training_configurations.md#TrainingArguments.get_optimizer_and_scheduler), [`scheduler`](../trainers/training_configurations.md#TrainingArguments.scheduler)

### `StoreTupleAction`
- def: [`easydel/infra/etils.py:561`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py#L561)
- members:
  - `__call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace, values: str, option_string: str | None = None)` — [`L603`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py#L603) — Parse and store the comma-separated string as a tuple of integers.
- used by: [`define_flags_with_default`](etils.md#define_flags_with_default)

## Functions
- `define_flags_with_default(_required_fields: list | None = None, **kwargs)` — [`L472`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py#L472) — Define command-line flags using argparse based on provided keyword arguments.

## Module values
- `AVAILABLE_ATTENTION_MECHANISMS` — [`L329`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py#L329)
- `AVAILABLE_GRADIENT_CHECKPOINTS` — [`L296`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py#L296)
- `AVAILABLE_GRADIENT_CHECKPOINT_TARGETS` — [`L365`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py#L365)
- `AVAILABLE_MOE_METHODS` — [`L325`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py#L325)
- `AVAILABLE_OPTIMIZERS` — [`L319`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py#L319)
- `AVAILABLE_SCHEDULERS` — [`L315`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py#L315)
- `AVAILABLE_SPARSE_MODULE_TYPES` — [`L360`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py#L360)
- `DEFAULT_ATTENTION_MECHANISM` — [`L349`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py#L349)
- `DEFAULT_MLA_ATTENTION_MECHANISM` — [`L353`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py#L353)
- `GRADIENT_CHECKPOINT_TARGETS` — [`L419`](../../../../../../raw/code/EasyDeL/easydel/infra/etils.py#L419)


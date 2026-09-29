---
title: 'Module: easydel/layers/quantization/_configs.py'
type: catalog
provenance: extracted
module: easydel/layers/quantization/_configs.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.layers.quantization._configs`/
symbols:
  QuantizationConfig: QuantizationConfig#
  QuantizationType: QuantizationType#
  resolve_ejkernel_quant_params: resolve_ejkernel_quant_params().
  QuantizationConfig.group_size: QuantizationConfig#group_size.
  QuantizationConfig.dtype: QuantizationConfig#dtype.
  QuantizationConfig.bits: QuantizationConfig#bits.
  QuantizationConfig.__post_init__: QuantizationConfig#__post_init__().
  resolve_jax_native_dtype: resolve_jax_native_dtype().
  QuantizationType.INT8: QuantizationType#INT8.
  QuantizationConfig.runtime_dtype: QuantizationConfig#runtime_dtype.
  QuantizationType.NF4: QuantizationType#NF4.
  DEFAULT_QUANTIZATION_PATTERN: DEFAULT_QUANTIZATION_PATTERN.
  QuantizationConfig.jax_native: QuantizationConfig#jax_native.
  QuantizationConfig.pattern: QuantizationConfig#pattern.
  QuantizationType.MXFP8: QuantizationType#MXFP8.
  QuantizationType.MXFP4: QuantizationType#MXFP4.
  QuantizationType.NVFP8: QuantizationType#NVFP8.
  QuantizationType.BINARY: QuantizationType#BINARY.
  QuantizationConfig.__hash__: QuantizationConfig#__hash__.
  QuantizationType.AFFINE: QuantizationType#AFFINE.
  QuantizationType.TERNARY: QuantizationType#TERNARY.
  QuantizationConfig.simulate: QuantizationConfig#simulate.
  QuantizationType.NVFP4: QuantizationType#NVFP4.
  QuantizationType.TURBOQUANT: QuantizationType#TURBOQUANT.
---
# Module: [`easydel/layers/quantization/_configs.py`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_configs.py)

## Classes
### `QuantizationConfig`
- def: [`easydel/layers/quantization/_configs.py:118`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_configs.py#L118) — documented in [easydel-layers-linears-_linear](../../../../concepts/easydel-layers-linears-_linear.md)
- doc: Configuration for model weight quantization behavior.
- signature: `class QuantizationConfig:`
- members:
  - `__post_init__(self)` — [`L202`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_configs.py#L202) — Post-initialization processing to normalize dtype values.
  - `bits` — [`L196`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_configs.py#L196)
  - `dtype` — [`L193`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_configs.py#L193)
  - `group_size` — [`L195`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_configs.py#L195)
  - `jax_native` — [`L198`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_configs.py#L198)
  - `pattern` — [`L200`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_configs.py#L200)
  - `runtime_dtype` — [`L194`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_configs.py#L194)
  - `simulate` — [`L197`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_configs.py#L197)
- protocol/private: `__hash__`[`L218`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_configs.py#L218)
- uses (calls/refs, reference-scoped): [`hash_fn`](../../utils/compiling_utils.md#hash_fn), [`QuantizationType`](_configs.md#QuantizationType), [`NF4`](_configs.md#QuantizationType.NF4), [`DEFAULT_QUANTIZATION_PATTERN`](_configs.md#DEFAULT_QUANTIZATION_PATTERN)
- used by: [`__init__`](../../infra/base_config.md#EasyDeLBaseConfig.__init__), [`_from_torch_pretrained`](../../infra/mixins/bridge.md#EasyBridgeMixin._from_torch_pretrained), [`_load_model_weights`](../../infra/mixins/bridge.md#EasyBridgeMixin._load_model_weights), [`from_pretrained`](../../infra/mixins/bridge.md#EasyBridgeMixin.from_pretrained), [`add_basic_configurations`](../../infra/base_config.md#EasyDeLBaseConfig.add_basic_configurations), [`straight_through`](_straight_through.md#straight_through), [`load_state`](../../infra/base_state.md#EasyDeLState.load_state), [`resolve_ejkernel_quant_params`](_configs.md#resolve_ejkernel_quant_params), [`to_quantized`](../linears/_linear.md#ParallelLinear.to_quantized), [`quantize`](_quants.md#quantize), [`from_pretrained`](../../modules/auto/auto_modeling.md#BaseAutoEasyModel.from_pretrained), [`make_default_tensor_straight_through`](../../trainers/training_utils.md#make_default_tensor_straight_through), [`from_pretrained`](../../modules/auto/auto_modeling.md#BaseAutoEasyState.from_pretrained), [`materialize_base_config`](../../infra/elarge/processing.md#materialize_base_config), [`to_load_state_kwargs`](../../infra/elarge/builders.md#to_load_state_kwargs), [`quantize`](../../infra/base_module.md#EasyDeLBaseModule.quantize), [`to_from_pretrained_kwargs`](../../infra/elarge/builders.md#to_from_pretrained_kwargs), [`pattern`](_quants.md#EasyQuantizer.pattern), [`_from_easydel_params`](../../modules/auto/auto_modeling.md#BaseAutoEasyModel._from_easydel_params), [`_from_torch_pretrained`](../../modules/auto/auto_modeling.md#BaseAutoEasyModel._from_torch_pretrained), [`config`](_quants.md#EasyQuantizer.config), [`_quantize_runtime`](../linears/_linear_quantized.md#ParallelLinearQuantized._quantize_runtime), [`_normalize_quantization_config`](../../infra/mixins/bridge.md#_normalize_quantization_config), [`_normalize_rebuild_quantization_config`](../../infra/base_module.md#EasyDeLBaseModule._normalize_rebuild_quantization_config), [`wqdtype`](../linears/_linear_quantized.md#ParallelLinearQuantized.wqdtype), [`__init__`](../linears/_linear_quantized.md#ParallelLinearQuantized.__init__), [`quantize`](../../infra/mixins/protocol.md#BaseModuleProtocol.quantize), [`quantize_linear_layers`](../../infra/utils.md#quantize_linear_layers), [`kv_cache`](../../infra/elarge/types/quantization.md#QuantizationCfg.kv_cache), [`model`](../../infra/elarge/types/quantization.md#QuantizationCfg.model), [`EasyDeLQuantizationConfig`](__init__.md#EasyDeLQuantizationConfig), [`logger`](../linears/_linear_quantized.md#logger), [`__init__`](_quants.md#EasyQuantizer.__init__), [`kv_cache_quantization_config`](../../infra/base_config.md#EasyDeLBaseConfigDict.kv_cache_quantization_config), [`quantization_config`](../../infra/base_config.md#EasyDeLBaseConfigDict.quantization_config)

### `QuantizationType`  ·  implements/extends StrEnum
- def: [`easydel/layers/quantization/_configs.py:66`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_configs.py#L66)
- doc: Enumeration of supported quantization data types.
- signature: `class QuantizationType(enum.StrEnum):`
- members:
  - `AFFINE` — [`L110`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_configs.py#L110)
  - `BINARY` — [`L113`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_configs.py#L113)
  - `INT8` — [`L111`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_configs.py#L111)
  - `MXFP4` — [`L106`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_configs.py#L106)
  - `MXFP8` — [`L105`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_configs.py#L105)
  - `NF4` — [`L109`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_configs.py#L109)
  - `NVFP4` — [`L108`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_configs.py#L108)
  - `NVFP8` — [`L107`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_configs.py#L107)
  - `TERNARY` — [`L112`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_configs.py#L112)
  - `TURBOQUANT` — [`L114`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_configs.py#L114)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`_load_model_weights`](../../infra/mixins/bridge.md#EasyBridgeMixin._load_model_weights), [`straight_through`](_straight_through.md#straight_through), [`resolve_ejkernel_quant_params`](_configs.md#resolve_ejkernel_quant_params), [`quantize`](_quants.md#quantize), [`dtype`](_configs.md#QuantizationConfig.dtype), [`quantize`](../../infra/base_module.md#EasyDeLBaseModule.quantize), [`__post_init__`](_configs.md#QuantizationConfig.__post_init__), [`resolve_jax_native_dtype`](_configs.md#resolve_jax_native_dtype), [`wqdtype`](../linears/_linear_quantized.md#ParallelLinearQuantized.wqdtype), [`runtime_dtype`](_configs.md#QuantizationConfig.runtime_dtype)

## Functions
- `resolve_ejkernel_quant_params(config: QuantizationConfig)` — [`L221`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_configs.py#L221) — Map EasyDeL quantization config to ejkernel quantization parameters.
- `resolve_jax_native_dtype(dtype: QuantizationType | str | None)` — [`L281`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_configs.py#L281) — Return the JAX-native dtype for supported quantization types.

## Module values
- `DEFAULT_QUANTIZATION_PATTERN` — [`L54`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_configs.py#L54)


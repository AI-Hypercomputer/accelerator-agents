---
title: 'Module: tpu_inference/layers/vllm/quantization/unquantized.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/vllm/quantization/unquantized.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.vllm.quantization.unquantized`/
symbols:
  VllmUnquantizedFusedMoEMethod.process_weights_after_loading: VllmUnquantizedFusedMoEMethod#process_weights_after_loading().
  VllmUnquantizedFusedMoEMethod.apply_monolithic: VllmUnquantizedFusedMoEMethod#apply_monolithic().
  VllmUnquantizedLinearMethod: VllmUnquantizedLinearMethod#
  VllmUnquantizedLinearMethod.process_weights_after_loading: VllmUnquantizedLinearMethod#process_weights_after_loading().
  VllmUnquantizedConfig.get_quant_method: VllmUnquantizedConfig#get_quant_method().
  VllmUnquantizedLinearMethod.process_unquantized_linear_weights: VllmUnquantizedLinearMethod#process_unquantized_linear_weights().
  _load_weight_for_layer: _load_weight_for_layer().
  VllmUnquantizedFusedMoEMethod: VllmUnquantizedFusedMoEMethod#
  VllmUnquantizedEmbeddingMethod.process_weights_after_loading: VllmUnquantizedEmbeddingMethod#process_weights_after_loading().
  VllmUnquantizedConfig: VllmUnquantizedConfig#
  VllmUnquantizedFusedMoEMethod.extra_backend_kwargs: VllmUnquantizedFusedMoEMethod#extra_backend_kwargs.
  logger: logger.
  VllmUnquantizedLinearMethod.apply: VllmUnquantizedLinearMethod#apply().
  VllmUnquantizedFusedMoEMethod.moe_backend: VllmUnquantizedFusedMoEMethod#moe_backend.
  VllmUnquantizedLinearMethod.maybe_process_weights: VllmUnquantizedLinearMethod#maybe_process_weights().
  VllmUnquantizedFusedMoEMethod.maybe_process_weights: VllmUnquantizedFusedMoEMethod#maybe_process_weights().
  VllmUnquantizedLinearMethod.__init__: VllmUnquantizedLinearMethod#__init__().
  P: P.
  VllmUnquantizedFusedMoEMethod.mesh: VllmUnquantizedFusedMoEMethod#mesh.
  VllmUnquantizedConfig.get_name: VllmUnquantizedConfig#get_name().
  VllmUnquantizedConfig.from_config: VllmUnquantizedConfig#from_config().
  VllmUnquantizedFusedMoEMethod._select_monolithic: VllmUnquantizedFusedMoEMethod#_select_monolithic().
  VllmUnquantizedEmbeddingMethod.mesh: VllmUnquantizedEmbeddingMethod#mesh.
  VllmUnquantizedEmbeddingMethod: VllmUnquantizedEmbeddingMethod#
  VllmUnquantizedConfig.get_supported_act_dtypes: VllmUnquantizedConfig#get_supported_act_dtypes().
  VllmUnquantizedConfig.get_min_capability: VllmUnquantizedConfig#get_min_capability().
  VllmUnquantizedConfig.get_config_filenames: VllmUnquantizedConfig#get_config_filenames().
  VllmUnquantizedEmbeddingMethod.__init__: VllmUnquantizedEmbeddingMethod#__init__().
  VllmUnquantizedFusedMoEMethod.__init__: VllmUnquantizedFusedMoEMethod#__init__().
  VllmUnquantizedFusedMoEMethod.is_monolithic: VllmUnquantizedFusedMoEMethod#is_monolithic().
---
# Module: [`tpu_inference/layers/vllm/quantization/unquantized.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/unquantized.py)

## Classes
### `VllmUnquantizedConfig`  ·  implements/extends VllmQuantConfig
- def: [`tpu_inference/layers/vllm/quantization/unquantized.py:124`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/unquantized.py#L124)
- signature: `class VllmUnquantizedConfig(QuantizationConfig, VllmQuantConfig):`
- members:
  - `from_config(cls, _: dict[str, Any])` — [`L143`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/unquantized.py#L143)
  - `get_config_filenames(cls)` — [`L139`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/unquantized.py#L139)
  - `get_min_capability(cls)` — [`L135`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/unquantized.py#L135)
  - `get_name(cls)` — [`L127`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/unquantized.py#L127)
  - `get_quant_method(self, layer: torch.nn.Module, prefix: str)` — [`L146`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/unquantized.py#L146)
  - `get_supported_act_dtypes(cls)` — [`L131`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/unquantized.py#L131)
- uses (calls/refs, reference-scoped): [`VllmQuantConfig`](configs.md#VllmQuantConfig), [`VllmUnquantizedLinearMethod`](unquantized.md#VllmUnquantizedLinearMethod), [`mesh`](configs.md#VllmQuantConfig.mesh), [`get_linear_config`](configs.md#VllmQuantConfig.get_linear_config), [`VllmUnquantizedFusedMoEMethod`](unquantized.md#VllmUnquantizedFusedMoEMethod), [`get_moe_config`](configs.md#VllmQuantConfig.get_moe_config), [`UNQUANTIZED`](../../common/quant_methods.md#UNQUANTIZED), [`VllmUnquantizedEmbeddingMethod`](unquantized.md#VllmUnquantizedEmbeddingMethod)
- used by: [`VllmQuantConfig`](configs.md#VllmQuantConfig), [`get_tpu_quantization_config`](__init__.md#get_tpu_quantization_config), [`get_quant_method`](compressed_tensors/compressed_tensors.md#VllmCompressedTensorsConfig.get_quant_method)

### `VllmUnquantizedEmbeddingMethod`
- def: [`tpu_inference/layers/vllm/quantization/unquantized.py:163`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/unquantized.py#L163)
- members:
  - `process_weights_after_loading(self, layer: torch.nn.Module)` — [`L168`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/unquantized.py#L168)
  - `mesh` — [`L166`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/unquantized.py#L166)
- protocol/private: `__init__`[`L165`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/unquantized.py#L165)
- uses (calls/refs, reference-scoped): [`ShardingAxisName`](../../common/sharding.md#ShardingAxisName), [`general_device_put`](../../common/utils.md#general_device_put), [`_load_weight_for_layer`](unquantized.md#_load_weight_for_layer), [`P`](unquantized.md#P)
- used by: [`get_quant_method`](unquantized.md#VllmUnquantizedConfig.get_quant_method)

### `VllmUnquantizedFusedMoEMethod`
- def: [`tpu_inference/layers/vllm/quantization/unquantized.py:349`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/unquantized.py#L349)
- members:
  - `apply_monolithic(self, layer: RoutedExperts, x: torch.Tensor, router_logits: torch.Tensor, input_ids: torch.Tensor | None = None)` — [`L442`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/unquantized.py#L442)
  - `is_monolithic(self)` — [`L370`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/unquantized.py#L370)
  - `maybe_process_weights(self, layer: torch.nn.Module, param_name: str, args, kwargs)` — [`L376`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/unquantized.py#L376) — Check if all weights are loaded for the layer. If so, process and shard the weights.
  - `process_weights_after_loading(self, layer: torch.nn.Module)` — [`L391`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/unquantized.py#L391)
  - `extra_backend_kwargs` — [`L363`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/unquantized.py#L363)
  - `mesh` — [`L360`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/unquantized.py#L360)
  - `moe_backend` — [`L361`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/unquantized.py#L361)
- protocol/private: `__init__`[`L353`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/unquantized.py#L353), `_select_monolithic`[`L373`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/unquantized.py#L373)
- uses (calls/refs, reference-scoped): [`ShardingAxisName`](../../common/sharding.md#ShardingAxisName), [`MoEBackend`](../../common/moe.md#MoEBackend), [`FusedMoEWeights`](../../common/process_weights/moe_weights.md#FusedMoEWeights), [`w13_weight_scale`](../../common/process_weights/moe_weights.md#FusedMoEWeights.w13_weight_scale), [`w2_weight_scale`](../../common/process_weights/moe_weights.md#FusedMoEWeights.w2_weight_scale), [`w13_weight`](../../common/process_weights/moe_weights.md#FusedMoEWeights.w13_weight), [`shard_moe_weights`](../../common/process_weights/moe_weights.md#shard_moe_weights), [`process_unquantized_moe_weights`](../../common/process_weights/moe_weights.md#process_unquantized_moe_weights), [`w13_bias`](../../common/process_weights/moe_weights.md#FusedMoEWeights.w13_bias), [`w2_weight`](../../common/process_weights/moe_weights.md#FusedMoEWeights.w2_weight), [`w2_bias`](../../common/process_weights/moe_weights.md#FusedMoEWeights.w2_bias), [`vllm_moe_apply`](../interface/moe.md#vllm_moe_apply), [`select_moe_backend_from_fused_moe_config`](../interface/moe.md#select_moe_backend_from_fused_moe_config), [`FUSED_MOE`](../../common/moe.md#MoEBackend.FUSED_MOE), [`_load_weight_for_layer`](unquantized.md#_load_weight_for_layer), [`logger`](unquantized.md#logger), [`UnquantizedFusedMoEMethod`](../../common/quantization/unquantized.md#UnquantizedFusedMoEMethod), [`_tensor_is_in_cpu`](../process_weights/cleanup_sharding.md#_tensor_is_in_cpu), [`VllmQuantizationMethod`](base.md#VllmQuantizationMethod), [`P`](unquantized.md#P)
- used by: [`get_quant_method`](deepseek_v4_fp8.md#VllmDeepseekV4Fp8Config.get_quant_method), [`get_quant_method`](fp8.md#VllmFp8Config.get_quant_method), [`get_quant_method`](nvfp4.md#VllmNvfp4Config.get_quant_method), [`get_quant_method`](unquantized.md#VllmUnquantizedConfig.get_quant_method), [`get_moe_method`](compressed_tensors/compressed_tensors_moe/compressed_tensors_moe.md#VllmCompressedTensorsMoEMethod.get_moe_method)

### `VllmUnquantizedLinearMethod`
- def: [`tpu_inference/layers/vllm/quantization/unquantized.py:199`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/unquantized.py#L199)
- members:
  - `apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor] = None)` — [`L314`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/unquantized.py#L314)
  - `maybe_process_weights(self, layer: torch.nn.Module, param_name: str, args, kwargs)` — [`L211`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/unquantized.py#L211) — Check if all weights are loaded for the layer. If so, process and shard the weights.
  - `process_unquantized_linear_weights(weight: jax.Array, bias: jax.Array | None)` — [`L281`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/unquantized.py#L281)
  - `process_weights_after_loading(self, layer: torch.nn.Module)` — [`L254`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/unquantized.py#L254)
- protocol/private: `__init__`[`L208`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/unquantized.py#L208)
- uses (calls/refs, reference-scoped): [`process_linear_weights`](../../common/process_weights/linear_weights.md#process_linear_weights), [`shard_linear_weights`](../../common/process_weights/linear_weights.md#shard_linear_weights), [`LinearWeights`](../../common/process_weights/linear_weights.md#LinearWeights), [`to_parameter_list`](../../common/process_weights/linear_weights.md#to_parameter_list), [`process_weights_after_loading`](nvfp4.md#VllmNvfp4LinearMethod.process_weights_after_loading), [`VllmQuantLinearConfig`](configs.md#VllmQuantLinearConfig), [`weight_scale`](../../common/process_weights/linear_weights.md#LinearWeights.weight_scale), [`_load_weight_for_layer`](unquantized.md#_load_weight_for_layer), [`bias`](../../common/process_weights/linear_weights.md#LinearWeights.bias), [`weight`](../../common/process_weights/linear_weights.md#LinearWeights.weight), [`zero_point`](../../common/process_weights/linear_weights.md#LinearWeights.zero_point), [`logger`](unquantized.md#logger), [`apply`](nvfp4.md#VllmNvfp4LinearMethod.apply), [`UnquantizedLinearMethod`](../../common/quantization/unquantized.md#UnquantizedLinearMethod), [`_tensor_is_in_cpu`](../process_weights/cleanup_sharding.md#_tensor_is_in_cpu), [`VllmQuantizationMethod`](base.md#VllmQuantizationMethod), [`_apply_fused`](../../common/quantization/unquantized.md#UnquantizedLinearMethod._apply_fused), [`_apply_split`](../../common/quantization/unquantized.md#UnquantizedLinearMethod._apply_split)
- used by: [`get_quant_method`](deepseek_v4_fp8.md#VllmDeepseekV4Fp8Config.get_quant_method), [`get_quant_method`](fp8.md#VllmFp8Config.get_quant_method), [`get_quant_method`](nvfp4.md#VllmNvfp4Config.get_quant_method), [`get_quant_method`](unquantized.md#VllmUnquantizedConfig.get_quant_method), [`get_quant_method`](awq.md#VllmAWQConfig.get_quant_method), [`get_quant_method`](mxfp4.md#VllmMxfp4Config.get_quant_method), [`__init__`](nvfp4.md#VllmNvfp4LinearMethod.__init__), [`VllmNvfp4LinearMethod`](nvfp4.md#VllmNvfp4LinearMethod)

## Functions
- `_load_weight_for_layer(layer: torch.nn.Module, param_name: str, sharding: NamedSharding)` — [`L66`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/unquantized.py#L66) — Load a layer's weight parameter onto the TPU mesh.

## Module values
- `P` — [`L61`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/unquantized.py#L61)
- `logger` — [`L63`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/unquantized.py#L63)


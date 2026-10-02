---
title: 'Module: tpu_inference/layers/jax/quantization/__init__.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/jax/quantization/__init__.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.jax.quantization`/
symbols:
  QuantizeMethodBase: QuantizeMethodBase#
  QuantizeMethodBase.apply_jax: QuantizeMethodBase#apply_jax().
  QuantizeMethodBase.create_weights_jax: QuantizeMethodBase#create_weights_jax().
  QuantizeMethodBase.process_weights_after_loading: QuantizeMethodBase#process_weights_after_loading().
  get_tpu_quantization_config: get_tpu_quantization_config().
---
# Module: [`tpu_inference/layers/jax/quantization/__init__.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/__init__.py)

## Classes
### `QuantizeMethodBase`  ·  implements/extends ABC
- def: [`tpu_inference/layers/jax/quantization/__init__.py:49`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/__init__.py#L49)
- doc: Base class for different quantized methods.
- signature: `class QuantizeMethodBase(ABC):`
- members:
  - `apply_jax(self, layer: JaxModule, *args, **kwargs)` — [`L60`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/__init__.py#L60) — Apply the weights in layer to the input tensor.
  - `create_weights_jax(self, layer: JaxModule, *weight_args, **extra_weight_attrs)` — [`L52`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/__init__.py#L52) — Create weights for a layer.
  - `process_weights_after_loading(self, layer: JaxModule, *args, **kwargs)` — [`L66`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/__init__.py#L66) — Processes weigths after loading.
- uses (calls/refs, reference-scoped): [`JaxModule`](../../__init__.md#JaxModule), [`apply_jax`](unquantized.md#UnquantizedFusedMoEMethod.apply_jax), [`apply_jax`](fp8.md#Fp8FusedMoEMethod.apply_jax), [`process_weights_after_loading`](fp8.md#Fp8BlockwiseLinearMethod.process_weights_after_loading), [`process_weights_after_loading`](fp8.md#Fp8FusedMoEMethod.process_weights_after_loading), [`create_weights_jax`](fp8.md#Fp8BlockwiseLinearMethod.create_weights_jax), [`apply_jax`](fp8.md#Fp8BlockwiseLinearMethod.apply_jax), [`create_weights_jax`](fp8.md#Fp8TensorwiseLinearMethod.create_weights_jax), [`apply_jax`](fp8.md#Fp8TensorwiseLinearMethod.apply_jax), [`apply_jax`](unquantized.md#UnquantizedLinearMethod.apply_jax), [`create_weights_jax`](fp8.md#Fp8FusedMoEMethod.create_weights_jax), [`create_weights_jax`](unquantized.md#UnquantizedMergedLinearMethod.create_weights_jax), [`process_weights_after_loading`](unquantized.md#UnquantizedFusedMoEMethod.process_weights_after_loading), [`UnquantizedLinearMethod`](unquantized.md#UnquantizedLinearMethod), [`UnquantizedFusedMoEMethod`](unquantized.md#UnquantizedFusedMoEMethod), [`Fp8BlockwiseLinearMethod`](fp8.md#Fp8BlockwiseLinearMethod), [`Fp8TensorwiseLinearMethod`](fp8.md#Fp8TensorwiseLinearMethod), [`Fp8FusedMoEMethod`](fp8.md#Fp8FusedMoEMethod)
- used by: [`get_quant_method`](unquantized.md#UnquantizedConfig.get_quant_method), [`get_quant_method`](fp8.md#Fp8Config.get_quant_method), [`quant_method`](../moe/moe.md#JaxMoE.quant_method), [`get_quant_method`](../../vllm/quantization/deepseek_v4_fp8.md#VllmDeepseekV4Fp8Config.get_quant_method), [`__call__`](../moe/moe.md#JaxMoE.__call__), [`_load_module`](../../../models/jax/utils/weight_utils.md#JaxAutoWeightsLoader._load_module), [`_process_weights_after_loading`](../../../models/jax/utils/weight_utils.md#JaxDummyModelLoader._process_weights_after_loading), [`quant_method`](../linear.md#JaxEinsum.quant_method), [`__call__`](../linear.md#JaxEinsum.__call__), [`get_quant_method`](configs.md#QuantizationConfig.get_quant_method), [`_process_weights_after_loading_jax`](../../../models/common/pathways_dummy_loader.md#_process_weights_after_loading_jax), [`quant_method`](../embed.md#JaxEmbed.quant_method), [`quant_method`](../norm.md#JaxRmsNorm.quant_method), [`UnquantizedLinearMethod`](unquantized.md#UnquantizedLinearMethod), [`__call__`](../moe/moe.md#JaxRoutedExperts.__call__), [`quant_method`](../moe/moe.md#JaxRoutedExperts.quant_method), [`UnquantizedFusedMoEMethod`](unquantized.md#UnquantizedFusedMoEMethod), [`Fp8BlockwiseLinearMethod`](fp8.md#Fp8BlockwiseLinearMethod), [`Fp8TensorwiseLinearMethod`](fp8.md#Fp8TensorwiseLinearMethod), [`__call__`](../embed.md#JaxEmbed.__call__), [`__call__`](../norm.md#JaxRmsNorm.__call__), [`quant_config`](../moe/moe.md#JaxRoutedExperts.quant_config), [`Fp8FusedMoEMethod`](fp8.md#Fp8FusedMoEMethod)

## Functions
- `get_tpu_quantization_config(vllm_config: VllmConfig)` — [`L24`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/__init__.py#L24)


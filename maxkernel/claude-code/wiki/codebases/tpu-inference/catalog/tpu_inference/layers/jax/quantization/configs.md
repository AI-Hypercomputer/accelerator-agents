---
title: 'Module: tpu_inference/layers/jax/quantization/configs.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/jax/quantization/configs.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.jax.quantization.configs`/
symbols:
  QuantizationConfig: QuantizationConfig#
  QuantizationConfig.get_quant_method: QuantizationConfig#get_quant_method().
  QuantLinearConfig.weight_sharding: QuantLinearConfig#weight_sharding.
  QuantLinearConfig.__init__: QuantLinearConfig#__init__().
  QuantLinearConfig.batch_sharding: QuantLinearConfig#batch_sharding.
  QuantLinearConfig: QuantLinearConfig#
  QuantizationConfig.get_from_keys: QuantizationConfig#get_from_keys().
  QuantizationConfig.is_layer_skipped: QuantizationConfig#is_layer_skipped().
  QuantLinearConfig.in_features: QuantLinearConfig#in_features.
  QuantLinearConfig.out_features: QuantLinearConfig#out_features.
  QuantLinearConfig.batch_features: QuantLinearConfig#batch_features.
  _to_partition_spec: _to_partition_spec().
  QuantLinearConfig.out_features_sharding: QuantLinearConfig#out_features_sharding.
  QuantizationConfig.prefix_full_match: QuantizationConfig#prefix_full_match().
  QuantLinearConfig.in_features_sharding: QuantLinearConfig#in_features_sharding.
  QuantLinearConfig.bias_sharding: QuantLinearConfig#bias_sharding.
  QuantizationConfig.__init__: QuantizationConfig#__init__().
---
# Module: [`tpu_inference/layers/jax/quantization/configs.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/configs.py)

## Classes
### `QuantLinearConfig`  ·  implements/extends QuantLinearConfig
- def: [`tpu_inference/layers/jax/quantization/configs.py:116`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/configs.py#L116)
- doc: Quantization config for jax linear layers.
- signature: `class QuantLinearConfig(CommonQuantLinearConfig):`
- members:
  - `batch_features` — [`L160`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/configs.py#L160)
  - `batch_sharding` — [`L166`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/configs.py#L166)
  - `bias_sharding` — [`L179`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/configs.py#L179)
  - `in_features` — [`L138`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/configs.py#L138)
  - `in_features_sharding` — [`L165`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/configs.py#L165)
  - `out_features` — [`L157`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/configs.py#L157)
  - `out_features_sharding` — [`L164`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/configs.py#L164)
  - `weight_sharding` — [`L173`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/configs.py#L173)
- protocol/private: `__init__`[`L119`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/configs.py#L119)
- uses (calls/refs, reference-scoped): [`JaxEinsum`](../linear.md#JaxEinsum), [`weight`](../linear.md#JaxEinsum.weight), [`QuantLinearConfig`](../../common/quantization/configs.md#QuantLinearConfig), `einsum_str`, [`__init__`](../../common/quantization/configs.md#QuantLinearConfig.__init__), [`_to_partition_spec`](configs.md#_to_partition_spec)
- used by: [`get_quant_method`](fp8.md#Fp8Config.get_quant_method), [`QuantLinearConfig`](../../common/quantization/configs.md#QuantLinearConfig), [`weight_sharding`](fp8.md#Fp8TensorwiseLinearMethod.weight_sharding), [`__init__`](fp8.md#Fp8BlockwiseLinearMethod.__init__), [`__init__`](fp8.md#Fp8TensorwiseLinearMethod.__init__), [`batch_sharding`](fp8.md#Fp8TensorwiseLinearMethod.batch_sharding), [`batch_features`](fp8.md#Fp8BlockwiseLinearMethod.batch_features), [`weight_sharding`](fp8.md#Fp8BlockwiseLinearMethod.weight_sharding), [`batch_features`](fp8.md#Fp8TensorwiseLinearMethod.batch_features), [`in_features`](fp8.md#Fp8BlockwiseLinearMethod.in_features), [`bias_sharding`](fp8.md#Fp8BlockwiseLinearMethod.bias_sharding), [`out_features`](fp8.md#Fp8BlockwiseLinearMethod.out_features), [`output_shape`](fp8.md#Fp8TensorwiseLinearMethod.output_shape), [`batch_sharding`](fp8.md#Fp8BlockwiseLinearMethod.batch_sharding)

### `QuantizationConfig`  ·  implements/extends ABC
- def: [`tpu_inference/layers/jax/quantization/configs.py:27`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/configs.py#L27)
- signature: `class QuantizationConfig(ABC):`
- members:
  - `get_from_keys(cls, config: dict, keys: list, *args)` — [`L38`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/configs.py#L38) — Get value from config using the first matching key.'
  - `get_quant_method(self, layer: JaxModule, prefix: str)` — [`L33`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/configs.py#L33)
  - `is_layer_skipped(cls, prefix: str, *, ignored_layers: list[str], fused_mapping: dict = dict())` — [`L53`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/configs.py#L53) — Check if a layer should be skipped from quantization.
  - `prefix_full_match(prefix: str, ignored_layers: list[str])` — [`L64`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/configs.py#L64)
- protocol/private: `__init__`[`L29`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/configs.py#L29)
- uses (calls/refs, reference-scoped): [`JaxModule`](../../__init__.md#JaxModule), [`QuantizeMethodBase`](__init__.md#QuantizeMethodBase), [`get_quant_method`](unquantized.md#UnquantizedConfig.get_quant_method), [`get_quant_method`](fp8.md#Fp8Config.get_quant_method), [`UnquantizedConfig`](unquantized.md#UnquantizedConfig)
- used by: [`get_quant_method`](fp8.md#Fp8Config.get_quant_method), [`quant_method`](../moe/moe.md#JaxMoE.quant_method), [`quant_method`](../linear.md#JaxEinsum.quant_method), [`quant_method`](../embed.md#JaxEmbed.quant_method), [`quant_method`](../norm.md#JaxRmsNorm.quant_method), [`quant_config`](../../../models/jax/deepseek_v3.md#DeepseekV3BaseAttention.quant_config), [`UnquantizedConfig`](unquantized.md#UnquantizedConfig), [`__init__`](../linear.md#JaxLinear.__init__), [`quant_config`](../moe/moe.md#JaxMoE.quant_config), [`__init__`](../../../models/jax/deepseek_v3.md#DeepSeekV3Router.__init__), [`__init__`](../../../models/jax/qwen3_moe.md#Qwen3MoeDecoderLayer.__init__), [`quant_config`](../../../models/jax/deepseek_v3.md#DeepseekV3MLP.quant_config), [`Fp8Config`](fp8.md#Fp8Config), [`is_checkpoint_fp8_serialized`](fp8.md#Fp8Config.is_checkpoint_fp8_serialized), [`__init__`](../linear.md#JaxEinsum.__init__), [`__init__`](../../../models/jax/deepseek_v3.md#MLAEinsum.__init__), [`__init__`](../linear.md#JaxMergedColumnParallelLinear.__init__), [`__init__`](../moe/moe.md#JaxRoutedExperts.__init__), [`__init__`](../embed.md#JaxEmbed.__init__), [`__init__`](../linear.md#JaxQKVParallelLinear.__init__), [`__init__`](../norm.md#JaxRmsNorm.__init__), [`__init__`](fp8.md#Fp8Config.__init__)

## Functions
- `_to_partition_spec(sharding)` — [`L101`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/configs.py#L101) — Convert a sharding value to a PartitionSpec.


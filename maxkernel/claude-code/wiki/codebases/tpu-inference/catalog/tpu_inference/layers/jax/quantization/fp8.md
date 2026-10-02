---
title: 'Module: tpu_inference/layers/jax/quantization/fp8.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/jax/quantization/fp8.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.jax.quantization.fp8`/
symbols:
  Fp8BlockwiseLinearMethod.process_weights_after_loading: Fp8BlockwiseLinearMethod#process_weights_after_loading().
  Fp8FusedMoEMethod.apply_jax: Fp8FusedMoEMethod#apply_jax().
  Fp8FusedMoEMethod.process_weights_after_loading: Fp8FusedMoEMethod#process_weights_after_loading().
  Fp8BlockwiseLinearMethod.create_weights_jax: Fp8BlockwiseLinearMethod#create_weights_jax().
  Fp8Config.get_quant_method: Fp8Config#get_quant_method().
  Fp8TensorwiseLinearMethod.create_weights_jax: Fp8TensorwiseLinearMethod#create_weights_jax().
  Fp8BlockwiseLinearMethod.apply_jax: Fp8BlockwiseLinearMethod#apply_jax().
  Fp8TensorwiseLinearMethod.apply_jax: Fp8TensorwiseLinearMethod#apply_jax().
  Fp8FusedMoEMethod.create_weights_jax: Fp8FusedMoEMethod#create_weights_jax().
  Fp8FusedMoEMethod.weight_scale_name: Fp8FusedMoEMethod#weight_scale_name.
  Fp8FusedMoEMethod.load_weights: Fp8FusedMoEMethod#load_weights().
  FP8_QUANT_METHOD_SUPPORTED_MOE_BACKENDS: FP8_QUANT_METHOD_SUPPORTED_MOE_BACKENDS.
  Fp8TensorwiseLinearMethod.weight_sharding: Fp8TensorwiseLinearMethod#weight_sharding.
  Fp8BlockwiseLinearMethod.kernel_shape: Fp8BlockwiseLinearMethod#kernel_shape.
  Fp8BlockwiseLinearMethod.__init__: Fp8BlockwiseLinearMethod#__init__().
  Fp8TensorwiseLinearMethod.__init__: Fp8TensorwiseLinearMethod#__init__().
  Fp8TensorwiseLinearMethod.batch_sharding: Fp8TensorwiseLinearMethod#batch_sharding.
  Fp8TensorwiseLinearMethod.kernel_shape: Fp8TensorwiseLinearMethod#kernel_shape.
  Fp8BlockwiseLinearMethod.batch_features: Fp8BlockwiseLinearMethod#batch_features.
  Fp8BlockwiseLinearMethod.weight_sharding: Fp8BlockwiseLinearMethod#weight_sharding.
  Fp8Config: Fp8Config#
  load_fp8_weight: load_fp8_weight().
  Fp8TensorwiseLinearMethod: Fp8TensorwiseLinearMethod#
  Fp8BlockwiseLinearMethod: Fp8BlockwiseLinearMethod#
  Fp8Config.weight_block_size: Fp8Config#weight_block_size.
  Fp8FusedMoEMethod.weight_block_size: Fp8FusedMoEMethod#weight_block_size.
  Fp8Config.is_checkpoint_fp8_serialized: Fp8Config#is_checkpoint_fp8_serialized.
  Fp8TensorwiseLinearMethod.batch_features: Fp8TensorwiseLinearMethod#batch_features.
  Fp8BlockwiseLinearMethod.in_features: Fp8BlockwiseLinearMethod#in_features.
  Fp8BlockwiseLinearMethod.bias_sharding: Fp8BlockwiseLinearMethod#bias_sharding.
  logger: logger.
  Fp8TensorwiseLinearMethod.output_shape: Fp8TensorwiseLinearMethod#output_shape.
  Fp8BlockwiseLinearMethod.out_features: Fp8BlockwiseLinearMethod#out_features.
  Fp8FusedMoEMethod.block_quant: Fp8FusedMoEMethod#block_quant.
  Fp8Config.ignored_layers: Fp8Config#ignored_layers.
  Fp8BlockwiseLinearMethod.quant_config: Fp8BlockwiseLinearMethod#quant_config.
  Fp8TensorwiseLinearMethod.einsum_str: Fp8TensorwiseLinearMethod#einsum_str.
  Fp8BlockwiseLinearMethod.einsum_str: Fp8BlockwiseLinearMethod#einsum_str.
  Fp8FusedMoEMethod: Fp8FusedMoEMethod#
  Fp8BlockwiseLinearMethod.batch_sharding: Fp8BlockwiseLinearMethod#batch_sharding.
  Fp8Config.__init__: Fp8Config#__init__().
  Fp8FusedMoEMethod.extra_backend_kwargs: Fp8FusedMoEMethod#extra_backend_kwargs.
  Fp8Config.ACTIVATION_SCHEMES: Fp8Config#ACTIVATION_SCHEMES.
  Fp8TensorwiseLinearMethod.in_features: Fp8TensorwiseLinearMethod#in_features.
  Fp8FusedMoEMethod.__init__: Fp8FusedMoEMethod#__init__().
  Fp8Config.activation_scheme: Fp8Config#activation_scheme.
---
# Module: [`tpu_inference/layers/jax/quantization/fp8.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/fp8.py)

## Classes
### `Fp8BlockwiseLinearMethod`  ·  implements/extends Fp8LinearMethod, QuantizeMethodBase
- def: [`tpu_inference/layers/jax/quantization/fp8.py:166`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/fp8.py#L166)
- doc: Block-wise Fp8 method for JAX Linear layer.
- signature: `class Fp8BlockwiseLinearMethod(QuantizeMethodBase, common_fp8.Fp8LinearMethod):`
- members:
  - `apply_jax(self, layer: JaxModule, x: jax.Array)` — [`L316`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/fp8.py#L316)
  - `create_weights_jax(self, layer: JaxModule, *weight_args, rngs, **extra_weight_attrs)` — [`L189`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/fp8.py#L189)
  - `process_weights_after_loading(self, layer: JaxEinsum)` — [`L258`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/fp8.py#L258)
  - `batch_features` — [`L177`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/fp8.py#L177)
  - `batch_sharding` — [`L178`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/fp8.py#L178)
  - `bias_sharding` — [`L180`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/fp8.py#L180)
  - `einsum_str` — [`L173`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/fp8.py#L173)
  - `in_features` — [`L176`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/fp8.py#L176)
  - `kernel_shape` — [`L184`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/fp8.py#L184)
  - `out_features` — [`L175`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/fp8.py#L175)
  - `quant_config` — [`L172`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/fp8.py#L172)
  - `weight_sharding` — [`L179`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/fp8.py#L179)
- protocol/private: `__init__`[`L169`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/fp8.py#L169)
- uses (calls/refs, reference-scoped): [`JaxModule`](../../__init__.md#JaxModule), [`JaxEinsum`](../linear.md#JaxEinsum), [`QuantizeMethodBase`](__init__.md#QuantizeMethodBase), [`shard_linear_weights`](../../common/process_weights/linear_weights.md#shard_linear_weights), [`process_blockwise_fp8_linear_weights`](../../common/quantization/fp8.md#process_blockwise_fp8_linear_weights), [`weight`](../linear.md#JaxEinsum.weight), [`load_nnx_param_from_reshaped_torch`](../../../models/jax/utils/weight_utils.md#load_nnx_param_from_reshaped_torch), [`_apply_fused`](../../common/quantization/fp8.md#Fp8LinearMethod._apply_fused), [`cpu_mesh_context`](../../common/utils.md#cpu_mesh_context), [`linear_config`](../../common/quantization/fp8.md#Fp8LinearMethod.linear_config), [`weight_scale`](../../common/process_weights/linear_weights.md#LinearWeights.weight_scale), [`bias`](../../common/process_weights/linear_weights.md#LinearWeights.bias), [`weight`](../../common/process_weights/linear_weights.md#LinearWeights.weight), [`weight_sharding`](configs.md#QuantLinearConfig.weight_sharding), `bias`, [`enable_quantized_matmul_kernel`](../../common/quantization/configs.md#QuantLinearConfig.enable_quantized_matmul_kernel), [`sharded_quantized_batched_matmul`](../../common/linear.md#sharded_quantized_batched_matmul), [`Fp8LinearMethod`](../../common/quantization/fp8.md#Fp8LinearMethod), [`batch_sharding`](configs.md#QuantLinearConfig.batch_sharding), [`cpu_mesh`](../../common/utils.md#cpu_mesh), [`requant_block_size`](../../common/quantization/configs.md#QuantLinearConfig.requant_block_size), `einsum_str`, [`output_sizes`](../../common/quantization/configs.md#QuantLinearConfig.output_sizes), [`prefix`](../linear.md#JaxEinsum.prefix), [`Fp8Config`](fp8.md#Fp8Config), [`QuantLinearConfig`](configs.md#QuantLinearConfig), [`n_shards`](../../common/quantization/configs.md#QuantLinearConfig.n_shards), [`weight_block_size`](fp8.md#Fp8Config.weight_block_size), [`weight_sharding`](../../common/quantization/configs.md#QuantLinearConfig.weight_sharding), [`__init__`](../../common/quantization/fp8.md#Fp8LinearMethod.__init__), [`in_features`](configs.md#QuantLinearConfig.in_features), [`fuse_matmuls`](../../common/quantization/configs.md#QuantLinearConfig.fuse_matmuls), [`bias_sharding`](../../common/quantization/configs.md#QuantLinearConfig.bias_sharding), [`batch_features`](configs.md#QuantLinearConfig.batch_features), `kernel_shape`, [`mesh`](../../common/quantization/configs.md#QuantLinearConfig.mesh), [`out_features`](configs.md#QuantLinearConfig.out_features), [`requant_weight_dtype`](../../common/quantization/configs.md#QuantLinearConfig.requant_weight_dtype), `dtype`, [`bias_sharding`](configs.md#QuantLinearConfig.bias_sharding)  (+1 more)
- used by: [`QuantizeMethodBase`](__init__.md#QuantizeMethodBase), [`get_quant_method`](fp8.md#Fp8Config.get_quant_method), [`apply_jax`](__init__.md#QuantizeMethodBase.apply_jax), [`create_weights_jax`](__init__.md#QuantizeMethodBase.create_weights_jax), [`process_weights_after_loading`](__init__.md#QuantizeMethodBase.process_weights_after_loading), [`Fp8LinearMethod`](../../common/quantization/fp8.md#Fp8LinearMethod)

### `Fp8Config`
- def: [`tpu_inference/layers/jax/quantization/fp8.py:621`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/fp8.py#L621)
- members:
  - `get_quant_method(self, layer: JaxModule, prefix: str)` — [`L661`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/fp8.py#L661)
  - `ACTIVATION_SCHEMES` — [`L623`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/fp8.py#L623)
  - `activation_scheme` — [`L644`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/fp8.py#L644)
  - `ignored_layers` — [`L645`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/fp8.py#L645)
  - `is_checkpoint_fp8_serialized` — [`L629`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/fp8.py#L629)
  - `weight_block_size` — [`L659`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/fp8.py#L659)
- protocol/private: `__init__`[`L625`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/fp8.py#L625)
- uses (calls/refs, reference-scoped): [`JaxModule`](../../__init__.md#JaxModule), [`JaxEinsum`](../linear.md#JaxEinsum), [`QuantizeMethodBase`](__init__.md#QuantizeMethodBase), [`JaxMoE`](../moe/moe.md#JaxMoE), [`QuantizationConfig`](configs.md#QuantizationConfig), [`JaxRoutedExperts`](../moe/moe.md#JaxRoutedExperts), [`UnquantizedLinearMethod`](unquantized.md#UnquantizedLinearMethod), [`UnquantizedFusedMoEMethod`](unquantized.md#UnquantizedFusedMoEMethod), [`QuantLinearConfig`](configs.md#QuantLinearConfig), [`Fp8BlockwiseLinearMethod`](fp8.md#Fp8BlockwiseLinearMethod), [`Fp8TensorwiseLinearMethod`](fp8.md#Fp8TensorwiseLinearMethod), [`get_from_keys`](configs.md#QuantizationConfig.get_from_keys), [`is_layer_skipped`](configs.md#QuantizationConfig.is_layer_skipped), [`Fp8FusedMoEMethod`](fp8.md#Fp8FusedMoEMethod)
- used by: [`process_weights_after_loading`](fp8.md#Fp8BlockwiseLinearMethod.process_weights_after_loading), [`create_weights_jax`](fp8.md#Fp8BlockwiseLinearMethod.create_weights_jax), [`create_weights_jax`](fp8.md#Fp8FusedMoEMethod.create_weights_jax), [`get_quant_method`](configs.md#QuantizationConfig.get_quant_method), [`__init__`](fp8.md#Fp8BlockwiseLinearMethod.__init__), [`get_tpu_quantization_config`](__init__.md#get_tpu_quantization_config)

### `Fp8FusedMoEMethod`  ·  implements/extends QuantizeMethodBase
- def: [`tpu_inference/layers/jax/quantization/fp8.py:340`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/fp8.py#L340)
- doc: Fp8 method for JAXMoE layer.
- signature: `class Fp8FusedMoEMethod(QuantizeMethodBase):`
- members:
  - `apply_jax(self, layer: JaxModule, x: jax.Array, *, router_logits: jax.Array)` — [`L566`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/fp8.py#L566) — Run the forward pass of the MoE layer.
  - `create_weights_jax(self, layer: JaxMoE, *weight_args, rngs, **extra_weight_attrs)` — [`L411`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/fp8.py#L411) — Create the quant method-specific weights.
  - `load_weights(self, *, layer: JaxMoE, original_load_weights_fn, weights: Iterable)` — [`L355`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/fp8.py#L355) — Load scale paramters and delegate the weight paramters to `original_load_weights_fn`
  - `process_weights_after_loading(self, layer: JaxMoE)` — [`L469`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/fp8.py#L469) — Process weights after loading.
  - `block_quant` — [`L351`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/fp8.py#L351)
  - `extra_backend_kwargs` — [`L349`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/fp8.py#L349)
  - `weight_block_size` — [`L350`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/fp8.py#L350)
  - `weight_scale_name` — [`L352`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/fp8.py#L352)
- protocol/private: `__init__`[`L347`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/fp8.py#L347)
- uses (calls/refs, reference-scoped): [`JaxModule`](../../__init__.md#JaxModule), [`moe_apply`](../../common/moe.md#moe_apply), [`MoEBackend`](../../common/moe.md#MoEBackend), [`FusedMoEWeights`](../../common/process_weights/moe_weights.md#FusedMoEWeights), [`w13_weight_scale`](../../common/process_weights/moe_weights.md#FusedMoEWeights.w13_weight_scale), [`w2_weight_scale`](../../common/process_weights/moe_weights.md#FusedMoEWeights.w2_weight_scale), [`w13_weight`](../../common/process_weights/moe_weights.md#FusedMoEWeights.w13_weight), [`w13_bias`](../../common/process_weights/moe_weights.md#FusedMoEWeights.w13_bias), [`w2_weight`](../../common/process_weights/moe_weights.md#FusedMoEWeights.w2_weight), [`QuantizeMethodBase`](__init__.md#QuantizeMethodBase), [`w2_bias`](../../common/process_weights/moe_weights.md#FusedMoEWeights.w2_bias), [`kernel_down_proj_EFD`](../moe/moe.md#JaxMoE.kernel_down_proj_EFD), [`process_quantized_moe_weights`](../../common/process_weights/moe_weights.md#process_quantized_moe_weights), [`JaxMoE`](../moe/moe.md#JaxMoE), [`cpu_mesh_context`](../../common/utils.md#cpu_mesh_context), [`kernel_gating_EDF`](../moe/moe.md#JaxMoE.kernel_gating_EDF), [`kernel_up_proj_EDF`](../moe/moe.md#JaxMoE.kernel_up_proj_EDF), [`moe_backend`](../moe/moe.md#JaxMoE.moe_backend), [`JaxRoutedExperts`](../moe/moe.md#JaxRoutedExperts), [`FUSED_MOE`](../../common/moe.md#MoEBackend.FUSED_MOE), [`FP8_QUANT_METHOD_SUPPORTED_MOE_BACKENDS`](fp8.md#FP8_QUANT_METHOD_SUPPORTED_MOE_BACKENDS), [`jax_array_from_reshaped_torch`](../../../models/jax/utils/weight_utils.md#jax_array_from_reshaped_torch), [`cpu_mesh`](../../common/utils.md#cpu_mesh), [`activation`](../moe/moe.md#JaxMoE.activation), [`quant_config`](../moe/moe.md#JaxMoE.quant_config), [`activation_ffw_td`](../moe/moe.md#JaxMoE.activation_ffw_td), [`dtype`](../moe/moe.md#JaxMoE.dtype), [`prefix`](../moe/moe.md#JaxMoE.prefix), [`Fp8Config`](fp8.md#Fp8Config), [`mesh`](../moe/moe.md#JaxMoE.mesh), [`logger`](fp8.md#logger)
- used by: [`QuantizeMethodBase`](__init__.md#QuantizeMethodBase), [`get_quant_method`](fp8.md#Fp8Config.get_quant_method), [`apply_jax`](__init__.md#QuantizeMethodBase.apply_jax), [`create_weights_jax`](__init__.md#QuantizeMethodBase.create_weights_jax), [`process_weights_after_loading`](__init__.md#QuantizeMethodBase.process_weights_after_loading)

### `Fp8TensorwiseLinearMethod`  ·  implements/extends Fp8LinearMethod, QuantizeMethodBase
- def: [`tpu_inference/layers/jax/quantization/fp8.py:82`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/fp8.py#L82)
- doc: Tensor-wise Fp8 method for JAX Linear layer.
- signature: `class Fp8TensorwiseLinearMethod(QuantizeMethodBase, common_fp8.Fp8LinearMethod):`
- members:
  - `apply_jax(self, layer: JaxModule, x: jax.Array)` — [`L142`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/fp8.py#L142)
  - `create_weights_jax(self, layer: JaxEinsum, *weight_args, rngs, **extra_weight_attrs)` — [`L106`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/fp8.py#L106)
  - `batch_features` — [`L92`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/fp8.py#L92)
  - `batch_sharding` — [`L93`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/fp8.py#L93)
  - `einsum_str` — [`L89`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/fp8.py#L89)
  - `in_features` — [`L104`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/fp8.py#L104)
  - `kernel_shape` — [`L100`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/fp8.py#L100)
  - `output_shape` — [`L91`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/fp8.py#L91)
  - `weight_sharding` — [`L96`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/fp8.py#L96)
- protocol/private: `__init__`[`L86`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/fp8.py#L86)
- uses (calls/refs, reference-scoped): [`JaxModule`](../../__init__.md#JaxModule), [`JaxEinsum`](../linear.md#JaxEinsum), [`create_param`](../base.md#create_param), [`QuantizeMethodBase`](__init__.md#QuantizeMethodBase), [`weight`](../linear.md#JaxEinsum.weight), [`_apply_fused`](../../common/quantization/fp8.md#Fp8LinearMethod._apply_fused), [`linear_config`](../../common/quantization/fp8.md#Fp8LinearMethod.linear_config), [`weight_sharding`](configs.md#QuantLinearConfig.weight_sharding), [`sharded_quantized_batched_matmul`](../../common/linear.md#sharded_quantized_batched_matmul), [`Fp8LinearMethod`](../../common/quantization/fp8.md#Fp8LinearMethod), [`batch_sharding`](configs.md#QuantLinearConfig.batch_sharding), `einsum_str`, [`output_sizes`](../../common/quantization/configs.md#QuantLinearConfig.output_sizes), [`prefix`](../linear.md#JaxEinsum.prefix), [`QuantLinearConfig`](configs.md#QuantLinearConfig), [`load_fp8_weight`](fp8.md#load_fp8_weight), [`__init__`](../../common/quantization/fp8.md#Fp8LinearMethod.__init__), [`in_features`](configs.md#QuantLinearConfig.in_features), [`batch_features`](configs.md#QuantLinearConfig.batch_features), `kernel_shape`, [`mesh`](../../common/quantization/configs.md#QuantLinearConfig.mesh), [`out_features`](configs.md#QuantLinearConfig.out_features)
- used by: [`QuantizeMethodBase`](__init__.md#QuantizeMethodBase), [`get_quant_method`](fp8.md#Fp8Config.get_quant_method), [`apply_jax`](__init__.md#QuantizeMethodBase.apply_jax), [`create_weights_jax`](__init__.md#QuantizeMethodBase.create_weights_jax), [`Fp8LinearMethod`](../../common/quantization/fp8.md#Fp8LinearMethod)

## Functions
- `load_fp8_weight(jax_param: nnx.Param, torch_weight: torch.Tensor, param_name: str)` — [`L57`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/fp8.py#L57) — Loads FP8 weights from a torch tensor into a JAX parameter.

## Module values
- `FP8_QUANT_METHOD_SUPPORTED_MOE_BACKENDS` — [`L52`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/fp8.py#L52)
- `logger` — [`L49`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/fp8.py#L49)


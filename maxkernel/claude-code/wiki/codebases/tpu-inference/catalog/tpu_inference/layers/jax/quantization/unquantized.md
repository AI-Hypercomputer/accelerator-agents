---
title: 'Module: tpu_inference/layers/jax/quantization/unquantized.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/jax/quantization/unquantized.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.jax.quantization.unquantized`/
symbols:
  UnquantizedFusedMoEMethod.apply_jax: UnquantizedFusedMoEMethod#apply_jax().
  UnquantizedFusedMoEMethod.extra_backend_kwargs: UnquantizedFusedMoEMethod#extra_backend_kwargs.
  UnquantizedConfig.get_quant_method: UnquantizedConfig#get_quant_method().
  UnquantizedLinearMethod.apply_jax: UnquantizedLinearMethod#apply_jax().
  UnquantizedMergedLinearMethod.create_weights_jax: UnquantizedMergedLinearMethod#create_weights_jax().
  UnquantizedFusedMoEMethod.process_weights_after_loading: UnquantizedFusedMoEMethod#process_weights_after_loading().
  UnquantizedMergedLinearMethod._load_merged_tensor: UnquantizedMergedLinearMethod#_load_merged_tensor().
  UnquantizedLinearMethod: UnquantizedLinearMethod#
  UnquantizedConfig: UnquantizedConfig#
  UnquantizedFusedMoEMethod: UnquantizedFusedMoEMethod#
  UnquantizedMergedLinearMethod: UnquantizedMergedLinearMethod#
  logger: logger.
---
# Module: [`tpu_inference/layers/jax/quantization/unquantized.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/unquantized.py)

## Classes
### `UnquantizedConfig`  ·  implements/extends QuantizationConfig
- def: [`tpu_inference/layers/jax/quantization/unquantized.py:342`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/unquantized.py#L342)
- signature: `class UnquantizedConfig(QuantizationConfig):`
- members:
  - `get_quant_method(self, layer: JaxModule, prefix: str)` — [`L344`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/unquantized.py#L344)
- uses (calls/refs, reference-scoped): [`JaxModule`](../../__init__.md#JaxModule), [`JaxEinsum`](../linear.md#JaxEinsum), [`QuantizeMethodBase`](__init__.md#QuantizeMethodBase), [`weight`](../linear.md#JaxEinsum.weight), [`JaxMoE`](../moe/moe.md#JaxMoE), [`QuantizationConfig`](configs.md#QuantizationConfig), [`JaxRoutedExperts`](../moe/moe.md#JaxRoutedExperts), [`QuantLinearConfig`](../../common/quantization/configs.md#QuantLinearConfig), [`UnquantizedLinearMethod`](unquantized.md#UnquantizedLinearMethod), [`JaxMergedColumnParallelLinear`](../linear.md#JaxMergedColumnParallelLinear), [`UnquantizedFusedMoEMethod`](unquantized.md#UnquantizedFusedMoEMethod), `einsum_str`, [`output_sizes`](../linear.md#JaxMergedColumnParallelLinear.output_sizes), [`UnquantizedMergedLinearMethod`](unquantized.md#UnquantizedMergedLinearMethod), `kernel_shape`
- used by: [`QuantizationConfig`](configs.md#QuantizationConfig), [`get_quant_method`](configs.md#QuantizationConfig.get_quant_method), [`get_tpu_quantization_config`](__init__.md#get_tpu_quantization_config), [`output_sizes`](../linear.md#JaxMergedColumnParallelLinear.output_sizes), [`quant_config`](../moe/moe.md#JaxRoutedExperts.quant_config), [`num_experts_per_tok`](../moe/moe.md#JaxRoutedExperts.num_experts_per_tok)

### `UnquantizedFusedMoEMethod`  ·  implements/extends QuantizeMethodBase, UnquantizedFusedMoEMethod
- def: [`tpu_inference/layers/jax/quantization/unquantized.py:169`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/unquantized.py#L169)
- doc: Unquantized method for JaxRoutedExperts layers.
- signature: `class UnquantizedFusedMoEMethod(QuantizeMethodBase, jax_common.UnquantizedFusedMoEMethod):`
- members:
  - `apply_jax(self, layer: JaxRoutedExperts, x: jax.Array, *, router_logits: jax.Array)` — [`L285`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/unquantized.py#L285) — Forward pass for MoE layer.
  - `process_weights_after_loading(self, layer: JaxRoutedExperts, *args, **kwargs)` — [`L175`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/unquantized.py#L175) — Process weights after loading.
  - `extra_backend_kwargs` — [`L212`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/unquantized.py#L212)
- uses (calls/refs, reference-scoped): [`moe_apply`](../../common/moe.md#moe_apply), [`MoEBackend`](../../common/moe.md#MoEBackend), [`FusedMoEWeights`](../../common/process_weights/moe_weights.md#FusedMoEWeights), [`w13_weight_scale`](../../common/process_weights/moe_weights.md#FusedMoEWeights.w13_weight_scale), [`w2_weight_scale`](../../common/process_weights/moe_weights.md#FusedMoEWeights.w2_weight_scale), [`w13_weight`](../../common/process_weights/moe_weights.md#FusedMoEWeights.w13_weight), [`shard_moe_weights`](../../common/process_weights/moe_weights.md#shard_moe_weights), [`process_unquantized_moe_weights`](../../common/process_weights/moe_weights.md#process_unquantized_moe_weights), [`w13_bias`](../../common/process_weights/moe_weights.md#FusedMoEWeights.w13_bias), [`w2_weight`](../../common/process_weights/moe_weights.md#FusedMoEWeights.w2_weight), [`QuantizeMethodBase`](__init__.md#QuantizeMethodBase), [`w2_bias`](../../common/process_weights/moe_weights.md#FusedMoEWeights.w2_bias), [`shard_put`](../../../models/jax/utils/weight_utils.md#shard_put), [`JaxRoutedExperts`](../moe/moe.md#JaxRoutedExperts), [`FUSED_MOE`](../../common/moe.md#MoEBackend.FUSED_MOE), [`kernel_gating_EDF`](../moe/moe.md#JaxRoutedExperts.kernel_gating_EDF), [`kernel_up_proj_EDF`](../moe/moe.md#JaxRoutedExperts.kernel_up_proj_EDF), [`kernel_down_proj_EFD`](../moe/moe.md#JaxRoutedExperts.kernel_down_proj_EFD), [`fused_moe_backends`](../../common/moe.md#MoEBackend.fused_moe_backends), [`moe_backend`](../moe/moe.md#JaxRoutedExperts.moe_backend), [`UnfusedMoEWeights`](../../common/process_weights/moe_weights.md#UnfusedMoEWeights), [`GMM_TP`](../../common/moe.md#MoEBackend.GMM_TP), [`GMM_EP`](../../common/moe.md#MoEBackend.GMM_EP), [`DENSE_MAT`](../../common/moe.md#MoEBackend.DENSE_MAT), [`MEGABLX_GMM`](../../common/moe.md#MoEBackend.MEGABLX_GMM), [`w1_weight`](../../common/process_weights/moe_weights.md#UnfusedMoEWeights.w1_weight), [`w2_weight`](../../common/process_weights/moe_weights.md#UnfusedMoEWeights.w2_weight), [`w3_weight`](../../common/process_weights/moe_weights.md#UnfusedMoEWeights.w3_weight), [`UnquantizedFusedMoEMethod`](../../common/quantization/unquantized.md#UnquantizedFusedMoEMethod), [`activation_ffw_td`](../moe/moe.md#JaxRoutedExperts.activation_ffw_td), [`activation`](../moe/moe.md#JaxRoutedExperts.activation), [`dtype`](../moe/moe.md#JaxRoutedExperts.dtype), [`mesh`](../moe/moe.md#JaxRoutedExperts.mesh), [`w1_bias`](../../common/process_weights/moe_weights.md#UnfusedMoEWeights.w1_bias), [`w1_weight_scale`](../../common/process_weights/moe_weights.md#UnfusedMoEWeights.w1_weight_scale), [`w2_bias`](../../common/process_weights/moe_weights.md#UnfusedMoEWeights.w2_bias), [`w2_weight_scale`](../../common/process_weights/moe_weights.md#UnfusedMoEWeights.w2_weight_scale), [`w3_bias`](../../common/process_weights/moe_weights.md#UnfusedMoEWeights.w3_bias), [`w3_weight_scale`](../../common/process_weights/moe_weights.md#UnfusedMoEWeights.w3_weight_scale)
- used by: [`QuantizeMethodBase`](__init__.md#QuantizeMethodBase), [`get_quant_method`](unquantized.md#UnquantizedConfig.get_quant_method), [`get_quant_method`](fp8.md#Fp8Config.get_quant_method), [`apply_jax`](__init__.md#QuantizeMethodBase.apply_jax), [`process_weights_after_loading`](__init__.md#QuantizeMethodBase.process_weights_after_loading), [`UnquantizedFusedMoEMethod`](../../common/quantization/unquantized.md#UnquantizedFusedMoEMethod)

### `UnquantizedLinearMethod`  ·  implements/extends QuantizeMethodBase, UnquantizedLinearMethod
- def: [`tpu_inference/layers/jax/quantization/unquantized.py:46`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/unquantized.py#L46)
- doc: Unquantized method for JAX Linear layer.
- signature: `class UnquantizedLinearMethod(QuantizeMethodBase, jax_common.UnquantizedLinearMethod):`
- members:
  - `apply_jax(self, layer: JaxModule, x: jax.Array)` — [`L51`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/unquantized.py#L51)
- uses (calls/refs, reference-scoped): [`JaxModule`](../../__init__.md#JaxModule), [`JaxEinsum`](../linear.md#JaxEinsum), [`QuantizeMethodBase`](__init__.md#QuantizeMethodBase), [`weight`](../linear.md#JaxEinsum.weight), `bias`, `einsum_str`, `linear_config`, [`UnquantizedLinearMethod`](../../common/quantization/unquantized.md#UnquantizedLinearMethod), [`fuse_matmuls`](../../common/quantization/configs.md#QuantLinearConfig.fuse_matmuls), [`UnquantizedMergedLinearMethod`](unquantized.md#UnquantizedMergedLinearMethod), [`_apply_fused`](../../common/quantization/unquantized.md#UnquantizedLinearMethod._apply_fused), [`_get_name`](../../__init__.md#JaxModule._get_name)
- used by: [`QuantizeMethodBase`](__init__.md#QuantizeMethodBase), [`get_quant_method`](unquantized.md#UnquantizedConfig.get_quant_method), [`get_quant_method`](fp8.md#Fp8Config.get_quant_method), [`apply_jax`](__init__.md#QuantizeMethodBase.apply_jax), [`UnquantizedLinearMethod`](../../common/quantization/unquantized.md#UnquantizedLinearMethod), [`UnquantizedMergedLinearMethod`](unquantized.md#UnquantizedMergedLinearMethod)

### `UnquantizedMergedLinearMethod`  ·  implements/extends UnquantizedLinearMethod
- def: [`tpu_inference/layers/jax/quantization/unquantized.py:68`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/unquantized.py#L68)
- doc: Unquantized method for JaxMerged*Linear layers.
- signature: `class UnquantizedMergedLinearMethod(UnquantizedLinearMethod):`
- members:
  - `_load_merged_tensor(param: nnx.Param, torch_tensor, shard_id: int = -1, *, n_shards: int, output_sizes: list, param_name: str)` — [`L118`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/unquantized.py#L118) — Accumulate one projection's checkpoint tensor, fuse when complete.
  - `create_weights_jax(self, layer: JaxEinsum, *weight_args, rngs, **extra_weight_attrs)` — [`L91`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/unquantized.py#L91)
- uses (calls/refs, reference-scoped): [`JaxEinsum`](../linear.md#JaxEinsum), [`weight`](../linear.md#JaxEinsum.weight), [`cpu_mesh_context`](../../common/utils.md#cpu_mesh_context), [`UnquantizedLinearMethod`](unquantized.md#UnquantizedLinearMethod), `bias`, [`reorder_concatenated_tensor_for_sharding`](../../common/utils.md#reorder_concatenated_tensor_for_sharding), [`jax_array_from_reshaped_torch`](../../../models/jax/utils/weight_utils.md#jax_array_from_reshaped_torch), [`assign_and_shard_param`](../../../models/jax/utils/weight_utils.md#assign_and_shard_param), [`output_sizes`](../../common/quantization/configs.md#QuantLinearConfig.output_sizes), [`prefix`](../linear.md#JaxEinsum.prefix), [`n_shards`](../../common/quantization/configs.md#QuantLinearConfig.n_shards), `linear_config`
- used by: [`get_quant_method`](unquantized.md#UnquantizedConfig.get_quant_method), [`create_weights_jax`](__init__.md#QuantizeMethodBase.create_weights_jax), [`UnquantizedLinearMethod`](unquantized.md#UnquantizedLinearMethod)

## Module values
- `logger` — [`L43`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/quantization/unquantized.py#L43)


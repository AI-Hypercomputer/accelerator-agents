---
title: 'Module: tpu_inference/layers/vllm/quantization/mxfp4.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/vllm/quantization/mxfp4.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.vllm.quantization.mxfp4`/
symbols:
  VllmMxfp4MoEMethod.process_mxfp4_moe_weights: VllmMxfp4MoEMethod#process_mxfp4_moe_weights().
  VllmMxfp4MoEMethod.apply_monolithic: VllmMxfp4MoEMethod#apply_monolithic().
  VllmMxfp4Config.get_quant_method: VllmMxfp4Config#get_quant_method().
  VllmMxfp4MoEMethod.extra_backend_kwargs: VllmMxfp4MoEMethod#extra_backend_kwargs.
  VllmMxfp4MoEMethod.process_weights_after_loading: VllmMxfp4MoEMethod#process_weights_after_loading().
  logger: logger.
  VllmMxfp4MoEMethod.moe_backend: VllmMxfp4MoEMethod#moe_backend.
  VllmMxfp4Config: VllmMxfp4Config#
  VllmMxfp4MoEMethod: VllmMxfp4MoEMethod#
  VllmMxfp4Config.get_name: VllmMxfp4Config#get_name().
  VllmMxfp4MoEMethod.mesh: VllmMxfp4MoEMethod#mesh.
  REQUANTIZED_BLOCK_SIZE: REQUANTIZED_BLOCK_SIZE.
  P: P.
  VllmMxfp4MoEMethod.__init__: VllmMxfp4MoEMethod#__init__().
  VllmMxfp4MoEMethod.mxfp4_backend: VllmMxfp4MoEMethod#mxfp4_backend.
  VllmMxfp4MoEMethod.get_fused_moe_quant_config: VllmMxfp4MoEMethod#get_fused_moe_quant_config().
  VllmMxfp4MoEMethod.is_monolithic: VllmMxfp4MoEMethod#is_monolithic().
---
# Module: [`tpu_inference/layers/vllm/quantization/mxfp4.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/mxfp4.py)

## Classes
### `VllmMxfp4Config`  ·  implements/extends VllmQuantConfig
- def: [`tpu_inference/layers/vllm/quantization/mxfp4.py:65`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/mxfp4.py#L65)
- signature: `class VllmMxfp4Config(Mxfp4Config, VllmQuantConfig):`
- members:
  - `get_name(cls)` — [`L68`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/mxfp4.py#L68)
  - `get_quant_method(self, layer: torch.nn.Module, prefix: str)` — [`L71`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/mxfp4.py#L71)
- uses (calls/refs, reference-scoped): [`VllmQuantConfig`](configs.md#VllmQuantConfig), [`VllmUnquantizedLinearMethod`](unquantized.md#VllmUnquantizedLinearMethod), [`mesh`](configs.md#VllmQuantConfig.mesh), [`get_linear_config`](configs.md#VllmQuantConfig.get_linear_config), [`get_moe_config`](configs.md#VllmQuantConfig.get_moe_config), [`MXFP4`](../../common/quant_methods.md#MXFP4), [`logger`](mxfp4.md#logger), [`VllmMxfp4MoEMethod`](mxfp4.md#VllmMxfp4MoEMethod)
- used by: [`VllmQuantConfig`](configs.md#VllmQuantConfig), [`get_tpu_quantization_config`](__init__.md#get_tpu_quantization_config)

### `VllmMxfp4MoEMethod`
- def: [`tpu_inference/layers/vllm/quantization/mxfp4.py:95`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/mxfp4.py#L95)
- members:
  - `apply_monolithic(self, layer: RoutedExperts, x: torch.Tensor, router_logits: torch.Tensor, input_ids: torch.Tensor | None = None)` — [`L204`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/mxfp4.py#L204)
  - `get_fused_moe_quant_config(self, layer: torch.nn.Module)` — [`L118`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/mxfp4.py#L118)
  - `is_monolithic(self)` — [`L128`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/mxfp4.py#L128)
  - `process_mxfp4_moe_weights(w13_weight: jax.Array, w13_weight_scale: jax.Array, w13_bias: jax.Array | None, w2_weight: jax.Array, w2_weight_scale: jax.Array, w2_bias: jax.Array | None)` — [`L144`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/mxfp4.py#L144)
  - `process_weights_after_loading(self, layer: torch.nn.Module)` — [`L131`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/mxfp4.py#L131)
  - `extra_backend_kwargs` — [`L112`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/mxfp4.py#L112)
  - `mesh` — [`L109`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/mxfp4.py#L109)
  - `moe_backend` — [`L110`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/mxfp4.py#L110)
  - `mxfp4_backend` — [`L107`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/mxfp4.py#L107)
- protocol/private: `__init__`[`L97`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/mxfp4.py#L97)
- uses (calls/refs, reference-scoped): [`ShardingAxisName`](../../common/sharding.md#ShardingAxisName), [`MoEBackend`](../../common/moe.md#MoEBackend), [`FusedMoEWeights`](../../common/process_weights/moe_weights.md#FusedMoEWeights), [`w13_weight_scale`](../../common/process_weights/moe_weights.md#FusedMoEWeights.w13_weight_scale), [`process_moe_weights`](../../common/process_weights/moe_weights.md#process_moe_weights), [`w2_weight_scale`](../../common/process_weights/moe_weights.md#FusedMoEWeights.w2_weight_scale), [`w13_weight`](../../common/process_weights/moe_weights.md#FusedMoEWeights.w13_weight), [`shard_moe_weights`](../../common/process_weights/moe_weights.md#shard_moe_weights), [`w13_bias`](../../common/process_weights/moe_weights.md#FusedMoEWeights.w13_bias), [`w2_weight`](../../common/process_weights/moe_weights.md#FusedMoEWeights.w2_weight), [`w2_bias`](../../common/process_weights/moe_weights.md#FusedMoEWeights.w2_bias), [`vllm_moe_apply`](../interface/moe.md#vllm_moe_apply), [`select_moe_backend_from_fused_moe_config`](../interface/moe.md#select_moe_backend_from_fused_moe_config), [`quantize_moe_weights`](../../common/process_weights/moe_weights.md#quantize_moe_weights), [`FUSED_MOE`](../../common/moe.md#MoEBackend.FUSED_MOE), [`dequantize_tensor_from_mxfp4_packed`](../../common/__init__.md#dequantize_tensor_from_mxfp4_packed), [`REQUANTIZED_BLOCK_SIZE`](mxfp4.md#REQUANTIZED_BLOCK_SIZE)
- used by: [`get_quant_method`](deepseek_v4_fp8.md#VllmDeepseekV4Fp8Config.get_quant_method), [`get_quant_method`](mxfp4.md#VllmMxfp4Config.get_quant_method)

## Module values
- `P` — [`L59`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/mxfp4.py#L59)
- `REQUANTIZED_BLOCK_SIZE` — [`L57`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/mxfp4.py#L57)
- `logger` — [`L61`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/mxfp4.py#L61)


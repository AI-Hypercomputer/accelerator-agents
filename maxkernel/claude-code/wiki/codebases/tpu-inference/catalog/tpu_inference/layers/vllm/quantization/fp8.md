---
title: 'Module: tpu_inference/layers/vllm/quantization/fp8.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/vllm/quantization/fp8.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.vllm.quantization.fp8`/
symbols:
  VllmFp8LinearMethod.process_weights_after_loading: VllmFp8LinearMethod#process_weights_after_loading().
  VllmFp8MoEMethod.process_weights_after_loading: VllmFp8MoEMethod#process_weights_after_loading().
  VllmFp8LinearMethod.linear_config: VllmFp8LinearMethod#linear_config.
  VllmFp8MoEMethod.apply_monolithic: VllmFp8MoEMethod#apply_monolithic().
  VllmFp8Config.get_quant_method: VllmFp8Config#get_quant_method().
  VllmFp8LinearMethod.apply: VllmFp8LinearMethod#apply().
  VllmFp8MoEMethod.extra_backend_kwargs: VllmFp8MoEMethod#extra_backend_kwargs.
  logger: logger.
  VllmFp8MoEMethod.weight_scale_name: VllmFp8MoEMethod#weight_scale_name.
  VllmFp8Config: VllmFp8Config#
  VllmFp8MoEMethod.weight_block_size: VllmFp8MoEMethod#weight_block_size.
  VllmFp8MoEMethod.moe_backend: VllmFp8MoEMethod#moe_backend.
  VllmFp8LinearMethod: VllmFp8LinearMethod#
  VllmFp8LinearMethod.__init__: VllmFp8LinearMethod#__init__().
  VllmFp8MoEMethod.block_quant: VllmFp8MoEMethod#block_quant.
  VllmFp8MoEMethod: VllmFp8MoEMethod#
  VllmFp8Config.get_name: VllmFp8Config#get_name().
  VllmFp8LinearMethod.create_weights: VllmFp8LinearMethod#create_weights().
  VllmFp8MoEMethod.mesh: VllmFp8MoEMethod#mesh.
  VllmFp8LinearMethod.use_marlin: VllmFp8LinearMethod#use_marlin.
  VllmFp8MoEMethod.quant_config: VllmFp8MoEMethod#quant_config.
  P: P.
  VllmFp8MoEMethod.__init__: VllmFp8MoEMethod#__init__().
  VllmFp8MoEMethod.fp8_backend: VllmFp8MoEMethod#fp8_backend.
  VllmFp8MoEMethod.is_monolithic: VllmFp8MoEMethod#is_monolithic().
---
# Module: [`tpu_inference/layers/vllm/quantization/fp8.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/fp8.py)

## Classes
### `VllmFp8Config`  ·  implements/extends VllmQuantConfig
- def: [`tpu_inference/layers/vllm/quantization/fp8.py:58`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/fp8.py#L58)
- signature: `class VllmFp8Config(vllm_fp8.Fp8Config, VllmQuantConfig):`
- members:
  - `get_name(cls)` — [`L61`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/fp8.py#L61)
  - `get_quant_method(self, layer: torch.nn.Module, prefix: str)` — [`L64`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/fp8.py#L64)
- uses (calls/refs, reference-scoped): [`VllmQuantConfig`](configs.md#VllmQuantConfig), [`VllmUnquantizedLinearMethod`](unquantized.md#VllmUnquantizedLinearMethod), [`mesh`](configs.md#VllmQuantConfig.mesh), [`get_linear_config`](configs.md#VllmQuantConfig.get_linear_config), [`VllmUnquantizedFusedMoEMethod`](unquantized.md#VllmUnquantizedFusedMoEMethod), [`get_moe_config`](configs.md#VllmQuantConfig.get_moe_config), [`logger`](fp8.md#logger), [`FP8`](../../common/quant_methods.md#FP8), [`VllmFp8LinearMethod`](fp8.md#VllmFp8LinearMethod), [`VllmFp8MoEMethod`](fp8.md#VllmFp8MoEMethod)
- used by: [`VllmQuantConfig`](configs.md#VllmQuantConfig), [`get_tpu_quantization_config`](__init__.md#get_tpu_quantization_config), [`__init__`](fp8.md#VllmFp8LinearMethod.__init__)

### `VllmFp8LinearMethod`
- def: [`tpu_inference/layers/vllm/quantization/fp8.py:98`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/fp8.py#L98)
- members:
  - `apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor] = None)` — [`L242`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/fp8.py#L242)
  - `create_weights(self, layer: torch.nn.Module, input_size_per_partition: int, output_partition_sizes: list[int], input_size: int, output_size: int, params_dtype: torch.dtype, **extra_weight_attrs)` — [`L132`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/fp8.py#L132)
  - `process_weights_after_loading(self, layer: torch.nn.Module)` — [`L152`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/fp8.py#L152)
  - `linear_config` — [`L122`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/fp8.py#L122)
  - `use_marlin` — [`L120`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/fp8.py#L120)
- protocol/private: `__init__`[`L106`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/fp8.py#L106)
- uses (calls/refs, reference-scoped): [`process_linear_weights`](../../common/process_weights/linear_weights.md#process_linear_weights), [`shard_linear_weights`](../../common/process_weights/linear_weights.md#shard_linear_weights), [`process_blockwise_fp8_linear_weights`](../../common/quantization/fp8.md#process_blockwise_fp8_linear_weights), [`LinearWeights`](../../common/process_weights/linear_weights.md#LinearWeights), [`fuse_matmuls`](configs.md#VllmQuantLinearConfig.fuse_matmuls), [`to_parameter_list`](../../common/process_weights/linear_weights.md#to_parameter_list), [`weight_sharding`](configs.md#VllmQuantLinearConfig.weight_sharding), [`_apply_fused`](../../common/quantization/fp8.md#Fp8LinearMethod._apply_fused), [`VllmQuantLinearConfig`](configs.md#VllmQuantLinearConfig), [`weight_scale`](../../common/process_weights/linear_weights.md#LinearWeights.weight_scale), [`n_shards`](configs.md#VllmQuantLinearConfig.n_shards), [`mesh`](configs.md#VllmQuantLinearConfig.mesh), [`bias_sharding`](configs.md#VllmQuantLinearConfig.bias_sharding), [`bias`](../../common/process_weights/linear_weights.md#LinearWeights.bias), [`output_sizes`](configs.md#VllmQuantLinearConfig.output_sizes), [`weight`](../../common/process_weights/linear_weights.md#LinearWeights.weight), [`zero_point`](../../common/process_weights/linear_weights.md#LinearWeights.zero_point), [`enable_quantized_matmul_kernel`](../../common/quantization/configs.md#QuantLinearConfig.enable_quantized_matmul_kernel), [`Fp8LinearMethod`](../../common/quantization/fp8.md#Fp8LinearMethod), [`_apply_split`](../../common/quantization/fp8.md#Fp8LinearMethod._apply_split), [`requant_block_size`](../../common/quantization/configs.md#QuantLinearConfig.requant_block_size), [`logger`](fp8.md#logger), [`VllmFp8Config`](fp8.md#VllmFp8Config), [`requant_weight_dtype`](../../common/quantization/configs.md#QuantLinearConfig.requant_weight_dtype)
- used by: [`get_quant_method`](deepseek_v4_fp8.md#VllmDeepseekV4Fp8Config.get_quant_method), [`get_quant_method`](fp8.md#VllmFp8Config.get_quant_method)

### `VllmFp8MoEMethod`
- def: [`tpu_inference/layers/vllm/quantization/fp8.py:274`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/fp8.py#L274)
- members:
  - `apply_monolithic(self, layer: RoutedExperts, x: torch.Tensor, router_logits: torch.Tensor, input_ids: torch.Tensor | None = None)` — [`L352`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/fp8.py#L352)
  - `is_monolithic(self)` — [`L297`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/fp8.py#L297)
  - `process_weights_after_loading(self, layer: torch.nn.Module)` — [`L300`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/fp8.py#L300)
  - `block_quant` — [`L284`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/fp8.py#L284)
  - `extra_backend_kwargs` — [`L292`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/fp8.py#L292)
  - `fp8_backend` — [`L287`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/fp8.py#L287)
  - `mesh` — [`L289`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/fp8.py#L289)
  - `moe_backend` — [`L290`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/fp8.py#L290)
  - `quant_config` — [`L282`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/fp8.py#L282)
  - `weight_block_size` — [`L283`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/fp8.py#L283)
  - `weight_scale_name` — [`L285`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/fp8.py#L285)
- protocol/private: `__init__`[`L276`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/fp8.py#L276)
- uses (calls/refs, reference-scoped): [`MoEBackend`](../../common/moe.md#MoEBackend), [`FusedMoEWeights`](../../common/process_weights/moe_weights.md#FusedMoEWeights), [`w13_weight_scale`](../../common/process_weights/moe_weights.md#FusedMoEWeights.w13_weight_scale), [`w2_weight_scale`](../../common/process_weights/moe_weights.md#FusedMoEWeights.w2_weight_scale), [`w13_weight`](../../common/process_weights/moe_weights.md#FusedMoEWeights.w13_weight), [`shard_moe_weights`](../../common/process_weights/moe_weights.md#shard_moe_weights), [`w13_bias`](../../common/process_weights/moe_weights.md#FusedMoEWeights.w13_bias), [`w2_weight`](../../common/process_weights/moe_weights.md#FusedMoEWeights.w2_weight), [`w2_bias`](../../common/process_weights/moe_weights.md#FusedMoEWeights.w2_bias), [`vllm_moe_apply`](../interface/moe.md#vllm_moe_apply), [`select_moe_backend_from_fused_moe_config`](../interface/moe.md#select_moe_backend_from_fused_moe_config), [`process_quantized_moe_weights`](../../common/process_weights/moe_weights.md#process_quantized_moe_weights), [`FUSED_MOE`](../../common/moe.md#MoEBackend.FUSED_MOE)
- used by: [`get_quant_method`](deepseek_v4_fp8.md#VllmDeepseekV4Fp8Config.get_quant_method), [`get_quant_method`](fp8.md#VllmFp8Config.get_quant_method)

## Module values
- `P` — [`L52`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/fp8.py#L52)
- `logger` — [`L54`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/fp8.py#L54)


---
title: 'Module: tpu_inference/layers/vllm/quantization/nvfp4.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/vllm/quantization/nvfp4.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.vllm.quantization.nvfp4`/
symbols:
  VllmNvfp4MoEMethod.process_weights_after_loading: VllmNvfp4MoEMethod#process_weights_after_loading().
  VllmNvfp4MoEMethod.unpack_and_process: VllmNvfp4MoEMethod#unpack_and_process().
  VllmNvfp4LinearMethod.process_weights_after_loading: VllmNvfp4LinearMethod#process_weights_after_loading().
  VllmNvfp4Config.get_quant_method: VllmNvfp4Config#get_quant_method().
  VllmNvfp4MoEMethod.apply_monolithic: VllmNvfp4MoEMethod#apply_monolithic().
  VllmNvfp4LinearMethod.__init__: VllmNvfp4LinearMethod#__init__().
  VllmNvfp4MoEMethod.extra_backend_kwargs: VllmNvfp4MoEMethod#extra_backend_kwargs.
  logger: logger.
  VllmNvfp4LinearMethod._apply_fused: VllmNvfp4LinearMethod#_apply_fused().
  VllmNvfp4Config: VllmNvfp4Config#
  VllmNvfp4LinearMethod.apply: VllmNvfp4LinearMethod#apply().
  VllmNvfp4MoEMethod.moe_backend: VllmNvfp4MoEMethod#moe_backend.
  VllmNvfp4LinearMethod._unpack_and_scale: VllmNvfp4LinearMethod#_unpack_and_scale().
  VllmNvfp4LinearMethod._apply_split: VllmNvfp4LinearMethod#_apply_split().
  VllmNvfp4LinearMethod: VllmNvfp4LinearMethod#
  VllmNvfp4Config.get_name: VllmNvfp4Config#get_name().
  VllmNvfp4Config.override_quantization_method: VllmNvfp4Config#override_quantization_method().
  VllmNvfp4LinearMethod.create_weights: VllmNvfp4LinearMethod#create_weights().
  VllmNvfp4MoEMethod.mesh: VllmNvfp4MoEMethod#mesh.
  VllmNvfp4LinearMethod.safe_expose_input_quant_key: VllmNvfp4LinearMethod#safe_expose_input_quant_key().
  VllmNvfp4LinearMethod.scalar_weight_loader: VllmNvfp4LinearMethod#scalar_weight_loader().
  VllmNvfp4MoEMethod: VllmNvfp4MoEMethod#
  VllmNvfp4LinearMethod.quant_config: VllmNvfp4LinearMethod#quant_config.
  VllmNvfp4LinearMethod.kernel: VllmNvfp4LinearMethod#kernel.
  VllmNvfp4MoEMethod.__init__: VllmNvfp4MoEMethod#__init__().
  VllmNvfp4MoEMethod.quant_config: VllmNvfp4MoEMethod#quant_config.
  VllmNvfp4MoEMethod.use_global_sf: VllmNvfp4MoEMethod#use_global_sf.
  VllmNvfp4MoEMethod.is_monolithic: VllmNvfp4MoEMethod#is_monolithic().
  VllmNvfp4MoEMethod.create_weights: VllmNvfp4MoEMethod#create_weights().
  VllmNvfp4MoEMethod.get_fused_moe_quant_config: VllmNvfp4MoEMethod#get_fused_moe_quant_config().
---
# Module: [`tpu_inference/layers/vllm/quantization/nvfp4.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/nvfp4.py)

## Classes
### `VllmNvfp4Config`  ·  implements/extends VllmQuantConfig
- def: [`tpu_inference/layers/vllm/quantization/nvfp4.py:75`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/nvfp4.py#L75)
- doc: NVFP4 config for TPU. Inherits config parsing from upstream.
- signature: `class VllmNvfp4Config(ModelOptNvFp4Config, VllmQuantConfig):`
- members:
  - `get_name(cls)` — [`L79`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/nvfp4.py#L79)
  - `get_quant_method(self, layer: torch.nn.Module, prefix: str)` — [`L96`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/nvfp4.py#L96)
  - `override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None)` — [`L83`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/nvfp4.py#L83)
- uses (calls/refs, reference-scoped): [`VllmQuantConfig`](configs.md#VllmQuantConfig), [`VllmUnquantizedLinearMethod`](unquantized.md#VllmUnquantizedLinearMethod), [`mesh`](configs.md#VllmQuantConfig.mesh), [`get_linear_config`](configs.md#VllmQuantConfig.get_linear_config), [`VllmUnquantizedFusedMoEMethod`](unquantized.md#VllmUnquantizedFusedMoEMethod), [`get_moe_config`](configs.md#VllmQuantConfig.get_moe_config), [`logger`](nvfp4.md#logger), [`NVFP4`](../../common/quant_methods.md#NVFP4), [`VllmNvfp4LinearMethod`](nvfp4.md#VllmNvfp4LinearMethod), [`VllmNvfp4MoEMethod`](nvfp4.md#VllmNvfp4MoEMethod)
- used by: [`VllmQuantConfig`](configs.md#VllmQuantConfig), [`get_tpu_quantization_config`](__init__.md#get_tpu_quantization_config), [`__init__`](nvfp4.md#VllmNvfp4LinearMethod.__init__)

### `VllmNvfp4LinearMethod`
- def: [`tpu_inference/layers/vllm/quantization/nvfp4.py:115`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/nvfp4.py#L115)
- members:
  - `apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor] = None)` — [`L229`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/nvfp4.py#L229)
  - `create_weights(self, layer, input_size_per_partition, output_partition_sizes, input_size, output_size, params_dtype, **extra_weight_attrs)` — [`L141`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/nvfp4.py#L141)
  - `process_weights_after_loading(self, layer)` — [`L157`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/nvfp4.py#L157)
  - `safe_expose_input_quant_key(layer, kernel)` — [`L130`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/nvfp4.py#L130)
  - `scalar_weight_loader(param, loaded_weight, *args, **kwargs)` — [`L148`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/nvfp4.py#L148)
  - `kernel` — [`L139`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/nvfp4.py#L139)
  - `quant_config` — [`L138`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/nvfp4.py#L138)
- protocol/private: `__init__`[`L123`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/nvfp4.py#L123), `_apply_fused`[`L257`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/nvfp4.py#L257), `_apply_split`[`L275`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/nvfp4.py#L275), `_unpack_and_scale`[`L183`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/nvfp4.py#L183)
- uses (calls/refs, reference-scoped): [`process_linear_weights`](../../common/process_weights/linear_weights.md#process_linear_weights), [`shard_linear_weights`](../../common/process_weights/linear_weights.md#shard_linear_weights), [`LinearWeights`](../../common/process_weights/linear_weights.md#LinearWeights), [`to_parameter_list`](../../common/process_weights/linear_weights.md#to_parameter_list), [`sharded_quantized_matmul`](../../common/linear.md#sharded_quantized_matmul), [`VllmUnquantizedLinearMethod`](unquantized.md#VllmUnquantizedLinearMethod), [`VllmQuantLinearConfig`](configs.md#VllmQuantLinearConfig), [`weight_scale`](../../common/process_weights/linear_weights.md#LinearWeights.weight_scale), [`bias`](../../common/process_weights/linear_weights.md#LinearWeights.bias), [`weight`](../../common/process_weights/linear_weights.md#LinearWeights.weight), [`slice_sharded_tensor_for_concatenation`](../../common/utils.md#slice_sharded_tensor_for_concatenation), [`zero_point`](../../common/process_weights/linear_weights.md#LinearWeights.zero_point), [`u8_unpack_e2m1`](../../common/__init__.md#u8_unpack_e2m1), [`VllmNvfp4Config`](nvfp4.md#VllmNvfp4Config), [`__init__`](unquantized.md#VllmUnquantizedLinearMethod.__init__)
- used by: [`get_quant_method`](nvfp4.md#VllmNvfp4Config.get_quant_method), [`process_weights_after_loading`](unquantized.md#VllmUnquantizedLinearMethod.process_weights_after_loading), [`apply`](unquantized.md#VllmUnquantizedLinearMethod.apply)

### `VllmNvfp4MoEMethod`
- def: [`tpu_inference/layers/vllm/quantization/nvfp4.py:296`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/nvfp4.py#L296)
- members:
  - `apply_monolithic(self, layer, x, router_logits, **kwargs)` — [`L470`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/nvfp4.py#L470)
  - `create_weights(self, layer, num_experts, hidden_size, intermediate_size_per_partition, params_dtype, **extra_weight_attrs)` — [`L324`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/nvfp4.py#L324)
  - `get_fused_moe_quant_config(self, layer)` — [`L364`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/nvfp4.py#L364)
  - `is_monolithic(self)` — [`L321`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/nvfp4.py#L321)
  - `process_weights_after_loading(self, layer)` — [`L367`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/nvfp4.py#L367)
  - `unpack_and_process(weights: FusedMoEWeights, w13_global_scale: jax.Array, w2_global_scale: jax.Array, desired_quant_dtype: jnp.dtype | None, requant_block_size: int | None)` — [`L405`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/nvfp4.py#L405)
  - `extra_backend_kwargs` — [`L316`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/nvfp4.py#L316)
  - `mesh` — [`L310`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/nvfp4.py#L310)
  - `moe_backend` — [`L311`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/nvfp4.py#L311)
  - `quant_config` — [`L309`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/nvfp4.py#L309)
  - `use_global_sf` — [`L315`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/nvfp4.py#L315)
- protocol/private: `__init__`[`L306`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/nvfp4.py#L306)
- uses (calls/refs, reference-scoped): [`MoEBackend`](../../common/moe.md#MoEBackend), [`FusedMoEWeights`](../../common/process_weights/moe_weights.md#FusedMoEWeights), [`w13_weight_scale`](../../common/process_weights/moe_weights.md#FusedMoEWeights.w13_weight_scale), [`w2_weight_scale`](../../common/process_weights/moe_weights.md#FusedMoEWeights.w2_weight_scale), [`w13_weight`](../../common/process_weights/moe_weights.md#FusedMoEWeights.w13_weight), [`w13_bias`](../../common/process_weights/moe_weights.md#FusedMoEWeights.w13_bias), [`w2_weight`](../../common/process_weights/moe_weights.md#FusedMoEWeights.w2_weight), [`w2_bias`](../../common/process_weights/moe_weights.md#FusedMoEWeights.w2_bias), [`vllm_moe_apply`](../interface/moe.md#vllm_moe_apply), [`select_moe_backend_from_fused_moe_config`](../interface/moe.md#select_moe_backend_from_fused_moe_config), [`process_quantized_moe_weights`](../../common/process_weights/moe_weights.md#process_quantized_moe_weights), [`FUSED_MOE`](../../common/moe.md#MoEBackend.FUSED_MOE), [`shard_moe_weights_to_tpu`](../../common/process_weights/moe_weights.md#shard_moe_weights_to_tpu), [`u8_unpack_e2m1`](../../common/__init__.md#u8_unpack_e2m1), [`MOE_REQUANTIZE_WEIGHT_DTYPE`](../../../envs.md#MOE_REQUANTIZE_WEIGHT_DTYPE), [`MOE_REQUANTIZE_BLOCK_SIZE`](../../../envs.md#MOE_REQUANTIZE_BLOCK_SIZE)
- used by: [`get_quant_method`](nvfp4.md#VllmNvfp4Config.get_quant_method)

## Module values
- `logger` — [`L71`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/nvfp4.py#L71)


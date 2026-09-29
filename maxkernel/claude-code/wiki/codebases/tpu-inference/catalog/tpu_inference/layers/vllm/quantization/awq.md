---
title: 'Module: tpu_inference/layers/vllm/quantization/awq.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/vllm/quantization/awq.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.vllm.quantization.awq`/
symbols:
  VllmAWQMoEMethod.process_awq_moe_weights: VllmAWQMoEMethod#process_awq_moe_weights().
  VllmAWQLinearMethod.process_awq_linear_weights: VllmAWQLinearMethod#process_awq_linear_weights().
  VllmAWQLinearMethod.process_weights_after_loading: VllmAWQLinearMethod#process_weights_after_loading().
  VllmAWQMoEMethod.apply_monolithic: VllmAWQMoEMethod#apply_monolithic().
  VllmAWQConfig.get_quant_method: VllmAWQConfig#get_quant_method().
  VllmAWQLinearMethod._apply_fused: VllmAWQLinearMethod#_apply_fused().
  VllmAWQMoEMethod.extra_backend_kwargs: VllmAWQMoEMethod#extra_backend_kwargs.
  VllmAWQLinearMethod.linear_config: VllmAWQLinearMethod#linear_config.
  VllmAWQLinearMethod.apply: VllmAWQLinearMethod#apply().
  VllmAWQMoEMethod.process_weights_after_loading: VllmAWQMoEMethod#process_weights_after_loading().
  VllmAWQMoEMethod.quant_config: VllmAWQMoEMethod#quant_config.
  VllmAWQConfig: VllmAWQConfig#
  logger: logger.
  VllmAWQMoEMethod.moe_backend: VllmAWQMoEMethod#moe_backend.
  VllmAWQLinearMethod.__init__: VllmAWQLinearMethod#__init__().
  VllmAWQConfig.get_name: VllmAWQConfig#get_name().
  VllmAWQMoEMethod.__init__: VllmAWQMoEMethod#__init__().
  VllmAWQMoEMethod.create_weights: VllmAWQMoEMethod#create_weights().
  VllmAWQMoEMethod.mesh: VllmAWQMoEMethod#mesh.
  VllmAWQLinearMethod._apply_split: VllmAWQLinearMethod#_apply_split().
  VllmAWQLinearMethod: VllmAWQLinearMethod#
  VllmAWQMoEMethod: VllmAWQMoEMethod#
  P: P.
  VllmAWQConfig.get_supported_act_dtypes: VllmAWQConfig#get_supported_act_dtypes().
  VllmAWQMoEMethod.is_monolithic: VllmAWQMoEMethod#is_monolithic().
  VllmAWQMoEMethod.get_fused_moe_quant_config: VllmAWQMoEMethod#get_fused_moe_quant_config().
---
# Module: [`tpu_inference/layers/vllm/quantization/awq.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/awq.py)

## Classes
### `VllmAWQConfig`  ·  implements/extends VllmQuantConfig
- def: [`tpu_inference/layers/vllm/quantization/awq.py:66`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/awq.py#L66)
- signature: `class VllmAWQConfig(AWQConfig, VllmQuantConfig):`
- members:
  - `get_name(cls)` — [`L69`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/awq.py#L69)
  - `get_quant_method(self, layer: torch.nn.Module, prefix: str)` — [`L78`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/awq.py#L78)
  - `get_supported_act_dtypes(self)` — [`L72`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/awq.py#L72)
- uses (calls/refs, reference-scoped): [`VllmQuantConfig`](configs.md#VllmQuantConfig), [`VllmUnquantizedLinearMethod`](unquantized.md#VllmUnquantizedLinearMethod), [`mesh`](configs.md#VllmQuantConfig.mesh), [`get_linear_config`](configs.md#VllmQuantConfig.get_linear_config), [`get_moe_config`](configs.md#VllmQuantConfig.get_moe_config), [`AWQ`](../../common/quant_methods.md#AWQ), [`VllmAWQLinearMethod`](awq.md#VllmAWQLinearMethod), [`VllmAWQMoEMethod`](awq.md#VllmAWQMoEMethod)
- used by: [`VllmQuantConfig`](configs.md#VllmQuantConfig), [`get_tpu_quantization_config`](__init__.md#get_tpu_quantization_config), [`__init__`](awq.md#VllmAWQLinearMethod.__init__), [`__init__`](awq.md#VllmAWQMoEMethod.__init__)

### `VllmAWQLinearMethod`
- def: [`tpu_inference/layers/vllm/quantization/awq.py:94`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/awq.py#L94)
- members:
  - `apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor] = None)` — [`L173`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/awq.py#L173)
  - `process_awq_linear_weights(weight: jax.Array, weight_scale: jax.Array, zero_point: jax.Array, bias: jax.Array | None)` — [`L126`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/awq.py#L126)
  - `process_weights_after_loading(self, layer: torch.nn.Module)` — [`L105`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/awq.py#L105)
  - `linear_config` — [`L103`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/awq.py#L103)
- protocol/private: `__init__`[`L100`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/awq.py#L100), `_apply_fused`[`L186`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/awq.py#L186), `_apply_split`[`L211`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/awq.py#L211)
- uses (calls/refs, reference-scoped): [`process_linear_weights`](../../common/process_weights/linear_weights.md#process_linear_weights), [`shard_linear_weights`](../../common/process_weights/linear_weights.md#shard_linear_weights), [`LinearWeights`](../../common/process_weights/linear_weights.md#LinearWeights), [`fuse_matmuls`](configs.md#VllmQuantLinearConfig.fuse_matmuls), [`to_parameter_list`](../../common/process_weights/linear_weights.md#to_parameter_list), [`weight_sharding`](configs.md#VllmQuantLinearConfig.weight_sharding), [`VllmQuantLinearConfig`](configs.md#VllmQuantLinearConfig), [`weight_scale`](../../common/process_weights/linear_weights.md#LinearWeights.weight_scale), [`n_shards`](configs.md#VllmQuantLinearConfig.n_shards), [`mesh`](configs.md#VllmQuantLinearConfig.mesh), [`bias_sharding`](configs.md#VllmQuantLinearConfig.bias_sharding), [`bias`](../../common/process_weights/linear_weights.md#LinearWeights.bias), [`output_sizes`](configs.md#VllmQuantLinearConfig.output_sizes), [`weight`](../../common/process_weights/linear_weights.md#LinearWeights.weight), [`slice_sharded_tensor_for_concatenation`](../../common/utils.md#slice_sharded_tensor_for_concatenation), [`zero_point`](../../common/process_weights/linear_weights.md#LinearWeights.zero_point), [`awq_u32_unpack_u4`](../../common/__init__.md#awq_u32_unpack_u4), [`VllmAWQConfig`](awq.md#VllmAWQConfig), [`logger`](awq.md#logger)
- used by: [`get_quant_method`](awq.md#VllmAWQConfig.get_quant_method)

### `VllmAWQMoEMethod`
- def: [`tpu_inference/layers/vllm/quantization/awq.py:240`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/awq.py#L240)
- members:
  - `apply_monolithic(self, layer: RoutedExperts, x: torch.Tensor, router_logits: torch.Tensor, input_ids: torch.Tensor | None = None)` — [`L487`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/awq.py#L487)
  - `create_weights(self, layer: torch.nn.Module, num_experts: int, hidden_size: int, intermediate_size_per_partition: int, params_dtype: torch.dtype, **extra_weight_attrs)` — [`L264`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/awq.py#L264)
  - `get_fused_moe_quant_config(self, layer: torch.nn.Module)` — [`L261`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/awq.py#L261)
  - `is_monolithic(self)` — [`L258`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/awq.py#L258)
  - `process_awq_moe_weights(w13_qweight: jax.Array, w13_scales: jax.Array, w13_qzeros: jax.Array, w13_bias: jax.Array | None, w2_qweight: jax.Array, w2_scales: jax.Array, w2_qzeros: jax.Array, w2_bias: jax.Array | None)` — [`L412`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/awq.py#L412)
  - `process_weights_after_loading(self, layer: torch.nn.Module)` — [`L382`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/awq.py#L382)
  - `extra_backend_kwargs` — [`L253`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/awq.py#L253)
  - `mesh` — [`L251`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/awq.py#L251)
  - `moe_backend` — [`L252`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/awq.py#L252)
  - `quant_config` — [`L250`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/awq.py#L250)
- protocol/private: `__init__`[`L242`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/awq.py#L242)
- uses (calls/refs, reference-scoped): [`ShardingAxisName`](../../common/sharding.md#ShardingAxisName), [`MoEBackend`](../../common/moe.md#MoEBackend), [`FusedMoEWeights`](../../common/process_weights/moe_weights.md#FusedMoEWeights), [`w13_weight_scale`](../../common/process_weights/moe_weights.md#FusedMoEWeights.w13_weight_scale), [`process_moe_weights`](../../common/process_weights/moe_weights.md#process_moe_weights), [`w2_weight_scale`](../../common/process_weights/moe_weights.md#FusedMoEWeights.w2_weight_scale), [`w13_weight`](../../common/process_weights/moe_weights.md#FusedMoEWeights.w13_weight), [`shard_moe_weights`](../../common/process_weights/moe_weights.md#shard_moe_weights), [`w13_bias`](../../common/process_weights/moe_weights.md#FusedMoEWeights.w13_bias), [`w2_weight`](../../common/process_weights/moe_weights.md#FusedMoEWeights.w2_weight), [`w2_bias`](../../common/process_weights/moe_weights.md#FusedMoEWeights.w2_bias), [`vllm_moe_apply`](../interface/moe.md#vllm_moe_apply), [`select_moe_backend_from_fused_moe_config`](../interface/moe.md#select_moe_backend_from_fused_moe_config), [`FUSED_MOE`](../../common/moe.md#MoEBackend.FUSED_MOE), [`awq_u32_unpack_u4`](../../common/__init__.md#awq_u32_unpack_u4), [`VllmAWQConfig`](awq.md#VllmAWQConfig)
- used by: [`get_quant_method`](awq.md#VllmAWQConfig.get_quant_method)

## Module values
- `P` — [`L60`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/awq.py#L60)
- `logger` — [`L62`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/awq.py#L62)


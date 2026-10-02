---
title: 'Module: tpu_inference/layers/vllm/quantization/configs.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/vllm/quantization/configs.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.vllm.quantization.configs`/
symbols:
  VllmQuantConfig: VllmQuantConfig#
  VllmQuantLinearConfig.fuse_matmuls: VllmQuantLinearConfig#fuse_matmuls.
  VllmQuantLinearConfig.weight_sharding: VllmQuantLinearConfig#weight_sharding.
  VllmQuantLinearConfig: VllmQuantLinearConfig#
  VllmQuantLinearConfig.n_shards: VllmQuantLinearConfig#n_shards.
  VllmQuantLinearConfig.mesh: VllmQuantLinearConfig#mesh.
  VllmQuantConfig.mesh: VllmQuantConfig#mesh.
  VllmQuantLinearConfig.bias_sharding: VllmQuantLinearConfig#bias_sharding.
  VllmQuantConfig.get_linear_config: VllmQuantConfig#get_linear_config().
  VllmQuantLinearConfig.output_sizes: VllmQuantLinearConfig#output_sizes.
  VllmQuantLinearConfig.output_sharding: VllmQuantLinearConfig#output_sharding.
  VllmQuantConfig.get_moe_config: VllmQuantConfig#get_moe_config().
  VllmQuantLinearConfig.get_input_sharding: VllmQuantLinearConfig#get_input_sharding().
  VllmQuantLinearConfig.get_output_sharding: VllmQuantLinearConfig#get_output_sharding().
  VllmQuantLinearConfig.tp_size: VllmQuantLinearConfig#tp_size.
  P: P.
  VllmQuantLinearConfig.input_sharding: VllmQuantLinearConfig#input_sharding.
  VllmQuantConfig.set_configs: VllmQuantConfig#set_configs().
  VllmQuantLinearConfig.num_proj: VllmQuantLinearConfig#num_proj.
  VllmQuantConfig.vllm_config: VllmQuantConfig#vllm_config.
  VllmQuantLinearConfig.__init__: VllmQuantLinearConfig#__init__().
  logger: logger.
---
# Module: [`tpu_inference/layers/vllm/quantization/configs.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/configs.py)

## Classes
### `VllmQuantConfig`
- def: [`tpu_inference/layers/vllm/quantization/configs.py:111`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/configs.py#L111)
- signature: `class VllmQuantConfig:`
- members:
  - `get_linear_config(self, layer: LinearBase)` — [`L120`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/configs.py#L120)
  - `get_moe_config(self, layer: RoutedExperts)` — [`L124`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/configs.py#L124)
  - `set_configs(cls, vllm_config: VllmConfig, mesh: Mesh)` — [`L116`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/configs.py#L116)
  - `mesh` — [`L113`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/configs.py#L113)
  - `vllm_config` — [`L112`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/configs.py#L112)
- uses (calls/refs, reference-scoped): [`VllmQuantLinearConfig`](configs.md#VllmQuantLinearConfig), [`VllmUnquantizedConfig`](unquantized.md#VllmUnquantizedConfig), [`VllmAWQConfig`](awq.md#VllmAWQConfig), [`VllmCompressedTensorsW8A8Fp8MoEMethod`](compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w8a8_fp8.md#VllmCompressedTensorsW8A8Fp8MoEMethod), [`VllmFp8Config`](fp8.md#VllmFp8Config), [`VllmNvfp4Config`](nvfp4.md#VllmNvfp4Config), [`VllmCompressedTensorsConfig`](compressed_tensors/compressed_tensors.md#VllmCompressedTensorsConfig), [`VllmCompressedTensorsW4A8MoEMethod`](compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a8.md#VllmCompressedTensorsW4A8MoEMethod), [`VllmDeepseekV4Fp8Config`](deepseek_v4_fp8.md#VllmDeepseekV4Fp8Config), [`VllmMxfp4Config`](mxfp4.md#VllmMxfp4Config)
- used by: [`get_tpu_quantization_config`](__init__.md#get_tpu_quantization_config), [`unpack_and_process`](compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a8.md#VllmCompressedTensorsW4A8MoEMethod.unpack_and_process), [`process_weights_after_loading`](compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a8.md#VllmCompressedTensorsW4A8MoEMethod.process_weights_after_loading), [`process_weights_after_loading`](compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w8a8_fp8.md#VllmCompressedTensorsW8A8Fp8MoEMethod.process_weights_after_loading), [`get_quant_method`](deepseek_v4_fp8.md#VllmDeepseekV4Fp8Config.get_quant_method), [`get_quant_method`](fp8.md#VllmFp8Config.get_quant_method), [`get_quant_method`](nvfp4.md#VllmNvfp4Config.get_quant_method), [`get_quant_method`](unquantized.md#VllmUnquantizedConfig.get_quant_method), [`get_quant_method`](awq.md#VllmAWQConfig.get_quant_method), [`get_quant_method`](compressed_tensors/compressed_tensors.md#VllmCompressedTensorsConfig.get_quant_method), [`get_quant_method`](mxfp4.md#VllmMxfp4Config.get_quant_method), [`get_scheme`](compressed_tensors/compressed_tensors.md#VllmCompressedTensorsConfig.get_scheme), [`VllmUnquantizedConfig`](unquantized.md#VllmUnquantizedConfig), [`__init__`](../../../models/jax/qwen2.md#Qwen2MLP.__init__), [`__init__`](../../../models/jax/qwen2.md#Qwen2DecoderLayer.__init__), [`__init__`](../../../models/jax/qwen3.md#Qwen3DecoderLayer.__init__), [`VllmAWQConfig`](awq.md#VllmAWQConfig), [`VllmCompressedTensorsW8A8Fp8MoEMethod`](compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w8a8_fp8.md#VllmCompressedTensorsW8A8Fp8MoEMethod), [`VllmFp8Config`](fp8.md#VllmFp8Config), [`VllmNvfp4Config`](nvfp4.md#VllmNvfp4Config), [`VllmCompressedTensorsConfig`](compressed_tensors/compressed_tensors.md#VllmCompressedTensorsConfig), [`VllmCompressedTensorsW4A8MoEMethod`](compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a8.md#VllmCompressedTensorsW4A8MoEMethod), [`VllmDeepseekV4Fp8Config`](deepseek_v4_fp8.md#VllmDeepseekV4Fp8Config), [`VllmMxfp4Config`](mxfp4.md#VllmMxfp4Config), [`__init__`](compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a8.md#VllmCompressedTensorsW4A8MoEMethod.__init__), [`__init__`](compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w8a8_fp8.md#VllmCompressedTensorsW8A8Fp8MoEMethod.__init__), [`__init__`](../../../models/jax/gemma4_mm.md#Gemma4MultimodalEmbedder.__init__), [`__init__`](../../../models/jax/gemma4_mm.md#Gemma4VisionEncoderLayer.__init__), [`__init__`](../../../models/jax/gemma4_mm.md#Gemma4VisionFlashAttention.__init__), [`__init__`](../../../models/jax/gemma4_mm.md#Gemma4VisionMLP.__init__), [`__init__`](../../../models/jax/gemma4_mm.md#Gemma4VisionModel.__init__), [`__init__`](../../../models/jax/gemma4_mtp.md#Gemma4MTPAttention.__init__), [`__init__`](../../../models/jax/gemma4_mtp.md#Gemma4MTPDecoderLayer.__init__), [`__init__`](../../../models/jax/gemma4.md#Gemma4Attention.__init__), [`__init__`](../../../models/jax/gemma4.md#Gemma4DecoderLayer.__init__), [`__init__`](../../../models/jax/gemma4.md#Gemma4MLP.__init__), [`__init__`](../../../models/jax/qwen2.md#Qwen2Attention.__init__), [`__init__`](../../../models/jax/qwen3.md#Qwen3Attention.__init__)

### `VllmQuantLinearConfig`  ·  implements/extends QuantLinearConfig
- def: [`tpu_inference/layers/vllm/quantization/configs.py:41`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/configs.py#L41)
- signature: `class VllmQuantLinearConfig(QuantLinearConfig):`
- members:
  - `get_input_sharding(self, x: torchax.tensor.Tensor)` — [`L93`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/configs.py#L93)
  - `get_output_sharding(self, x: torchax.tensor.Tensor)` — [`L102`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/configs.py#L102)
  - `bias_sharding` — [`L89`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/configs.py#L89)
  - `fuse_matmuls` — [`L70`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/configs.py#L70)
  - `input_sharding` — [`L64`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/configs.py#L64)
  - `mesh` — [`L49`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/configs.py#L49)
  - `n_shards` — [`L53`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/configs.py#L53)
  - `num_proj` — [`L83`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/configs.py#L83)
  - `output_sharding` — [`L59`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/configs.py#L59)
  - `output_sizes` — [`L68`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/configs.py#L68)
  - `tp_size` — [`L50`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/configs.py#L50)
  - `weight_sharding` — [`L57`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/configs.py#L57)
- protocol/private: `__init__`[`L43`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/configs.py#L43)
- uses (calls/refs, reference-scoped): [`ShardingAxisName`](../../common/sharding.md#ShardingAxisName), [`QuantLinearConfig`](../../common/quantization/configs.md#QuantLinearConfig), [`P`](configs.md#P), [`__init__`](../../common/quantization/configs.md#QuantLinearConfig.__init__), [`get_model_matmul_fusion_assignment`](../../common/process_weights/linear_weights.md#get_model_matmul_fusion_assignment), [`enable_sp`](../../common/quantization/configs.md#QuantLinearConfig.enable_sp), [`logger`](configs.md#logger)
- used by: [`process_weights_after_loading`](fp8.md#VllmFp8LinearMethod.process_weights_after_loading), [`process_weights_after_loading`](compressed_tensors/schemes/compressed_tensors_w8a8_fp8.md#VllmCompressedTensorsW8A8Fp8.process_weights_after_loading), [`process_fp8_linear_weights`](compressed_tensors/schemes/compressed_tensors_w8a8_fp8.md#VllmCompressedTensorsW8A8Fp8.process_fp8_linear_weights), [`process_awq_linear_weights`](awq.md#VllmAWQLinearMethod.process_awq_linear_weights), [`process_int8_linear_weights`](compressed_tensors/schemes/compressed_tensors_w8a8_int8.md#VllmCompressedTensorsW8A8Int8.process_int8_linear_weights), [`process_uint4_linear_weights`](compressed_tensors/schemes/compressed_tensors_w4a8_fp8.md#VllmCompressedTensorsW4A8Fp8.process_uint4_linear_weights), [`process_weights_after_loading`](compressed_tensors/schemes/compressed_tensors_w4a8_fp8.md#VllmCompressedTensorsW4A8Fp8.process_weights_after_loading), [`process_weights_after_loading`](compressed_tensors/schemes/compressed_tensors_w8a8_int8.md#VllmCompressedTensorsW8A8Int8.process_weights_after_loading), [`process_weights_after_loading`](awq.md#VllmAWQLinearMethod.process_weights_after_loading), [`_apply_fused`](compressed_tensors/schemes/compressed_tensors_w4a8_fp8.md#VllmCompressedTensorsW4A8Fp8._apply_fused), [`_apply_fused`](compressed_tensors/schemes/compressed_tensors_w8a8_fp8.md#VllmCompressedTensorsW8A8Fp8._apply_fused), [`_apply_fused`](compressed_tensors/schemes/compressed_tensors_w8a8_int8.md#VllmCompressedTensorsW8A8Int8._apply_fused), [`get_linear_config`](configs.md#VllmQuantConfig.get_linear_config), [`QuantLinearConfig`](../../common/quantization/configs.md#QuantLinearConfig), [`__init__`](nvfp4.md#VllmNvfp4LinearMethod.__init__), [`apply`](fp8.md#VllmFp8LinearMethod.apply), [`_apply_fused`](awq.md#VllmAWQLinearMethod._apply_fused), [`_apply_split`](compressed_tensors/schemes/compressed_tensors_w4a8_fp8.md#VllmCompressedTensorsW4A8Fp8._apply_split), [`_apply_split`](compressed_tensors/schemes/compressed_tensors_w8a8_fp8.md#VllmCompressedTensorsW8A8Fp8._apply_split), [`_apply_split`](compressed_tensors/schemes/compressed_tensors_w8a8_int8.md#VllmCompressedTensorsW8A8Int8._apply_split), [`apply`](awq.md#VllmAWQLinearMethod.apply), [`apply_weights`](compressed_tensors/schemes/compressed_tensors_w4a8_fp8.md#VllmCompressedTensorsW4A8Fp8.apply_weights), [`apply_weights`](compressed_tensors/schemes/compressed_tensors_w8a8_fp8.md#VllmCompressedTensorsW8A8Fp8.apply_weights), [`apply_weights`](compressed_tensors/schemes/compressed_tensors_w8a8_int8.md#VllmCompressedTensorsW8A8Int8.apply_weights), [`__init__`](awq.md#VllmAWQLinearMethod.__init__), [`__init__`](compressed_tensors/schemes/compressed_tensors_w8a8_fp8.md#VllmCompressedTensorsW8A8Fp8.__init__), [`__init__`](fp8.md#VllmFp8LinearMethod.__init__), [`__init__`](unquantized.md#VllmUnquantizedLinearMethod.__init__), [`__init__`](compressed_tensors/schemes/compressed_tensors_w4a8_fp8.md#VllmCompressedTensorsW4A8Fp8.__init__), [`__init__`](compressed_tensors/schemes/compressed_tensors_w8a8_int8.md#VllmCompressedTensorsW8A8Int8.__init__)

## Module values
- `P` — [`L36`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/configs.py#L36)
- `logger` — [`L38`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/configs.py#L38)


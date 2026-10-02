---
title: 'Module: tpu_inference/layers/common/quantization/configs.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/common/quantization/configs.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.common.quantization.configs`/QuantLinearConfig#
symbols:
  QuantLinearConfig: ''
  QuantLinearConfig.enable_quantized_matmul_kernel: enable_quantized_matmul_kernel.
  QuantLinearConfig.requant_block_size: requant_block_size.
  QuantLinearConfig.output_sizes: output_sizes.
  QuantLinearConfig.n_shards: n_shards.
  QuantLinearConfig.weight_sharding: weight_sharding.
  QuantLinearConfig.__init__: __init__().
  QuantLinearConfig.fuse_matmuls: fuse_matmuls.
  QuantLinearConfig.enable_sp: enable_sp.
  QuantLinearConfig.bias_sharding: bias_sharding.
  QuantLinearConfig.mesh: mesh.
  QuantLinearConfig.requant_weight_dtype: requant_weight_dtype.
  QuantLinearConfig.input_sharding: input_sharding.
  QuantLinearConfig.output_sharding: output_sharding.
---
# Module: [`tpu_inference/layers/common/quantization/configs.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/quantization/configs.py)

## Classes
### `QuantLinearConfig`
- def: [`tpu_inference/layers/common/quantization/configs.py:22`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/quantization/configs.py#L22)
- signature: `class QuantLinearConfig:`
- members:
  - `bias_sharding` — [`L39`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/quantization/configs.py#L39)
  - `enable_quantized_matmul_kernel` — [`L45`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/quantization/configs.py#L45)
  - `enable_sp` — [`L34`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/quantization/configs.py#L34)
  - `fuse_matmuls` — [`L33`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/quantization/configs.py#L33)
  - `input_sharding` — [`L35`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/quantization/configs.py#L35)
  - `mesh` — [`L37`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/quantization/configs.py#L37)
  - `n_shards` — [`L43`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/quantization/configs.py#L43)
  - `output_sharding` — [`L36`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/quantization/configs.py#L36)
  - `output_sizes` — [`L30`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/quantization/configs.py#L30)
  - `requant_block_size` — [`L46`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/quantization/configs.py#L46)
  - `requant_weight_dtype` — [`L47`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/quantization/configs.py#L47)
  - `weight_sharding` — [`L31`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/quantization/configs.py#L31)
- protocol/private: `__init__`[`L24`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/quantization/configs.py#L24)
- uses (calls/refs, reference-scoped): [`VllmQuantLinearConfig`](../../vllm/quantization/configs.md#VllmQuantLinearConfig), [`QuantLinearConfig`](../../jax/quantization/configs.md#QuantLinearConfig), [`ENABLE_QUANTIZED_MATMUL_KERNEL`](../../../envs.md#ENABLE_QUANTIZED_MATMUL_KERNEL), [`REQUANTIZE_BLOCK_SIZE`](../../../envs.md#REQUANTIZE_BLOCK_SIZE), [`REQUANTIZE_WEIGHT_DTYPE`](../../../envs.md#REQUANTIZE_WEIGHT_DTYPE)
- used by: [`process_weights_after_loading`](../../jax/quantization/fp8.md#Fp8BlockwiseLinearMethod.process_weights_after_loading), [`process_weights_after_loading`](../../vllm/quantization/fp8.md#VllmFp8LinearMethod.process_weights_after_loading), [`create_weights_jax`](../../jax/quantization/fp8.md#Fp8BlockwiseLinearMethod.create_weights_jax), [`process_weights_after_loading`](../../vllm/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.md#VllmCompressedTensorsW8A8Fp8.process_weights_after_loading), [`get_quant_method`](../../jax/quantization/unquantized.md#UnquantizedConfig.get_quant_method), [`process_fp8_linear_weights`](../../vllm/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.md#VllmCompressedTensorsW8A8Fp8.process_fp8_linear_weights), [`linear_config`](../../vllm/quantization/fp8.md#VllmFp8LinearMethod.linear_config), [`process_int8_linear_weights`](../../vllm/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.md#VllmCompressedTensorsW8A8Int8.process_int8_linear_weights), [`process_uint4_linear_weights`](../../vllm/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8.md#VllmCompressedTensorsW4A8Fp8.process_uint4_linear_weights), [`apply_jax`](../../jax/quantization/fp8.md#Fp8BlockwiseLinearMethod.apply_jax), [`create_weights_jax`](../../jax/quantization/fp8.md#Fp8TensorwiseLinearMethod.create_weights_jax), [`weight_sharding`](../../vllm/quantization/configs.md#VllmQuantLinearConfig.weight_sharding), [`apply_jax`](../../jax/quantization/fp8.md#Fp8TensorwiseLinearMethod.apply_jax), [`apply_jax`](../../jax/quantization/unquantized.md#UnquantizedLinearMethod.apply_jax), [`_apply_fused`](fp8.md#Fp8LinearMethod._apply_fused), [`create_weights_jax`](../../jax/quantization/unquantized.md#UnquantizedMergedLinearMethod.create_weights_jax), [`VllmQuantLinearConfig`](../../vllm/quantization/configs.md#VllmQuantLinearConfig), [`output_sharding`](../../vllm/quantization/configs.md#VllmQuantLinearConfig.output_sharding), [`batch_sharding`](../../jax/quantization/configs.md#QuantLinearConfig.batch_sharding), [`_apply_split`](fp8.md#Fp8LinearMethod._apply_split), [`get_input_sharding`](../../vllm/quantization/configs.md#VllmQuantLinearConfig.get_input_sharding), [`get_output_sharding`](../../vllm/quantization/configs.md#VllmQuantLinearConfig.get_output_sharding), [`QuantLinearConfig`](../../jax/quantization/configs.md#QuantLinearConfig), [`__init__`](fp8.md#Fp8LinearMethod.__init__), [`__init__`](../../vllm/quantization/configs.md#VllmQuantLinearConfig.__init__)


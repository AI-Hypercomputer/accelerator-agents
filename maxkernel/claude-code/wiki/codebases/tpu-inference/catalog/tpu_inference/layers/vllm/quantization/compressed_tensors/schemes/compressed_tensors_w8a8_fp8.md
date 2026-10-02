---
title: 'Module: tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.vllm.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8`/
symbols:
  VllmCompressedTensorsW8A8Fp8.process_weights_after_loading: VllmCompressedTensorsW8A8Fp8#process_weights_after_loading().
  VllmCompressedTensorsW8A8Fp8.process_fp8_linear_weights: VllmCompressedTensorsW8A8Fp8#process_fp8_linear_weights().
  VllmCompressedTensorsW8A8Fp8.linear_config: VllmCompressedTensorsW8A8Fp8#linear_config.
  VllmCompressedTensorsW8A8Fp8._apply_fused: VllmCompressedTensorsW8A8Fp8#_apply_fused().
  VllmCompressedTensorsW8A8Fp8._apply_split: VllmCompressedTensorsW8A8Fp8#_apply_split().
  VllmCompressedTensorsW8A8Fp8.apply_weights: VllmCompressedTensorsW8A8Fp8#apply_weights().
  VllmCompressedTensorsW8A8Fp8.__init__: VllmCompressedTensorsW8A8Fp8#__init__().
  logger: logger.
  VllmCompressedTensorsW8A8Fp8.create_weights: VllmCompressedTensorsW8A8Fp8#create_weights().
  VllmCompressedTensorsW8A8Fp8: VllmCompressedTensorsW8A8Fp8#
  P: P.
  VllmCompressedTensorsW8A8Fp8.safe_expose_input_quant_key: VllmCompressedTensorsW8A8Fp8#safe_expose_input_quant_key().
  VllmCompressedTensorsW8A8Fp8.use_marlin: VllmCompressedTensorsW8A8Fp8#use_marlin.
---
# Module: [`tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py)

## Classes
### `VllmCompressedTensorsW8A8Fp8`
- def: [`tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py:48`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py#L48)
- signature: `class VllmCompressedTensorsW8A8Fp8(CompressedTensorsW8A8Fp8):`
- members:
  - `apply_weights(self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor])` — [`L219`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py#L219)
  - `create_weights(self, layer: torch.nn.Module, input_size_per_partition: int, output_partition_sizes: list[int], input_size: int, output_size: int, params_dtype: torch.dtype, **extra_weight_attrs)` — [`L84`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py#L84)
  - `process_fp8_linear_weights(weight: jax.Array, weight_scale: jax.Array, bias: jax.Array | None)` — [`L123`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py#L123)
  - `process_weights_after_loading(self, layer: torch.nn.Module)` — [`L104`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py#L104)
  - `safe_expose_input_quant_key(layer, kernel)` — [`L69`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py#L69)
  - `linear_config` — [`L82`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py#L82)
  - `use_marlin` — [`L80`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py#L80)
- protocol/private: `__init__`[`L50`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py#L50), `_apply_fused`[`L227`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py#L227), `_apply_split`[`L263`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py#L263)
- uses (calls/refs, reference-scoped): [`process_linear_weights`](../../../../common/process_weights/linear_weights.md#process_linear_weights), [`shard_linear_weights`](../../../../common/process_weights/linear_weights.md#shard_linear_weights), [`process_blockwise_fp8_linear_weights`](../../../../common/quantization/fp8.md#process_blockwise_fp8_linear_weights), [`LinearWeights`](../../../../common/process_weights/linear_weights.md#LinearWeights), [`fuse_matmuls`](../../configs.md#VllmQuantLinearConfig.fuse_matmuls), [`to_parameter_list`](../../../../common/process_weights/linear_weights.md#to_parameter_list), [`weight_sharding`](../../configs.md#VllmQuantLinearConfig.weight_sharding), [`sharded_quantized_matmul`](../../../../common/linear.md#sharded_quantized_matmul), [`VllmQuantLinearConfig`](../../configs.md#VllmQuantLinearConfig), [`weight_scale`](../../../../common/process_weights/linear_weights.md#LinearWeights.weight_scale), [`n_shards`](../../configs.md#VllmQuantLinearConfig.n_shards), [`mesh`](../../configs.md#VllmQuantLinearConfig.mesh), [`bias_sharding`](../../configs.md#VllmQuantLinearConfig.bias_sharding), [`quantize_tensor`](../../../../common/__init__.md#quantize_tensor), [`bias`](../../../../common/process_weights/linear_weights.md#LinearWeights.bias), [`dequantize_tensor`](../../../../common/__init__.md#dequantize_tensor), [`output_sizes`](../../configs.md#VllmQuantLinearConfig.output_sizes), [`weight`](../../../../common/process_weights/linear_weights.md#LinearWeights.weight), [`slice_sharded_tensor_for_concatenation`](../../../../common/utils.md#slice_sharded_tensor_for_concatenation), [`zero_point`](../../../../common/process_weights/linear_weights.md#LinearWeights.zero_point), [`enable_quantized_matmul_kernel`](../../../../common/quantization/configs.md#QuantLinearConfig.enable_quantized_matmul_kernel), [`requant_block_size`](../../../../common/quantization/configs.md#QuantLinearConfig.requant_block_size), [`logger`](compressed_tensors_w8a8_fp8.md#logger), [`requant_weight_dtype`](../../../../common/quantization/configs.md#QuantLinearConfig.requant_weight_dtype), [`P`](compressed_tensors_w8a8_fp8.md#P)
- used by: [`get_scheme`](../compressed_tensors.md#VllmCompressedTensorsConfig.get_scheme)

## Module values
- `P` — [`L43`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py#L43)
- `logger` — [`L45`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py#L45)


---
title: 'Module: tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.vllm.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8`/
symbols:
  VllmCompressedTensorsW8A8Int8.process_int8_linear_weights: VllmCompressedTensorsW8A8Int8#process_int8_linear_weights().
  VllmCompressedTensorsW8A8Int8.process_weights_after_loading: VllmCompressedTensorsW8A8Int8#process_weights_after_loading().
  VllmCompressedTensorsW8A8Int8._apply_fused: VllmCompressedTensorsW8A8Int8#_apply_fused().
  VllmCompressedTensorsW8A8Int8.linear_config: VllmCompressedTensorsW8A8Int8#linear_config.
  VllmCompressedTensorsW8A8Int8._apply_split: VllmCompressedTensorsW8A8Int8#_apply_split().
  VllmCompressedTensorsW8A8Int8.apply_weights: VllmCompressedTensorsW8A8Int8#apply_weights().
  logger: logger.
  VllmCompressedTensorsW8A8Int8.__init__: VllmCompressedTensorsW8A8Int8#__init__().
  VllmCompressedTensorsW8A8Int8: VllmCompressedTensorsW8A8Int8#
  VllmCompressedTensorsW8A8Int8.is_channelwise: VllmCompressedTensorsW8A8Int8#is_channelwise.
  P: P.
  VllmCompressedTensorsW8A8Int8.create_weights: VllmCompressedTensorsW8A8Int8#create_weights().
---
# Module: [`tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py)

## Classes
### `VllmCompressedTensorsW8A8Int8`
- def: [`tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py:49`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py#L49)
- signature: `class VllmCompressedTensorsW8A8Int8(CompressedTensorsW8A8Int8):`
- members:
  - `apply_weights(self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor])` — [`L182`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py#L182)
  - `create_weights(self, layer: torch.nn.Module, output_partition_sizes: list[int], input_size_per_partition: int, params_dtype: torch.dtype, weight_loader: Callable, **kwargs)` — [`L58`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py#L58)
  - `process_int8_linear_weights(weight: jax.Array, weight_scale: jax.Array, bias: jax.Array | None)` — [`L137`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py#L137)
  - `process_weights_after_loading(self, layer: torch.nn.Module)` — [`L114`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py#L114)
  - `is_channelwise` — [`L56`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py#L56)
  - `linear_config` — [`L55`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py#L55)
- protocol/private: `__init__`[`L51`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py#L51), `_apply_fused`[`L192`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py#L192), `_apply_split`[`L213`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py#L213)
- uses (calls/refs, reference-scoped): [`process_linear_weights`](../../../../common/process_weights/linear_weights.md#process_linear_weights), [`shard_linear_weights`](../../../../common/process_weights/linear_weights.md#shard_linear_weights), [`LinearWeights`](../../../../common/process_weights/linear_weights.md#LinearWeights), [`fuse_matmuls`](../../configs.md#VllmQuantLinearConfig.fuse_matmuls), [`to_parameter_list`](../../../../common/process_weights/linear_weights.md#to_parameter_list), [`weight_sharding`](../../configs.md#VllmQuantLinearConfig.weight_sharding), [`sharded_quantized_matmul`](../../../../common/linear.md#sharded_quantized_matmul), [`VllmQuantLinearConfig`](../../configs.md#VllmQuantLinearConfig), [`weight_scale`](../../../../common/process_weights/linear_weights.md#LinearWeights.weight_scale), [`n_shards`](../../configs.md#VllmQuantLinearConfig.n_shards), [`mesh`](../../configs.md#VllmQuantLinearConfig.mesh), [`bias_sharding`](../../configs.md#VllmQuantLinearConfig.bias_sharding), [`bias`](../../../../common/process_weights/linear_weights.md#LinearWeights.bias), [`output_sizes`](../../configs.md#VllmQuantLinearConfig.output_sizes), [`weight`](../../../../common/process_weights/linear_weights.md#LinearWeights.weight), [`slice_sharded_tensor_for_concatenation`](../../../../common/utils.md#slice_sharded_tensor_for_concatenation), [`zero_point`](../../../../common/process_weights/linear_weights.md#LinearWeights.zero_point), [`enable_quantized_matmul_kernel`](../../../../common/quantization/configs.md#QuantLinearConfig.enable_quantized_matmul_kernel), [`logger`](compressed_tensors_w8a8_int8.md#logger)
- used by: [`get_scheme`](../compressed_tensors.md#VllmCompressedTensorsConfig.get_scheme)

## Module values
- `P` — [`L45`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py#L45)
- `logger` — [`L46`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py#L46)


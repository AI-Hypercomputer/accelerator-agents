---
title: 'Module: tpu_inference/layers/common/process_weights/linear_weights.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/common/process_weights/linear_weights.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.common.process_weights.linear_weights`/
symbols:
  process_linear_weights: process_linear_weights().
  shard_linear_weights: shard_linear_weights().
  LinearWeights: LinearWeights#
  to_parameter_list: to_parameter_list().
  LinearWeights.weight_scale: LinearWeights#weight_scale.
  LinearWeights.weight: LinearWeights#weight.
  LinearWeights.bias: LinearWeights#bias.
  LinearWeights.zero_point: LinearWeights#zero_point.
  format_linear_scale: format_linear_scale().
  get_model_matmul_fusion_assignment: get_model_matmul_fusion_assignment().
  process_linear_weights.slice_tensor: process_linear_weights().slice_tensor().
  P: P.
  logger: logger.
  MODEL_MATMUL_FUSION_TRUTH_TABLE: MODEL_MATMUL_FUSION_TRUTH_TABLE.
---
# Module: [`tpu_inference/layers/common/process_weights/linear_weights.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/linear_weights.py)

## Classes
### `LinearWeights`
- def: [`tpu_inference/layers/common/process_weights/linear_weights.py:38`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/linear_weights.py#L38)
- signature: `class LinearWeights:`
- members:
  - `bias` — [`L42`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/linear_weights.py#L42)
  - `weight` — [`L39`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/linear_weights.py#L39)
  - `weight_scale` — [`L40`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/linear_weights.py#L40)
  - `zero_point` — [`L41`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/linear_weights.py#L41)
- used by: [`process_weights_after_loading`](../../jax/quantization/fp8.md#Fp8BlockwiseLinearMethod.process_weights_after_loading), [`process_weights_after_loading`](../../vllm/quantization/fp8.md#VllmFp8LinearMethod.process_weights_after_loading), [`process_linear_weights`](linear_weights.md#process_linear_weights), [`process_fp8_linear_weights`](../../vllm/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.md#VllmCompressedTensorsW8A8Fp8.process_fp8_linear_weights), [`shard_linear_weights`](linear_weights.md#shard_linear_weights), [`process_blockwise_fp8_linear_weights`](../quantization/fp8.md#process_blockwise_fp8_linear_weights), [`process_awq_linear_weights`](../../vllm/quantization/awq.md#VllmAWQLinearMethod.process_awq_linear_weights), [`process_int8_linear_weights`](../../vllm/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.md#VllmCompressedTensorsW8A8Int8.process_int8_linear_weights), [`process_uint4_linear_weights`](../../vllm/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8.md#VllmCompressedTensorsW4A8Fp8.process_uint4_linear_weights), [`process_weights_after_loading`](../../vllm/quantization/nvfp4.md#VllmNvfp4LinearMethod.process_weights_after_loading), [`process_unquantized_linear_weights`](../../vllm/quantization/unquantized.md#VllmUnquantizedLinearMethod.process_unquantized_linear_weights)

## Functions
- `format_linear_scale(weight_scale: jax.Array | Tensor | list | None, enable_kernel: bool)` — [`L84`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/linear_weights.py#L84)
- `get_model_matmul_fusion_assignment(model_name: str, batch_size: int, tp_size: int, layer_name: str)` — [`L78`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/linear_weights.py#L78)
- `process_linear_weights(weights: LinearWeights, fused: bool = False, output_sizes: list[int] | None = None, reorder_size: int | None = None, per_tensor: bool = False, enable_kernel: bool = False)` — [`L99`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/linear_weights.py#L99)
- `shard_linear_weights(weights: LinearWeights, mesh: Mesh | None, weight_p_spec: PartitionSpec, bias_p_spec: PartitionSpec, per_tensor: bool = False)` — [`L165`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/linear_weights.py#L165)
- `slice_tensor(tensor)` — [`L132`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/linear_weights.py#L132)
- `to_parameter_list(tensor: list[torch.Tensor])` — [`L73`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/linear_weights.py#L73)

## Module values
- `MODEL_MATMUL_FUSION_TRUTH_TABLE` — [`L45`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/linear_weights.py#L45)
- `P` — [`L31`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/linear_weights.py#L31)
- `logger` — [`L33`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/process_weights/linear_weights.py#L33)


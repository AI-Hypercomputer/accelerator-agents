---
title: 'Module: tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.vllm.quantization.compressed_tensors.schemes.compressed_tensors_w4a8_fp8`/
symbols:
  VllmCompressedTensorsW4A8Fp8.process_uint4_linear_weights: VllmCompressedTensorsW4A8Fp8#process_uint4_linear_weights().
  VllmCompressedTensorsW4A8Fp8.process_weights_after_loading: VllmCompressedTensorsW4A8Fp8#process_weights_after_loading().
  VllmCompressedTensorsW4A8Fp8._apply_fused: VllmCompressedTensorsW4A8Fp8#_apply_fused().
  VllmCompressedTensorsW4A8Fp8.linear_config: VllmCompressedTensorsW4A8Fp8#linear_config.
  VllmCompressedTensorsW4A8Fp8._apply_split: VllmCompressedTensorsW4A8Fp8#_apply_split().
  VllmCompressedTensorsW4A8Fp8.apply_weights: VllmCompressedTensorsW4A8Fp8#apply_weights().
  VllmCompressedTensorsW4A8Fp8.create_weights: VllmCompressedTensorsW4A8Fp8#create_weights().
  logger: logger.
  VllmCompressedTensorsW4A8Fp8.__init__: VllmCompressedTensorsW4A8Fp8#__init__().
  VllmCompressedTensorsW4A8Fp8.has_g_idx: VllmCompressedTensorsW4A8Fp8#has_g_idx.
  VllmCompressedTensorsW4A8Fp8.weight_block_size: VllmCompressedTensorsW4A8Fp8#weight_block_size.
  VllmCompressedTensorsW4A8Fp8: VllmCompressedTensorsW4A8Fp8#
  VllmCompressedTensorsW4A8Fp8.pack_factor: VllmCompressedTensorsW4A8Fp8#pack_factor.
  VllmCompressedTensorsW4A8Fp8.group_size: VllmCompressedTensorsW4A8Fp8#group_size.
  VllmCompressedTensorsW4A8Fp8.weight_quant: VllmCompressedTensorsW4A8Fp8#weight_quant.
  VllmCompressedTensorsW4A8Fp8.strategy: VllmCompressedTensorsW4A8Fp8#strategy.
  VllmCompressedTensorsW4A8Fp8.num_bits: VllmCompressedTensorsW4A8Fp8#num_bits.
  VllmCompressedTensorsW4A8Fp8.wtype: VllmCompressedTensorsW4A8Fp8#wtype.
  P: P.
  VllmCompressedTensorsW4A8Fp8.symmetric: VllmCompressedTensorsW4A8Fp8#symmetric.
  VllmCompressedTensorsW4A8Fp8.actorder: VllmCompressedTensorsW4A8Fp8#actorder.
  VllmCompressedTensorsW4A8Fp8.quant_type: VllmCompressedTensorsW4A8Fp8#quant_type.
  VllmCompressedTensorsW4A8Fp8.out_dtype: VllmCompressedTensorsW4A8Fp8#out_dtype.
  VllmCompressedTensorsW4A8Fp8.is_static_input_scheme: VllmCompressedTensorsW4A8Fp8#is_static_input_scheme.
---
# Module: [`tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8.py`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8.py)

## Classes
### `VllmCompressedTensorsW4A8Fp8`
- def: [`tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8.py:51`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8.py#L51)
- signature: `class VllmCompressedTensorsW4A8Fp8(CompressedTensorsW4A8Fp8):`
- members:
  - `apply_weights(self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor])` — [`L229`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8.py#L229)
  - `create_weights(self, layer: torch.nn.Module, output_size: int, input_size: int, output_partition_sizes: list[int], input_size_per_partition: int, params_dtype: torch.dtype, weight_loader: Callable, **kwargs)` — [`L92`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8.py#L92)
  - `process_uint4_linear_weights(uint_weight: jax.Array, weight_scale: jax.Array, bias: jax.Array | None)` — [`L177`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8.py#L177)
  - `process_weights_after_loading(self, layer: torch.nn.Module)` — [`L157`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8.py#L157)
  - `actorder` — [`L65`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8.py#L65)
  - `group_size` — [`L66`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8.py#L66)
  - `has_g_idx` — [`L67`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8.py#L67)
  - `is_static_input_scheme` — [`L87`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8.py#L87)
  - `linear_config` — [`L90`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8.py#L90)
  - `num_bits` — [`L63`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8.py#L63)
  - `out_dtype` — [`L86`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8.py#L86)
  - `pack_factor` — [`L61`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8.py#L61)
  - `quant_type` — [`L83`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8.py#L83)
  - `strategy` — [`L62`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8.py#L62)
  - `symmetric` — [`L64`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8.py#L64)
  - `weight_block_size` — [`L88`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8.py#L88)
  - `weight_quant` — [`L85`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8.py#L85)
  - `wtype` — [`L70`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8.py#L70)
- protocol/private: `__init__`[`L53`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8.py#L53), `_apply_fused`[`L237`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8.py#L237), `_apply_split`[`L256`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8.py#L256)
- uses (calls/refs, reference-scoped): [`process_linear_weights`](../../../../common/process_weights/linear_weights.md#process_linear_weights), [`shard_linear_weights`](../../../../common/process_weights/linear_weights.md#shard_linear_weights), [`LinearWeights`](../../../../common/process_weights/linear_weights.md#LinearWeights), [`fuse_matmuls`](../../configs.md#VllmQuantLinearConfig.fuse_matmuls), [`to_parameter_list`](../../../../common/process_weights/linear_weights.md#to_parameter_list), [`weight_sharding`](../../configs.md#VllmQuantLinearConfig.weight_sharding), [`sharded_quantized_matmul`](../../../../common/linear.md#sharded_quantized_matmul), [`VllmQuantLinearConfig`](../../configs.md#VllmQuantLinearConfig), [`weight_scale`](../../../../common/process_weights/linear_weights.md#LinearWeights.weight_scale), [`n_shards`](../../configs.md#VllmQuantLinearConfig.n_shards), [`mesh`](../../configs.md#VllmQuantLinearConfig.mesh), [`bias_sharding`](../../configs.md#VllmQuantLinearConfig.bias_sharding), [`bias`](../../../../common/process_weights/linear_weights.md#LinearWeights.bias), [`output_sizes`](../../configs.md#VllmQuantLinearConfig.output_sizes), [`weight`](../../../../common/process_weights/linear_weights.md#LinearWeights.weight), [`slice_sharded_tensor_for_concatenation`](../../../../common/utils.md#slice_sharded_tensor_for_concatenation), [`zero_point`](../../../../common/process_weights/linear_weights.md#LinearWeights.zero_point), [`enable_quantized_matmul_kernel`](../../../../common/quantization/configs.md#QuantLinearConfig.enable_quantized_matmul_kernel), [`logger`](compressed_tensors_w4a8_fp8.md#logger)
- used by: [`get_scheme`](../compressed_tensors.md#VllmCompressedTensorsConfig.get_scheme)

## Module values
- `P` — [`L47`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8.py#L47)
- `logger` — [`L48`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8.py#L48)


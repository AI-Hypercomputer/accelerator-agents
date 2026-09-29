---
title: 'Module: tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a8.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a8.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.vllm.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a8`/
symbols:
  VllmCompressedTensorsW4A8MoEMethod.unpack_and_process: VllmCompressedTensorsW4A8MoEMethod#unpack_and_process().
  VllmCompressedTensorsW4A8MoEMethod.process_weights_after_loading: VllmCompressedTensorsW4A8MoEMethod#process_weights_after_loading().
  VllmCompressedTensorsW4A8MoEMethod.apply_monolithic: VllmCompressedTensorsW4A8MoEMethod#apply_monolithic().
  VllmCompressedTensorsW4A8MoEMethod.extra_backend_kwargs: VllmCompressedTensorsW4A8MoEMethod#extra_backend_kwargs.
  VllmCompressedTensorsW4A8MoEMethod.packed_factor: VllmCompressedTensorsW4A8MoEMethod#packed_factor.
  VllmCompressedTensorsW4A8MoEMethod.group_size: VllmCompressedTensorsW4A8MoEMethod#group_size.
  VllmCompressedTensorsW4A8MoEMethod.create_weights: VllmCompressedTensorsW4A8MoEMethod#create_weights().
  VllmCompressedTensorsW4A8MoEMethod: VllmCompressedTensorsW4A8MoEMethod#
  VllmCompressedTensorsW4A8MoEMethod.moe_backend: VllmCompressedTensorsW4A8MoEMethod#moe_backend.
  VllmCompressedTensorsW4A8MoEMethod.weight_quant: VllmCompressedTensorsW4A8MoEMethod#weight_quant.
  VllmCompressedTensorsW4A8MoEMethod.num_bits: VllmCompressedTensorsW4A8MoEMethod#num_bits.
  logger: logger.
  VllmCompressedTensorsW4A8MoEMethod.__init__: VllmCompressedTensorsW4A8MoEMethod#__init__().
  VllmCompressedTensorsW4A8MoEMethod.moe_quant_config: VllmCompressedTensorsW4A8MoEMethod#moe_quant_config.
  VllmCompressedTensorsW4A8MoEMethod.get_fused_moe_quant_config: VllmCompressedTensorsW4A8MoEMethod#get_fused_moe_quant_config().
  VllmCompressedTensorsW4A8MoEMethod.wtype: VllmCompressedTensorsW4A8MoEMethod#wtype.
  VllmCompressedTensorsW4A8MoEMethod.input_quant: VllmCompressedTensorsW4A8MoEMethod#input_quant.
  VllmCompressedTensorsW4A8MoEMethod.is_monolithic: VllmCompressedTensorsW4A8MoEMethod#is_monolithic().
---
# Module: [`tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a8.py`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a8.py)

## Classes
### `VllmCompressedTensorsW4A8MoEMethod`  ·  implements/extends VllmQuantConfig
- def: [`tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a8.py:45`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a8.py#L45)
- doc: MoE method for int4 weights and 8 bit activations.
- signature: `class VllmCompressedTensorsW4A8MoEMethod(CompressedTensorsMoEMethod, VllmQuantConfig):`
- members:
  - `apply_monolithic(self, layer: RoutedExperts, x: torch.Tensor, router_logits: torch.Tensor, input_ids: torch.Tensor | None = None)` — [`L306`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a8.py#L306)
  - `create_weights(self, layer: torch.nn.Module, num_experts: int, hidden_size: int, intermediate_size_per_partition: int, params_dtype: torch.dtype, **extra_weight_attrs)` — [`L88`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a8.py#L88) — Initializes the weights and scales for the FusedMoE layer.
  - `get_fused_moe_quant_config(self, layer: torch.nn.Module)` — [`L301`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a8.py#L301)
  - `is_monolithic(self)` — [`L84`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a8.py#L84) — Indicates if the MoE operation is monolithic.
  - `process_weights_after_loading(self, layer: torch.nn.Module)` — [`L189`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a8.py#L189) — Processes and shards MoE weights after loading.
  - `unpack_and_process(weights: FusedMoEWeights, desired_quant_dtype: jnp.dtype | None, requant_block_size: int | None)` — [`L240`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a8.py#L240)
  - `extra_backend_kwargs` — [`L67`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a8.py#L67)
  - `group_size` — [`L75`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a8.py#L75)
  - `input_quant` — [`L73`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a8.py#L73)
  - `moe_backend` — [`L65`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a8.py#L65)
  - `moe_quant_config` — [`L299`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a8.py#L299)
  - `num_bits` — [`L76`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a8.py#L76)
  - `packed_factor` — [`L77`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a8.py#L77)
  - `weight_quant` — [`L72`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a8.py#L72)
  - `wtype` — [`L70`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a8.py#L70)
- protocol/private: `__init__`[`L54`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a8.py#L54)
- uses (calls/refs, reference-scoped): [`MoEBackend`](../../../../common/moe.md#MoEBackend), [`FusedMoEWeights`](../../../../common/process_weights/moe_weights.md#FusedMoEWeights), [`VllmQuantConfig`](../../configs.md#VllmQuantConfig), [`w13_weight_scale`](../../../../common/process_weights/moe_weights.md#FusedMoEWeights.w13_weight_scale), [`w2_weight_scale`](../../../../common/process_weights/moe_weights.md#FusedMoEWeights.w2_weight_scale), [`w13_weight`](../../../../common/process_weights/moe_weights.md#FusedMoEWeights.w13_weight), [`w13_bias`](../../../../common/process_weights/moe_weights.md#FusedMoEWeights.w13_bias), [`w2_weight`](../../../../common/process_weights/moe_weights.md#FusedMoEWeights.w2_weight), [`w2_bias`](../../../../common/process_weights/moe_weights.md#FusedMoEWeights.w2_bias), [`vllm_moe_apply`](../../../interface/moe.md#vllm_moe_apply), [`select_moe_backend_from_fused_moe_config`](../../../interface/moe.md#select_moe_backend_from_fused_moe_config), [`process_quantized_moe_weights`](../../../../common/process_weights/moe_weights.md#process_quantized_moe_weights), [`FUSED_MOE`](../../../../common/moe.md#MoEBackend.FUSED_MOE), [`mesh`](../../configs.md#VllmQuantConfig.mesh), [`shard_moe_weights_to_tpu`](../../../../common/process_weights/moe_weights.md#shard_moe_weights_to_tpu), [`MOE_REQUANTIZE_WEIGHT_DTYPE`](../../../../../envs.md#MOE_REQUANTIZE_WEIGHT_DTYPE), [`MOE_REQUANTIZE_BLOCK_SIZE`](../../../../../envs.md#MOE_REQUANTIZE_BLOCK_SIZE), [`u32_unpack_i4`](../../../../common/__init__.md#u32_unpack_i4)
- used by: [`VllmQuantConfig`](../../configs.md#VllmQuantConfig), [`get_moe_method`](compressed_tensors_moe.md#VllmCompressedTensorsMoEMethod.get_moe_method)

## Module values
- `logger` — [`L42`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a8.py#L42)


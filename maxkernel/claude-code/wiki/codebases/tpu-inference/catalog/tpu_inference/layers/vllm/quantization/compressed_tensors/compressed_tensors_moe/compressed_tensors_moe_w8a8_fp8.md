---
title: 'Module: tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w8a8_fp8.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w8a8_fp8.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.vllm.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_fp8`/
symbols:
  VllmCompressedTensorsW8A8Fp8MoEMethod.process_weights_after_loading: VllmCompressedTensorsW8A8Fp8MoEMethod#process_weights_after_loading().
  VllmCompressedTensorsW8A8Fp8MoEMethod.apply_monolithic: VllmCompressedTensorsW8A8Fp8MoEMethod#apply_monolithic().
  VllmCompressedTensorsW8A8Fp8MoEMethod.extra_backend_kwargs: VllmCompressedTensorsW8A8Fp8MoEMethod#extra_backend_kwargs.
  VllmCompressedTensorsW8A8Fp8MoEMethod: VllmCompressedTensorsW8A8Fp8MoEMethod#
  VllmCompressedTensorsW8A8Fp8MoEMethod.moe_backend: VllmCompressedTensorsW8A8Fp8MoEMethod#moe_backend.
  logger: logger.
  VllmCompressedTensorsW8A8Fp8MoEMethod.__init__: VllmCompressedTensorsW8A8Fp8MoEMethod#__init__().
  VllmCompressedTensorsW8A8Fp8MoEMethod.is_monolithic: VllmCompressedTensorsW8A8Fp8MoEMethod#is_monolithic().
---
# Module: [`tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w8a8_fp8.py`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w8a8_fp8.py)

## Classes
### `VllmCompressedTensorsW8A8Fp8MoEMethod`  ·  implements/extends VllmQuantConfig
- def: [`tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w8a8_fp8.py:36`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w8a8_fp8.py#L36)
- signature: `class VllmCompressedTensorsW8A8Fp8MoEMethod(CompressedTensorsW8A8Fp8MoEMethod, VllmQuantConfig):`
- members:
  - `apply_monolithic(self, layer: RoutedExperts, x: torch.Tensor, router_logits: torch.Tensor, input_ids: torch.Tensor | None = None)` — [`L130`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w8a8_fp8.py#L130)
  - `is_monolithic(self)` — [`L55`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w8a8_fp8.py#L55)
  - `process_weights_after_loading(self, layer: torch.nn.Module)` — [`L58`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w8a8_fp8.py#L58) — Docstring for process_weights_after_loading
  - `extra_backend_kwargs` — [`L50`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w8a8_fp8.py#L50)
  - `moe_backend` — [`L48`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w8a8_fp8.py#L48)
- protocol/private: `__init__`[`L39`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w8a8_fp8.py#L39)
- uses (calls/refs, reference-scoped): [`MoEBackend`](../../../../common/moe.md#MoEBackend), [`FusedMoEWeights`](../../../../common/process_weights/moe_weights.md#FusedMoEWeights), [`VllmQuantConfig`](../../configs.md#VllmQuantConfig), [`w13_weight_scale`](../../../../common/process_weights/moe_weights.md#FusedMoEWeights.w13_weight_scale), [`w2_weight_scale`](../../../../common/process_weights/moe_weights.md#FusedMoEWeights.w2_weight_scale), [`w13_weight`](../../../../common/process_weights/moe_weights.md#FusedMoEWeights.w13_weight), [`shard_moe_weights`](../../../../common/process_weights/moe_weights.md#shard_moe_weights), [`w13_bias`](../../../../common/process_weights/moe_weights.md#FusedMoEWeights.w13_bias), [`w2_weight`](../../../../common/process_weights/moe_weights.md#FusedMoEWeights.w2_weight), [`w2_bias`](../../../../common/process_weights/moe_weights.md#FusedMoEWeights.w2_bias), [`vllm_moe_apply`](../../../interface/moe.md#vllm_moe_apply), [`select_moe_backend_from_fused_moe_config`](../../../interface/moe.md#select_moe_backend_from_fused_moe_config), [`process_quantized_moe_weights`](../../../../common/process_weights/moe_weights.md#process_quantized_moe_weights), [`FUSED_MOE`](../../../../common/moe.md#MoEBackend.FUSED_MOE), [`mesh`](../../configs.md#VllmQuantConfig.mesh)
- used by: [`VllmQuantConfig`](../../configs.md#VllmQuantConfig), [`get_moe_method`](compressed_tensors_moe.md#VllmCompressedTensorsMoEMethod.get_moe_method)

## Module values
- `logger` — [`L33`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w8a8_fp8.py#L33)


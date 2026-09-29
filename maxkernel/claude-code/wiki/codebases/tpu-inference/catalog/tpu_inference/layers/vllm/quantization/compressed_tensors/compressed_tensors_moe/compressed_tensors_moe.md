---
title: 'Module: tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.vllm.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe`/
symbols:
  VllmCompressedTensorsMoEMethod.get_moe_method: VllmCompressedTensorsMoEMethod#get_moe_method().
  VllmCompressedTensorsMoEMethod: VllmCompressedTensorsMoEMethod#
  logger: logger.
---
# Module: [`tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe.py`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe.py)

## Classes
### `VllmCompressedTensorsMoEMethod`
- def: [`tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe.py:29`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe.py#L29)
- signature: `class VllmCompressedTensorsMoEMethod(CompressedTensorsMoEMethod):`
- members:
  - `get_moe_method(quant_config: VllmCompressedTensorsConfig, layer: torch.nn.Module, layer_name: str)` — [`L32`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe.py#L32)
- uses (calls/refs, reference-scoped): [`VllmUnquantizedFusedMoEMethod`](../../unquantized.md#VllmUnquantizedFusedMoEMethod), [`VllmCompressedTensorsW8A8Fp8MoEMethod`](compressed_tensors_moe_w8a8_fp8.md#VllmCompressedTensorsW8A8Fp8MoEMethod), [`VllmCompressedTensorsW4A8MoEMethod`](compressed_tensors_moe_w4a8.md#VllmCompressedTensorsW4A8MoEMethod)
- used by: [`get_quant_method`](../compressed_tensors.md#VllmCompressedTensorsConfig.get_quant_method)

## Module values
- `logger` — [`L26`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe.py#L26)


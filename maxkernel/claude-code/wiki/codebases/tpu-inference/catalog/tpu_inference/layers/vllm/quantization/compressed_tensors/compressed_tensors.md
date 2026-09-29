---
title: 'Module: tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.vllm.quantization.compressed_tensors.compressed_tensors`/
symbols:
  VllmCompressedTensorsConfig.get_quant_method: VllmCompressedTensorsConfig#get_quant_method().
  VllmCompressedTensorsConfig.get_scheme: VllmCompressedTensorsConfig#get_scheme().
  logger: logger.
  VllmCompressedTensorsConfig: VllmCompressedTensorsConfig#
  VllmCompressedTensorsConfig.get_name: VllmCompressedTensorsConfig#get_name().
  P: P.
---
# Module: [`tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors.py`](../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors.py)

## Classes
### `VllmCompressedTensorsConfig`  ·  implements/extends VllmQuantConfig
- def: [`tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors.py:51`](../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors.py#L51)
- signature: `class VllmCompressedTensorsConfig(CompressedTensorsConfig, VllmQuantConfig):`
- members:
  - `get_name(cls)` — [`L54`](../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors.py#L54)
  - `get_quant_method(self, layer: torch.nn.Module, prefix: str)` — [`L124`](../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors.py#L124)
  - `get_scheme(self, layer: torch.nn.Module, layer_name: Optional[str] = None)` — [`L57`](../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors.py#L57) — compressed-tensors supports non uniform in the following way:
- uses (calls/refs, reference-scoped): [`VllmQuantConfig`](../configs.md#VllmQuantConfig), [`get_quant_method`](../unquantized.md#VllmUnquantizedConfig.get_quant_method), [`get_linear_config`](../configs.md#VllmQuantConfig.get_linear_config), [`VllmUnquantizedConfig`](../unquantized.md#VllmUnquantizedConfig), [`get_moe_config`](../configs.md#VllmQuantConfig.get_moe_config), [`get_moe_method`](compressed_tensors_moe/compressed_tensors_moe.md#VllmCompressedTensorsMoEMethod.get_moe_method), [`logger`](compressed_tensors.md#logger), [`COMPRESSED_TENSORS`](../../../common/quant_methods.md#COMPRESSED_TENSORS), [`VllmCompressedTensorsMoEMethod`](compressed_tensors_moe/compressed_tensors_moe.md#VllmCompressedTensorsMoEMethod), [`VllmCompressedTensorsW4A8Fp8`](schemes/compressed_tensors_w4a8_fp8.md#VllmCompressedTensorsW4A8Fp8), [`VllmCompressedTensorsW8A8Fp8`](schemes/compressed_tensors_w8a8_fp8.md#VllmCompressedTensorsW8A8Fp8), [`VllmCompressedTensorsW8A8Int8`](schemes/compressed_tensors_w8a8_int8.md#VllmCompressedTensorsW8A8Int8)
- used by: [`VllmQuantConfig`](../configs.md#VllmQuantConfig), [`get_tpu_quantization_config`](../__init__.md#get_tpu_quantization_config)

## Module values
- `P` — [`L46`](../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors.py#L46)
- `logger` — [`L47`](../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors.py#L47)


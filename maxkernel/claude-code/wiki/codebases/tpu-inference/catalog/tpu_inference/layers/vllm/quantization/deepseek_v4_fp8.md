---
title: 'Module: tpu_inference/layers/vllm/quantization/deepseek_v4_fp8.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/vllm/quantization/deepseek_v4_fp8.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.vllm.quantization.deepseek_v4_fp8`/
symbols:
  VllmDeepseekV4Fp8Config.get_quant_method: VllmDeepseekV4Fp8Config#get_quant_method().
  logger: logger.
  VllmDeepseekV4Fp8Config: VllmDeepseekV4Fp8Config#
  VllmDeepseekV4Fp8Config.get_name: VllmDeepseekV4Fp8Config#get_name().
---
# Module: [`tpu_inference/layers/vllm/quantization/deepseek_v4_fp8.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/deepseek_v4_fp8.py)

## Classes
### `VllmDeepseekV4Fp8Config`  ·  implements/extends VllmQuantConfig
- def: [`tpu_inference/layers/vllm/quantization/deepseek_v4_fp8.py:40`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/deepseek_v4_fp8.py#L40)
- doc: TPU quantization config for "deepseek_v4_fp8" format.
- signature: `class VllmDeepseekV4Fp8Config(DeepseekV4FP8Config, VllmQuantConfig):`
- members:
  - `get_name(cls)` — [`L50`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/deepseek_v4_fp8.py#L50)
  - `get_quant_method(self, layer: torch.nn.Module, prefix: str)` — [`L53`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/deepseek_v4_fp8.py#L53)
- uses (calls/refs, reference-scoped): [`VllmQuantConfig`](configs.md#VllmQuantConfig), [`QuantizeMethodBase`](../../jax/quantization/__init__.md#QuantizeMethodBase), [`VllmUnquantizedLinearMethod`](unquantized.md#VllmUnquantizedLinearMethod), [`mesh`](configs.md#VllmQuantConfig.mesh), [`get_linear_config`](configs.md#VllmQuantConfig.get_linear_config), [`VllmUnquantizedFusedMoEMethod`](unquantized.md#VllmUnquantizedFusedMoEMethod), [`get_moe_config`](configs.md#VllmQuantConfig.get_moe_config), [`VllmFp8LinearMethod`](fp8.md#VllmFp8LinearMethod), [`DSV4_FP8`](../../common/quant_methods.md#DSV4_FP8), [`VllmFp8MoEMethod`](fp8.md#VllmFp8MoEMethod), [`VllmMxfp4MoEMethod`](mxfp4.md#VllmMxfp4MoEMethod)
- used by: [`VllmQuantConfig`](configs.md#VllmQuantConfig), [`get_tpu_quantization_config`](__init__.md#get_tpu_quantization_config)

## Module values
- `logger` — [`L36`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/deepseek_v4_fp8.py#L36)


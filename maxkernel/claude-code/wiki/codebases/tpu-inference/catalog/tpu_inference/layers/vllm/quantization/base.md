---
title: 'Module: tpu_inference/layers/vllm/quantization/base.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/vllm/quantization/base.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.vllm.quantization.base`/VllmQuantizationMethod#
symbols:
  VllmQuantizationMethod: ''
  VllmQuantizationMethod.maybe_process_weights: maybe_process_weights().
---
# Module: [`tpu_inference/layers/vllm/quantization/base.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/base.py)

## Classes
### `VllmQuantizationMethod`  ·  implements/extends ABC
- def: [`tpu_inference/layers/vllm/quantization/base.py:20`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/base.py#L20)
- signature: `class VllmQuantizationMethod(ABC):`
- members:
  - `maybe_process_weights(self, layer: torch.nn.Module, param_name: str, args, kwargs)` — [`L23`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/quantization/base.py#L23)
- used by: [`VllmUnquantizedLinearMethod`](unquantized.md#VllmUnquantizedLinearMethod), [`VllmUnquantizedFusedMoEMethod`](unquantized.md#VllmUnquantizedFusedMoEMethod)


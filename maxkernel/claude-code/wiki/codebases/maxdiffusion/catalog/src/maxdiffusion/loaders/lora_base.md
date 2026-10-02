---
title: 'Module: src/maxdiffusion/loaders/lora_base.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/loaders/lora_base.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.loaders.lora_base`/LoRABaseMixin#
symbols:
  LoRABaseMixin: ''
  LoRABaseMixin.load_lora_weights: load_lora_weights().
  LoRABaseMixin._fetch_state_dict: _fetch_state_dict().
  LoRABaseMixin._lora_lodable_modules: _lora_lodable_modules.
  LoRABaseMixin.num_fused_loras: num_fused_loras.
---
# Module: [`src/maxdiffusion/loaders/lora_base.py`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/lora_base.py)

## Classes
### `LoRABaseMixin`
- def: [`src/maxdiffusion/loaders/lora_base.py:21`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/lora_base.py#L21)
- doc: Utility class for handing LoRAs
- signature: `class LoRABaseMixin:`
- members:
  - `load_lora_weights(self, **kwargs)` — [`L27`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/lora_base.py#L27)
  - `num_fused_loras` — [`L25`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/lora_base.py#L25)
- protocol/private: `_fetch_state_dict`[`L31`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/lora_base.py#L31), `_lora_lodable_modules`[`L24`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/lora_base.py#L24)
- uses (calls/refs, reference-scoped): [`_get_model_file`](../utils/hub_utils.md#_get_model_file), [`StableDiffusionLoraLoaderMixin`](lora_pipeline.md#StableDiffusionLoraLoaderMixin), [`load_lora_weights`](wan_lora_nnx_loader.md#Wan2_1NNXLoraLoader.load_lora_weights), [`load_lora_weights`](wan_lora_nnx_loader.md#Wan2_2NNXLoraLoader.load_lora_weights), [`load_state_dict`](../models/modeling_utils.md#load_state_dict), [`load_lora_weights`](flux_lora_pipeline.md#FluxLoraLoaderMixin.load_lora_weights), [`load_lora_weights`](lora_pipeline.md#StableDiffusionLoraLoaderMixin.load_lora_weights), [`FluxLoraLoaderMixin`](flux_lora_pipeline.md#FluxLoraLoaderMixin), [`Wan2_1NNXLoraLoader`](wan_lora_nnx_loader.md#Wan2_1NNXLoraLoader), [`Wan2_2NNXLoraLoader`](wan_lora_nnx_loader.md#Wan2_2NNXLoraLoader), [`LORA_WEIGHT_NAME_SAFE`](lora_pipeline.md#LORA_WEIGHT_NAME_SAFE)
- used by: [`StableDiffusionLoraLoaderMixin`](lora_pipeline.md#StableDiffusionLoraLoaderMixin), [`lora_state_dict`](lora_pipeline.md#StableDiffusionLoraLoaderMixin.lora_state_dict), [`FluxLoraLoaderMixin`](flux_lora_pipeline.md#FluxLoraLoaderMixin), [`Wan2_1NNXLoraLoader`](wan_lora_nnx_loader.md#Wan2_1NNXLoraLoader), [`Wan2_2NNXLoraLoader`](wan_lora_nnx_loader.md#Wan2_2NNXLoraLoader), [`lora_state_dict`](flux_lora_pipeline.md#FluxLoraLoaderMixin.lora_state_dict)


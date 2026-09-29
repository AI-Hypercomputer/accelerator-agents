---
title: 'Module: src/maxdiffusion/loaders/wan_lora_nnx_loader.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/loaders/wan_lora_nnx_loader.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.loaders.wan_lora_nnx_loader`/Wan2_
symbols:
  Wan2_1NNXLoraLoader.load_lora_weights: 1NNXLoraLoader#load_lora_weights().
  Wan2_2NNXLoraLoader.load_lora_weights: 2NNXLoraLoader#load_lora_weights().
  Wan2_2NNXLoraLoader.translate_fn: 2NNXLoraLoader#translate_fn().
  Wan2_1NNXLoraLoader: 1NNXLoraLoader#
  Wan2_1NNXLoraLoader.translate_fn: 1NNXLoraLoader#translate_fn().
  Wan2_2NNXLoraLoader: 2NNXLoraLoader#
---
# Module: [`src/maxdiffusion/loaders/wan_lora_nnx_loader.py`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/wan_lora_nnx_loader.py)

## Classes
### `Wan2_1NNXLoraLoader`  ·  implements/extends LoRABaseMixin
- def: [`src/maxdiffusion/loaders/wan_lora_nnx_loader.py:25`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/wan_lora_nnx_loader.py#L25)
- doc: Handles loading LoRA weights into NNX-based WAN 2.1 model.
- signature: `class Wan2_1NNXLoraLoader(LoRABaseMixin):`
- members:
  - `load_lora_weights(self, pipeline: nnx.Module, lora_model_path: str, transformer_weight_name: str, rank: int, scale: float = 1, scan_layers: bool = False, dtype: str = "float32", **kwargs)` — [`L32`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/wan_lora_nnx_loader.py#L32) — Merges LoRA weights into the pipeline from a checkpoint.
  - `translate_fn(nnx_path_str)` — [`L50`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/wan_lora_nnx_loader.py#L50)
- uses (calls/refs, reference-scoped): [`merge_lora_for_scanned`](../models/lora_nnx.md#merge_lora_for_scanned), [`LoRABaseMixin`](lora_base.md#LoRABaseMixin), [`StableDiffusionLoraLoaderMixin`](lora_pipeline.md#StableDiffusionLoraLoaderMixin), [`lora_state_dict`](lora_pipeline.md#StableDiffusionLoraLoaderMixin.lora_state_dict), [`merge_lora`](../models/lora_nnx.md#merge_lora), [`preprocess_wan_lora_dict`](lora_conversion_utils.md#preprocess_wan_lora_dict), [`translate_wan_nnx_path_to_diffusers_lora`](lora_conversion_utils.md#translate_wan_nnx_path_to_diffusers_lora)
- used by: [`LoRABaseMixin`](lora_base.md#LoRABaseMixin), [`load_lora_weights`](lora_base.md#LoRABaseMixin.load_lora_weights)

### `Wan2_2NNXLoraLoader`  ·  implements/extends LoRABaseMixin
- def: [`src/maxdiffusion/loaders/wan_lora_nnx_loader.py:64`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/wan_lora_nnx_loader.py#L64)
- doc: Handles loading LoRA weights into NNX-based WAN 2.2 model.
- signature: `class Wan2_2NNXLoraLoader(LoRABaseMixin):`
- members:
  - `load_lora_weights(self, pipeline: nnx.Module, lora_model_path: str, high_noise_weight_name: str, low_noise_weight_name: str, rank: int, scale: float = 1, scan_layers: bool = False, dtype: str = "float32", **kwargs)` — [`L71`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/wan_lora_nnx_loader.py#L71) — Merges LoRA weights into the pipeline from a checkpoint.
  - `translate_fn(nnx_path_str: str)` — [`L90`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/wan_lora_nnx_loader.py#L90)
- uses (calls/refs, reference-scoped): [`merge_lora_for_scanned`](../models/lora_nnx.md#merge_lora_for_scanned), [`LoRABaseMixin`](lora_base.md#LoRABaseMixin), [`StableDiffusionLoraLoaderMixin`](lora_pipeline.md#StableDiffusionLoraLoaderMixin), [`lora_state_dict`](lora_pipeline.md#StableDiffusionLoraLoaderMixin.lora_state_dict), [`merge_lora`](../models/lora_nnx.md#merge_lora), [`preprocess_wan_lora_dict`](lora_conversion_utils.md#preprocess_wan_lora_dict), [`translate_wan_nnx_path_to_diffusers_lora`](lora_conversion_utils.md#translate_wan_nnx_path_to_diffusers_lora)
- used by: [`LoRABaseMixin`](lora_base.md#LoRABaseMixin), [`load_lora_weights`](lora_base.md#LoRABaseMixin.load_lora_weights)


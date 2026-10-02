---
title: 'Module: src/maxdiffusion/loaders/lora_pipeline.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/loaders/lora_pipeline.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.loaders.lora_pipeline`/
symbols:
  StableDiffusionLoraLoaderMixin._get_lora_layer: StableDiffusionLoraLoaderMixin#_get_lora_layer().
  StableDiffusionLoraLoaderMixin: StableDiffusionLoraLoaderMixin#
  StableDiffusionLoraLoaderMixin.lora_state_dict: StableDiffusionLoraLoaderMixin#lora_state_dict().
  StableDiffusionLoraLoaderMixin._intercept: StableDiffusionLoraLoaderMixin#_intercept().
  StableDiffusionLoraLoaderMixin.load_lora_weights: StableDiffusionLoraLoaderMixin#load_lora_weights().
  StableDiffusionLoraLoaderMixin.make_lora_interceptor: StableDiffusionLoraLoaderMixin#make_lora_interceptor().
  StableDiffusionLoraLoaderMixin.load_lora: StableDiffusionLoraLoaderMixin#load_lora().
  LORA_WEIGHT_NAME_SAFE: LORA_WEIGHT_NAME_SAFE.
  StableDiffusionLoraLoaderMixin.rename_for_interceptor: StableDiffusionLoraLoaderMixin#rename_for_interceptor().
  StableDiffusionLoraLoaderMixin.unet_name: StableDiffusionLoraLoaderMixin#unet_name.
  StableDiffusionLoraLoaderMixin.text_encoder_name: StableDiffusionLoraLoaderMixin#text_encoder_name.
  TEXT_ENCODER_NAME: TEXT_ENCODER_NAME.
  UNET_NAME: UNET_NAME.
  TRANSFORMER_NAME: TRANSFORMER_NAME.
  LORA_WEIGHT_NAME: LORA_WEIGHT_NAME.
  StableDiffusionLoraLoaderMixin._lora_lodable_modules: StableDiffusionLoraLoaderMixin#_lora_lodable_modules.
---
# Module: [`src/maxdiffusion/loaders/lora_pipeline.py`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/lora_pipeline.py)

## Classes
### `StableDiffusionLoraLoaderMixin`  ·  implements/extends LoRABaseMixin
- def: [`src/maxdiffusion/loaders/lora_pipeline.py:35`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/lora_pipeline.py#L35)
- doc: Load LoRA layers into Stable Diffusion \[`UNet2DConditionModel`\] and
- signature: `class StableDiffusionLoraLoaderMixin(LoRABaseMixin):`
- members:
  - `load_lora(cls, state_dict, network_alphas, params, adapter_name=None, _pipeline=None)` — [`L274`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/lora_pipeline.py#L274) — This will load the LoRA layers specified in `state_dict` into `unet`.
  - `load_lora_weights(self, pretrained_model_name_or_path_or_dict: Union[str, Dict[str, jnp.ndarray]], params, adapter_name=None, **kwargs)` — [`L45`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/lora_pipeline.py#L45) — Load LoRA weights specified in `pretrained_model_name_or_path_or_dict` into `self.unet` and
  - `lora_state_dict(cls, pretrained_model_name_or_path: str, **kwargs)` — [`L173`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/lora_pipeline.py#L173) — Return state dict for lora weights and the network alphas.
  - `make_lora_interceptor(cls, params, rank, network_alphas, adapter_name)` — [`L136`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/lora_pipeline.py#L136)
  - `rename_for_interceptor(params_keys, network_alphas, adapter_name)` — [`L122`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/lora_pipeline.py#L122)
  - `text_encoder_name` — [`L43`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/lora_pipeline.py#L43)
  - `unet_name` — [`L42`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/lora_pipeline.py#L42)
- protocol/private: `_get_lora_layer`[`L91`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/lora_pipeline.py#L91), `_intercept`[`L155`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/lora_pipeline.py#L155), `_lora_lodable_modules`[`L41`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/lora_pipeline.py#L41)
- uses (calls/refs, reference-scoped): [`LoRABaseMixin`](lora_base.md#LoRABaseMixin), [`BaseLoRALayer`](../models/lora.md#BaseLoRALayer), [`_convert_non_diffusers_lora_to_diffusers`](lora_conversion_utils.md#_convert_non_diffusers_lora_to_diffusers), [`_fetch_state_dict`](lora_base.md#LoRABaseMixin._fetch_state_dict), [`FlaxStableDiffusionPipeline`](../pipelines/stable_diffusion/pipeline_flax_stable_diffusion.md#FlaxStableDiffusionPipeline), [`FlaxStableDiffusionXLPipeline`](../pipelines/stable_diffusion_xl/pipeline_flax_stable_diffusion_xl.md#FlaxStableDiffusionXLPipeline), [`LoRALinearLayer`](../models/lora.md#LoRALinearLayer), [`rank`](../models/lora.md#LoRALinearLayer.rank), [`out_features`](../models/lora.md#LoRALinearLayer.out_features), [`LoRAConv2DLayer`](../models/lora.md#LoRAConv2DLayer), [`convert_lora_pytorch_state_dict_to_flax`](../models/modeling_flax_pytorch_utils.md#convert_lora_pytorch_state_dict_to_flax), [`dtype`](../models/lora.md#LoRALinearLayer.dtype), [`network_alpha`](../models/lora.md#LoRALinearLayer.network_alpha), [`precision`](../models/lora.md#LoRALinearLayer.precision), [`weights_dtype`](../models/lora.md#LoRALinearLayer.weights_dtype), [`dtype`](../models/lora.md#LoRAConv2DLayer.dtype), [`network_alpha`](../models/lora.md#LoRAConv2DLayer.network_alpha), [`precision`](../models/lora.md#LoRAConv2DLayer.precision), [`rank`](../models/lora.md#LoRAConv2DLayer.rank), [`weights_dtype`](../models/lora.md#LoRAConv2DLayer.weights_dtype), [`_maybe_map_sgm_blocks_to_diffusers`](lora_conversion_utils.md#_maybe_map_sgm_blocks_to_diffusers), [`feature_group_count`](../models/lora.md#LoRAConv2DLayer.feature_group_count), [`input_dilation`](../models/lora.md#LoRAConv2DLayer.input_dilation), [`kernel_dilation`](../models/lora.md#LoRAConv2DLayer.kernel_dilation), [`kernel_size`](../models/lora.md#LoRAConv2DLayer.kernel_size), [`out_features`](../models/lora.md#LoRAConv2DLayer.out_features), [`padding`](../models/lora.md#LoRAConv2DLayer.padding), [`strides`](../models/lora.md#LoRAConv2DLayer.strides), [`TEXT_ENCODER_NAME`](lora_pipeline.md#TEXT_ENCODER_NAME), [`UNET_NAME`](lora_pipeline.md#UNET_NAME)
- used by: [`LoRABaseMixin`](lora_base.md#LoRABaseMixin), [`load_lora_weights`](wan_lora_nnx_loader.md#Wan2_1NNXLoraLoader.load_lora_weights), [`load_lora_weights`](wan_lora_nnx_loader.md#Wan2_2NNXLoraLoader.load_lora_weights), [`load_lora_weights`](lora_base.md#LoRABaseMixin.load_lora_weights), [`FlaxStableDiffusionPipeline`](../pipelines/stable_diffusion/pipeline_flax_stable_diffusion.md#FlaxStableDiffusionPipeline), [`FlaxStableDiffusionXLPipeline`](../pipelines/stable_diffusion_xl/pipeline_flax_stable_diffusion_xl.md#FlaxStableDiffusionXLPipeline)

## Module values
- `LORA_WEIGHT_NAME` — [`L31`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/lora_pipeline.py#L31)
- `LORA_WEIGHT_NAME_SAFE` — [`L32`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/lora_pipeline.py#L32)
- `TEXT_ENCODER_NAME` — [`L27`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/lora_pipeline.py#L27)
- `TRANSFORMER_NAME` — [`L29`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/lora_pipeline.py#L29)
- `UNET_NAME` — [`L28`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/lora_pipeline.py#L28)


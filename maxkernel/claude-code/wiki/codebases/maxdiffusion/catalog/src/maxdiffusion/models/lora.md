---
title: 'Module: src/maxdiffusion/models/lora.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/models/lora.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.models.lora`/
symbols:
  LoRAConv2DLayer.__call__: LoRAConv2DLayer#__call__().
  LoRALinearLayer.__call__: LoRALinearLayer#__call__().
  BaseLoRALayer: BaseLoRALayer#
  LoRALinearLayer: LoRALinearLayer#
  LoRALinearLayer.rank: LoRALinearLayer#rank.
  LoRALinearLayer.out_features: LoRALinearLayer#out_features.
  LoRAConv2DLayer: LoRAConv2DLayer#
  LoRALinearLayer.network_alpha: LoRALinearLayer#network_alpha.
  LoRALinearLayer.dtype: LoRALinearLayer#dtype.
  LoRALinearLayer.weights_dtype: LoRALinearLayer#weights_dtype.
  LoRALinearLayer.precision: LoRALinearLayer#precision.
  LoRAConv2DLayer.rank: LoRAConv2DLayer#rank.
  LoRAConv2DLayer.network_alpha: LoRAConv2DLayer#network_alpha.
  LoRAConv2DLayer.dtype: LoRAConv2DLayer#dtype.
  LoRAConv2DLayer.weights_dtype: LoRAConv2DLayer#weights_dtype.
  LoRAConv2DLayer.precision: LoRAConv2DLayer#precision.
  LoRAConv2DLayer.out_features: LoRAConv2DLayer#out_features.
  LoRAConv2DLayer.kernel_size: LoRAConv2DLayer#kernel_size.
  LoRAConv2DLayer.strides: LoRAConv2DLayer#strides.
  LoRAConv2DLayer.padding: LoRAConv2DLayer#padding.
  LoRAConv2DLayer.input_dilation: LoRAConv2DLayer#input_dilation.
  LoRAConv2DLayer.kernel_dilation: LoRAConv2DLayer#kernel_dilation.
  LoRAConv2DLayer.feature_group_count: LoRAConv2DLayer#feature_group_count.
  LoRALinearLayer.lora_scale: LoRALinearLayer#lora_scale.
  LoRAConv2DLayer.lora_scale: LoRAConv2DLayer#lora_scale.
---
# Module: [`src/maxdiffusion/models/lora.py`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/lora.py)

## Classes
### `BaseLoRALayer`
- def: [`src/maxdiffusion/models/lora.py:23`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/lora.py#L23)
- doc: Base LoRA layer class for all LoRA layer implementation
- signature: `class BaseLoRALayer:`
- uses (calls/refs, reference-scoped): [`LoRALinearLayer`](lora.md#LoRALinearLayer), [`LoRAConv2DLayer`](lora.md#LoRAConv2DLayer)
- used by: [`LoRALinearLayer`](lora.md#LoRALinearLayer), [`_intercept`](../loaders/flux_lora_pipeline.md#FluxLoraLoaderMixin._intercept), [`_intercept`](../loaders/lora_pipeline.md#StableDiffusionLoraLoaderMixin._intercept), [`LoRAConv2DLayer`](lora.md#LoRAConv2DLayer)

### `LoRAConv2DLayer`  ·  implements/extends BaseLoRALayer, Module
- def: [`src/maxdiffusion/models/lora.py:73`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/lora.py#L73)
- doc: Implements LoRA Conv layer
- signature: `class LoRAConv2DLayer(nn.Module, BaseLoRALayer):`
- members:
  - `dtype` — [`L87`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/lora.py#L87)
  - `feature_group_count` — [`L85`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/lora.py#L85)
  - `input_dilation` — [`L83`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/lora.py#L83)
  - `kernel_dilation` — [`L84`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/lora.py#L84)
  - `kernel_size` — [`L80`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/lora.py#L80)
  - `lora_scale` — [`L90`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/lora.py#L90)
  - `network_alpha` — [`L86`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/lora.py#L86)
  - `out_features` — [`L78`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/lora.py#L78)
  - `padding` — [`L82`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/lora.py#L82)
  - `precision` — [`L89`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/lora.py#L89)
  - `rank` — [`L79`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/lora.py#L79)
  - `strides` — [`L81`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/lora.py#L81)
  - `weights_dtype` — [`L88`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/lora.py#L88)
- protocol/private: `__call__`[`L93`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/lora.py#L93)
- uses (calls/refs, reference-scoped): [`BaseLoRALayer`](lora.md#BaseLoRALayer)
- used by: [`_get_lora_layer`](../loaders/lora_pipeline.md#StableDiffusionLoraLoaderMixin._get_lora_layer), [`BaseLoRALayer`](lora.md#BaseLoRALayer)

### `LoRALinearLayer`  ·  implements/extends BaseLoRALayer, Module
- def: [`src/maxdiffusion/models/lora.py:31`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/lora.py#L31)
- doc: Implements LoRA linear layer
- signature: `class LoRALinearLayer(nn.Module, BaseLoRALayer):`
- members:
  - `dtype` — [`L39`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/lora.py#L39)
  - `lora_scale` — [`L42`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/lora.py#L42)
  - `network_alpha` — [`L38`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/lora.py#L38)
  - `out_features` — [`L36`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/lora.py#L36)
  - `precision` — [`L41`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/lora.py#L41)
  - `rank` — [`L37`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/lora.py#L37)
  - `weights_dtype` — [`L40`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/lora.py#L40)
- protocol/private: `__call__`[`L45`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/lora.py#L45)
- uses (calls/refs, reference-scoped): [`BaseLoRALayer`](lora.md#BaseLoRALayer)
- used by: [`_get_lora_layer`](../loaders/lora_pipeline.md#StableDiffusionLoraLoaderMixin._get_lora_layer), [`_get_lora_layer`](../loaders/flux_lora_pipeline.md#FluxLoraLoaderMixin._get_lora_layer), [`BaseLoRALayer`](lora.md#BaseLoRALayer)


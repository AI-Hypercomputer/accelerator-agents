---
title: 'Module: src/maxdiffusion/loaders/flux_lora_pipeline.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/loaders/flux_lora_pipeline.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.loaders.flux_lora_pipeline`/FluxLoraLoaderMixin#
symbols:
  FluxLoraLoaderMixin._get_lora_layer: _get_lora_layer().
  FluxLoraLoaderMixin._intercept: _intercept().
  FluxLoraLoaderMixin.load_lora_weights: load_lora_weights().
  FluxLoraLoaderMixin.make_lora_interceptor: make_lora_interceptor().
  FluxLoraLoaderMixin: ''
  FluxLoraLoaderMixin.lora_state_dict: lora_state_dict().
  FluxLoraLoaderMixin.load_lora: load_lora().
  FluxLoraLoaderMixin.rename_for_interceptor: rename_for_interceptor().
  FluxLoraLoaderMixin._lora_lodable_modules: _lora_lodable_modules.
---
# Module: [`src/maxdiffusion/loaders/flux_lora_pipeline.py`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/flux_lora_pipeline.py)

## Classes
### `FluxLoraLoaderMixin`  ·  implements/extends LoRABaseMixin
- def: [`src/maxdiffusion/loaders/flux_lora_pipeline.py:24`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/flux_lora_pipeline.py#L24)
- signature: `class FluxLoraLoaderMixin(LoRABaseMixin):`
- members:
  - `load_lora(cls, config, state_dict, params, adapter_name=None)` — [`L140`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/flux_lora_pipeline.py#L140)
  - `load_lora_weights(self, config, pretrained_model_name_or_path_or_dict: Union[str, Dict[str, jnp.ndarray]], params, adapter_name=None, **kwargs)` — [`L27`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/flux_lora_pipeline.py#L27)
  - `lora_state_dict(cls, pretrained_model_name_or_path: str, **kwargs)` — [`L99`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/flux_lora_pipeline.py#L99)
  - `make_lora_interceptor(cls, params, rank, network_alphas, adapter_name)` — [`L60`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/flux_lora_pipeline.py#L60)
  - `rename_for_interceptor(params_keys, network_alphas, adapter_name)` — [`L46`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/flux_lora_pipeline.py#L46)
- protocol/private: `_get_lora_layer`[`L84`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/flux_lora_pipeline.py#L84), `_intercept`[`L67`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/flux_lora_pipeline.py#L67), `_lora_lodable_modules`[`L25`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/flux_lora_pipeline.py#L25)
- uses (calls/refs, reference-scoped): [`LoRABaseMixin`](lora_base.md#LoRABaseMixin), [`BaseLoRALayer`](../models/lora.md#BaseLoRALayer), [`_fetch_state_dict`](lora_base.md#LoRABaseMixin._fetch_state_dict), [`LoRALinearLayer`](../models/lora.md#LoRALinearLayer), [`rank`](../models/lora.md#LoRALinearLayer.rank), [`out_features`](../models/lora.md#LoRALinearLayer.out_features), [`convert_flux_lora_pytorch_state_dict_to_flax`](../models/modeling_flax_pytorch_utils.md#convert_flux_lora_pytorch_state_dict_to_flax), [`dtype`](../models/lora.md#LoRALinearLayer.dtype), [`network_alpha`](../models/lora.md#LoRALinearLayer.network_alpha), [`precision`](../models/lora.md#LoRALinearLayer.precision), [`weights_dtype`](../models/lora.md#LoRALinearLayer.weights_dtype)
- used by: [`LoRABaseMixin`](lora_base.md#LoRABaseMixin), [`load_lora_weights`](lora_base.md#LoRABaseMixin.load_lora_weights)


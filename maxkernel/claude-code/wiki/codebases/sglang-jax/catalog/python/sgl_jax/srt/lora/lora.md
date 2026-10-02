---
title: 'Module: python/sgl_jax/srt/lora/lora.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/lora/lora.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.lora.lora`/
symbols:
  LoRAAdapter.initialize_weights: LoRAAdapter#initialize_weights().
  LoRAAdapter.config: LoRAAdapter#config.
  LoRAAdapter.scaling: LoRAAdapter#scaling.
  LoRAAdapter.__init__: LoRAAdapter#__init__().
  LoRAAdapter.layers: LoRAAdapter#layers.
  LoRAAdapter: LoRAAdapter#
  logger: logger.
  LoRAAdapter.load_config: LoRAAdapter#load_config.
  LoRALayer.weights: LoRALayer#weights.
  LoRALayer.__init__: LoRALayer#__init__().
  LoRALayer.config: LoRALayer#config.
  LoRAAdapter.lora_backend: LoRAAdapter#lora_backend.
  LoRALayer: LoRALayer#
  LoRAAdapter.weights: LoRAAdapter#weights.
  LoRALayer.base_hf_config: LoRALayer#base_hf_config.
  LoRAAdapter.uid: LoRAAdapter#uid.
  LoRAAdapter.base_hf_config: LoRAAdapter#base_hf_config.
---
# Module: [`python/sgl_jax/srt/lora/lora.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora.py)

## Classes
### `LoRAAdapter`  ·  implements/extends Module
- def: [`python/sgl_jax/srt/lora/lora.py:47`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora.py#L47)
- signature: `class LoRAAdapter(nnx.Module):`
- members:
  - `initialize_weights(self)` — [`L73`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora.py#L73)
  - `base_hf_config` — [`L61`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora.py#L61)
  - `config` — [`L59`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora.py#L59)
  - `layers` — [`L66`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora.py#L66)
  - `load_config` — [`L62`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora.py#L62)
  - `lora_backend` — [`L63`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora.py#L63)
  - `scaling` — [`L64`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora.py#L64)
  - `uid` — [`L58`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora.py#L58)
  - `weights` — [`L70`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora.py#L70)
- protocol/private: `__init__`[`L49`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora.py#L49)
- uses (calls/refs, reference-scoped): [`_get_weights_iterator`](../model_loader/loader.md#DefaultModelLoader._get_weights_iterator), [`LoadConfig`](../configs/load_config.md#LoadConfig), [`BaseLoRABackend`](backend/base_backend.md#BaseLoRABackend), [`LoRAConfig`](lora_config.md#LoRAConfig), [`DefaultModelLoader`](../model_loader/loader.md#DefaultModelLoader), [`hf_config`](lora_config.md#LoRAConfig.hf_config), [`r`](lora_config.md#LoRAConfig.r), [`lora_alpha`](lora_config.md#LoRAConfig.lora_alpha), [`logger`](lora.md#logger), [`path`](lora_config.md#LoRAConfig.path), [`weights`](lora.md#LoRALayer.weights), [`LoRALayer`](lora.md#LoRALayer), [`Source`](../model_loader/loader.md#DefaultModelLoader.Source), [`revision`](../model_loader/loader.md#DefaultModelLoader.Source.revision), [`fall_back_to_pt`](../model_loader/loader.md#DefaultModelLoader.Source.fall_back_to_pt)
- used by: [`load_lora_weights`](lora_manager.md#LoRAManager.load_lora_weights), [`load_lora_weight_to_buffer`](lora_memory_pool.md#LoRAMemoryPool.load_lora_weight_to_buffer), [`prepare_dynamic_lora_batch`](lora_manager.md#LoRAManager.prepare_dynamic_lora_batch), [`prepare_lora_batch`](lora_memory_pool.md#LoRAMemoryPool.prepare_lora_batch), [`loras`](lora_manager.md#LoRAManager.loras)

### `LoRALayer`  ·  implements/extends Module
- def: [`python/sgl_jax/srt/lora/lora.py:37`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora.py#L37)
- signature: `class LoRALayer(nnx.Module):`
- members:
  - `base_hf_config` — [`L41`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora.py#L41)
  - `config` — [`L40`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora.py#L40)
  - `weights` — [`L44`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora.py#L44)
- protocol/private: `__init__`[`L38`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora.py#L38)
- uses (calls/refs, reference-scoped): [`LoRAConfig`](lora_config.md#LoRAConfig)
- used by: [`load_lora_weights`](lora_manager.md#LoRAManager.load_lora_weights), [`load_lora_weight_to_buffer`](lora_memory_pool.md#LoRAMemoryPool.load_lora_weight_to_buffer), [`initialize_weights`](lora.md#LoRAAdapter.initialize_weights), [`layers`](lora.md#LoRAAdapter.layers)

## Module values
- `logger` — [`L34`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora.py#L34)


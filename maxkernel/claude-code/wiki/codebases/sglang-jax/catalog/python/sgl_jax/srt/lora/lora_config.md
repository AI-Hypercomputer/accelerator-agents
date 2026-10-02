---
title: 'Module: python/sgl_jax/srt/lora/lora_config.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/lora/lora_config.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.lora.lora_config`/LoRAConfig#
symbols:
  LoRAConfig: ''
  LoRAConfig.target_modules: target_modules.
  LoRAConfig.hf_config: hf_config.
  LoRAConfig.r: r.
  LoRAConfig.lora_alpha: lora_alpha.
  LoRAConfig.path: path.
  LoRAConfig.get_lora_config: get_lora_config().
  LoRAConfig.__init__: __init__().
---
# Module: [`python/sgl_jax/srt/lora/lora_config.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_config.py)

## Classes
### `LoRAConfig`
- def: [`python/sgl_jax/srt/lora/lora_config.py:22`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_config.py#L22)
- signature: `class LoRAConfig:`
- members:
  - `get_lora_config(self, dummy=False)` — [`L38`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_config.py#L38)
  - `hf_config` — [`L28`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_config.py#L28)
  - `lora_alpha` — [`L36`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_config.py#L36)
  - `path` — [`L27`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_config.py#L27)
  - `r` — [`L35`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_config.py#L35)
  - `target_modules` — [`L29`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_config.py#L29)
- protocol/private: `__init__`[`L23`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_config.py#L23)
- used by: [`load_lora_config`](lora_manager.md#LoRAManager.load_lora_config), [`initialize_weights`](lora.md#LoRAAdapter.initialize_weights), [`prepare_dynamic_lora_batch`](lora_manager.md#LoRAManager.prepare_dynamic_lora_batch), [`target_modules`](lora_manager.md#LoRAManager.target_modules), [`max_lora_rank`](lora_manager.md#LoRAManager.max_lora_rank), [`can_support`](lora_memory_pool.md#LoRAMemoryPool.can_support), [`config`](lora.md#LoRAAdapter.config), [`scaling`](lora.md#LoRAAdapter.scaling), [`__init__`](lora.md#LoRAAdapter.__init__), [`can_support`](lora_manager.md#LoRAManager.can_support), [`configs`](lora_manager.md#LoRAManager.configs), [`__init__`](lora.md#LoRALayer.__init__), [`config`](lora.md#LoRALayer.config)


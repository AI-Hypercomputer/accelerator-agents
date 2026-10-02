---
title: 'Module: python/sgl_jax/srt/lora/lora_manager.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/lora/lora_manager.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.lora.lora_manager`/
symbols:
  LoRAManager.memory_pool: LoRAManager#memory_pool.
  LoRAManager.load_lora_weights: LoRAManager#load_lora_weights().
  LoRAManager.load_lora_config: LoRAManager#load_lora_config().
  LoRAManager.init_state: LoRAManager#init_state().
  LoRAManager.init_memory_pool: LoRAManager#init_memory_pool().
  LoRAManager.target_modules: LoRAManager#target_modules.
  LoRAManager.prepare_dynamic_lora_batch: LoRAManager#prepare_dynamic_lora_batch().
  LoRAManager.lora_modules: LoRAManager#lora_modules.
  LoRAManager.prepare_lora_batch: LoRAManager#prepare_lora_batch().
  LoRAManager.max_lora_rank: LoRAManager#max_lora_rank.
  LoRAManager.update_lora_info: LoRAManager#update_lora_info().
  LoRAManager.lora_backend: LoRAManager#lora_backend.
  LoRAManager.prepare_static_lora_batch: LoRAManager#prepare_static_lora_batch().
  LoRAManager.tp_size: LoRAManager#tp_size.
  LoRAManager.max_loras_per_batch: LoRAManager#max_loras_per_batch.
  LoRAManager.head_dim: LoRAManager#head_dim.
  logger: logger.
  LoRAManager.apply_lora_surgery: LoRAManager#apply_lora_surgery().
  LoRAManager.num_attention_heads: LoRAManager#num_attention_heads.
  LoRAManager._replace_with_lora: LoRAManager#_replace_with_lora().
  LoRAManager.base_model: LoRAManager#base_model.
  LoRAManager.configs: LoRAManager#configs.
  LoRAManager.can_support: LoRAManager#can_support().
  LoRAManager.num_kv_heads: LoRAManager#num_kv_heads.
  LoRAManager.loras: LoRAManager#loras.
  LoRAManager.original_num_kv_heads: LoRAManager#original_num_kv_heads.
  LoRAManager.num_pinned_loras: LoRAManager#num_pinned_loras.
  LoRAManager.hidden_size: LoRAManager#hidden_size.
  LoRAManager.get_buffer_id: LoRAManager#get_buffer_id().
  LoRAManager.intermediate_size: LoRAManager#intermediate_size.
  LoRAManager.lora_refs: LoRAManager#lora_refs.
  LoRAManager.init_lora_adapters: LoRAManager#init_lora_adapters().
  LoRAManager.num_layers: LoRAManager#num_layers.
  LoRAManager.__init__: LoRAManager#__init__().
  LoRAManager: LoRAManager#
  LoRAManager.base_hf_config: LoRAManager#base_hf_config.
  LoRAManager.mesh: LoRAManager#mesh.
  LoRAManager.static_lora: LoRAManager#static_lora.
  LoRAManager.dtype: LoRAManager#dtype.
  LoRAManager.server_args: LoRAManager#server_args.
  LoRAManager.init_lora_shapes: LoRAManager#init_lora_shapes().
  LoRAManager.has_new_weights: LoRAManager#has_new_weights.
  LoRAManager.model_config: LoRAManager#model_config.
  LoRAManager._get_nested_attr: LoRAManager#_get_nested_attr().
  LoRAManager._set_nested_attr: LoRAManager#_set_nested_attr().
---
# Module: [`python/sgl_jax/srt/lora/lora_manager.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_manager.py)

## Classes
### `LoRAManager`
- def: [`python/sgl_jax/srt/lora/lora_manager.py:37`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_manager.py#L37)
- doc: LoRA manager for JAX-based inference.
- signature: `class LoRAManager:`
- members:
  - `__init__(self, base_model, base_hf_config, max_loras_per_batch: int, dtype: jnp.dtype, mesh: Mesh, max_lora_rank: int | None = None, target_modules: set[str] | None = None, lora_paths: list[LoRARef] | None = None, server_args=None, model_config=None)` — [`L66`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_manager.py#L66) — Initialize LoRA manager.
  - `_get_nested_attr(self, obj, attr_path: str)` — [`L627`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_manager.py#L627) — Get nested attribute using dot notation.
  - `_replace_with_lora(self, parent_module, attr_name: str, full_path: str, layer_idx: int)` — [`L579`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_manager.py#L579) — Replace a Linear layer with LoRALinear wrapper.
  - `_set_nested_attr(self, obj, attr_path: str, value)` — [`L642`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_manager.py#L642) — Set nested attribute using dot notation.
  - `apply_lora_surgery(self)` — [`L483`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_manager.py#L483) — Apply Flax Model Surgery to add LoRA layers to the base model.
  - `can_support(self, config: LoRAConfig)` — [`L407`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_manager.py#L407) — Check if memory pool can support the given LoRA config.
  - `get_buffer_id(self, lora_id: str | None)` — [`L479`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_manager.py#L479) — Get buffer slot ID for a given LoRA adapter ID.
  - `init_lora_adapters(self, lora_paths: list[LoRARef] | None = None)` — [`L183`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_manager.py#L183) — Initialize adapter storage and optionally load adapters.
  - `init_lora_shapes(self, max_lora_rank: int | None = None, target_modules: set[str] | None = None)` — [`L211`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_manager.py#L211) — Infer LoRA target modules and max_lora_rank from loaded adapters if not provided.
  - `init_memory_pool(self)` — [`L263`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_manager.py#L263) — Initialize the LoRA memory pool with proper sharding.
  - `init_state(self, max_lora_rank: int | None = None, target_modules: set[str] | None = None, lora_paths: list[LoRARef] | None = None)` — [`L133`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_manager.py#L133) — Initialize internal state of LoRAManager.
  - `load_lora_config(self, lora_ref: LoRARef)` — [`L320`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_manager.py#L320) — Load only the config of a LoRA adapter (not the weights).
  - `load_lora_weights(self, lora_ref: LoRARef)` — [`L357`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_manager.py#L357) — Load LoRA weights from disk to CPU memory.
  - `prepare_dynamic_lora_batch()` — [`L442`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_manager.py#L442)
  - `prepare_lora_batch(self, model_worker_batch: ModelWorkerBatch)` — [`L411`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_manager.py#L411) — Prepare LoRA batch for inference.
  - `prepare_static_lora_batch()` — [`L434`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_manager.py#L434)
  - `update_lora_info(self)` — [`L298`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_manager.py#L298) — Update all LoRA modules to associate them with the latest memory buffer.
  - `base_hf_config` — [`L95`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_manager.py#L95)
  - `base_model` — [`L94`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_manager.py#L94)
  - `configs` — [`L195`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_manager.py#L195)
  - `dtype` — [`L97`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_manager.py#L97)
  - `has_new_weights` — [`L477`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_manager.py#L477)
  - `head_dim` — [`L110`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_manager.py#L110)
  - `hidden_size` — [`L104`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_manager.py#L104)
  - `intermediate_size` — [`L105`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_manager.py#L105)
  - `lora_backend` — [`L610`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_manager.py#L610)
  - `lora_modules` — [`L517`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_manager.py#L517)
  - `lora_refs` — [`L201`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_manager.py#L201)
  - `loras` — [`L198`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_manager.py#L198)
  - `max_lora_rank` — [`L250`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_manager.py#L250)
  - `max_loras_per_batch` — [`L96`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_manager.py#L96)
  - `memory_pool` — [`L281`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_manager.py#L281)
  - `mesh` — [`L98`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_manager.py#L98)
  - `model_config` — [`L100`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_manager.py#L100)
  - `num_attention_heads` — [`L106`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_manager.py#L106)
  - `num_kv_heads` — [`L109`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_manager.py#L109)
  - `num_layers` — [`L103`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_manager.py#L103)
  - `num_pinned_loras` — [`L204`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_manager.py#L204)
  - `original_num_kv_heads` — [`L117`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_manager.py#L117)
  - `server_args` — [`L99`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_manager.py#L99)
  - `static_lora` — [`L113`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_manager.py#L113)
  - `target_modules` — [`L225`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_manager.py#L225)
  - `tp_size` — [`L120`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_manager.py#L120)
- uses (calls/refs, reference-scoped): `ModelWorkerBatch`, [`init_buffers`](lora_memory_pool.md#LoRAMemoryPool.init_buffers), [`initialize_weights`](lora.md#LoRAAdapter.initialize_weights), [`prepare_lora_batch`](backend/bgmv_backend.md#BgmvLoRABackend.prepare_lora_batch), [`LinearBase`](../layers/linear.md#LinearBase), [`LoRARef`](lora_registry.md#LoRARef), [`prepare_lora_batch`](lora_memory_pool.md#LoRAMemoryPool.prepare_lora_batch), [`can_support`](lora_memory_pool.md#LoRAMemoryPool.can_support), [`config`](lora.md#LoRAAdapter.config), [`LoadConfig`](../configs/load_config.md#LoadConfig), [`lora_id`](lora_registry.md#LoRARef.lora_id), [`LoRAConfig`](lora_config.md#LoRAConfig), [`logger`](lora_manager.md#logger), [`lora_name`](lora_registry.md#LoRARef.lora_name), [`target_modules`](lora_memory_pool.md#LoRAMemoryPool.target_modules), `lora_ids`, [`BaseLayerWithLoRA`](layers.md#BaseLayerWithLoRA), [`target_modules`](lora_config.md#LoRAConfig.target_modules), [`scaling`](lora.md#LoRAAdapter.scaling), [`layers`](lora.md#LoRAAdapter.layers), [`r`](lora_config.md#LoRAConfig.r), [`get_array`](lora_memory_pool.md#LoRAMemoryPool.get_array), [`BgmvLoRABackend`](backend/bgmv_backend.md#BgmvLoRABackend), [`LoRAAdapter`](lora.md#LoRAAdapter), [`pinned`](lora_registry.md#LoRARef.pinned), [`lora_path`](lora_registry.md#LoRARef.lora_path), [`LoRALinear`](layers.md#LoRALinear), [`get_buffer_id`](lora_memory_pool.md#LoRAMemoryPool.get_buffer_id), [`lora_alpha`](lora_config.md#LoRAConfig.lora_alpha), [`set_lora_info`](layers.md#BaseLayerWithLoRA.set_lora_info), [`weights`](lora.md#LoRALayer.weights), [`LoRAMemoryPool`](lora_memory_pool.md#LoRAMemoryPool), [`get_normalized_target_modules`](utils.md#get_normalized_target_modules), [`get_target_module_name`](utils.md#get_target_module_name)
- used by: [`lora_manager`](../model_executor/model_runner.md#ModelRunner.lora_manager), [`prepare_lora_batch`](../managers/tp_worker.md#ModelWorker.prepare_lora_batch), [`init_lora_manager`](../model_executor/model_runner.md#ModelRunner.init_lora_manager)

## Module values
- `logger` — [`L34`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_manager.py#L34)


---
title: 'Module: python/sgl_jax/srt/lora/lora_registry.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/lora/lora_registry.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.lora.lora_registry`/LoRARe
symbols:
  LoRARef: f#
  LoRARegistry._register_adapter: gistry#_register_adapter().
  LoRARegistry._registry: gistry#_registry.
  LoRARegistry._counters: gistry#_counters.
  LoRARegistry.acquire: gistry#acquire().
  LoRARef.lora_id: f#lora_id.
  LoRARef.lora_name: f#lora_name.
  LoRARegistry.register: gistry#register().
  LoRARegistry.unregister: gistry#unregister().
  LoRARegistry.release: gistry#release().
  LoRARegistry.wait_for_unload: gistry#wait_for_unload().
  LoRARegistry._lookup: gistry#_lookup().
  LoRARegistry._registry_lock: gistry#_registry_lock.
  LoRARef.pinned: f#pinned.
  LoRARef.lora_path: f#lora_path.
  LoRARegistry.get_all_adapters: gistry#get_all_adapters().
  LoRARef.__post_init__: f#__post_init__().
  LoRARegistry.__init__: gistry#__init__().
  LoRARegistry.num_registered_loras: gistry#num_registered_loras().
  LoRARef.__str__: f#__str__().
  LoRARegistry: gistry#
---
# Module: [`python/sgl_jax/srt/lora/lora_registry.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_registry.py)

## Classes
### `LoRARef`
- def: [`python/sgl_jax/srt/lora/lora_registry.py:25`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_registry.py#L25) — documented in [python-sgl_jax-srt-server_args](../../../../../concepts/python-sgl_jax-srt-server_args.md)
- doc: Reference record for a LoRA model.
- signature: `class LoRARef:`
- members:
  - `lora_id` — [`L34`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_registry.py#L34)
  - `lora_name` — [`L35`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_registry.py#L35)
  - `lora_path` — [`L36`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_registry.py#L36)
  - `pinned` — [`L37`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_registry.py#L37)
- protocol/private: `__post_init__`[`L39`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_registry.py#L39), `__str__`[`L43`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_registry.py#L43)
- used by: [`load_lora_weights`](lora_manager.md#LoRAManager.load_lora_weights), [`load_lora_config`](lora_manager.md#LoRAManager.load_lora_config), [`init_state`](lora_manager.md#LoRAManager.init_state), [`check_lora_server_args`](../server_args.md#ServerArgs.check_lora_server_args), [`target_modules`](lora_manager.md#LoRAManager.target_modules), [`check_dynamic_lora_args`](../server_args.md#ServerArgs.check_dynamic_lora_args), [`_register_adapter`](lora_registry.md#LoRARegistry._register_adapter), [`_registry`](lora_registry.md#LoRARegistry._registry), [`register`](lora_registry.md#LoRARegistry.register), [`unregister`](lora_registry.md#LoRARegistry.unregister), [`_lookup`](lora_registry.md#LoRARegistry._lookup), [`get_all_adapters`](lora_registry.md#LoRARegistry.get_all_adapters), [`lora_refs`](lora_manager.md#LoRAManager.lora_refs), [`init_lora_adapters`](lora_manager.md#LoRAManager.init_lora_adapters), [`__init__`](lora_manager.md#LoRAManager.__init__), [`__init__`](lora_registry.md#LoRARegistry.__init__)

### `LoRARegistry`
- def: [`python/sgl_jax/srt/lora/lora_registry.py:52`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_registry.py#L52)
- doc: The central registry to keep track of available LoRA adapters and ongoing LoRA requests.
- signature: `class LoRARegistry:`
- members:
  - `_register_adapter(self, lora_ref: LoRARef)` — [`L180`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_registry.py#L180) — Internal helper method to register a LoRA adapter.
  - `acquire(self, lora_name: str | list[str])` — [`L108`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_registry.py#L108) — Queries registry for LoRA IDs based on LoRA names and start tracking the usage of the corresponding LoRA adapters
  - `get_all_adapters(self)` — [`L200`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_registry.py#L200) — Returns a dictionary of all registered LoRA adapters.
  - `num_registered_loras(self)` — [`L194`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_registry.py#L194) — Returns the total number of LoRA adapters currently registered.
  - `register(self, lora_ref: LoRARef)` — [`L81`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_registry.py#L81) — Register a new LoRARef object in the registry.
  - `release(self, lora_id: str | list[str])` — [`L146`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_registry.py#L146) — Decrements the usage counter for a LoRA adapter, indicating that it is no longer in use.
  - `unregister(self, lora_name: str)` — [`L91`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_registry.py#L91) — Unregister a LoRARef object from the registry and returns the removed LoRA ID.
  - `wait_for_unload(self, lora_id: str)` — [`L161`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_registry.py#L161) — Waits until the usage counter for a LoRA adapter reaches zero, indicating that it is no longer in use.
- protocol/private: `__init__`[`L61`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_registry.py#L61), `_counters`[`L74`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_registry.py#L74), `_lookup`[`L114`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_registry.py#L114), `_registry`[`L72`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_registry.py#L72), `_registry_lock`[`L70`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/lora/lora_registry.py#L70)
- uses (calls/refs, reference-scoped): [`LoRARef`](lora_registry.md#LoRARef), [`lora_id`](lora_registry.md#LoRARef.lora_id), [`lora_name`](lora_registry.md#LoRARef.lora_name), [`decrement`](../utils/common_utils.md#ConcurrentCounter.decrement), [`increment`](../utils/common_utils.md#ConcurrentCounter.increment), [`RWLock`](../utils/common_utils.md#RWLock), [`reader_lock`](../utils/common_utils.md#RWLock.reader_lock), [`writer_lock`](../utils/common_utils.md#RWLock.writer_lock), [`ConcurrentCounter`](../utils/common_utils.md#ConcurrentCounter), [`wait_for_zero`](../utils/common_utils.md#ConcurrentCounter.wait_for_zero)


---
title: 'Module: easydel/inference/esurge/runners/executors/sampler_executor.py'
type: catalog
provenance: extracted
module: easydel/inference/esurge/runners/executors/sampler_executor.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.esurge.runners.executors.sampler_executor`/SamplerExecutor#
symbols:
  SamplerExecutor.compile: compile().
  SamplerExecutor._empty_sharding: _empty_sharding.
  SamplerExecutor._sampling_fn: _sampling_fn().
  SamplerExecutor._cache: _cache.
  SamplerExecutor.get_compiled: get_compiled().
  SamplerExecutor._cache_put: _cache_put().
  SamplerExecutor.cache_keys: cache_keys().
  SamplerExecutor.has: has().
  SamplerExecutor.model: model.
  SamplerExecutor.clear_cache: clear_cache().
  SamplerExecutor._cache_get: _cache_get().
  SamplerExecutor._build_sampling_fn: _build_sampling_fn().
  SamplerExecutor: ''
  SamplerExecutor.use_aot_forward: use_aot_forward.
  SamplerExecutor.__init__: __init__().
  SamplerExecutor.max_model_len: max_model_len.
---
# Module: [`easydel/inference/esurge/runners/executors/sampler_executor.py`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/sampler_executor.py)

## Classes
### `SamplerExecutor`
- def: [`easydel/inference/esurge/runners/executors/sampler_executor.py:86`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/sampler_executor.py#L86)
- doc: Compile, cache, and execute the token sampling step.
- signature: `class SamplerExecutor:`
- members:
  - `__init__(self, *, model: EasyDeLBaseModule, max_model_len: int, empty_sharding: jax.sharding.Sharding, use_aot_forward: bool, cache_capacity: int = 64)` — [`L126`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/sampler_executor.py#L126) — Initialize the SamplerExecutor.
  - `_build_sampling_fn(self)` — [`L310`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/sampler_executor.py#L310) — Build the JIT-compiled sampling function.
  - `_cache_get(self, key: tuple[int, int, str, str])` — [`L179`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/sampler_executor.py#L179) — Retrieve a compiled function from the cache.
  - `_cache_put(self, key: tuple[int, int, str, str], value: tp.Any)` — [`L168`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/sampler_executor.py#L168) — Add a compiled function to the cache.
  - `cache_keys(self)` — [`L199`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/sampler_executor.py#L199) — Get all keys currently in the cache.
  - `clear_cache(self)` — [`L159`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/sampler_executor.py#L159) — Clear all cached compiled functions.
  - `compile(self, *, num_tokens: int, padded_num_reqs: int, inputs: StepFunctionInputs, metadata)` — [`L236`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/sampler_executor.py#L236) — Compile and cache a sampler function for specific dimensions.
  - `get_compiled(self, *, num_tokens: int, padded_num_reqs: int)` — [`L218`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/sampler_executor.py#L218) — Retrieve a pre-compiled sampler function for given dimensions.
  - `has(self, key: tuple[int, int, str, str])` — [`L207`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/sampler_executor.py#L207) — Check if a key exists in the cache.
  - `max_model_len` — [`L151`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/sampler_executor.py#L151)
  - `model` — [`L150`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/sampler_executor.py#L150)
  - `use_aot_forward` — [`L153`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/sampler_executor.py#L153)
- protocol/private: `_cache`[`L157`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/sampler_executor.py#L157), `_empty_sharding`[`L152`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/sampler_executor.py#L152), `_sampling_fn`[`L156`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/sampler_executor.py#L156), `_sampling_fn`[`L345`](../../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/executors/sampler_executor.py#L345)
- uses (calls/refs, reference-scoped): [`config`](../../../../infra/base_module.md#EasyDeLBaseModule.config), [`EasyDeLBaseModule`](../../../../infra/base_module.md#EasyDeLBaseModule), [`dtype`](../../../../infra/base_module.md#EasyDeLBaseModule.dtype), [`sample_tokens`](../../core/sampler.md#sample_tokens), [`StepFunctionInputs`](../execution_types.md#StepFunctionInputs), [`SamplingMetadata`](../../core/sampling_metadata.md#SamplingMetadata), [`apply_history_penalties_from_counts`](../../core/sampler.md#apply_history_penalties_from_counts), [`req_num_tokens_full`](../execution_types.md#StepFunctionInputs.req_num_tokens_full), [`rng_key`](../execution_types.md#StepFunctionInputs.rng_key), [`update_token_counts`](../../core/sampler.md#update_token_counts)
- used by: [`_compile_sampler_variant`](../execution_manager.md#ExecutionManager._compile_sampler_variant), [`update_graphs`](../execution_manager.md#ExecutionManager.update_graphs), [`release_model_state`](../model_runner.md#eSurgeRunner.release_model_state), [`get_compiled_key`](../execution_manager.md#ExecutionManager.get_compiled_key), [`sample_tokens`](../execution_manager.md#ExecutionManager.sample_tokens), [`_sampler_executor`](../execution_manager.md#ExecutionManager._sampler_executor), [`clear_cache`](../execution_manager.md#ExecutionManager.clear_cache), [`has_compiled_variants`](../execution_manager.md#ExecutionManager.has_compiled_variants)


---
title: 'Module: python/sgl_jax/srt/model_executor/model_runner_kv_cache_mixin.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/model_executor/model_runner_kv_cache_mixin.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.model_executor.model_runner_kv_cache_mixin`/
symbols:
  ModelRunnerKVCacheMixin._init_pools: ModelRunnerKVCacheMixin#_init_pools().
  ModelRunnerKVCacheMixin.init_memory_pool: ModelRunnerKVCacheMixin#init_memory_pool().
  ModelRunnerKVCacheMixin._compute_cell_size: ModelRunnerKVCacheMixin#_compute_cell_size().
  _build_hybrid_pools: _build_hybrid_pools().
  ModelRunnerKVCacheMixin.handle_recurrent_cache: ModelRunnerKVCacheMixin#handle_recurrent_cache().
  ModelRunnerKVCacheMixin._resolve_max_num_reqs: ModelRunnerKVCacheMixin#_resolve_max_num_reqs().
  _per_req_state_bytes_from_config: _per_req_state_bytes_from_config().
  ModelRunnerKVCacheMixin.linear_recurrent_config: ModelRunnerKVCacheMixin#linear_recurrent_config().
  ModelRunnerKVCacheMixin._maybe_wrap_hybrid_kv_pool: ModelRunnerKVCacheMixin#_maybe_wrap_hybrid_kv_pool().
  _linear_state_params_from_config: _linear_state_params_from_config().
  ModelRunnerKVCacheMixin.qwen3_5_hybrid_config: ModelRunnerKVCacheMixin#qwen3_5_hybrid_config().
  ModelRunnerKVCacheMixin._profile_available_bytes: ModelRunnerKVCacheMixin#_profile_available_bytes().
  ModelRunnerKVCacheMixin._init_kv_cache_dtype: ModelRunnerKVCacheMixin#_init_kv_cache_dtype().
  ModelRunnerKVCacheMixin.kimi_linear_config: ModelRunnerKVCacheMixin#kimi_linear_config().
  ModelRunnerKVCacheMixin.lightning_config: ModelRunnerKVCacheMixin#lightning_config().
  ModelRunnerKVCacheMixin._kv_pool_layer_count: ModelRunnerKVCacheMixin#_kv_pool_layer_count().
  ModelRunnerKVCacheMixin.profile_max_num_token: ModelRunnerKVCacheMixin#profile_max_num_token().
  ModelRunnerKVCacheMixin._apply_token_constraints: ModelRunnerKVCacheMixin#_apply_token_constraints().
  logger: logger.
  _build_non_hybrid_memory_pools: _build_non_hybrid_memory_pools().
  ModelRunnerKVCacheMixin.align128: ModelRunnerKVCacheMixin#align128().
  ModelRunnerKVCacheMixin: ModelRunnerKVCacheMixin#
  _compute_recurrent_per_req_bytes: _compute_recurrent_per_req_bytes().
  _split_state_kv_budget: _split_state_kv_budget().
  _enforce_recurrent_state_server_constraints: _enforce_recurrent_state_server_constraints().
  _recurrent_slot_factor: _recurrent_slot_factor().
---
# Module: [`python/sgl_jax/srt/model_executor/model_runner_kv_cache_mixin.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner_kv_cache_mixin.py)

## Classes
### `ModelRunnerKVCacheMixin`
- def: [`python/sgl_jax/srt/model_executor/model_runner_kv_cache_mixin.py:204`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner_kv_cache_mixin.py#L204)
- signature: `class ModelRunnerKVCacheMixin:`
- members:
  - `_apply_token_constraints(self: ModelRunner, token_capacity: int, max_total_tokens: int | None, dp_size: int)` — [`L360`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner_kv_cache_mixin.py#L360) — Apply external constraints to token capacity.
  - `_compute_cell_size(self: ModelRunner)` — [`L206`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner_kv_cache_mixin.py#L206) — Per-token KV cache cost in bytes per device, summed across layers. — documented in [python-sgl_jax-srt-configs-model_config](../../../../../concepts/python-sgl_jax-srt-configs-model_config.md)
  - `_init_kv_cache_dtype(self: ModelRunner)` — [`L350`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner_kv_cache_mixin.py#L350) — Resolve kv_cache_dtype from server_args.
  - `_init_pools(self: ModelRunner, max_num_reqs: int, dp_size: int)` — [`L464`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner_kv_cache_mixin.py#L464) — Create ReqToTokenPool, KV pool, allocator, and MemoryPools. — documented in [python-sgl_jax-srt-configs-model_config](../../../../../concepts/python-sgl_jax-srt-configs-model_config.md)
  - `_kv_pool_layer_count(self: ModelRunner)` — [`L685`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner_kv_cache_mixin.py#L685) — Layer count for KV pool sizing.
  - `_maybe_wrap_hybrid_kv_pool(self: ModelRunner, token_to_kv_pool_class: type, **kvcache_kwargs)` — [`L428`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner_kv_cache_mixin.py#L428) — Wrap KV pool with HybridLinearKVPool if has_recurrent_state.
  - `_profile_available_bytes(self: ModelRunner, total_device_memory: int)` — [`L261`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner_kv_cache_mixin.py#L261) — Profile available bytes for KV cache (+ recurrent state).
  - `_resolve_max_num_reqs(self: ModelRunner, max_num_reqs: int | None)` — [`L396`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner_kv_cache_mixin.py#L396) — Compute max concurrent requests. — documented in [python-sgl_jax-srt-model_executor-model_runner](../../../../../concepts/python-sgl_jax-srt-model_executor-model_runner.md)
  - `align128(x: int)` — [`L209`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner_kv_cache_mixin.py#L209)
  - `handle_recurrent_cache(self: ModelRunner, total_rest_memory: int)` — [`L273`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner_kv_cache_mixin.py#L273) — Split HBM between recurrent state and KV cache.
  - `init_memory_pool(self: ModelRunner, max_num_reqs: int | None = None, max_total_tokens: int | None = None, total_device_memory: int | None = None, dp_size: int = 1)` — [`L588`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner_kv_cache_mixin.py#L588) — Initialize memory pool for KV cache (+ recurrent state if hybrid). — documented in [python-sgl_jax-srt-model_executor-model_runner](../../../../../concepts/python-sgl_jax-srt-model_executor-model_runner.md)
  - `kimi_linear_config(self: ModelRunner)` — [`L659`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner_kv_cache_mixin.py#L659)
  - `lightning_config(self: ModelRunner)` — [`L665`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner_kv_cache_mixin.py#L665)
  - `linear_recurrent_config(self: ModelRunner)` — [`L677`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner_kv_cache_mixin.py#L677) — Return linear recurrent config if the model has linear attention, else None.
  - `profile_max_num_token(self: ModelRunner, total_device_memory: int)` — [`L334`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner_kv_cache_mixin.py#L334) — Profile the maximum number of tokens that can fit in memory.
  - `qwen3_5_hybrid_config(self: ModelRunner)` — [`L671`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner_kv_cache_mixin.py#L671)
- uses (calls/refs, reference-scoped): [`hf_config`](../configs/model_config.md#ModelConfig.hf_config), [`model_config`](model_runner.md#ModelRunner.model_config), [`server_args`](model_runner.md#ModelRunner.server_args), [`hf_text_config`](../configs/model_config.md#ModelConfig.hf_text_config), [`_build_hybrid_pools`](model_runner_kv_cache_mixin.md#_build_hybrid_pools), [`set_num_token_hybrid`](model_runner.md#ModelRunner.set_num_token_hybrid), [`ModelRunner`](model_runner.md#ModelRunner), [`ReqToTokenPool`](../mem_cache/memory_pool.md#ReqToTokenPool), [`_per_req_state_bytes_from_config`](model_runner_kv_cache_mixin.md#_per_req_state_bytes_from_config), [`max_total_num_tokens`](model_runner.md#ModelRunner.max_total_num_tokens), [`mesh`](model_runner.md#ModelRunner.mesh), [`adjust_layer_num`](model_runner.md#ModelRunner.adjust_layer_num), [`is_none`](../speculative/spec_info.md#SpeculativeAlgorithm.is_none), [`swa_max_total_num_tokens`](model_runner.md#ModelRunner.swa_max_total_num_tokens), [`attention_tp_size`](model_runner.md#ModelRunner.attention_tp_size), [`attn_backend`](model_runner.md#ModelRunner.attn_backend), [`get_available_device_memory`](model_runner.md#ModelRunner.get_available_device_memory), [`SWATokenToKVPoolAllocator`](../mem_cache/allocator.md#SWATokenToKVPoolAllocator), [`get_num_kv_heads`](../configs/model_config.md#ModelConfig.get_num_kv_heads), [`head_dim`](../configs/model_config.md#ModelConfig.head_dim), [`page_size`](model_runner.md#ModelRunner.page_size), [`use_mla_backend`](model_runner.md#ModelRunner.use_mla_backend), [`max_recurrent_state_size`](../server_args.md#ServerArgs.max_recurrent_state_size), [`page_size`](../server_args.md#ServerArgs.page_size), [`context_len`](../configs/model_config.md#ModelConfig.context_len), [`spec_algorithm`](model_runner.md#ModelRunner.spec_algorithm), [`get_num_kv_heads_by_tp`](../utils/jax_utils.md#get_num_kv_heads_by_tp), [`max_running_requests`](../server_args.md#ServerArgs.max_running_requests), [`MHATokenToKVPool`](../mem_cache/memory_pool.md#MHATokenToKVPool), [`get_bailing_hybrid_config`](../configs/bailing_hybrid.md#get_bailing_hybrid_config), [`full_max_total_num_tokens`](model_runner.md#ModelRunner.full_max_total_num_tokens), [`req_to_token_pool`](model_runner.md#ModelRunner.req_to_token_pool), [`token_to_kv_pool_allocator`](model_runner.md#ModelRunner.token_to_kv_pool_allocator), [`full_attention_layer_ids`](../configs/kimi_linear.md#KimiLinearConfig.full_attention_layer_ids), [`is_hybrid`](model_runner.md#ModelRunner.is_hybrid), [`SWAKVPool`](../mem_cache/memory_pool.md#SWAKVPool), [`get_total_num_kv_heads_with_replication`](../configs/model_config.md#ModelConfig.get_total_num_kv_heads_with_replication), [`dtype`](model_runner.md#ModelRunner.dtype), [`get_hybrid_layer_counts`](../configs/model_config.md#ModelConfig.get_hybrid_layer_counts), [`get_kimi_linear_config`](../configs/kimi_linear.md#get_kimi_linear_config)  (+21 more)
- used by: [`compilation_manager`](../managers/tp_worker.md#ModelWorker.compilation_manager), [`attn_backend_wrapper`](../layers/attention/hybrid_linear_attn_backend.md#attn_backend_wrapper), [`ModelRunner`](model_runner.md#ModelRunner), [`_sampler_step`](model_runner.md#ModelRunner._sampler_step)

## Functions
- `_build_hybrid_pools(cfg, max_num_reqs: int, max_context_len: int, tp_size: int, token_to_kv_pool, mesh, dp_size: int = 1, state_size: int | None = None)` — [`L132`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner_kv_cache_mixin.py#L132) — Build RecurrentStatePool + HybridReqToTokenPool + MemoryPools.
- `_build_non_hybrid_memory_pools(token_to_kv_pool)` — [`L194`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner_kv_cache_mixin.py#L194) — Wrap a single KV pool in MemoryPools.
- `_compute_recurrent_per_req_bytes(num_layers: int, num_heads: int, head_dim: int, conv_kernel_size: int, tp_size: int, temporal_dtype_bytes: int, conv_dtype_bytes: int, num_k_heads: int | None = None, head_k_dim: int | None = None)` — [`L32`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner_kv_cache_mixin.py#L32) — Per-device per-request recurrent + conv buffer size in bytes.
- `_enforce_recurrent_state_server_constraints(server_args)` — [`L113`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner_kv_cache_mixin.py#L113) — Assert server constraints for hybrid recurrent state models.
- `_linear_state_params_from_config(cfg)` — [`L77`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner_kv_cache_mixin.py#L77)
- `_per_req_state_bytes_from_config(cfg, tp_size: int)` — [`L97`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner_kv_cache_mixin.py#L97) — Per-request recurrent + conv state bytes for a hybrid recurrent model.
- `_recurrent_slot_factor(server_args)` — [`L124`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner_kv_cache_mixin.py#L124) — Slots per concurrent request: 2 under radix caching (running slot +
- `_split_state_kv_budget(available_bytes: int, ratio: float, per_req_state_bytes: int)` — [`L58`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner_kv_cache_mixin.py#L58) — Split available HBM into (state_max_reqs, kv_budget).

## Module values
- `logger` — [`L24`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner_kv_cache_mixin.py#L24)


---
title: 'Module: easydel/inference/esurge/config.py'
type: catalog
provenance: extracted
module: easydel/inference/esurge/config.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.esurge.config`/
symbols:
  SchedulerConfig.__post_init__: SchedulerConfig#__post_init__().
  SchedulerConfig.long_prefill_token_threshold: SchedulerConfig#long_prefill_token_threshold.
  SchedulerConfig.max_num_batched_tokens: SchedulerConfig#max_num_batched_tokens.
  Config: Config#
  SchedulerConfig.max_model_len: SchedulerConfig#max_model_len.
  SchedulerConfig.max_num_seq_buckets: SchedulerConfig#max_num_seq_buckets.
  SpeculativeConfig.use_eagle: SpeculativeConfig#use_eagle().
  CacheConfig.num_pages: CacheConfig#num_pages.
  CacheConfig.__post_init__: CacheConfig#__post_init__().
  Config.scheduler_config: Config#scheduler_config.
  Config.cache_config: Config#cache_config.
  SchedulerConfig: SchedulerConfig#
  SchedulerConfig.max_num_seqs: SchedulerConfig#max_num_seqs.
  SchedulerConfig.token_safety_margin: SchedulerConfig#token_safety_margin.
  CacheConfig: CacheConfig#
  CacheConfig.page_size: CacheConfig#page_size.
  SchedulerConfig.policy: SchedulerConfig#policy.
  SchedulerConfig.async_scheduling: SchedulerConfig#async_scheduling.
  CacheConfig.enable_prefix_caching: CacheConfig#enable_prefix_caching.
  SpeculativeConfig.num_speculative_tokens: SpeculativeConfig#num_speculative_tokens.
  SchedulerConfig.chunked_prefill_enabled: SchedulerConfig#chunked_prefill_enabled.
  SpeculativeConfig: SpeculativeConfig#
  SpeculativeConfig.speculative_model: SpeculativeConfig#speculative_model.
  Config.speculative_config: Config#speculative_config.
  LONG_PREFILL_TRS.LONG_PREFILL_TRS: LONG_PREFILL_TRS.LONG_PREFILL_TRS.
---
# Module: [`easydel/inference/esurge/config.py`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/config.py)

## Classes
### `CacheConfig`
- def: [`easydel/inference/esurge/config.py:175`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/config.py#L175)
- doc: Configuration for the KV (key-value) cache.
- signature: `class CacheConfig:`
- members:
  - `__post_init__(self)` — [`L215`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/config.py#L215) — Validate configuration parameters.
  - `enable_prefix_caching` — [`L212`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/config.py#L212) — ---
  - `num_pages` — [`L206`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/config.py#L206) — ---
  - `page_size` — [`L209`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/config.py#L209) — ---
- used by: [`from_runner`](scheduler/scheduler.md#Scheduler.from_runner), [`kv_cache_manager`](scheduler/scheduler.md#Scheduler.kv_cache_manager), [`page_size`](scheduler/scheduler.md#Scheduler.page_size), [`data_parallel_size`](scheduler/scheduler.md#Scheduler.data_parallel_size), [`cache_config`](config.md#Config.cache_config)

### `Config`
- def: [`easydel/inference/esurge/config.py:272`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/config.py#L272)
- doc: Unified configuration for the eSurge engine.
- signature: `class Config:`
- members:
  - `cache_config` — [`L310`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/config.py#L310) — ---
  - `scheduler_config` — [`L307`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/config.py#L307) — ---
  - `speculative_config` — [`L313`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/config.py#L313) — ---
- uses (calls/refs, reference-scoped): [`CacheConfig`](config.md#CacheConfig), [`SchedulerConfig`](config.md#SchedulerConfig), [`SpeculativeConfig`](config.md#SpeculativeConfig)
- used by: [`from_runner`](scheduler/scheduler.md#Scheduler.from_runner), [`scheduler_config`](scheduler/scheduler.md#Scheduler.scheduler_config), [`cache_config`](scheduler/scheduler.md#Scheduler.cache_config), [`__init__`](scheduler/scheduler.md#Scheduler.__init__), [`finished_recving_kv_req_ids`](scheduler/scheduler.md#Scheduler.finished_recving_kv_req_ids)

### `SchedulerConfig`
- def: [`easydel/inference/esurge/config.py:65`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/config.py#L65)
- doc: Configuration for the request scheduler.
- signature: `class SchedulerConfig:`
- members:
  - `__post_init__(self)` — [`L134`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/config.py#L134) — Validate configuration parameters.
  - `async_scheduling` — [`L131`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/config.py#L131) — ---
  - `chunked_prefill_enabled` — [`L122`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/config.py#L122) — ---
  - `long_prefill_token_threshold` — [`L119`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/config.py#L119) — ---
  - `max_model_len` — [`L113`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/config.py#L113) — ---
  - `max_num_batched_tokens` — [`L110`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/config.py#L110) — ---
  - `max_num_seq_buckets` — [`L128`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/config.py#L128) — ---
  - `max_num_seqs` — [`L107`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/config.py#L107) — ---
  - `policy` — [`L116`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/config.py#L116) — ---
  - `token_safety_margin` — [`L125`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/config.py#L125) — ---
- uses (calls/refs, reference-scoped): [`LONG_PREFILL_TRS`](config.md#LONG_PREFILL_TRS.LONG_PREFILL_TRS)
- used by: [`schedule`](scheduler/scheduler.md#Scheduler.schedule), [`from_runner`](scheduler/scheduler.md#Scheduler.from_runner), [`kv_cache_manager`](scheduler/scheduler.md#Scheduler.kv_cache_manager), [`policy`](scheduler/scheduler.md#Scheduler.policy), [`max_num_scheduled_tokens`](scheduler/scheduler.md#Scheduler.max_num_scheduled_tokens), [`requests`](scheduler/scheduler.md#Scheduler.requests), [`max_model_len`](scheduler/scheduler.md#Scheduler.max_model_len), [`max_num_running_reqs`](scheduler/scheduler.md#Scheduler.max_num_running_reqs), [`page_size`](scheduler/scheduler.md#Scheduler.page_size), [`scheduler_config`](config.md#Config.scheduler_config)

### `SpeculativeConfig`
- def: [`easydel/inference/esurge/config.py:229`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/config.py#L229)
- doc: Configuration for speculative decoding.
- signature: `class SpeculativeConfig:`
- members:
  - `use_eagle(self)` — [`L261`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/config.py#L261) — Check if Eagle speculative decoding is enabled.
  - `num_speculative_tokens` — [`L255`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/config.py#L255) — ---
  - `speculative_model` — [`L258`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/config.py#L258) — ---
- used by: [`num_lookahead_tokens`](scheduler/scheduler.md#Scheduler.num_lookahead_tokens), [`Config`](config.md#Config)

## Module values
- `LONG_PREFILL_TRS` — [`L61`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/config.py#L61)


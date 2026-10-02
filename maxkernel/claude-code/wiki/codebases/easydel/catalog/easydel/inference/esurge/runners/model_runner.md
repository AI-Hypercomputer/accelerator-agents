---
title: 'Module: easydel/inference/esurge/runners/model_runner.py'
type: catalog
provenance: extracted
module: easydel/inference/esurge/runners/model_runner.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.esurge.runners.model_runner`/
symbols:
  eSurgeRunner._execute_model_impl: eSurgeRunner#_execute_model_impl().
  eSurgeRunner._update_states: eSurgeRunner#_update_states().
  eSurgeRunner.sequence_buffer: eSurgeRunner#sequence_buffer.
  eSurgeRunner._precompute_vlm_prefill: eSurgeRunner#_precompute_vlm_prefill().
  eSurgeRunner.metadata: eSurgeRunner#metadata.
  eSurgeRunner._get_window_state_views: eSurgeRunner#_get_window_state_views().
  logger: logger.
  eSurgeRunner.requests: eSurgeRunner#requests.
  eSurgeRunner.model: eSurgeRunner#model.
  eSurgeRunner._modify_prev_results: eSurgeRunner#_modify_prev_results().
  eSurgeRunner._log_startup_summary: eSurgeRunner#_log_startup_summary().
  eSurgeRunner.initialize_kv_cache: eSurgeRunner#initialize_kv_cache().
  _AsyncExecutionHandle.get_output: _AsyncExecutionHandle#get_output().
  eSurgeRunner._apply_window_aware_runtime_cap: eSurgeRunner#_apply_window_aware_runtime_cap().
  eSurgeRunner.compile: eSurgeRunner#compile().
  eSurgeRunner.executor_manager: eSurgeRunner#executor_manager.
  eSurgeRunner.release_model_state: eSurgeRunner#release_model_state().
  eSurgeRunner._precompile_jitted_helpers: eSurgeRunner#_precompile_jitted_helpers().
  eSurgeRunner._reorder_decode_first_per_shard: eSurgeRunner#_reorder_decode_first_per_shard().
  eSurgeRunner.update_model_weights: eSurgeRunner#update_model_weights().
  eSurgeRunner.max_num_reqs: eSurgeRunner#max_num_reqs.
  eSurgeRunner._update_placeholder: eSurgeRunner#_update_placeholder().
  eSurgeRunner._reorder_decode_first: eSurgeRunner#_reorder_decode_first().
  eSurgeRunner.max_model_len: eSurgeRunner#max_model_len.
  eSurgeRunner._build_kv_cache_groups: eSurgeRunner#_build_kv_cache_groups().
  eSurgeRunner._window_row_indices_cpu: eSurgeRunner#_window_row_indices_cpu.
  eSurgeRunner.num_reqs_max_model_len: eSurgeRunner#num_reqs_max_model_len.
  eSurgeRunner.num_tokens_paddings: eSurgeRunner#num_tokens_paddings.
  eSurgeRunner.max_pages_per_req: eSurgeRunner#max_pages_per_req.
  eSurgeRunner.max_num_tokens: eSurgeRunner#max_num_tokens.
  eSurgeRunner._executor: eSurgeRunner#_executor.
  eSurgeRunner._finalize_sync_runner_state: eSurgeRunner#_finalize_sync_runner_state().
  eSurgeRunner.reset_state: eSurgeRunner#reset_state().
  eSurgeRunner._empty_sharding: eSurgeRunner#_empty_sharding().
  eSurgeRunner.active_num_seq_buckets: eSurgeRunner#active_num_seq_buckets.
  eSurgeRunner._pre_async_results: eSurgeRunner#_pre_async_results.
  eSurgeRunner.max_num_seq_buckets: eSurgeRunner#max_num_seq_buckets.
  eSurgeRunner._get_full_attention_page_table_index: eSurgeRunner#_get_full_attention_page_table_index().
  eSurgeRunner._get_vlm_cpu_buffers: eSurgeRunner#_get_vlm_cpu_buffers().
  eSurgeRunner.execute_model: eSurgeRunner#execute_model().
  eSurgeRunner: eSurgeRunner#
  eSurgeRunner._allow_sparse_window_packing: eSurgeRunner#_allow_sparse_window_packing.
  eSurgeRunner.destroy_kv_cache: eSurgeRunner#destroy_kv_cache().
  eSurgeRunner.execute_model_async: eSurgeRunner#execute_model_async().
  eSurgeRunner.wait_for_execution: eSurgeRunner#wait_for_execution().
  eSurgeRunner.input_ids_buf: eSurgeRunner#input_ids_buf.
  eSurgeRunner.position_ids_buf: eSurgeRunner#position_ids_buf.
  eSurgeRunner.req_num_tokens_full_buf: eSurgeRunner#req_num_tokens_full_buf.
  eSurgeRunner.kv_cache_groups: eSurgeRunner#kv_cache_groups.
  _AsyncExecutionHandle: _AsyncExecutionHandle#
  eSurgeRunner.log_it: eSurgeRunner#log_it.
  eSurgeRunner._perf_tps_ema: eSurgeRunner#_perf_tps_ema.
  eSurgeRunner.mesh: eSurgeRunner#mesh().
  eSurgeRunner._window_temperature_cpu: eSurgeRunner#_window_temperature_cpu.
  eSurgeRunner._window_top_p_cpu: eSurgeRunner#_window_top_p_cpu.
  eSurgeRunner._window_top_k_cpu: eSurgeRunner#_window_top_k_cpu.
  eSurgeRunner._window_min_p_cpu: eSurgeRunner#_window_min_p_cpu.
  eSurgeRunner._window_frequency_penalties_cpu: eSurgeRunner#_window_frequency_penalties_cpu.
  eSurgeRunner._window_presence_penalties_cpu: eSurgeRunner#_window_presence_penalties_cpu.
  eSurgeRunner._window_repetition_penalties_cpu: eSurgeRunner#_window_repetition_penalties_cpu.
  eSurgeRunner._collect_schedulable_window_rows: eSurgeRunner#_collect_schedulable_window_rows().
  eSurgeRunner._find_reuse_index_in_shard: eSurgeRunner#_find_reuse_index_in_shard().
  eSurgeRunner.shutdown: eSurgeRunner#shutdown().
  _AsyncExecutionHandle._resolved_output: _AsyncExecutionHandle#_resolved_output.
  _AsyncExecutionHandle.__init__: _AsyncExecutionHandle#__init__().
  eSurgeRunner.window_aware_runtime_estimate: eSurgeRunner#window_aware_runtime_estimate.
  eSurgeRunner.page_size: eSurgeRunner#page_size.
  eSurgeRunner.num_tokens_paddings_arr: eSurgeRunner#num_tokens_paddings_arr.
  eSurgeRunner.scheduled_full_buf: eSurgeRunner#scheduled_full_buf.
  eSurgeRunner.active_mask_full_buf: eSurgeRunner#active_mask_full_buf.
  eSurgeRunner.initialize_async_executor: eSurgeRunner#initialize_async_executor().
  eSurgeRunner.min_input_pad: eSurgeRunner#min_input_pad.
  eSurgeRunner._scheduled_full_cpu: eSurgeRunner#_scheduled_full_cpu.
  eSurgeRunner._active_mask_full_cpu: eSurgeRunner#_active_mask_full_cpu.
  eSurgeRunner._perf_history: eSurgeRunner#_perf_history.
  eSurgeRunner._clear_window_aware_runtime_cap_metadata: eSurgeRunner#_clear_window_aware_runtime_cap_metadata().
  eSurgeRunner._get_token_paddings: eSurgeRunner#_get_token_paddings().
  eSurgeRunner._init_seq_buckets: eSurgeRunner#_init_seq_buckets().
  eSurgeRunner._get_current_bucket: eSurgeRunner#_get_current_bucket().
  eSurgeRunner._req_num_tokens_cpu: eSurgeRunner#_req_num_tokens_cpu.
  eSurgeRunner.infer_req_shard: eSurgeRunner#infer_req_shard().
  _AsyncExecutionHandle._finalize: _AsyncExecutionHandle#_finalize.
  eSurgeRunner.enable_sampler_metrics: eSurgeRunner#enable_sampler_metrics.
  eSurgeRunner._perf_iteration: eSurgeRunner#_perf_iteration.
  eSurgeRunner.__init__: eSurgeRunner#__init__().
  eSurgeRunner.num_reqs_most_model_len: eSurgeRunner#num_reqs_most_model_len.
  eSurgeRunner.arange: eSurgeRunner#arange.
  eSurgeRunner.arange_np: eSurgeRunner#arange_np.
  RunnerPerfSample: RunnerPerfSample#
  eSurgeRunner._perf_alpha: eSurgeRunner#_perf_alpha.
  eSurgeRunner._setup_variables: eSurgeRunner#_setup_variables().
  eSurgeRunner._vlm_cpu_buffers: eSurgeRunner#_vlm_cpu_buffers.
  eSurgeRunner._fmt_bucket: eSurgeRunner#_fmt_bucket().
  RunnerPerfSample.iteration: RunnerPerfSample#iteration.
  RunnerPerfSample.total_tokens: RunnerPerfSample#total_tokens.
  RunnerPerfSample.num_scheduled_reqs: RunnerPerfSample#num_scheduled_reqs.
  RunnerPerfSample.num_new: RunnerPerfSample#num_new.
  RunnerPerfSample.num_cached: RunnerPerfSample#num_cached.
  RunnerPerfSample.num_finished: RunnerPerfSample#num_finished.
  RunnerPerfSample.total_time: RunnerPerfSample#total_time.
  RunnerPerfSample.agg_tps: RunnerPerfSample#agg_tps.
  RunnerPerfSample.req_tps: RunnerPerfSample#req_tps.
  RunnerPerfSample.ema_tps: RunnerPerfSample#ema_tps.
  _AsyncExecutionHandle._model_runner_output: _AsyncExecutionHandle#_model_runner_output.
  _AsyncExecutionHandle._windows: _AsyncExecutionHandle#_windows.
  eSurgeRunner.max_num_batched_tokens: eSurgeRunner#max_num_batched_tokens.
  eSurgeRunner.enable_window_aware_runtime_cap: eSurgeRunner#enable_window_aware_runtime_cap.
  eSurgeRunner.max_num_seqs: eSurgeRunner#max_num_seqs.
  eSurgeRunner.async_scheduling: eSurgeRunner#async_scheduling.
  eSurgeRunner._perf_last_agg_tps: eSurgeRunner#_perf_last_agg_tps.
  eSurgeRunner._perf_last_req_tps: eSurgeRunner#_perf_last_req_tps.
  eSurgeRunner._perf_last_total_time: eSurgeRunner#_perf_last_total_time.
  eSurgeRunner._perf_last_total_tokens: eSurgeRunner#_perf_last_total_tokens.
  eSurgeRunner._get_request_paddings: eSurgeRunner#_get_request_paddings().
  eSurgeRunner._clamp_request_buckets_to_runtime_cap: eSurgeRunner#_clamp_request_buckets_to_runtime_cap().
  _get_padded_num_reqs_with_upper_limit: _get_padded_num_reqs_with_upper_limit().
  eSurgeRunner.enable_overlap_execution: eSurgeRunner#enable_overlap_execution.
---
# Module: [`easydel/inference/esurge/runners/model_runner.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py)

## Classes
### `RunnerPerfSample`
- def: [`easydel/inference/esurge/runners/model_runner.py:104`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L104)
- signature: `class RunnerPerfSample:`
- members:
  - `agg_tps` — [`L112`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L112)
  - `ema_tps` — [`L114`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L114)
  - `iteration` — [`L105`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L105)
  - `num_cached` — [`L109`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L109)
  - `num_finished` — [`L110`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L110)
  - `num_new` — [`L108`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L108)
  - `num_scheduled_reqs` — [`L107`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L107)
  - `req_tps` — [`L113`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L113)
  - `total_time` — [`L111`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L111)
  - `total_tokens` — [`L106`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L106)
- used by: [`_execute_model_impl`](model_runner.md#eSurgeRunner._execute_model_impl), [`_perf_history`](model_runner.md#eSurgeRunner._perf_history)

### `_AsyncExecutionHandle`
- def: [`easydel/inference/esurge/runners/model_runner.py:117`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L117)
- doc: Deferred host-materialized model output for overlap execution.
- signature: `class _AsyncExecutionHandle:`
- members:
  - `get_output(self)` — [`L131`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L131)
- protocol/private: `__init__`[`L120`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L120), `_finalize`[`L128`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L128), `_model_runner_output`[`L126`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L126), `_resolved_output`[`L129`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L129), `_windows`[`L127`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L127)
- uses (calls/refs, reference-scoped): [`ModelRunnerOutput`](../outputs.md#ModelRunnerOutput), [`AsyncWindowResult`](async_types.md#AsyncWindowResult), [`sampled_token_ids`](../outputs.md#ModelRunnerOutput.sampled_token_ids), [`token_logprobs`](../outputs.md#ModelRunnerOutput.token_logprobs), [`row_positions`](async_types.md#AsyncWindowResult.row_positions), [`sampled_token_ids`](async_types.md#AsyncWindowResult.sampled_token_ids), [`token_logprobs`](async_types.md#AsyncWindowResult.token_logprobs), [`valid_mask`](async_types.md#AsyncWindowResult.valid_mask), [`req_ids`](async_types.md#AsyncWindowResult.req_ids)
- used by: [`_execute_model_impl`](model_runner.md#eSurgeRunner._execute_model_impl), [`execute_model_async`](model_runner.md#eSurgeRunner.execute_model_async), [`wait_for_execution`](model_runner.md#eSurgeRunner.wait_for_execution)

### `eSurgeRunner`
- def: [`easydel/inference/esurge/runners/model_runner.py:194`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L194)
- doc: High-performance model runner for efficient batched inference.
- signature: `class eSurgeRunner:`
- members:
  - `__init__(self, model: EasyDeLBaseModule, hbm_utilization: float = 0.5, page_size: int = 128, max_model_len: int = 2 ** 13, max_num_batched_tokens: int | None = None, min_input_pad: int = 256, min_token_pad: int | None = None, max_num_seqs: int = 16, max_num_seq_buckets: list[int] | None = None, async_scheduling: bool = True, use_aot_forward: bool = True, bind_graphstate_for_aot: bool = False, verbose: bool = False, enable_overlap_execution: bool = True, enable_sampler_metrics: bool = False, enable_window_aware_runtime_cap: bool = False)` — [`L251`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L251) — Initialize the model runner.
  - `_apply_window_aware_runtime_cap(self, max_num_batched_tokens: int)` — [`L473`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L473) — Attach a hybrid full/sliding runtime-cap estimate to cache metadata.
  - `_build_kv_cache_groups(self)` — [`L397`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L397) — Build cache-group specs for runtime-cap and scheduler estimation.
  - `_clamp_request_buckets_to_runtime_cap(buckets: list[int], runtime_cap: int)` — [`L726`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L726) — Clamp request-count buckets to the runtime execution cap.
  - `_clear_window_aware_runtime_cap_metadata(self)` — [`L463`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L463) — Reset runtime-cap metadata to the default non-window-aware state.
  - `_collect_schedulable_window_rows(self, *, start_index: int, stop_index: int, scheduled_tokens_by_req: dict[str, int], allow_sparse_packing: bool)` — [`L1024`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L1024) — Collect runnable rows for a window, compacting interior zero-token gaps.
  - `_empty_sharding(self)` — [`L620`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L620) — Get empty sharding for replicated arrays.
  - `_execute_model_impl(self, scheduler_output: SchedulerOutput, *, return_async_output: bool = False)` — [`L1907`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L1907) — Execute the model on scheduled requests.
  - `_get_current_bucket(self, num_reqs: int)` — [`L708`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L708) — Select the smallest bucket that can accommodate num_reqs.
  - `_get_full_attention_page_table_index(self)` — [`L444`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L444) — Return the page table group index for the full-attention cache group.
  - `_get_request_paddings(min_bucket: int, max_bucket: int)` — [`L663`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L663) — Generate request count buckets using exponential growth.
  - `_get_token_paddings(min_token_size: int, max_token_size: int, padding_gap: int)` — [`L629`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L629) — Generate padding sizes for efficient compilation.
  - `_get_vlm_cpu_buffers(self, *, num_tokens_static: int, uses_mrope_model: bool)` — [`L831`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L831) — Get or create cached CPU buffers for VLM prefill data.
  - `_get_window_state_views(self, *, start_index: int, row_count: int, page_table_cpu: np.ndarray, page_table_version: int | None, row_indices: np.ndarray | None = None)` — [`L909`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L909) — Return CPU-side state views aligned to the active scheduler window.
  - `_init_seq_buckets(self, user_buckets: list[int] | None, max_num_seqs: int, min_input_pad: int)` — [`L684`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L684) — Initialize sequence count buckets for compilation.
  - `_log_startup_summary(self)` — [`L514`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L514) — Log a consolidated startup summary to the logger.
  - `_modify_prev_results(self)` — [`L1684`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L1684) — Apply previous iteration's tokens to sequence buffer.
  - `_precompile_jitted_helpers(self, reqs_padds: list[int], prompt_len_buckets: list[int], precompile_allowed_mask: bool = False, allowed_max: int = 512)` — [`L1096`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L1096) — Precompile JIT helper kernels for various input configurations.
  - `_precompute_vlm_prefill(self, req_state: CachedRequestState)` — [`L1361`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L1361) — Precompute prompt embeddings (+ optional mRoPE indices) for VLM requests.
  - `_reorder_decode_first(self, scheduler_output: SchedulerOutput)` — [`L1789`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L1789) — Reorder active requests so decode requests are placed first.
  - `_reorder_decode_first_per_shard(self, scheduler_output: SchedulerOutput, dp_size: int)` — [`L1829`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L1829) — Reorder decode requests first within each DP shard's row range.
  - `_setup_variables(self)` — [`L750`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L750) — Initialize internal variables and preallocate reusable buffers.
  - `_update_placeholder(self, discard_sampled_tokens_req_indices: list[int], request_seq_lens: list[tuple[int, int, CachedRequestState, int]])` — [`L1738`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L1738) — Set placeholders for tokens not yet generated.
  - `_update_states(self, scheduler_output: SchedulerOutput)` — [`L1433`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L1433) — Update internal states based on scheduler output.
  - `compile(self, *, max_num_batched_tokens: int | None = None)` — [`L1203`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L1203) — Compile the model for token/request bucket sizes.
  - `destroy_kv_cache(self)` — [`L1326`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L1326) — Destroy the current ragged KV cache to release memory.
  - `execute_model(self, scheduler_output: SchedulerOutput)` — [`L2566`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L2566) — Execute the model synchronously on scheduled requests.
  - `execute_model_async(self, scheduler_output: SchedulerOutput)` — [`L2587`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L2587) — Dispatch model work and defer the host-side token materialization.
  - `infer_req_shard(page_ids: tuple[list[int], ...])` — [`L1478`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L1478) — Infer the DP shard index that owns a request based on its page IDs.
  - `initialize_async_executor(self)` — [`L2598`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L2598) — Retained for API compatibility.
  - `initialize_kv_cache(self)` — [`L1331`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L1331) — Reinitialize the ragged KV cache if it has been destroyed.
  - `mesh(self)` — [`L611`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L611) — Get the JAX sharding mesh from the model.
  - `release_model_state(self, *, clear_compiled_cache: bool = False)` — [`L1298`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L1298) — Drop model/graph references held by the runner to free memory.
  - `reset_state(self)` — [`L2611`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L2611) — Clear sequence state and request bookkeeping.
  - `shutdown(self)` — [`L2638`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L2638) — Cleanup resources including async executor if present.
  - `update_model_weights(self, model: EasyDeLBaseModule | None = None, *, graphdef=None, graphstate=None, graphother=None, reset_state: bool = True)` — [`L1249`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L1249) — Update the runner's model weights/graphs and optionally reset state.
  - `wait_for_execution(self, future: Future | _AsyncExecutionHandle)` — [`L2621`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L2621) — Wait for an async execution to complete and return the result.
  - `active_mask_full_buf` — [`L799`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L799)
  - `active_num_seq_buckets` — [`L764`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L764)
  - `arange` — [`L791`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L791)
  - `arange_np` — [`L792`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L792)
  - `async_scheduling` — [`L350`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L350)
  - `enable_overlap_execution` — [`L378`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L378)
  - `enable_sampler_metrics` — [`L379`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L379)
  - `enable_window_aware_runtime_cap` — [`L343`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L343)
  - `executor_manager` — [`L365`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L365)
  - `input_ids_buf` — [`L794`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L794)
  - `kv_cache_groups` — [`L345`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L345)
  - `log_it` — [`L376`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L376)
  - `max_model_len` — [`L344`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L344)
  - `max_num_batched_tokens` — [`L338`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L338)
  - `max_num_reqs` — [`L349`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L349)
  - `max_num_seq_buckets` — [`L347`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L347)
  - `max_num_seqs` — [`L348`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L348)
  - `max_num_tokens` — [`L362`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L362)
  - `max_pages_per_req` — [`L353`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L353)
  - `metadata` — [`L327`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L327)
  - `min_input_pad` — [`L351`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L351)
  - `model` — [`L296`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L296)
  - `num_reqs_max_model_len` — [`L759`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L759)
  - `num_reqs_most_model_len` — [`L760`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L760)
  - `num_tokens_paddings` — [`L357`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L357)
  - `num_tokens_paddings_arr` — [`L796`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L796)
  - `page_size` — [`L352`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L352)
  - `position_ids_buf` — [`L795`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L795)
  - `req_num_tokens_full_buf` — [`L798`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L798)
  - `requests` — [`L768`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L768)
  - `scheduled_full_buf` — [`L797`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L797)
  - `sequence_buffer` — [`L782`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L782)
  - `window_aware_runtime_estimate` — [`L346`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L346)
- protocol/private: `_active_mask_full_cpu`[`L803`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L803), `_allow_sparse_window_packing`[`L761`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L761), `_executor`[`L393`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L393), `_finalize_sync_runner_state`[`L2414`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L2414), `_find_reuse_index_in_shard`[`L1607`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L1607), `_fmt_bucket`[`L2488`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L2488), `_perf_alpha`[`L384`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L384), `_perf_history`[`L389`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L389), `_perf_iteration`[`L382`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L382), `_perf_last_agg_tps`[`L385`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L385), `_perf_last_req_tps`[`L386`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L386), `_perf_last_total_time`[`L387`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L387), `_perf_last_total_tokens`[`L388`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L388), `_perf_tps_ema`[`L383`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L383), `_pre_async_results`[`L392`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L392), `_req_num_tokens_cpu`[`L804`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L804), `_scheduled_full_cpu`[`L802`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L802), `_vlm_cpu_buffers`[`L820`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L820), `_window_frequency_penalties_cpu`[`L809`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L809), `_window_min_p_cpu`[`L808`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L808), `_window_presence_penalties_cpu`[`L810`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L810), `_window_repetition_penalties_cpu`[`L811`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L811), `_window_row_indices_cpu`[`L812`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L812), `_window_temperature_cpu`[`L805`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L805), `_window_top_k_cpu`[`L807`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L807), `_window_top_p_cpu`[`L806`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L806)
- uses (calls/refs, reference-scoped): [`config`](../../../infra/base_module.md#EasyDeLBaseModule.config), [`EasyDeLBaseModule`](../../../infra/base_module.md#EasyDeLBaseModule), [`remove_request`](sequence_buffer.md#SequenceBuffer.remove_request), [`clear`](sequence_buffer.md#SequenceBuffer.clear), [`execute`](execution_manager.md#ExecutionManager.execute), [`RaggedPagesCacheConfig`](../../../caching/ragged_page/cache.md#RaggedPagesCacheConfig), [`init_operations_cache`](../../../infra/mixins/generation.md#EasyGenerationMixin.init_operations_cache), [`swap_states`](sequence_buffer.md#SequenceBuffer.swap_states), [`create_kv_cache_specs_from_config`](../core/interface.md#create_kv_cache_specs_from_config), [`mesh`](../../../infra/base_module.md#EasyDeLBaseModule.mesh), [`add_request`](sequence_buffer.md#SequenceBuffer.add_request), [`_model_executor`](execution_manager.md#ExecutionManager._model_executor), [`logger`](model_runner.md#logger), [`update_graphs`](execution_manager.md#ExecutionManager.update_graphs), [`UnifiedAttentionCacheConfig`](../../../caching/unified_attention/cache.md#UnifiedAttentionCacheConfig), [`get_output`](model_runner.md#_AsyncExecutionHandle.get_output), [`estimate_runtime_page_budget`](../core/interface.md#estimate_runtime_page_budget), [`compile`](execution_manager.md#ExecutionManager.compile), [`SchedulerOutput`](../scheduler/output.md#SchedulerOutput), [`kv_pages`](execution_manager.md#ExecutionManager.kv_pages), [`record_runner_metrics`](../metrics.md#MetricsCollector.record_runner_metrics), [`create_unified_attention_cache_config`](../../../infra/mixins/generation.md#EasyGenerationMixin.create_unified_attention_cache_config), [`get_metrics_collector`](../metrics.md#get_metrics_collector), [`_sampler_executor`](execution_manager.md#ExecutionManager._sampler_executor), [`create_ragged_page_cache_config`](../../../infra/mixins/generation.md#EasyGenerationMixin.create_ragged_page_cache_config), [`clear_recurrent_slots`](execution_manager.md#ExecutionManager.clear_recurrent_slots), [`esurge_compatible_model`](../../../infra/mixins/generation.md#EasyGenerationMixin.esurge_compatible_model), [`get_operations_cache_info`](../../../infra/mixins/operation_cache.md#OperationCacheMixin.get_operations_cache_info), [`num_computed_tokens`](sequence_buffer.md#SequenceBuffer.num_computed_tokens), [`kv_cache_spec`](../core/interface.md#CacheGroupSpec.kv_cache_spec), [`token_ids`](sequence_buffer.md#SequenceBuffer.token_ids), [`compute_embedding_with_info`](../../../infra/base_module.md#EasyDeLBaseModule.compute_embedding_with_info), [`TurboQuantConfig`](../../../layers/quantization/_turboquant.md#TurboQuantConfig), [`layers`](../../../infra/mixins/operation_cache.md#OperationsCacheInfo.layers), [`clear_vision_data`](states.md#CachedRequestState.clear_vision_data), [`top_k`](sequence_buffer.md#SequenceBuffer.top_k), [`min_p`](sequence_buffer.md#SequenceBuffer.min_p), [`pages_per_dp_shard`](../core/dp_sharding.md#pages_per_dp_shard), [`req_id_to_index`](sequence_buffer.md#SequenceBuffer.req_id_to_index), [`swap_rows`](sequence_buffer.md#swap_rows)  (+129 more)
- used by: [`_import_structure`](../../../__init__.md#_import_structure), [`from_runner`](../scheduler/scheduler.md#Scheduler.from_runner), [`__del__`](../esurge_engine.md#eSurge.__del__), [`runner`](../esurge_engine.md#eSurge.runner), [`_distributed_execute_step`](../esurge_engine.md#eSurge._distributed_execute_step), [`_overlap_execution`](../esurge_engine.md#eSurge._overlap_execution)

## Functions
- `_get_padded_num_reqs_with_upper_limit(x: int, upper_limit: int, min_input_pad: int)` — [`L169`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L169) — Calculate padded request count for compilation efficiency.

## Module values
- `logger` — [`L100`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/model_runner.py#L100)

